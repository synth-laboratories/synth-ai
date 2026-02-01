//! Job streamer for polling and streaming job events.

use super::{
    config::StreamConfig,
    endpoints::StreamEndpoints,
    handler::StreamHandler,
    types::{StreamMessage, StreamType},
};
use crate::errors::CoreError;
use crate::http::HttpClient;
use serde_json::Value;
use std::collections::HashSet;
use std::sync::Arc;

/// Default timeout in seconds for streaming requests.
const DEFAULT_TIMEOUT_SECS: u64 = 60;

/// Terminal job statuses.
const TERMINAL_STATUSES: &[&str] = &[
    "succeeded",
    "failed",
    "cancelled",
    "canceled",
    "completed",
    "error",
];

/// Job streamer that polls endpoints and dispatches to handlers.
pub struct JobStreamer {
    base_url: String,
    api_key: String,
    job_id: String,
    endpoints: StreamEndpoints,
    config: StreamConfig,
    handlers: Vec<Arc<dyn StreamHandler>>,
    seen_messages: HashSet<String>,
    last_event_seq: Option<i64>,
}

impl JobStreamer {
    /// Create a new job streamer.
    pub fn new(
        base_url: impl Into<String>,
        api_key: impl Into<String>,
        job_id: impl Into<String>,
    ) -> Self {
        let job_id = job_id.into();
        Self {
            base_url: base_url.into().trim_end_matches('/').to_string(),
            api_key: api_key.into(),
            job_id: job_id.clone(),
            endpoints: StreamEndpoints::learning(&job_id),
            config: StreamConfig::default(),
            handlers: vec![],
            seen_messages: HashSet::new(),
            last_event_seq: None,
        }
    }

    /// Set the stream endpoints.
    pub fn with_endpoints(mut self, endpoints: StreamEndpoints) -> Self {
        self.endpoints = endpoints;
        self
    }

    /// Set the stream config.
    pub fn with_config(mut self, config: StreamConfig) -> Self {
        self.config = config;
        self
    }

    /// Add a handler.
    pub fn with_handler(mut self, handler: Arc<dyn StreamHandler>) -> Self {
        self.handlers.push(handler);
        self
    }

    /// Add a handler (convenience method).
    pub fn add_handler<H: StreamHandler + 'static>(&mut self, handler: H) {
        self.handlers.push(Arc::new(handler));
    }

    /// Poll status once.
    pub async fn poll_status(&mut self) -> Result<Option<Value>, CoreError> {
        let client = self.create_client()?;

        for endpoint in self.endpoints.all_status_endpoints() {
            match client.get::<Value>(endpoint, None).await {
                Ok(status) => {
                    self.dispatch_status(&status);
                    return Ok(Some(status));
                }
                Err(e) => {
                    // Check if it's a 404 - try next fallback
                    if let Some(404) = e.status() {
                        continue;
                    }
                    return Err(e.into());
                }
            }
        }

        Ok(None)
    }

    /// Poll events once.
    pub async fn poll_events(&mut self) -> Result<Vec<StreamMessage>, CoreError> {
        if !self.config.is_stream_enabled(StreamType::Events) {
            return Ok(vec![]);
        }

        let client = self.create_client()?;
        let mut all_messages = vec![];
        let mut total_events: usize = 0;

        for endpoint in self.endpoints.all_event_endpoints() {
            // Add since_seq parameter if we have one
            let url = if let Some(seq) = self.last_event_seq {
                format!("{}?since_seq={}", endpoint, seq)
            } else {
                endpoint.to_string()
            };

            match client.get::<Value>(&url, None).await {
                Ok(response) => {
                    let events_list = response
                        .get("events")
                        .and_then(|v| v.as_array())
                        .or_else(|| response.as_array());
                    if let Some(events) = events_list {
                        for event in events {
                            if self.config.should_include_event(event) {
                                let seq = event.get("seq").and_then(|v| v.as_i64());
                                let msg = StreamMessage::event(
                                    &self.job_id,
                                    event.clone(),
                                    seq.unwrap_or(0),
                                );

                                // Update last seen seq
                                if let Some(s) = seq {
                                    self.last_event_seq =
                                        Some(self.last_event_seq.map(|l| l.max(s)).unwrap_or(s));
                                }

                                self.dispatch_message(&msg);
                                all_messages.push(msg);
                                total_events += 1;
                                if let Some(max_events) = self.config.max_events_per_poll {
                                    if total_events >= max_events {
                                        return Ok(all_messages);
                                    }
                                }
                            }
                        }
                    }
                    break; // Success, don't try fallbacks
                }
                Err(e) => {
                    if let Some(404) = e.status() {
                        continue; // Try next fallback
                    }
                    return Err(e.into());
                }
            }
        }

        Ok(all_messages)
    }

    /// Poll metrics once.
    pub async fn poll_metrics(&mut self) -> Result<Vec<StreamMessage>, CoreError> {
        if !self.config.is_stream_enabled(StreamType::Metrics) {
            return Ok(vec![]);
        }

        let client = self.create_client()?;
        let mut all_messages = vec![];

        let metric_endpoints = self.endpoints.all_metric_endpoints();
        if metric_endpoints.is_empty() {
            return Ok(all_messages);
        }

        for endpoint in metric_endpoints {
            match client.get::<Value>(endpoint, None).await {
                Ok(response) => {
                    let mut metrics: Option<&Vec<Value>> = None;
                    if let Some(items) = response.get("points").and_then(|v| v.as_array()) {
                        metrics = Some(items);
                    } else if let Some(items) = response.get("metrics").and_then(|v| v.as_array()) {
                        metrics = Some(items);
                    } else if let Some(items) = response.as_array() {
                        metrics = Some(items);
                    }

                    if let Some(metrics) = metrics {
                        for metric in metrics {
                            if self.config.should_include_metric(metric) {
                                let step = metric.get("step").and_then(|v| v.as_i64()).unwrap_or(0);
                                let msg =
                                    StreamMessage::metrics(&self.job_id, metric.clone(), step);
                                self.dispatch_message(&msg);
                                all_messages.push(msg);
                            }
                        }
                    }
                    break; // Success, don't try fallbacks
                }
                Err(e) => {
                    if let Some(404) = e.status() {
                        continue;
                    }
                    return Err(e.into());
                }
            }
        }

        Ok(all_messages)
    }

    /// Poll timeline once.
    pub async fn poll_timeline(&mut self) -> Result<Vec<StreamMessage>, CoreError> {
        if !self.config.is_stream_enabled(StreamType::Timeline) {
            return Ok(vec![]);
        }

        let client = self.create_client()?;
        let mut all_messages = vec![];
        let timeline_endpoints = self.endpoints.all_timeline_endpoints();
        if timeline_endpoints.is_empty() {
            return Ok(all_messages);
        }

        for endpoint in timeline_endpoints {
            match client.get::<Value>(endpoint, None).await {
                Ok(response) => {
                    let mut entries: Option<&Vec<Value>> = None;
                    if let Some(items) = response.get("events").and_then(|v| v.as_array()) {
                        entries = Some(items);
                    } else if let Some(items) = response.get("timeline").and_then(|v| v.as_array())
                    {
                        entries = Some(items);
                    } else if let Some(items) = response.as_array() {
                        entries = Some(items);
                    }

                    if let Some(entries) = entries {
                        for entry in entries {
                            if !self.config.should_include_timeline(entry) {
                                continue;
                            }
                            let phase = entry.get("phase").and_then(|v| v.as_str()).unwrap_or("");
                            let job_id = entry
                                .get("job_id")
                                .and_then(|v| v.as_str())
                                .unwrap_or(&self.job_id);
                            let msg = StreamMessage::timeline(job_id, phase, entry.clone());
                            self.dispatch_message(&msg);
                            all_messages.push(msg);
                        }
                    }
                    break; // Success, don't try fallbacks
                }
                Err(e) => {
                    if let Some(404) = e.status() {
                        continue;
                    }
                    return Err(e.into());
                }
            }
        }

        Ok(all_messages)
    }

    /// Stream until the job reaches a terminal state.
    pub async fn stream_until_terminal(&mut self) -> Result<Value, CoreError> {
        // Notify handlers of start
        for handler in &self.handlers {
            handler.on_start(&self.job_id);
        }

        loop {
            // Poll status
            if let Some(status) = self.poll_status().await? {
                if Self::is_terminal(&status) {
                    let final_status = status.get("status").and_then(|v| v.as_str());

                    // Notify handlers of end
                    for handler in &self.handlers {
                        handler.on_end(&self.job_id, final_status);
                        handler.flush();
                    }

                    return Ok(status);
                }
            }

            // Poll events
            let _ = self.poll_events().await?;

            // Poll metrics (less frequently)
            let _ = self.poll_metrics().await?;
            let _ = self.poll_timeline().await?;

            // Wait before next poll
            tokio::time::sleep(tokio::time::Duration::from_secs_f64(
                self.config.poll_interval_seconds,
            ))
            .await;
        }
    }

    /// Stream for a maximum duration, returning early if terminal.
    pub async fn stream_for_duration(
        &mut self,
        max_seconds: f64,
    ) -> Result<Option<Value>, CoreError> {
        let start = std::time::Instant::now();
        let max_duration = std::time::Duration::from_secs_f64(max_seconds);

        for handler in &self.handlers {
            handler.on_start(&self.job_id);
        }

        loop {
            if start.elapsed() >= max_duration {
                for handler in &self.handlers {
                    handler.on_end(&self.job_id, Some("timeout"));
                    handler.flush();
                }
                return Ok(None);
            }

            if let Some(status) = self.poll_status().await? {
                if Self::is_terminal(&status) {
                    let final_status = status.get("status").and_then(|v| v.as_str());
                    for handler in &self.handlers {
                        handler.on_end(&self.job_id, final_status);
                        handler.flush();
                    }
                    return Ok(Some(status));
                }
            }

            let _ = self.poll_events().await?;
            let _ = self.poll_metrics().await?;
            let _ = self.poll_timeline().await?;

            tokio::time::sleep(tokio::time::Duration::from_secs_f64(
                self.config.poll_interval_seconds,
            ))
            .await;
        }
    }

    fn create_client(&self) -> Result<HttpClient, CoreError> {
        HttpClient::new(&self.base_url, &self.api_key, DEFAULT_TIMEOUT_SECS)
            .map_err(|e| CoreError::Internal(format!("Failed to create HTTP client: {}", e)))
    }

    fn dispatch_status(&mut self, status: &Value) {
        let msg = StreamMessage::status(&self.job_id, status.clone());
        self.dispatch_message(&msg);
    }

    fn dispatch_message(&mut self, message: &StreamMessage) {
        // Deduplication
        if self.config.deduplicate {
            let key = message.key();
            if self.seen_messages.contains(&key) {
                return;
            }
            self.seen_messages.insert(key);
        }

        // Dispatch to handlers
        for handler in &self.handlers {
            if handler.should_handle(message) {
                handler.handle(message);
            }
        }
    }

    fn is_terminal(status: &Value) -> bool {
        status
            .get("status")
            .and_then(|v| v.as_str())
            .map(|s| TERMINAL_STATUSES.contains(&s))
            .unwrap_or(false)
    }

    /// Get the job ID.
    pub fn job_id(&self) -> &str {
        &self.job_id
    }

    /// Get the last event sequence number.
    pub fn last_event_seq(&self) -> Option<i64> {
        self.last_event_seq
    }

    /// Clear seen messages (for re-streaming).
    pub fn clear_seen(&mut self) {
        self.seen_messages.clear();
        self.last_event_seq = None;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_terminal_detection() {
        assert!(JobStreamer::is_terminal(
            &serde_json::json!({"status": "succeeded"})
        ));
        assert!(JobStreamer::is_terminal(
            &serde_json::json!({"status": "failed"})
        ));
        assert!(JobStreamer::is_terminal(
            &serde_json::json!({"status": "cancelled"})
        ));
        assert!(!JobStreamer::is_terminal(
            &serde_json::json!({"status": "running"})
        ));
        assert!(!JobStreamer::is_terminal(
            &serde_json::json!({"status": "pending"})
        ));
    }

    #[test]
    fn test_streamer_creation() {
        let streamer = JobStreamer::new("https://api.example.com", "sk-test", "job-123")
            .with_config(StreamConfig::minimal())
            .with_endpoints(StreamEndpoints::prompt_learning("job-123"));

        assert_eq!(streamer.job_id(), "job-123");
        assert!(streamer.last_event_seq().is_none());
    }

    #[test]
    fn test_clear_seen() {
        let mut streamer = JobStreamer::new("https://api.example.com", "sk-test", "job-123");

        streamer.seen_messages.insert("test".to_string());
        streamer.last_event_seq = Some(42);

        streamer.clear_seen();

        assert!(streamer.seen_messages.is_empty());
        assert!(streamer.last_event_seq.is_none());
    }
}
