//! Event streaming for optimization jobs.
//!
//! This module provides SSE-style event streaming with deduplication
//! for optimization jobs.

use std::collections::HashSet;
use std::time::{Duration, Instant};

use serde_json::Value;

use crate::errors::CoreError;
use crate::http::HttpClient;

use super::events::{EventParser, ParsedEvent};

/// Event stream for polling job events.
pub struct EventStream {
    /// HTTP client reference
    client: HttpClient,
    /// Job ID to stream events for
    job_id: String,
    /// Base URL for API
    base_url: String,
    /// Last seen sequence number
    last_seq: i64,
    /// Whether to deduplicate events
    deduplicate: bool,
    /// Set of seen sequence numbers
    seen_seqs: HashSet<i64>,
    /// Maximum events per poll
    max_events_per_poll: i32,
}

impl EventStream {
    /// Create a new event stream for a job.
    pub fn new(client: HttpClient, base_url: &str, job_id: &str) -> Self {
        Self {
            client,
            job_id: job_id.to_string(),
            base_url: base_url.trim_end_matches('/').to_string(),
            last_seq: 0,
            deduplicate: true,
            seen_seqs: HashSet::new(),
            max_events_per_poll: 500,
        }
    }

    /// Set the starting sequence number.
    pub fn with_start_seq(mut self, seq: i64) -> Self {
        self.last_seq = seq;
        self
    }

    /// Enable or disable deduplication.
    pub fn with_deduplicate(mut self, dedupe: bool) -> Self {
        self.deduplicate = dedupe;
        self
    }

    /// Set max events per poll.
    pub fn with_max_events(mut self, max: i32) -> Self {
        self.max_events_per_poll = max;
        self
    }

    /// Get the last seen sequence number.
    pub fn last_seq(&self) -> i64 {
        self.last_seq
    }

    /// Poll for new events.
    ///
    /// Returns events since the last sequence number.
    pub async fn poll_events(&mut self) -> Result<Vec<ParsedEvent>, CoreError> {
        let url = format!(
            "{}/api/policy-optimization/online/jobs/{}/events",
            self.base_url, self.job_id
        );

        let params = [
            ("since_seq", self.last_seq.to_string()),
            ("limit", self.max_events_per_poll.to_string()),
        ];

        let params_slice: &[(&str, &str)] = &[
            ("since_seq", &params[0].1),
            ("limit", &params[1].1),
        ];

        let response: Value = self
            .client
            .get(&url, Some(params_slice))
            .await
            .map_err(|e| CoreError::Internal(format!("failed to fetch events: {}", e)))?;

        // Parse events array
        let events_array = response
            .get("events")
            .and_then(|v| v.as_array())
            .cloned()
            .unwrap_or_default();

        let mut parsed_events = Vec::new();

        for event_value in events_array {
            let parsed = EventParser::parse(&event_value);

            // Update last_seq
            if let Some(seq) = parsed.seq {
                if seq > self.last_seq {
                    self.last_seq = seq;
                }

                // Deduplication
                if self.deduplicate {
                    if self.seen_seqs.contains(&seq) {
                        continue;
                    }
                    self.seen_seqs.insert(seq);

                    // Limit seen_seqs size to prevent memory growth
                    if self.seen_seqs.len() > 10000 {
                        // Keep only recent sequences
                        let threshold = self.last_seq - 5000;
                        self.seen_seqs.retain(|&s| s > threshold);
                    }
                }
            }

            parsed_events.push(parsed);
        }

        Ok(parsed_events)
    }

    /// Stream events until a terminal condition with callback.
    ///
    /// # Arguments
    ///
    /// * `on_event` - Callback for each event
    /// * `timeout` - Maximum time to stream
    /// * `poll_interval` - Time between polls
    /// * `is_terminal` - Optional check for terminal status
    pub async fn stream_until<F, T>(
        &mut self,
        mut on_event: F,
        timeout: Duration,
        poll_interval: Duration,
        mut is_terminal: T,
    ) -> Result<(), CoreError>
    where
        F: FnMut(&ParsedEvent),
        T: FnMut() -> bool,
    {
        let start = Instant::now();
        let mut last_event_time = Instant::now();

        loop {
            // Check timeout
            if start.elapsed() > timeout {
                return Err(CoreError::Timeout(format!(
                    "event stream timed out after {:.0} seconds",
                    timeout.as_secs_f64()
                )));
            }

            // Check terminal condition
            if is_terminal() {
                return Ok(());
            }

            // Poll events
            match self.poll_events().await {
                Ok(events) => {
                    if !events.is_empty() {
                        last_event_time = Instant::now();
                    }

                    for event in &events {
                        on_event(event);

                        // Check for terminal events
                        if event.category.is_terminal() {
                            return Ok(());
                        }
                    }
                }
                Err(e) => {
                    // Log error but continue streaming
                    // Allow some grace period for transient errors
                    if last_event_time.elapsed() > Duration::from_secs(120) {
                        return Err(e);
                    }
                }
            }

            // Wait before next poll
            tokio::time::sleep(poll_interval).await;
        }
    }
}

/// Stream configuration.
#[derive(Debug, Clone)]
pub struct StreamConfig {
    /// Poll interval in seconds
    pub poll_interval_secs: f64,
    /// Maximum events per poll
    pub max_events_per_poll: i32,
    /// Whether to deduplicate events
    pub deduplicate: bool,
    /// Timeout in seconds
    pub timeout_secs: f64,
}

impl Default for StreamConfig {
    fn default() -> Self {
        Self {
            poll_interval_secs: 5.0,
            max_events_per_poll: 500,
            deduplicate: true,
            timeout_secs: 3600.0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_stream_config_default() {
        let config = StreamConfig::default();
        assert_eq!(config.poll_interval_secs, 5.0);
        assert_eq!(config.max_events_per_poll, 500);
        assert!(config.deduplicate);
    }
}
