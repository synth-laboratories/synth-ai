//! Streaming types for job event streams.
//!
//! Core types for stream messages and stream type identification.

use serde::{Deserialize, Serialize};
use serde_json::Value;

/// Type of stream data.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum StreamType {
    /// Job status updates.
    Status,
    /// Job events (progress, errors, etc.).
    Events,
    /// Metrics (loss curves, reward, etc.).
    Metrics,
    /// Timeline events.
    Timeline,
}

impl StreamType {
    /// Get the endpoint suffix for this stream type.
    pub fn endpoint_suffix(&self) -> &'static str {
        match self {
            StreamType::Status => "",
            StreamType::Events => "/events",
            StreamType::Metrics => "/metrics",
            StreamType::Timeline => "/timeline",
        }
    }

    /// Get all stream types.
    pub fn all() -> Vec<StreamType> {
        vec![
            StreamType::Status,
            StreamType::Events,
            StreamType::Metrics,
            StreamType::Timeline,
        ]
    }
}

impl Default for StreamType {
    fn default() -> Self {
        Self::Events
    }
}

/// A message from a job stream.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StreamMessage {
    /// Type of stream this message came from.
    pub stream_type: StreamType,
    /// ISO 8601 timestamp.
    pub timestamp: String,
    /// Job ID this message belongs to.
    pub job_id: String,
    /// Message payload.
    pub data: Value,
    /// Sequence number (for events).
    #[serde(default)]
    pub seq: Option<i64>,
    /// Step/iteration number (for metrics).
    #[serde(default)]
    pub step: Option<i64>,
    /// Phase name (for timeline).
    #[serde(default)]
    pub phase: Option<String>,
    /// Event type string (for events).
    #[serde(default)]
    pub event_type: Option<String>,
    /// Log level (for events).
    #[serde(default)]
    pub level: Option<String>,
}

impl StreamMessage {
    /// Create a new stream message.
    pub fn new(stream_type: StreamType, job_id: impl Into<String>, data: Value) -> Self {
        Self {
            stream_type,
            timestamp: chrono::Utc::now().to_rfc3339(),
            job_id: job_id.into(),
            data,
            seq: None,
            step: None,
            phase: None,
            event_type: None,
            level: None,
        }
    }

    /// Create a status message.
    pub fn status(job_id: impl Into<String>, data: Value) -> Self {
        Self::new(StreamType::Status, job_id, data)
    }

    /// Create an event message.
    pub fn event(job_id: impl Into<String>, data: Value, seq: i64) -> Self {
        let mut msg = Self::new(StreamType::Events, job_id, data.clone());
        msg.seq = Some(seq);
        msg.event_type = data.get("type").and_then(|v| v.as_str()).map(String::from);
        msg.level = data.get("level").and_then(|v| v.as_str()).map(String::from);
        msg
    }

    /// Create a metrics message.
    pub fn metrics(job_id: impl Into<String>, data: Value, step: i64) -> Self {
        let mut msg = Self::new(StreamType::Metrics, job_id, data);
        msg.step = Some(step);
        msg
    }

    /// Create a timeline message.
    pub fn timeline(job_id: impl Into<String>, phase: impl Into<String>, data: Value) -> Self {
        let mut msg = Self::new(StreamType::Timeline, job_id, data);
        msg.phase = Some(phase.into());
        msg
    }

    /// Generate a unique key for deduplication.
    pub fn key(&self) -> String {
        match self.stream_type {
            StreamType::Events => {
                format!("event:{}", self.seq.unwrap_or(0))
            }
            StreamType::Metrics => {
                let name = self
                    .data
                    .get("name")
                    .and_then(|v| v.as_str())
                    .unwrap_or("unknown");
                format!("metric:{}:{}", name, self.step.unwrap_or(0))
            }
            StreamType::Timeline => {
                format!(
                    "timeline:{}:{}",
                    self.phase.as_deref().unwrap_or(""),
                    self.timestamp
                )
            }
            StreamType::Status => {
                let status = self
                    .data
                    .get("status")
                    .and_then(|v| v.as_str())
                    .unwrap_or("unknown");
                format!("status:{}:{}", status, self.timestamp)
            }
        }
    }

    /// Check if this is an error event.
    pub fn is_error(&self) -> bool {
        self.level.as_deref() == Some("error")
    }

    /// Check if this is a warning event.
    pub fn is_warning(&self) -> bool {
        self.level.as_deref() == Some("warning")
    }

    /// Get the message text if present.
    pub fn message(&self) -> Option<&str> {
        self.data.get("message").and_then(|v| v.as_str())
    }

    /// Get a field from the data payload.
    pub fn get(&self, key: &str) -> Option<&Value> {
        self.data.get(key)
    }

    /// Get a string field from the data payload.
    pub fn get_str(&self, key: &str) -> Option<&str> {
        self.data.get(key).and_then(|v| v.as_str())
    }

    /// Get an i64 field from the data payload.
    pub fn get_i64(&self, key: &str) -> Option<i64> {
        self.data.get(key).and_then(|v| v.as_i64())
    }

    /// Get an f64 field from the data payload.
    pub fn get_f64(&self, key: &str) -> Option<f64> {
        self.data.get(key).and_then(|v| v.as_f64())
    }
}

impl Default for StreamMessage {
    fn default() -> Self {
        Self::new(StreamType::Events, "", Value::Null)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_stream_type_suffix() {
        assert_eq!(StreamType::Status.endpoint_suffix(), "");
        assert_eq!(StreamType::Events.endpoint_suffix(), "/events");
        assert_eq!(StreamType::Metrics.endpoint_suffix(), "/metrics");
        assert_eq!(StreamType::Timeline.endpoint_suffix(), "/timeline");
    }

    #[test]
    fn test_stream_message_creation() {
        let msg = StreamMessage::status("job-123", serde_json::json!({"status": "running"}));

        assert_eq!(msg.stream_type, StreamType::Status);
        assert_eq!(msg.job_id, "job-123");
        assert_eq!(msg.get_str("status"), Some("running"));
    }

    #[test]
    fn test_event_message() {
        let data = serde_json::json!({
            "type": "progress",
            "level": "info",
            "message": "Step completed"
        });
        let msg = StreamMessage::event("job-123", data, 42);

        assert_eq!(msg.stream_type, StreamType::Events);
        assert_eq!(msg.seq, Some(42));
        assert_eq!(msg.event_type, Some("progress".to_string()));
        assert_eq!(msg.level, Some("info".to_string()));
    }

    #[test]
    fn test_dedup_key() {
        let msg1 = StreamMessage::event("job-1", serde_json::json!({}), 1);
        let msg2 = StreamMessage::event("job-1", serde_json::json!({}), 1);
        let msg3 = StreamMessage::event("job-1", serde_json::json!({}), 2);

        assert_eq!(msg1.key(), msg2.key());
        assert_ne!(msg1.key(), msg3.key());
    }

    #[test]
    fn test_error_detection() {
        let error_msg = StreamMessage {
            level: Some("error".to_string()),
            ..StreamMessage::default()
        };
        let warning_msg = StreamMessage {
            level: Some("warning".to_string()),
            ..StreamMessage::default()
        };
        let info_msg = StreamMessage {
            level: Some("info".to_string()),
            ..StreamMessage::default()
        };

        assert!(error_msg.is_error());
        assert!(!error_msg.is_warning());
        assert!(warning_msg.is_warning());
        assert!(!info_msg.is_error());
    }

    #[test]
    fn test_serde() {
        let msg = StreamMessage::status("job-123", serde_json::json!({"status": "running"}));

        let json = serde_json::to_string(&msg).unwrap();
        let parsed: StreamMessage = serde_json::from_str(&json).unwrap();

        assert_eq!(parsed.job_id, "job-123");
        assert_eq!(parsed.stream_type, StreamType::Status);
    }
}
