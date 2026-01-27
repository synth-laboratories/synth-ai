//! Shared DTOs for Synth core and SDKs.

use serde::{Deserialize, Serialize};
use serde_json::Value;

/// Core event structure returned by the Rust core.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CoreEvent {
    /// Sequence number (monotonic, if provided by backend).
    pub seq: i64,
    /// Event type string.
    #[serde(rename = "type")]
    pub event_type: String,
    /// Optional human-readable message.
    pub message: Option<String>,
    /// Event payload (backend-specific).
    pub data_json: Value,
    /// Optional timestamp string.
    pub ts: Option<String>,
}

/// Request for polling events.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EventPollRequest {
    pub kind: String,
    pub job_id: String,
    pub since_seq: Option<i64>,
    pub limit: Option<usize>,
}

/// Response for polling events.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EventPollResponse {
    pub events: Vec<CoreEvent>,
    pub next_seq: Option<i64>,
    pub has_more: Option<bool>,
}

/// Backend error envelope.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BackendError {
    pub status: u16,
    pub code: Option<String>,
    pub message: String,
    pub retryable: bool,
}

/// Tunnel creation request (placeholder for core bindings).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TunnelCreate {
    pub local_port: u16,
    pub backend: String,
    pub backend_url: Option<String>,
    pub api_key: Option<String>,
}

/// Tunnel status response (placeholder for core bindings).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TunnelStatus {
    pub public_url: String,
    pub local_url: String,
    pub backend: String,
    pub lease_id: Option<String>,
    pub process_id: Option<u32>,
    pub started_at: Option<String>,
}

/// Tunnel lifecycle event (placeholder for core bindings).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TunnelEvent {
    pub status: String,
    pub message: Option<String>,
    pub ts: Option<String>,
}
