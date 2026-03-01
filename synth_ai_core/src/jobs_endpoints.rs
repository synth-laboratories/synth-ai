//! Canonical job endpoint constants and event schema (Single Source of Truth).
//!
//! All endpoint paths and event schemas are defined here and exported to:
//! - `synth_ai_core::streaming::endpoints` (client-side streaming)
//! - `synth_ai_core::api::jobs` (client-side submission)
//! - Python SDK via `synth_ai_py` bindings
//! - `rust_backend` routing (server-side)
//!
//! # Endpoint Families
//!
//! Canonical: `/api/v1/offline/jobs/*`
//!
//! Route construction delegates to `api::routes` which owns the versioned
//! path segments. This module prepends `/api` as required by the backend.

use crate::api::routes::{self, ApiVersion};

// ---------------------------------------------------------------------------
// Canonical job endpoint family
// ---------------------------------------------------------------------------

/// Convenience: `/api` prefix + the versioned route.
fn api_prefix(route: &str) -> String {
    format!("/api{}", route)
}

/// Root for all canonical job endpoints.
pub const JOBS_ROOT: &str = "/api/v1/offline/jobs";

/// Create/submit endpoints by domain.
pub const JOBS_CREATE_GEPA: &str = "/api/v1/offline/jobs";
pub const JOBS_CREATE_MIPRO: &str = "/api/v1/offline/jobs";
pub const JOBS_CREATE_EVAL: &str = "/api/v1/offline/jobs";
pub const JOBS_CREATE_GRAPH: &str = "/api/v1/offline/jobs";
pub const JOBS_CREATE_VERIFIER: &str = "/api/v1/offline/jobs";

/// Format helpers for per-job endpoints.
pub fn jobs_status(job_id: &str) -> String {
    api_prefix(&routes::offline_job_path(job_id, ApiVersion::V1))
}

pub fn jobs_events(job_id: &str) -> String {
    api_prefix(&routes::offline_job_subpath(job_id, "events", ApiVersion::V1))
}

pub fn jobs_events_stream(job_id: &str) -> String {
    api_prefix(&routes::offline_job_subpath(job_id, "events/stream", ApiVersion::V1))
}

pub fn jobs_artifacts(job_id: &str) -> String {
    api_prefix(&routes::offline_job_subpath(job_id, "artifacts", ApiVersion::V1))
}

pub fn jobs_cancel(job_id: &str) -> String {
    jobs_status(job_id)
}

pub fn jobs_metrics(job_id: &str) -> String {
    api_prefix(&routes::offline_job_subpath(job_id, "metrics", ApiVersion::V1))
}

// ---------------------------------------------------------------------------
// Terminal verbs and canonical status mapping
// ---------------------------------------------------------------------------

/// Canonical terminal verbs. An event type ending with any of these indicates
/// the job has reached a final state.
pub const TERMINAL_VERB_COMPLETED: &str = "job.completed";
pub const TERMINAL_VERB_FAILED: &str = "job.failed";
pub const TERMINAL_VERB_CANCELLED: &str = "job.cancelled";

/// All terminal verb suffixes.
pub const TERMINAL_VERBS: &[&str] = &[
    TERMINAL_VERB_COMPLETED,
    TERMINAL_VERB_FAILED,
    TERMINAL_VERB_CANCELLED,
];

/// Check if an event type string is terminal by suffix matching.
pub fn is_terminal_event_type(event_type: &str) -> bool {
    let lower = event_type.to_lowercase();
    TERMINAL_VERBS.iter().any(|verb| lower.ends_with(verb))
}

/// Map a terminal event type to a canonical status.
pub fn terminal_event_to_status(event_type: &str) -> Option<&'static str> {
    let lower = event_type.to_lowercase();
    if lower.ends_with(TERMINAL_VERB_COMPLETED) {
        Some("succeeded")
    } else if lower.ends_with(TERMINAL_VERB_FAILED) {
        Some("failed")
    } else if lower.ends_with(TERMINAL_VERB_CANCELLED) {
        Some("cancelled")
    } else {
        None
    }
}

/// Canonical terminal status strings (from status endpoint).
pub const TERMINAL_STATUSES: &[&str] = &[
    "succeeded",
    "failed",
    "cancelled",
    "canceled",
    "completed",
];

/// Check if a status string is terminal.
pub fn is_terminal_status(status: &str) -> bool {
    TERMINAL_STATUSES.contains(&status.to_lowercase().as_str())
}

// ---------------------------------------------------------------------------
// Event schema: minimum required fields
// ---------------------------------------------------------------------------

/// Required fields for any SSE event from the backend.
pub const EVENT_REQUIRED_FIELDS: &[&str] = &["type"];

/// Recommended fields for SSE events (should be present, not strictly required).
pub const EVENT_RECOMMENDED_FIELDS: &[&str] = &["job_id", "seq", "data"];

/// Optional fields that may or may not be present.
pub const EVENT_OPTIONAL_FIELDS: &[&str] = &["run_id", "timestamp", "message", "level", "ts"];

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_canonical_endpoints() {
        assert_eq!(jobs_status("abc"), "/api/v1/offline/jobs/abc");
        assert_eq!(jobs_events("abc"), "/api/v1/offline/jobs/abc/events");
        assert_eq!(
            jobs_events_stream("abc"),
            "/api/v1/offline/jobs/abc/events/stream"
        );
        assert_eq!(jobs_artifacts("abc"), "/api/v1/offline/jobs/abc/artifacts");
        assert_eq!(jobs_cancel("abc"), "/api/v1/offline/jobs/abc");
    }

    #[test]
    fn test_terminal_detection_by_suffix() {
        assert!(is_terminal_event_type("learning.policy.gepa.job.completed"));
        assert!(is_terminal_event_type("eval.policy.job.failed"));
        assert!(is_terminal_event_type("some.new.domain.job.cancelled"));
        assert!(!is_terminal_event_type("learning.policy.gepa.candidate.evaluated"));
        assert!(!is_terminal_event_type("learning.policy.gepa.generation.completed"));
    }

    #[test]
    fn test_terminal_event_to_status() {
        assert_eq!(
            terminal_event_to_status("learning.policy.gepa.job.completed"),
            Some("succeeded")
        );
        assert_eq!(
            terminal_event_to_status("eval.policy.job.failed"),
            Some("failed")
        );
        assert_eq!(
            terminal_event_to_status("some.job.cancelled"),
            Some("cancelled")
        );
        assert_eq!(
            terminal_event_to_status("learning.policy.gepa.candidate.evaluated"),
            None
        );
    }

    #[test]
    fn test_terminal_status() {
        assert!(is_terminal_status("succeeded"));
        assert!(is_terminal_status("failed"));
        assert!(is_terminal_status("cancelled"));
        assert!(is_terminal_status("canceled"));
        assert!(is_terminal_status("completed"));
        assert!(is_terminal_status("Completed")); // case insensitive
        assert!(!is_terminal_status("running"));
        assert!(!is_terminal_status("pending"));
    }

    #[test]
    fn test_create_endpoints() {
        assert_eq!(JOBS_CREATE_GEPA, "/api/v1/offline/jobs");
        assert_eq!(JOBS_CREATE_EVAL, "/api/v1/offline/jobs");
    }
}
