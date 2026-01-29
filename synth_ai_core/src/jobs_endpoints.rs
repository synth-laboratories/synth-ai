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
//! Canonical: `/api/jobs/*`
//! Legacy aliases (kept during migration):
//! - `/api/prompt-learning/online/jobs/*`
//! - `/api/policy-optimization/online/jobs/*`
//! - `/api/eval/jobs/*`

// ---------------------------------------------------------------------------
// Canonical job endpoint family
// ---------------------------------------------------------------------------

/// Root for all canonical job endpoints.
pub const JOBS_ROOT: &str = "/api/jobs";

/// Create/submit endpoints by domain.
pub const JOBS_CREATE_GEPA: &str = "/api/jobs/gepa";
pub const JOBS_CREATE_MIPRO: &str = "/api/jobs/mipro";
pub const JOBS_CREATE_EVAL: &str = "/api/jobs/eval";
pub const JOBS_CREATE_GRAPH: &str = "/api/jobs/graph";
pub const JOBS_CREATE_VERIFIER: &str = "/api/jobs/verifier";

/// Format helpers for per-job endpoints.
pub fn jobs_status(job_id: &str) -> String {
    format!("{}/{}", JOBS_ROOT, job_id)
}

pub fn jobs_events(job_id: &str) -> String {
    format!("{}/{}/events", JOBS_ROOT, job_id)
}

pub fn jobs_events_stream(job_id: &str) -> String {
    format!("{}/{}/events/stream", JOBS_ROOT, job_id)
}

pub fn jobs_artifacts(job_id: &str) -> String {
    format!("{}/{}/artifacts", JOBS_ROOT, job_id)
}

pub fn jobs_cancel(job_id: &str) -> String {
    format!("{}/{}/cancel", JOBS_ROOT, job_id)
}

pub fn jobs_metrics(job_id: &str) -> String {
    format!("{}/{}/metrics", JOBS_ROOT, job_id)
}

// ---------------------------------------------------------------------------
// Legacy alias endpoints (kept during migration)
// ---------------------------------------------------------------------------

/// Legacy prompt-learning endpoint root.
pub const LEGACY_PROMPT_LEARNING_ROOT: &str = "/api/prompt-learning/online/jobs";

/// Legacy policy-optimization endpoint root.
pub const LEGACY_POLICY_OPTIMIZATION_ROOT: &str = "/api/policy-optimization/online/jobs";

/// Legacy eval endpoint root.
pub const LEGACY_EVAL_ROOT: &str = "/api/eval/jobs";

/// Legacy learning endpoint root (used by some older SDKs).
pub const LEGACY_LEARNING_ROOT: &str = "/api/learning/jobs";

/// Legacy orchestration endpoint root.
pub const LEGACY_ORCHESTRATION_ROOT: &str = "/api/orchestration/jobs";

/// Format helpers for legacy prompt-learning endpoints.
pub fn legacy_prompt_learning_status(job_id: &str) -> String {
    format!("{}/{}", LEGACY_PROMPT_LEARNING_ROOT, job_id)
}

pub fn legacy_prompt_learning_events(job_id: &str) -> String {
    format!("{}/{}/events", LEGACY_PROMPT_LEARNING_ROOT, job_id)
}

pub fn legacy_prompt_learning_events_stream(job_id: &str) -> String {
    format!("{}/{}/events/stream", LEGACY_PROMPT_LEARNING_ROOT, job_id)
}

pub fn legacy_prompt_learning_metrics(job_id: &str) -> String {
    format!("{}/{}/metrics", LEGACY_PROMPT_LEARNING_ROOT, job_id)
}

/// Format helpers for legacy eval endpoints.
pub fn legacy_eval_status(job_id: &str) -> String {
    format!("{}/{}", LEGACY_EVAL_ROOT, job_id)
}

pub fn legacy_eval_events(job_id: &str) -> String {
    format!("{}/{}/events", LEGACY_EVAL_ROOT, job_id)
}

pub fn legacy_eval_events_stream(job_id: &str) -> String {
    format!("{}/{}/events/stream", LEGACY_EVAL_ROOT, job_id)
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
        assert_eq!(jobs_status("abc"), "/api/jobs/abc");
        assert_eq!(jobs_events("abc"), "/api/jobs/abc/events");
        assert_eq!(jobs_events_stream("abc"), "/api/jobs/abc/events/stream");
        assert_eq!(jobs_artifacts("abc"), "/api/jobs/abc/artifacts");
        assert_eq!(jobs_cancel("abc"), "/api/jobs/abc/cancel");
    }

    #[test]
    fn test_legacy_endpoints() {
        assert_eq!(
            legacy_prompt_learning_status("abc"),
            "/api/prompt-learning/online/jobs/abc"
        );
        assert_eq!(
            legacy_eval_status("abc"),
            "/api/eval/jobs/abc"
        );
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
        assert_eq!(JOBS_CREATE_GEPA, "/api/jobs/gepa");
        assert_eq!(JOBS_CREATE_EVAL, "/api/jobs/eval");
    }
}
