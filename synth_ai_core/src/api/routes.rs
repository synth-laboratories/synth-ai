//! Canonical versioned route builders for the optimization API.
//!
//! All paths are produced **without** the `/api` prefix. Callers (e.g.
//! `jobs_endpoints.rs`, streaming, etc.) prepend `/api` as needed.

use std::fmt;

/// Supported API versions.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ApiVersion {
    V1,
    V2,
}

impl ApiVersion {
    pub fn as_str(&self) -> &'static str {
        match self {
            ApiVersion::V1 => "v1",
            ApiVersion::V2 => "v2",
        }
    }

    pub fn from_str(s: &str) -> Option<Self> {
        match s.trim().to_lowercase().as_str() {
            "v1" => Some(ApiVersion::V1),
            "v2" => Some(ApiVersion::V2),
            _ => None,
        }
    }
}

impl fmt::Display for ApiVersion {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

// ---------------------------------------------------------------------------
// Default API versions per domain
// ---------------------------------------------------------------------------

pub const GEPA_API_VERSION: ApiVersion = ApiVersion::V2;
pub const MIPRO_API_VERSION: ApiVersion = ApiVersion::V1;
pub const EVAL_API_VERSION: ApiVersion = ApiVersion::V2;

// ---------------------------------------------------------------------------
// Offline jobs
// ---------------------------------------------------------------------------

/// e.g. `/v1/offline/jobs`
pub fn offline_jobs_base(api_version: ApiVersion) -> String {
    format!("/{}/offline/jobs", api_version.as_str())
}

/// e.g. `/v1/offline/jobs/{job_id}`
pub fn offline_job_path(job_id: &str, api_version: ApiVersion) -> String {
    format!("{}/{}", offline_jobs_base(api_version), job_id)
}

/// e.g. `/v1/offline/jobs/{job_id}/events`
pub fn offline_job_subpath(job_id: &str, suffix: &str, api_version: ApiVersion) -> String {
    let suffix = if suffix.starts_with('/') {
        suffix.to_string()
    } else {
        format!("/{}", suffix)
    };
    format!("{}{}", offline_job_path(job_id, api_version), suffix)
}

// ---------------------------------------------------------------------------
// Online sessions
// ---------------------------------------------------------------------------

/// e.g. `/v1/online/sessions`
pub fn online_sessions_base(api_version: ApiVersion) -> String {
    format!("/{}/online/sessions", api_version.as_str())
}

/// e.g. `/v1/online/sessions/{session_id}`
pub fn online_session_path(session_id: &str, api_version: ApiVersion) -> String {
    format!("{}/{}", online_sessions_base(api_version), session_id)
}

/// e.g. `/v1/online/sessions/{session_id}/reward`
pub fn online_session_subpath(session_id: &str, suffix: &str, api_version: ApiVersion) -> String {
    let suffix = if suffix.starts_with('/') {
        suffix.to_string()
    } else {
        format!("/{}", suffix)
    };
    format!("{}{}", online_session_path(session_id, api_version), suffix)
}

// ---------------------------------------------------------------------------
// Policy systems
// ---------------------------------------------------------------------------

/// e.g. `/v1/policy-optimization/systems`
pub fn policy_systems_base(api_version: ApiVersion) -> String {
    format!("/{}/policy-optimization/systems", api_version.as_str())
}

/// e.g. `/v1/policy-optimization/systems/{system_id}`
pub fn policy_system_path(system_id: &str, api_version: ApiVersion) -> String {
    format!("{}/{}", policy_systems_base(api_version), system_id)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_api_version_roundtrip() {
        assert_eq!(ApiVersion::from_str("v1"), Some(ApiVersion::V1));
        assert_eq!(ApiVersion::from_str("V2"), Some(ApiVersion::V2));
        assert_eq!(ApiVersion::from_str(" v1 "), Some(ApiVersion::V1));
        assert_eq!(ApiVersion::from_str("v3"), None);
        assert_eq!(ApiVersion::V1.as_str(), "v1");
        assert_eq!(ApiVersion::V2.as_str(), "v2");
    }

    #[test]
    fn test_default_versions() {
        assert_eq!(GEPA_API_VERSION, ApiVersion::V2);
        assert_eq!(MIPRO_API_VERSION, ApiVersion::V1);
        assert_eq!(EVAL_API_VERSION, ApiVersion::V2);
    }

    #[test]
    fn test_offline_jobs_base() {
        assert_eq!(offline_jobs_base(ApiVersion::V1), "/v1/offline/jobs");
        assert_eq!(offline_jobs_base(ApiVersion::V2), "/v2/offline/jobs");
    }

    #[test]
    fn test_offline_job_path() {
        assert_eq!(
            offline_job_path("abc-123", ApiVersion::V1),
            "/v1/offline/jobs/abc-123"
        );
        assert_eq!(
            offline_job_path("xyz", ApiVersion::V2),
            "/v2/offline/jobs/xyz"
        );
    }

    #[test]
    fn test_offline_job_subpath() {
        assert_eq!(
            offline_job_subpath("abc", "events", ApiVersion::V1),
            "/v1/offline/jobs/abc/events"
        );
        assert_eq!(
            offline_job_subpath("abc", "/events", ApiVersion::V1),
            "/v1/offline/jobs/abc/events"
        );
        assert_eq!(
            offline_job_subpath("abc", "events/stream", ApiVersion::V2),
            "/v2/offline/jobs/abc/events/stream"
        );
        assert_eq!(
            offline_job_subpath("abc", "artifacts", ApiVersion::V2),
            "/v2/offline/jobs/abc/artifacts"
        );
        assert_eq!(
            offline_job_subpath("abc", "metrics", ApiVersion::V1),
            "/v1/offline/jobs/abc/metrics"
        );
    }

    #[test]
    fn test_online_sessions_base() {
        assert_eq!(online_sessions_base(ApiVersion::V1), "/v1/online/sessions");
        assert_eq!(online_sessions_base(ApiVersion::V2), "/v2/online/sessions");
    }

    #[test]
    fn test_online_session_path() {
        assert_eq!(
            online_session_path("sess-1", ApiVersion::V1),
            "/v1/online/sessions/sess-1"
        );
    }

    #[test]
    fn test_online_session_subpath() {
        assert_eq!(
            online_session_subpath("s1", "reward", ApiVersion::V1),
            "/v1/online/sessions/s1/reward"
        );
        assert_eq!(
            online_session_subpath("s1", "/events", ApiVersion::V2),
            "/v2/online/sessions/s1/events"
        );
    }

    #[test]
    fn test_policy_systems_base() {
        assert_eq!(
            policy_systems_base(ApiVersion::V1),
            "/v1/policy-optimization/systems"
        );
        assert_eq!(
            policy_systems_base(ApiVersion::V2),
            "/v2/policy-optimization/systems"
        );
    }

    #[test]
    fn test_policy_system_path() {
        assert_eq!(
            policy_system_path("sys-1", ApiVersion::V1),
            "/v1/policy-optimization/systems/sys-1"
        );
    }

    // Parity check: the paths that jobs_endpoints.rs currently hardcodes
    // should be reproducible via this module.
    #[test]
    fn test_parity_with_jobs_endpoints() {
        // JOBS_ROOT is "/api/v1/offline/jobs"
        assert_eq!(
            format!("/api{}", offline_jobs_base(ApiVersion::V1)),
            "/api/v1/offline/jobs"
        );
        // jobs_events("abc") == "/api/v1/offline/jobs/abc/events"
        assert_eq!(
            format!(
                "/api{}",
                offline_job_subpath("abc", "events", ApiVersion::V1)
            ),
            "/api/v1/offline/jobs/abc/events"
        );
        // jobs_events_stream("abc") == "/api/v1/offline/jobs/abc/events/stream"
        assert_eq!(
            format!(
                "/api{}",
                offline_job_subpath("abc", "events/stream", ApiVersion::V1)
            ),
            "/api/v1/offline/jobs/abc/events/stream"
        );
    }
}
