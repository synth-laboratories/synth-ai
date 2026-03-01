use crate::data::{Artifact, ContextOverride};
use crate::errors::CoreError;
use serde_json::Value;

pub const MAX_INLINE_ARTIFACT_BYTES: usize = 64 * 1024;
pub const MAX_TOTAL_INLINE_ARTIFACTS_BYTES: usize = 256 * 1024;
pub const MAX_ARTIFACTS_PER_ROLLOUT: usize = 10;
pub const MAX_ARTIFACT_METADATA_BYTES: usize = 16 * 1024;
pub const MAX_ARTIFACT_CONTENT_TYPE_LENGTH: usize = 128;
pub const MAX_CONTEXT_SNAPSHOT_BYTES: usize = 1 * 1024 * 1024;
pub const MAX_CONTEXT_OVERRIDES_PER_ROLLOUT: usize = 20;

pub fn validate_artifact_size(artifact: &Artifact, max_bytes: usize) -> Result<(), CoreError> {
    artifact
        .validate_size(max_bytes as i64)
        .map_err(CoreError::InvalidInput)
}

pub fn validate_artifacts_list(artifacts: &[Artifact]) -> Result<(), CoreError> {
    if artifacts.len() > MAX_ARTIFACTS_PER_ROLLOUT {
        return Err(CoreError::InvalidInput(format!(
            "Too many artifacts: {} > {}",
            artifacts.len(),
            MAX_ARTIFACTS_PER_ROLLOUT
        )));
    }

    let mut total_size = 0usize;

    for artifact in artifacts {
        let size = match &artifact.content {
            crate::data::ArtifactContent::Text(text) => text.as_bytes().len(),
            crate::data::ArtifactContent::Structured(map) => {
                serde_json::to_string(map).map(|s| s.len()).unwrap_or(0)
            }
        };
        total_size += size;

        if let Some(content_type) = &artifact.content_type {
            if content_type.len() > MAX_ARTIFACT_CONTENT_TYPE_LENGTH {
                return Err(CoreError::InvalidInput(format!(
                    "Artifact content_type too long: {} > {}",
                    content_type.len(),
                    MAX_ARTIFACT_CONTENT_TYPE_LENGTH
                )));
            }
        }

        if !artifact.metadata.is_empty() {
            let metadata_size = serde_json::to_string(&artifact.metadata)
                .map(|s| s.len())
                .unwrap_or(0);
            if metadata_size > MAX_ARTIFACT_METADATA_BYTES {
                return Err(CoreError::InvalidInput(format!(
                    "Artifact metadata too large: {} > {}",
                    metadata_size, MAX_ARTIFACT_METADATA_BYTES
                )));
            }
        }
    }

    if total_size > MAX_TOTAL_INLINE_ARTIFACTS_BYTES {
        return Err(CoreError::InvalidInput(format!(
            "Total artifacts size {} exceeds {} bytes",
            total_size, MAX_TOTAL_INLINE_ARTIFACTS_BYTES
        )));
    }

    Ok(())
}

pub fn validate_context_overrides(overrides: &[ContextOverride]) -> Result<(), CoreError> {
    if overrides.len() > MAX_CONTEXT_OVERRIDES_PER_ROLLOUT {
        return Err(CoreError::InvalidInput(format!(
            "Too many context overrides: {} > {}",
            overrides.len(),
            MAX_CONTEXT_OVERRIDES_PER_ROLLOUT
        )));
    }

    let mut total_size = 0usize;
    let mut total_files = 0usize;
    for override_item in overrides {
        total_size += override_item.size_bytes();
        total_files += override_item.file_artifacts.len();
    }

    if total_size > MAX_CONTEXT_SNAPSHOT_BYTES {
        return Err(CoreError::InvalidInput(format!(
            "Total context override size {} exceeds {} bytes",
            total_size, MAX_CONTEXT_SNAPSHOT_BYTES
        )));
    }

    let max_files = 50usize;
    if total_files > max_files {
        return Err(CoreError::InvalidInput(format!(
            "Too many file artifacts across overrides: {} > {}",
            total_files, max_files
        )));
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// URL classification helpers (GEPA container-auth policy)
// ---------------------------------------------------------------------------

/// Local host set matching Python `_is_local_http_container_url`.
const LOCAL_HOSTS: &[&str] = &["localhost", "127.0.0.1", "0.0.0.0", "host.docker.internal"];

/// Returns `true` if `url` is an HTTP (not HTTPS) URL pointing at a local host.
///
/// Matches Python `_is_local_http_container_url` in builders.py / prompt_learning.py.
pub fn is_local_http_container_url(url: &str) -> bool {
    let trimmed = url.trim();
    let parsed = match url::Url::parse(trimmed) {
        Ok(u) => u,
        Err(_) => return false,
    };
    if parsed.scheme() != "http" {
        return false;
    }
    let host = match parsed.host_str() {
        Some(h) => h.to_ascii_lowercase(),
        None => return false,
    };
    LOCAL_HOSTS.iter().any(|&local| host == local)
}

/// Default SynthTunnel trusted host patterns.
const SYNTHTUNNEL_DEFAULT_PATTERNS: &[&str] = &[
    "st.usesynth.ai",
    "*.st.usesynth.ai",
    "infra-api-dev.usesynth.ai",
    "infra-api.usesynth.ai",
    "api-dev.usesynth.ai",
    "api.usesynth.ai",
    "localhost",
    "127.0.0.1",
    "::1",
];

/// Match a hostname against a pattern (exact, `*.suffix`, or `.suffix`).
fn host_matches_pattern(host: &str, pattern: &str) -> bool {
    if pattern.is_empty() {
        return false;
    }
    if let Some(suffix) = pattern.strip_prefix("*.") {
        let dot_suffix = format!(".{suffix}");
        host.ends_with(&dot_suffix) && host.len() > dot_suffix.len()
    } else if pattern.starts_with('.') {
        host.ends_with(pattern)
    } else {
        host == pattern
    }
}

/// Returns `true` if the URL targets the SynthTunnel gateway.
///
/// Exactly mirrors Python `is_synthtunnel_url` in `synth_ai/core/utils/urls.py`.
pub fn is_synthtunnel_url(url: &str) -> bool {
    let parsed = match url::Url::parse(url) {
        Ok(u) => u,
        Err(_) => return false,
    };

    let path = parsed.path();
    if !path.starts_with("/s/rt_") {
        return false;
    }

    let hostname = match parsed.host_str() {
        Some(h) => h.to_ascii_lowercase(),
        None => return false,
    };

    let mut patterns: Vec<String> = SYNTHTUNNEL_DEFAULT_PATTERNS
        .iter()
        .map(|s| s.to_string())
        .collect();

    if let Ok(extra) = std::env::var("SYNTH_TUNNEL_TRUSTED_HOSTS") {
        let extra = extra.trim().to_string();
        if !extra.is_empty() {
            for raw in extra.split(',') {
                let value = raw.trim().to_lowercase();
                if !value.is_empty() && !patterns.contains(&value) {
                    patterns.push(value);
                }
            }
        }
    }

    patterns.iter().any(|p| host_matches_pattern(&hostname, p))
}

/// Result of GEPA container auth validation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GepaAuthRequirement {
    /// Auth is fine as-is.
    Ok,
    /// URL goes through SynthTunnel but no signing key is configured.
    SynthTunnelSignerRequired,
    /// URL is a non-local remote endpoint without a signing key.
    RemoteSignerRequired,
}

/// Validate whether the container URL + signing key combination is acceptable
/// for GEPA jobs.
pub fn validate_gepa_container_auth(
    container_url: &str,
    has_signing_key: bool,
) -> GepaAuthRequirement {
    if is_local_http_container_url(container_url) {
        return GepaAuthRequirement::Ok;
    }
    if is_synthtunnel_url(container_url) {
        if has_signing_key {
            GepaAuthRequirement::Ok
        } else {
            GepaAuthRequirement::SynthTunnelSignerRequired
        }
    } else if has_signing_key {
        GepaAuthRequirement::Ok
    } else {
        GepaAuthRequirement::RemoteSignerRequired
    }
}

pub fn validate_context_snapshot(snapshot_data: &Value) -> Result<(), CoreError> {
    let size = serde_json::to_string(snapshot_data)
        .map(|s| s.len())
        .unwrap_or(0);
    if size > MAX_CONTEXT_SNAPSHOT_BYTES {
        return Err(CoreError::InvalidInput(format!(
            "Context snapshot size {} exceeds {} bytes",
            size, MAX_CONTEXT_SNAPSHOT_BYTES
        )));
    }
    Ok(())
}

#[cfg(test)]
mod url_classification_tests {
    use super::*;

    // ---- is_local_http_container_url ----

    #[test]
    fn test_local_urls_true() {
        assert!(is_local_http_container_url("http://localhost:8000"));
        assert!(is_local_http_container_url("http://127.0.0.1:8000"));
        assert!(is_local_http_container_url("http://0.0.0.0:9000"));
        assert!(is_local_http_container_url("http://host.docker.internal:8000"));
        assert!(is_local_http_container_url("http://localhost"));
        assert!(is_local_http_container_url("http://localhost:8080/api"));
    }

    #[test]
    fn test_non_local_urls_false() {
        assert!(!is_local_http_container_url("https://localhost:8000"));
        assert!(!is_local_http_container_url("http://example.com"));
        assert!(!is_local_http_container_url(""));
        assert!(!is_local_http_container_url("http://api.usesynth.ai"));
        assert!(!is_local_http_container_url("ftp://localhost:8000"));
        assert!(!is_local_http_container_url("http://192.168.1.1:8000"));
        assert!(!is_local_http_container_url("not-a-url"));
    }

    #[test]
    fn test_whitespace_trimmed() {
        assert!(is_local_http_container_url("  http://localhost:8000  "));
    }

    // ---- is_synthtunnel_url ----

    #[test]
    fn test_synthtunnel_basic_match() {
        assert!(is_synthtunnel_url(
            "http://st.usesynth.ai/s/rt_abc123/rollout"
        ));
        assert!(is_synthtunnel_url(
            "https://infra-api-dev.usesynth.ai/s/rt_token/path"
        ));
        assert!(is_synthtunnel_url(
            "http://localhost/s/rt_token"
        ));
        assert!(is_synthtunnel_url(
            "http://127.0.0.1/s/rt_token"
        ));
    }

    #[test]
    fn test_synthtunnel_wildcard_match() {
        assert!(is_synthtunnel_url(
            "https://foo.st.usesynth.ai/s/rt_abc"
        ));
    }

    #[test]
    fn test_synthtunnel_no_match() {
        // Missing /s/rt_ prefix
        assert!(!is_synthtunnel_url("https://st.usesynth.ai/other/path"));
        // Wrong host
        assert!(!is_synthtunnel_url(
            "https://evil.com/s/rt_abc"
        ));
        // Empty
        assert!(!is_synthtunnel_url(""));
        // No path
        assert!(!is_synthtunnel_url("https://st.usesynth.ai"));
    }

    // ---- validate_gepa_container_auth ----

    #[test]
    fn test_gepa_auth_local_always_ok() {
        assert_eq!(
            validate_gepa_container_auth("http://localhost:8000", false),
            GepaAuthRequirement::Ok
        );
        assert_eq!(
            validate_gepa_container_auth("http://localhost:8000", true),
            GepaAuthRequirement::Ok
        );
    }

    #[test]
    fn test_gepa_auth_synthtunnel_needs_signer() {
        assert_eq!(
            validate_gepa_container_auth("https://st.usesynth.ai/s/rt_abc", false),
            GepaAuthRequirement::SynthTunnelSignerRequired
        );
        assert_eq!(
            validate_gepa_container_auth("https://st.usesynth.ai/s/rt_abc", true),
            GepaAuthRequirement::Ok
        );
    }

    #[test]
    fn test_gepa_auth_remote_needs_signer() {
        assert_eq!(
            validate_gepa_container_auth("https://example.com/container", false),
            GepaAuthRequirement::RemoteSignerRequired
        );
        assert_eq!(
            validate_gepa_container_auth("https://example.com/container", true),
            GepaAuthRequirement::Ok
        );
    }

    // ---- host_matches_pattern ----

    #[test]
    fn test_host_matches_pattern() {
        assert!(host_matches_pattern("localhost", "localhost"));
        assert!(host_matches_pattern("foo.st.usesynth.ai", "*.st.usesynth.ai"));
        assert!(!host_matches_pattern("st.usesynth.ai", "*.st.usesynth.ai"));
        assert!(host_matches_pattern("foo.bar", ".bar"));
        assert!(!host_matches_pattern("", "localhost"));
        assert!(!host_matches_pattern("localhost", ""));
    }
}
