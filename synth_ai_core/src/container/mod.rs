//! Local API client for container communication.
//!
//! This module provides the client for communicating with containers:
//! - HTTP client with authentication
//! - Data contracts (request/response types)
//! - Health checking and waiting
//! - Rollout execution

pub mod auth;
pub mod client;
pub mod datasets;
pub mod health;
pub mod helpers;
pub mod llm_guards;
pub mod override_helpers;
pub mod proxy;
pub mod rollout_helpers;
pub mod trace_helpers;
pub mod tracing_utils;
pub mod types;
pub mod validation;
pub mod validators;
pub mod vendors;

pub use auth::{
    encrypt_for_backend, CONTAINER_AUTHORIZATION_HEADER_NAME, SEALED_BOX_ALGORITHM,
    SYNTH_CONTAINER_AUTH_AUDIENCE, SYNTH_CONTAINER_AUTH_DEFAULT_ISSUER,
    SYNTH_CONTAINER_AUTH_ISSUER_NAME, SYNTH_CONTAINER_TRUSTED_PUBKEYS_NAME,
};
pub use client::{ContainerClient, EnvClient};
pub use datasets::{ensure_split, normalise_seed, TaskDatasetSpec};
pub use health::container_health;
pub use helpers::{
    extract_api_key, get_default_max_completion_tokens, normalize_chat_completion_url,
    parse_tool_calls_from_response,
};
pub use llm_guards::is_direct_provider_call;
pub use override_helpers::{
    apply_context_overrides, get_agent_skills_path, get_applied_env_vars, MAX_ENV_VARS,
    MAX_ENV_VAR_VALUE_LENGTH, MAX_FILES_PER_OVERRIDE, MAX_FILE_SIZE_BYTES, MAX_TOTAL_SIZE_BYTES,
    PREFLIGHT_SCRIPT_TIMEOUT_SECONDS,
};
pub use proxy::{
    extract_message_text, inject_system_hint, normalize_response_format_for_groq,
    parse_tool_call_from_text, prepare_for_groq, prepare_for_openai,
    synthesize_tool_call_if_missing,
};
pub use rollout_helpers::build_rollout_response;
pub use trace_helpers::{
    build_trace_payload, build_trajectory_trace, extract_trace_correlation_id,
    include_event_history_in_response, include_event_history_in_trajectories,
    include_trace_correlation_id_in_response, validate_trace_correlation_id,
    verify_trace_correlation_id_in_response,
};
pub use tracing_utils::{
    resolve_sft_output_dir, resolve_tracing_db_url, tracing_env_enabled, unique_sft_path,
};
pub use types::{
    AuthInfo, DatasetInfo, HealthResponse, InferenceInfo, InfoResponse, LimitsInfo, RolloutEnvSpec,
    RolloutMetrics, RolloutPolicySpec, RolloutRequest, RolloutResponse, RolloutSafetyConfig,
    TaskDescriptor, TaskInfo,
};
pub use validation::{
    is_local_http_container_url, is_synth_managed_ngrok_url, is_synthtunnel_url,
    validate_artifact_size, validate_artifacts_list, validate_context_overrides,
    validate_context_snapshot, validate_gepa_container_auth, GepaAuthRequirement,
    MAX_ARTIFACTS_PER_ROLLOUT, MAX_ARTIFACT_CONTENT_TYPE_LENGTH, MAX_ARTIFACT_METADATA_BYTES,
    MAX_CONTEXT_OVERRIDES_PER_ROLLOUT, MAX_CONTEXT_SNAPSHOT_BYTES, MAX_INLINE_ARTIFACT_BYTES,
    MAX_TOTAL_INLINE_ARTIFACTS_BYTES,
};
pub use validators::{
    normalize_inference_url, validate_container_url, validate_rollout_response_for_rl,
};
pub use vendors::{get_groq_key, get_openai_key, normalize_single, normalize_vendor_keys};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_module_exports() {
        // Verify all types are accessible
        let _ = ContainerClient::new("http://localhost:8000", None);
        let _ = RolloutRequest::new("test");
        let _ = TaskDescriptor::new("id", "name");
    }
}
