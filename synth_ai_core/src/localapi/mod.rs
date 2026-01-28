//! Local API client for task app communication.
//!
//! This module provides the client for communicating with task apps:
//! - HTTP client with authentication
//! - Data contracts (request/response types)
//! - Health checking and waiting
//! - Rollout execution

pub mod client;
pub mod auth;
pub mod helpers;
pub mod datasets;
pub mod override_helpers;
pub mod tracing_utils;
pub mod llm_guards;
pub mod health;
pub mod proxy;
pub mod trace_helpers;
pub mod validation;
pub mod rollout_helpers;
pub mod validators;
pub mod vendors;
pub mod types;

pub use client::{EnvClient, TaskAppClient};
pub use auth::{
    ensure_localapi_auth, encrypt_for_backend, mint_environment_api_key,
    setup_environment_api_key, DEV_ENVIRONMENT_API_KEY_NAME, ENVIRONMENT_API_KEY_NAME,
    ENVIRONMENT_API_KEY_ALIASES_NAME, MAX_ENVIRONMENT_API_KEY_BYTES, SEALED_BOX_ALGORITHM,
    normalize_environment_api_key, allowed_environment_api_keys, is_api_key_header_authorized,
};
pub use proxy::{
    prepare_for_openai, prepare_for_groq, normalize_response_format_for_groq,
    inject_system_hint, extract_message_text, parse_tool_call_from_text,
    synthesize_tool_call_if_missing,
};
pub use validators::{
    validate_rollout_response_for_rl, normalize_inference_url, validate_task_app_url,
};
pub use helpers::{
    normalize_chat_completion_url, get_default_max_completion_tokens,
    extract_api_key, parse_tool_calls_from_response,
};
pub use datasets::{TaskDatasetSpec, ensure_split, normalise_seed};
pub use tracing_utils::{tracing_env_enabled, resolve_tracing_db_url, resolve_sft_output_dir, unique_sft_path};
pub use llm_guards::is_direct_provider_call;
pub use health::task_app_health;
pub use override_helpers::{
    get_agent_skills_path, apply_context_overrides, get_applied_env_vars,
    MAX_FILE_SIZE_BYTES, MAX_TOTAL_SIZE_BYTES, MAX_FILES_PER_OVERRIDE, MAX_ENV_VARS,
    MAX_ENV_VAR_VALUE_LENGTH, PREFLIGHT_SCRIPT_TIMEOUT_SECONDS,
};
pub use trace_helpers::{
    extract_trace_correlation_id, validate_trace_correlation_id,
    include_trace_correlation_id_in_response, build_trace_payload, build_trajectory_trace,
    include_event_history_in_response, include_event_history_in_trajectories,
    verify_trace_correlation_id_in_response,
};
pub use validation::{
    MAX_INLINE_ARTIFACT_BYTES, MAX_TOTAL_INLINE_ARTIFACTS_BYTES, MAX_ARTIFACTS_PER_ROLLOUT,
    MAX_ARTIFACT_METADATA_BYTES, MAX_ARTIFACT_CONTENT_TYPE_LENGTH, MAX_CONTEXT_SNAPSHOT_BYTES,
    MAX_CONTEXT_OVERRIDES_PER_ROLLOUT, validate_artifact_size, validate_artifacts_list,
    validate_context_overrides, validate_context_snapshot,
};
pub use rollout_helpers::build_rollout_response;
pub use vendors::{normalize_vendor_keys, get_openai_key, get_groq_key, normalize_single};
pub use types::{
    AuthInfo, DatasetInfo, HealthResponse, InfoResponse, InferenceInfo, LimitsInfo,
    RolloutEnvSpec, RolloutMetrics, RolloutPolicySpec, RolloutRequest, RolloutResponse,
    RolloutSafetyConfig, TaskDescriptor, TaskInfo,
};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_module_exports() {
        // Verify all types are accessible
        let _ = TaskAppClient::new("http://localhost:8000", None);
        let _ = RolloutRequest::new("test");
        let _ = TaskDescriptor::new("id", "name");
    }
}
