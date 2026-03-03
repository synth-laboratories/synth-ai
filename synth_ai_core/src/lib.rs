//! Synth core library.
//!
//! This crate provides the core functionality for the Synth SDK:
//! - API client for backend communication
//! - Authentication and credential management
//! - Configuration handling
//! - HTTP client utilities
//! - Job orchestration and streaming
//! - Tracing and storage
//! - Tunnel management
//! - Data types (enums, rubrics, objectives, etc.)
//! - Streaming framework
//! - Local API client for containers

pub mod api;
pub mod auth;
pub mod config;
pub mod container;
pub mod data;
pub mod errors;
pub mod events;
pub mod http;
pub mod jobs;
pub mod jobs_endpoints;
pub mod models;
pub mod orchestration;
pub mod polling;
pub mod shared_client;
pub mod sse;
pub mod streaming;
pub mod trace_upload;
pub mod tracing;
pub mod tunnels;
pub mod urls;
pub mod utils;
pub mod x402;

// Re-export core types at crate root for convenience
pub use errors::{CoreError, CoreResult, HttpErrorInfo, JobErrorInfo, UsageLimitInfo};
pub use jobs::{CandidateStatus, JobEvent, JobEventType, JobLifecycle, JobStatus};

// Re-export API types for convenience
pub use api::{
    ContainerDeployClient, ContainerDeployResponse, ContainerDeploySpec, ContainerDeployStatus,
    ContainerDeploymentInfo, ContainerLimits, EvalJobStatus, PolicyJobStatus, SynthClient,
};

// Re-export route builders
pub use api::routes::{
    offline_job_path as routes_offline_job_path, offline_job_subpath as routes_offline_job_subpath,
    offline_jobs_base as routes_offline_jobs_base,
    online_session_path as routes_online_session_path,
    online_session_subpath as routes_online_session_subpath,
    online_sessions_base as routes_online_sessions_base,
    policy_system_path as routes_policy_system_path,
    policy_systems_base as routes_policy_systems_base, ApiVersion, EVAL_API_VERSION,
    GEPA_API_VERSION, MIPRO_API_VERSION,
};

// Re-export model helpers
pub use models::{detect_model_provider, normalize_model_identifier, supported_models};

// Re-export orchestration types
pub use orchestration::{
    base_event_schemas, base_job_event_schema, build_program_candidate,
    build_prompt_learning_payload, event_enum_values, extract_program_candidate_content,
    extract_stages_from_candidate, get_base_schema, is_valid_event_type, merge_event_schema,
    normalize_transformation, seed_reward_entry, seed_score_entry, validate_event_type,
    validate_prompt_learning_config, validate_prompt_learning_config_strict, CandidateInfo,
    EventCategory, EventParser, EventStream, GEPAProgress, MutationSummary, MutationTypeStats,
    ParsedEvent, PhaseSummary, ProgramCandidate, ProgressTracker, PromptLearningJob,
    PromptLearningResult, PromptLearningValidationResult, PromptResults, SeedAnalysis, SeedInfo,
    StageInfo, TokenUsage, MAX_INSTRUCTION_LENGTH, MAX_ROLLOUT_SAMPLES, MAX_SEED_INFO_COUNT,
};

// Re-export tracing types
pub use tracing::{
    BaseEventFields, EnvironmentEvent, EventReward, EventType, HookCallback, HookContext,
    HookEvent, HookManager, LLMCallRecord, LLMContentPart, LLMMessage, LLMUsage, LMCAISEvent,
    MarkovBlanketMessage, MessageContent, OutcomeReward, RuntimeEvent, SessionTimeStep,
    SessionTrace, SessionTracer, StorageConfig, TimeRecord, ToolCallResult, ToolCallSpec,
    TraceStorage, TracingError, TracingEvent,
};

#[cfg(feature = "libsql")]
pub use tracing::LibsqlTraceStorage;

// Re-export data types
pub use data::{
    // Enum values mapping
    data_enum_values,
    lever_sensor_v1_contract_schema,
    AdaptiveBatchLevel,
    AdaptiveCurriculumLevel,
    ApplicationErrorType,
    ApplicationStatus,
    // Artifacts
    Artifact,
    ArtifactBundle,
    ArtifactContent,
    // Context overrides
    ContextOverride,
    ContextOverrideStatus,
    Criterion,
    CriterionScoreData,
    EventObjectiveAssignment,
    GraphType,
    InferenceMode,
    JobStatus as DataJobStatus,
    // Enums
    JobType,
    // Judgements
    Judgement,
    // Levers + Sensors
    Lever,
    LeverActor,
    LeverConstraints,
    LeverFormat,
    LeverKind,
    LeverMutability,
    LeverMutation,
    LeverProvenance,
    LeverSnapshot,
    MiproLeverSummary,
    ObjectiveDirection,
    ObjectiveKey,
    // Objectives
    ObjectiveSpec,
    OptimizationMode,
    OutcomeObjectiveAssignment,
    OutputMode,
    ProviderName,
    RewardObservation,
    RewardScope,
    RewardSource,
    RewardType,
    // Rubrics
    Rubric,
    RubricAssignment,
    ScopeKey,
    ScopeKind,
    Sensor,
    SensorFrame,
    SensorFrameSummary,
    SensorKind,
    SuccessStatus,
    TrainingType,
    VerifierMode,
};

// Re-export streaming types
pub use streaming::{
    BufferedHandler, CallbackHandler, JobStreamer, JsonHandler, MultiHandler, StreamConfig,
    StreamEndpoints, StreamHandler, StreamMessage, StreamType,
};

// Re-export SSE helpers
pub use sse::{stream_sse, stream_sse_request, SseEvent, SseStream};

// Re-export local API types
pub use container::{
    apply_context_overrides, build_rollout_response, build_trace_payload, build_trajectory_trace,
    container_health, encrypt_for_backend, ensure_split, extract_api_key, extract_message_text,
    extract_trace_correlation_id, get_agent_skills_path, get_applied_env_vars,
    get_default_max_completion_tokens, get_groq_key, get_openai_key,
    include_event_history_in_response, include_event_history_in_trajectories,
    include_trace_correlation_id_in_response, inject_system_hint, is_direct_provider_call,
    is_local_http_container_url, is_synth_managed_ngrok_url, is_synthtunnel_url, normalise_seed,
    normalize_chat_completion_url, normalize_inference_url, normalize_response_format_for_groq,
    normalize_vendor_keys, parse_tool_call_from_text, parse_tool_calls_from_response,
    prepare_for_groq, prepare_for_openai, resolve_sft_output_dir, resolve_tracing_db_url,
    synthesize_tool_call_if_missing, tracing_env_enabled, unique_sft_path, validate_artifact_size,
    validate_artifacts_list, validate_container_url, validate_context_overrides,
    validate_context_snapshot, validate_gepa_container_auth, validate_rollout_response_for_rl,
    validate_trace_correlation_id, verify_trace_correlation_id_in_response, ContainerClient,
    DatasetInfo, GepaAuthRequirement, HealthResponse, InferenceInfo, InfoResponse, LimitsInfo,
    RolloutEnvSpec, RolloutMetrics, RolloutPolicySpec, RolloutRequest, RolloutResponse,
    RolloutSafetyConfig, TaskDatasetSpec, TaskDescriptor, TaskInfo, MAX_ARTIFACTS_PER_ROLLOUT,
    MAX_ARTIFACT_CONTENT_TYPE_LENGTH, MAX_ARTIFACT_METADATA_BYTES,
    MAX_CONTEXT_OVERRIDES_PER_ROLLOUT, MAX_CONTEXT_SNAPSHOT_BYTES, MAX_ENV_VARS,
    MAX_ENV_VAR_VALUE_LENGTH, MAX_FILES_PER_OVERRIDE, MAX_FILE_SIZE_BYTES,
    MAX_INLINE_ARTIFACT_BYTES, MAX_TOTAL_INLINE_ARTIFACTS_BYTES, MAX_TOTAL_SIZE_BYTES,
    PREFLIGHT_SCRIPT_TIMEOUT_SECONDS, SEALED_BOX_ALGORITHM,
};

// Re-export trace upload types
pub use trace_upload::{TraceUploadClient, UploadUrlResponse};

// Re-export utility helpers
pub use utils::{
    cleanup_paths, compute_import_paths, create_and_write_json, ensure_private_dir,
    find_config_path, get_bin_path, get_home_config_file_paths, is_file_type, is_hidden_path,
    load_json_to_value, repo_root, should_filter_log_line, strip_json_comments, synth_bin_dir,
    synth_container_config_path, synth_home_dir, synth_user_config_path, validate_file_type,
    write_private_json, write_private_text,
};
