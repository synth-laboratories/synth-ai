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
//! - Local API client for task apps

pub mod api;
pub mod auth;
pub mod config;
pub mod data;
pub mod errors;
pub mod events;
pub mod http;
pub mod jobs;
pub mod localapi;
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

// Re-export core types at crate root for convenience
pub use errors::{CoreError, CoreResult, HttpErrorInfo, JobErrorInfo, UsageLimitInfo};
pub use jobs::{CandidateStatus, JobEvent, JobEventType, JobLifecycle, JobStatus};

// Re-export API types for convenience
pub use api::{
    EvalJobStatus, GraphEvolveClient, InferenceClient, LocalApiDeployClient,
    LocalApiDeployResponse, LocalApiDeploySpec, LocalApiDeployStatus, LocalApiDeploymentInfo,
    LocalApiLimits, PolicyJobStatus, SynthClient,
};

// Re-export model helpers
pub use models::{detect_model_provider, normalize_model_identifier, supported_models};

// Re-export orchestration types
pub use orchestration::{
    base_event_schemas, base_job_event_schema, build_graph_evolve_config,
    build_graph_evolve_graph_record_payload, build_graph_evolve_inference_payload,
    build_graph_evolve_payload, build_graph_evolve_placeholder_dataset, build_program_candidate,
    build_prompt_learning_payload, convert_openai_sft, event_enum_values,
    extract_program_candidate_content, extract_stages_from_candidate, get_base_schema,
    graph_opt_supported_models, is_valid_event_type, load_graph_evolve_dataset,
    load_graph_job_toml, load_graphgen_taskset, merge_event_schema,
    normalize_graph_evolve_policy_models, normalize_transformation, parse_graph_evolve_dataset,
    parse_graphgen_taskset, resolve_graph_evolve_snapshot_id, seed_reward_entry, seed_score_entry,
    validate_event_type, validate_graph_job_payload, validate_graph_job_section,
    validate_graphgen_job_config, validate_graphgen_taskset, validate_prompt_learning_config,
    validate_prompt_learning_config_strict, CandidateInfo, EventCategory, EventParser, EventStream,
    GEPAProgress, GraphEvolveJob, GraphGenValidationResult, MutationSummary, MutationTypeStats,
    ParsedEvent, PhaseSummary, ProgramCandidate, ProgressTracker, PromptLearningJob,
    PromptLearningResult, PromptLearningValidationResult, PromptResults, RankedPrompt,
    SeedAnalysis, SeedInfo, StageInfo, TokenUsage, MAX_INSTRUCTION_LENGTH, MAX_ROLLOUT_SAMPLES,
    MAX_SEED_INFO_COUNT,
};

// Re-export tracing types
pub use tracing::{
    BaseEventFields, EnvironmentEvent, EventReward, EventType, HookCallback, HookContext,
    HookEvent, HookManager, LLMCallRecord, LLMContentPart, LLMMessage, LLMUsage, LMCAISEvent,
    LibsqlTraceStorage, MarkovBlanketMessage, MessageContent, OutcomeReward, RuntimeEvent,
    SessionTimeStep, SessionTrace, SessionTracer, StorageConfig, TimeRecord, ToolCallResult,
    ToolCallSpec, TraceStorage, TracingError, TracingEvent,
};

// Re-export data types
pub use data::{
    // Enum values mapping
    data_enum_values,
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
pub use localapi::{
    allowed_environment_api_keys, apply_context_overrides, build_rollout_response,
    build_trace_payload, build_trajectory_trace, encrypt_for_backend, ensure_localapi_auth,
    ensure_split, extract_api_key, extract_message_text, extract_trace_correlation_id,
    get_agent_skills_path, get_applied_env_vars, get_default_max_completion_tokens, get_groq_key,
    get_openai_key, include_event_history_in_response, include_event_history_in_trajectories,
    include_trace_correlation_id_in_response, inject_system_hint, is_api_key_header_authorized,
    is_direct_provider_call, mint_environment_api_key, normalise_seed,
    normalize_chat_completion_url, normalize_environment_api_key, normalize_inference_url,
    normalize_response_format_for_groq, normalize_vendor_keys, parse_tool_call_from_text,
    parse_tool_calls_from_response, prepare_for_groq, prepare_for_openai, resolve_sft_output_dir,
    resolve_tracing_db_url, setup_environment_api_key, synthesize_tool_call_if_missing,
    task_app_health, tracing_env_enabled, unique_sft_path, validate_artifact_size,
    validate_artifacts_list, validate_context_overrides, validate_context_snapshot,
    validate_rollout_response_for_rl, validate_task_app_url, validate_trace_correlation_id,
    verify_trace_correlation_id_in_response, DatasetInfo, HealthResponse, InferenceInfo,
    InfoResponse, LimitsInfo, RolloutEnvSpec, RolloutMetrics, RolloutPolicySpec, RolloutRequest,
    RolloutResponse, RolloutSafetyConfig, TaskAppClient, TaskDatasetSpec, TaskDescriptor, TaskInfo,
    DEV_ENVIRONMENT_API_KEY_NAME, ENVIRONMENT_API_KEY_ALIASES_NAME, ENVIRONMENT_API_KEY_NAME,
    MAX_ARTIFACTS_PER_ROLLOUT, MAX_ARTIFACT_CONTENT_TYPE_LENGTH, MAX_ARTIFACT_METADATA_BYTES,
    MAX_CONTEXT_OVERRIDES_PER_ROLLOUT, MAX_CONTEXT_SNAPSHOT_BYTES, MAX_ENVIRONMENT_API_KEY_BYTES,
    MAX_ENV_VARS, MAX_ENV_VAR_VALUE_LENGTH, MAX_FILES_PER_OVERRIDE, MAX_FILE_SIZE_BYTES,
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
    synth_home_dir, synth_localapi_config_path, synth_user_config_path, validate_file_type,
    write_private_json, write_private_text,
};
