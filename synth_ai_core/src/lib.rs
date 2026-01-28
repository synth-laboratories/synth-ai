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
pub mod sse;
pub mod streaming;
pub mod tracing;
pub mod tunnels;
pub mod trace_upload;
pub mod urls;
pub mod utils;

// Re-export core types at crate root for convenience
pub use errors::{CoreError, CoreResult, HttpErrorInfo, JobErrorInfo, UsageLimitInfo};
pub use jobs::{CandidateStatus, JobEvent, JobEventType, JobLifecycle, JobStatus};

// Re-export API types for convenience
pub use api::{SynthClient, PolicyJobStatus, EvalJobStatus, InferenceClient, GraphEvolveClient};

// Re-export model helpers
pub use models::{normalize_model_identifier, detect_model_provider, supported_models};

// Re-export orchestration types
pub use orchestration::{
    EventCategory, EventParser, ParsedEvent,
    GEPAProgress, ProgressTracker, CandidateInfo, TokenUsage,
    ProgramCandidate, StageInfo, SeedInfo,
    MutationTypeStats, MutationSummary, SeedAnalysis, PhaseSummary,
    MAX_INSTRUCTION_LENGTH, MAX_ROLLOUT_SAMPLES, MAX_SEED_INFO_COUNT,
    PromptLearningJob, PromptLearningResult, PromptResults, RankedPrompt,
    build_prompt_learning_payload, validate_prompt_learning_config, validate_prompt_learning_config_strict, PromptLearningValidationResult,
    validate_graphgen_job_config, validate_graph_job_section, load_graph_job_toml, validate_graph_job_payload, GraphGenValidationResult,
    graph_opt_supported_models, validate_graphgen_taskset, parse_graphgen_taskset, load_graphgen_taskset,
    convert_openai_sft,
    seed_score_entry, extract_stages_from_candidate, extract_program_candidate_content,
    normalize_transformation, build_program_candidate,
    event_enum_values, is_valid_event_type, validate_event_type,
    base_event_schemas, base_job_event_schema, get_base_schema, merge_event_schema,
    parse_graph_evolve_dataset, load_graph_evolve_dataset, normalize_graph_evolve_policy_models,
    build_graph_evolve_config, build_graph_evolve_payload, resolve_graph_evolve_snapshot_id,
    build_graph_evolve_graph_record_payload, build_graph_evolve_inference_payload,
    build_graph_evolve_placeholder_dataset, GraphEvolveJob,
    EventStream,
};

// Re-export tracing types
pub use tracing::{
    SessionTracer, LibsqlTraceStorage, TraceStorage, StorageConfig, TracingError,
    SessionTrace, SessionTimeStep, TracingEvent, EventType,
    LMCAISEvent, EnvironmentEvent, RuntimeEvent,
    BaseEventFields, TimeRecord, MessageContent, MarkovBlanketMessage,
    LLMCallRecord, LLMMessage, LLMUsage, LLMContentPart, ToolCallSpec, ToolCallResult,
    OutcomeReward, EventReward,
    HookManager, HookEvent, HookContext, HookCallback,
};

// Re-export data types
pub use data::{
    // Enums
    JobType, JobStatus as DataJobStatus, ProviderName, InferenceMode,
    RewardSource, RewardType, RewardScope,
    ObjectiveKey, ObjectiveDirection, OutputMode, SuccessStatus,
    GraphType, OptimizationMode, VerifierMode, TrainingType,
    AdaptiveCurriculumLevel, AdaptiveBatchLevel,
    // Rubrics
    Rubric, Criterion,
    // Objectives
    ObjectiveSpec, RewardObservation, OutcomeObjectiveAssignment, EventObjectiveAssignment,
    // Judgements
    Judgement, RubricAssignment, CriterionScoreData,
    // Artifacts
    Artifact, ArtifactBundle, ArtifactContent,
    // Context overrides
    ContextOverride, ContextOverrideStatus, ApplicationStatus, ApplicationErrorType,
    // Enum values mapping
    data_enum_values,
};

// Re-export streaming types
pub use streaming::{
    StreamType, StreamMessage, StreamConfig, StreamEndpoints,
    StreamHandler, CallbackHandler, JsonHandler, BufferedHandler, MultiHandler,
    JobStreamer,
};

// Re-export SSE helpers
pub use sse::{stream_sse, stream_sse_request, SseEvent, SseStream};

// Re-export local API types
pub use localapi::{
    TaskAppClient, RolloutRequest, RolloutResponse, RolloutMetrics,
    TaskInfo, TaskDescriptor, DatasetInfo, InferenceInfo, LimitsInfo,
    RolloutEnvSpec, RolloutPolicySpec, RolloutSafetyConfig,
    HealthResponse, InfoResponse,
    ensure_localapi_auth, encrypt_for_backend, mint_environment_api_key,
    setup_environment_api_key, DEV_ENVIRONMENT_API_KEY_NAME, ENVIRONMENT_API_KEY_NAME,
    ENVIRONMENT_API_KEY_ALIASES_NAME, MAX_ENVIRONMENT_API_KEY_BYTES, SEALED_BOX_ALGORITHM,
    normalize_environment_api_key, allowed_environment_api_keys, is_api_key_header_authorized,
    prepare_for_openai, prepare_for_groq, normalize_response_format_for_groq,
    inject_system_hint, extract_message_text, parse_tool_call_from_text,
    synthesize_tool_call_if_missing,
    validate_rollout_response_for_rl, normalize_inference_url, validate_task_app_url,
    normalize_vendor_keys, get_openai_key, get_groq_key,
    normalize_chat_completion_url, get_default_max_completion_tokens,
    extract_api_key, parse_tool_calls_from_response,
    extract_trace_correlation_id, validate_trace_correlation_id,
    include_trace_correlation_id_in_response, build_trace_payload, build_trajectory_trace,
    include_event_history_in_response, include_event_history_in_trajectories,
    verify_trace_correlation_id_in_response,
    MAX_INLINE_ARTIFACT_BYTES, MAX_TOTAL_INLINE_ARTIFACTS_BYTES, MAX_ARTIFACTS_PER_ROLLOUT,
    MAX_ARTIFACT_METADATA_BYTES, MAX_ARTIFACT_CONTENT_TYPE_LENGTH, MAX_CONTEXT_SNAPSHOT_BYTES,
    MAX_CONTEXT_OVERRIDES_PER_ROLLOUT, validate_artifact_size, validate_artifacts_list,
    validate_context_overrides, validate_context_snapshot,
    build_rollout_response,
    get_agent_skills_path, apply_context_overrides, get_applied_env_vars,
    MAX_FILE_SIZE_BYTES, MAX_TOTAL_SIZE_BYTES, MAX_FILES_PER_OVERRIDE, MAX_ENV_VARS,
    MAX_ENV_VAR_VALUE_LENGTH, PREFLIGHT_SCRIPT_TIMEOUT_SECONDS,
    task_app_health,
    TaskDatasetSpec, ensure_split, normalise_seed,
    tracing_env_enabled, resolve_tracing_db_url, resolve_sft_output_dir, unique_sft_path,
    is_direct_provider_call,
};

// Re-export trace upload types
pub use trace_upload::{TraceUploadClient, UploadUrlResponse};

// Re-export utility helpers
pub use utils::{
    strip_json_comments, create_and_write_json, load_json_to_value,
    repo_root, synth_home_dir, synth_user_config_path, synth_localapi_config_path, synth_bin_dir,
    is_file_type, validate_file_type, is_hidden_path, get_bin_path,
    get_home_config_file_paths, find_config_path, compute_import_paths, cleanup_paths,
    ensure_private_dir, write_private_text, write_private_json, should_filter_log_line,
};
