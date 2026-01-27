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
pub mod orchestration;
pub mod polling;
pub mod streaming;
pub mod tracing;
pub mod tunnels;
pub mod urls;

// Re-export core types at crate root for convenience
pub use errors::{CoreError, CoreResult, HttpErrorInfo, JobErrorInfo, UsageLimitInfo};
pub use jobs::{CandidateStatus, JobEvent, JobEventType, JobLifecycle, JobStatus};

// Re-export API types for convenience
pub use api::{SynthClient, PolicyJobStatus, EvalJobStatus};

// Re-export orchestration types
pub use orchestration::{
    EventCategory, EventParser, ParsedEvent,
    GEPAProgress, ProgressTracker, CandidateInfo, TokenUsage,
    PromptLearningJob, PromptLearningResult, PromptResults, RankedPrompt,
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
};

// Re-export streaming types
pub use streaming::{
    StreamType, StreamMessage, StreamConfig, StreamEndpoints,
    StreamHandler, CallbackHandler, JsonHandler, BufferedHandler, MultiHandler,
    JobStreamer,
};

// Re-export local API types
pub use localapi::{
    TaskAppClient, RolloutRequest, RolloutResponse, RolloutMetrics,
    TaskInfo, TaskDescriptor, DatasetInfo, InferenceInfo, LimitsInfo,
    RolloutEnvSpec, RolloutPolicySpec, RolloutSafetyConfig,
    HealthResponse, InfoResponse,
};
