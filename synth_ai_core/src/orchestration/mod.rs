//! Job orchestration module.
//!
//! This module provides high-level job orchestration with event streaming
//! and progress tracking for optimization jobs.
//!
//! # Example
//!
//! ```ignore
//! use synth_ai_core::orchestration::PromptLearningJob;
//!
//! let mut job = PromptLearningJob::from_dict(
//!     serde_json::json!({
//!         "algorithm": "gepa",
//!         "task_app_url": "http://localhost:8000",
//!         "env_name": "default",
//!         "policy": { "model": "gpt-4o-mini", "provider": "openai" },
//!     }),
//!     None,
//!     None,
//! )?;
//!
//! let job_id = job.submit().await?;
//! let result = job.stream_until_complete(3600.0, None::<fn(_)>).await?;
//! println!("Best score: {:?}", result.best_score);
//! ```

pub mod events;
pub mod job_events;
pub mod progress;
pub mod event_extraction;
pub mod event_types;
pub mod event_schemas;
pub mod base_events;
pub mod schemas;
pub mod prompt_learning;
pub mod streaming;
pub mod builders;
pub mod validation;
pub mod graph_validation;
pub mod graph_convert;
pub mod graph_evolve;

// Re-export main types
pub use events::{EventCategory, EventParser, ParsedEvent};
pub use job_events::{parse_job_event, validate_base_event, ParsedJobEvent, ValidationError, ValidationResult};
pub use progress::{CandidateInfo, GEPAProgress, ProgressTracker, TokenUsage};
pub use event_extraction::{
    seed_score_entry,
    extract_stages_from_candidate,
    extract_program_candidate_content,
    normalize_transformation,
    build_program_candidate,
};
pub use event_types::{event_enum_values, is_valid_event_type, validate_event_type};
pub use event_schemas::{base_event_schemas, base_job_event_schema, get_base_schema, merge_event_schema};
pub use base_events::{BaseJobEvent, JobEvent, CandidateEvent};
pub use schemas::{
    MutationTypeStats,
    MutationSummary,
    SeedAnalysis,
    PhaseSummary,
    ProgramCandidate,
    StageInfo,
    SeedInfo,
    MAX_INSTRUCTION_LENGTH,
    MAX_ROLLOUT_SAMPLES,
    MAX_SEED_INFO_COUNT,
};
pub use prompt_learning::{PromptLearningJob, PromptLearningResult, PromptResults, RankedPrompt};
pub use builders::build_prompt_learning_payload;
pub use validation::{validate_prompt_learning_config, validate_prompt_learning_config_strict, PromptLearningValidationResult};
pub use graph_validation::{validate_graphgen_job_config, validate_graph_job_section, load_graph_job_toml, validate_graph_job_payload, GraphGenValidationResult};
pub use graph_validation::{graph_opt_supported_models, validate_graphgen_taskset, parse_graphgen_taskset, load_graphgen_taskset};
pub use graph_convert::convert_openai_sft;
pub use graph_evolve::{
    parse_graph_evolve_dataset,
    load_graph_evolve_dataset,
    normalize_graph_evolve_policy_models,
    build_graph_evolve_config,
    build_graph_evolve_payload,
    resolve_graph_evolve_snapshot_id,
    build_graph_evolve_graph_record_payload,
    build_graph_evolve_inference_payload,
    build_graph_evolve_placeholder_dataset,
    GraphEvolveJob,
};
pub use streaming::EventStream;
