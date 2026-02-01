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
//!     None,
//! )?;
//!
//! let job_id = job.submit().await?;
//! let result = job.stream_until_complete(3600.0, None::<fn(_)>).await?;
//! println!("Best score: {:?}", result.best_score);
//! ```

pub mod base_events;
pub mod builders;
pub mod event_extraction;
pub mod event_schemas;
pub mod event_types;
pub mod events;
pub mod graph_convert;
pub mod graph_evolve;
pub mod graph_validation;
pub mod job_events;
pub mod progress;
pub mod prompt_learning;
pub mod schemas;
pub mod streaming;
pub mod validation;

// Re-export main types
pub use base_events::{BaseJobEvent, CandidateEvent, JobEvent};
pub use builders::build_prompt_learning_payload;
pub use event_extraction::{
    build_program_candidate, extract_program_candidate_content, extract_stages_from_candidate,
    normalize_transformation, seed_reward_entry, seed_score_entry,
};
pub use event_schemas::{
    base_event_schemas, base_job_event_schema, get_base_schema, merge_event_schema,
};
pub use event_types::{event_enum_values, is_valid_event_type, validate_event_type};
pub use events::{EventCategory, EventParser, ParsedEvent};
pub use graph_convert::convert_openai_sft;
pub use graph_evolve::{
    build_graph_evolve_config, build_graph_evolve_graph_record_payload,
    build_graph_evolve_inference_payload, build_graph_evolve_payload,
    build_graph_evolve_placeholder_dataset, load_graph_evolve_dataset,
    normalize_graph_evolve_policy_models, parse_graph_evolve_dataset,
    resolve_graph_evolve_snapshot_id, GraphEvolveJob,
};
pub use graph_validation::{
    graph_opt_supported_models, load_graphgen_taskset, parse_graphgen_taskset,
    validate_graphgen_taskset,
};
pub use graph_validation::{
    load_graph_job_toml, validate_graph_job_payload, validate_graph_job_section,
    validate_graphgen_job_config, GraphGenValidationResult,
};
pub use job_events::{
    parse_job_event, validate_base_event, ParsedJobEvent, ValidationError, ValidationResult,
};
pub use progress::{CandidateInfo, GEPAProgress, ProgressTracker, TokenUsage};
pub use prompt_learning::{PromptLearningJob, PromptLearningResult, PromptResults, RankedPrompt};
pub use schemas::{
    MutationSummary, MutationTypeStats, PhaseSummary, ProgramCandidate, SeedAnalysis, SeedInfo,
    StageInfo, MAX_INSTRUCTION_LENGTH, MAX_ROLLOUT_SAMPLES, MAX_SEED_INFO_COUNT,
};
pub use streaming::EventStream;
pub use validation::{
    validate_prompt_learning_config, validate_prompt_learning_config_strict,
    PromptLearningValidationResult,
};
