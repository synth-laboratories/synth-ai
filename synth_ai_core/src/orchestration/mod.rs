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
//!         "container_url": "http://localhost:8000",
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
//! println!("Best score: {:?}", result.best_reward);
//! ```

pub mod base_events;
pub mod builders;
pub mod event_extraction;
pub mod event_schemas;
pub mod event_types;
pub mod events;
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
pub use job_events::{
    parse_job_event, validate_base_event, ParsedJobEvent, ValidationError, ValidationResult,
};
pub use progress::{CandidateInfo, GEPAProgress, ProgressTracker, TokenUsage};
pub use prompt_learning::{PromptLearningJob, PromptLearningResult, PromptResults};
pub use schemas::{
    MutationSummary, MutationTypeStats, PhaseSummary, ProgramCandidate, SeedAnalysis, SeedInfo,
    StageInfo, MAX_INSTRUCTION_LENGTH, MAX_ROLLOUT_SAMPLES, MAX_SEED_INFO_COUNT,
};
pub use streaming::EventStream;
pub use validation::{
    validate_prompt_learning_config, validate_prompt_learning_config_strict,
    PromptLearningValidationResult,
};
