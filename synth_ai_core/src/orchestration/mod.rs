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
pub mod progress;
pub mod prompt_learning;
pub mod streaming;

// Re-export main types
pub use events::{EventCategory, EventParser, EventPath, ParsedEvent, TerminalStatus};
pub use progress::{CandidateInfo, GEPAProgress, ProgressTracker, TokenUsage};
pub use prompt_learning::{PromptLearningJob, PromptLearningResult, PromptResults, RankedPrompt};
pub use streaming::EventStream;
