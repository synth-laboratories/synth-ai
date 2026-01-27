//! Synth API client.
//!
//! This module provides a client for interacting with the Synth AI API.
//!
//! # Example
//!
//! ```ignore
//! use synth_ai_core::api::{SynthClient, GepaJobRequest, PolicyConfig};
//!
//! #[tokio::main]
//! async fn main() -> Result<(), Box<dyn std::error::Error>> {
//!     let client = SynthClient::from_env()?;
//!
//!     // Submit a GEPA job
//!     let job_id = client.jobs().submit_gepa(GepaJobRequest {
//!         task_app_url: "http://localhost:8000".into(),
//!         env_name: "default".into(),
//!         policy: PolicyConfig::default(),
//!         ..Default::default()
//!     }).await?;
//!
//!     // Poll until complete
//!     let result = client.jobs().poll_until_complete(&job_id, 3600.0, 15.0).await?;
//!     println!("Best score: {:?}", result.best_score);
//!
//!     Ok(())
//! }
//! ```

pub mod types;
pub mod client;
pub mod jobs;
pub mod eval;
pub mod graphs;

// Re-export main types for convenience
pub use client::SynthClient;
pub use types::{
    // Job status
    PolicyJobStatus,
    EvalJobStatus,
    // Config types
    PolicyConfig,
    GepaConfig,
    MiproConfig,
    // Request types
    GepaJobRequest,
    MiproJobRequest,
    EvalJobRequest,
    // Response types
    JobSubmitResponse,
    PromptLearningResult,
    EvalResult,
    // Graph types
    GraphCompletionRequest,
    GraphCompletionResponse,
    VerifierOptions,
    VerifierResponse,
    RlmOptions,
    Usage,
};
pub use jobs::JobsClient;
pub use eval::EvalClient;
pub use graphs::GraphsClient;
