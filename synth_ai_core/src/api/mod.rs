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

pub mod client;
pub mod eval;
pub mod graph_evolve;
pub mod graphs;
pub mod inference;
pub mod jobs;
pub mod localapi;
pub mod types;

// Re-export main types for convenience
pub use client::SynthClient;
pub use eval::EvalClient;
pub use graph_evolve::GraphEvolveClient;
pub use graphs::{build_verifier_request, resolve_graph_job_id, GraphsClient};
pub use inference::InferenceClient;
pub use jobs::JobsClient;
pub use localapi::LocalApiDeployClient;
pub use types::{
    EvalJobRequest,
    EvalJobStatus,
    EvalResult,
    GepaConfig,
    // Request types
    GepaJobRequest,
    // Graph types
    GraphCompletionRequest,
    GraphCompletionResponse,
    // Response types
    JobSubmitResponse,
    LocalApiDeployResponse,
    LocalApiDeploySpec,
    LocalApiDeployStatus,
    LocalApiDeploymentInfo,
    LocalApiLimits,
    MiproConfig,
    MiproJobRequest,
    // Config types
    PolicyConfig,
    // Job status
    PolicyJobStatus,
    PromptLearningResult,
    RlmOptions,
    Usage,
    VerifierOptions,
    VerifierResponse,
};
