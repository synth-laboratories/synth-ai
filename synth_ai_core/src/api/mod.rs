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
//!         container_url: "http://localhost:8000".into(),
//!         env_name: "default".into(),
//!         policy: PolicyConfig::default(),
//!         ..Default::default()
//!     }).await?;
//!
//!     // Poll until complete
//!     let result = client.jobs().poll_until_complete(&job_id, 3600.0, 15.0).await?;
//!     println!("Best score: {:?}", result.best_reward);
//!
//!     Ok(())
//! }
//! ```

pub mod client;
pub mod container;
pub mod jobs;
pub mod routes;
pub mod types;

// Re-export main types for convenience
pub use client::SynthClient;
pub use container::ContainerDeployClient;
pub use jobs::JobsClient;
pub use routes::{
    ApiVersion, offline_job_path, offline_job_subpath, offline_jobs_base, online_session_path,
    online_session_subpath, online_sessions_base, policy_system_path, policy_systems_base,
    EVAL_API_VERSION, GEPA_API_VERSION, MIPRO_API_VERSION,
};
pub use types::{
    ContainerDeployResponse,
    ContainerDeploySpec,
    ContainerDeployStatus,
    ContainerDeploymentInfo,
    ContainerLimits,
    EvalJobRequest,
    EvalJobStatus,
    EvalResult,
    GepaConfig,
    // Request types
    GepaJobRequest,
    // Graph types (kept — shared type definitions)
    GraphCompletionRequest,
    GraphCompletionResponse,
    // Response types
    JobSubmitResponse,
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
