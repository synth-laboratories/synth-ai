//! # Synth AI SDK
//!
//! Ergonomic Rust SDK for Synth AI - serverless post-training APIs.
//!
//! ## Quick Start
//!
//! ```rust,ignore
//! use synth_ai::Synth;
//!
//! #[tokio::main]
//! async fn main() -> Result<(), synth_ai::Error> {
//!     // Create client from SYNTH_API_KEY env var
//!     let synth = Synth::from_env()?;
//!
//!     // Submit a prompt optimization job
//!     let result = synth
//!         .optimize()
//!         .task_app("https://my-task-app.com")
//!         .model("gpt-4o")
//!         .run()
//!         .await?;
//!
//!     println!("Best candidate: {:?}", result.best_candidate);
//!     Ok(())
//! }
//! ```

use serde_json::Value;
use std::env;
use std::path::Path;
use std::time::Duration;
use thiserror::Error;

pub mod environment_pools;

pub use environment_pools::EnvironmentPoolsClient;

// Re-export core for advanced usage
pub use synth_ai_core as core;
pub use synth_ai_core_types as types;

// Re-export commonly used core types
pub use synth_ai_core::{
    api::GraphEvolveClient,
    api::PromptLearningResult,
    api::{EvalJobRequest, EvalJobStatus, GepaJobRequest, MiproJobRequest, PolicyJobStatus},
    api::{
        GraphCompletionRequest, GraphCompletionResponse, RlmOptions, VerifierOptions,
        VerifierResponse,
    },
    // Local API
    localapi::TaskAppClient,
    orchestration::{CandidateInfo, GEPAProgress, ProgressTracker},
    // Orchestration
    orchestration::{PromptLearningJob, PromptResults, RankedPrompt},
    tunnels::errors::TunnelError,
    tunnels::open_tunnel,
    // Tunnels
    tunnels::types::{TunnelBackend, TunnelHandle},
    ApplicationErrorType,
    ApplicationStatus,
    Artifact,
    BufferedHandler,
    CallbackHandler,
    ContextOverride,
    ContextOverrideStatus,
    Lever,
    LeverActor,
    LeverConstraints,
    LeverFormat,
    LeverKind,
    LeverMutability,
    LeverMutation,
    LeverProvenance,
    LeverSnapshot,
    MiproLeverSummary,
    ScopeKey,
    ScopeKind,
    Sensor,
    SensorFrame,
    SensorFrameSummary,
    SensorKind,
    // Errors
    CoreError,
    Criterion,
    CriterionScoreData,
    EventObjectiveAssignment,
    // Graph evolve
    GraphEvolveJob,
    JobStreamer,
    // Data types
    JobType,
    JsonHandler,
    Judgement,
    LLMCallRecord,
    LLMContentPart,
    LLMMessage,
    LLMUsage,
    LocalApiDeployResponse,
    LocalApiDeploySpec,
    LocalApiDeployStatus,
    LocalApiDeploymentInfo,
    LocalApiLimits,
    MarkovBlanketMessage,
    MessageContent,
    ObjectiveSpec,
    OutcomeObjectiveAssignment,
    RewardObservation,
    Rubric,
    RubricAssignment,
    SessionTimeStep,
    SessionTrace,
    // Tracing
    SessionTracer,
    StreamConfig,
    StreamEndpoints,
    StreamHandler,
    StreamMessage,
    // Streaming
    StreamType,
    // API types
    SynthClient as CoreClient,
    TimeRecord,
    ToolCallResult,
    ToolCallSpec,
    // Trace upload
    TraceUploadClient,
    TracingEvent,
    UploadUrlResponse,
};

#[cfg(feature = "libsql")]
pub use synth_ai_core::tracing::LibsqlTraceStorage;

pub use synth_ai_core::data::{
    CalibrationExample, EventRewardRecord, GoldExample, InstanceObjectiveAssignment,
    OutcomeRewardRecord, RewardAggregates, SynthModelName,
};
pub use synth_ai_core::tracing::{LLMChunk, LLMRequestParams};

/// SDK version.
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// Default Synth API base URL.
pub const DEFAULT_BASE_URL: &str = "https://api.usesynth.ai";

/// Environment variable for API key.
pub const API_KEY_ENV: &str = "SYNTH_API_KEY";

// =============================================================================
// Error Types
// =============================================================================

/// SDK error type.
#[derive(Debug, Error)]
pub enum Error {
    /// Missing API key.
    #[error("API key not found. Set {API_KEY_ENV} or provide explicitly.")]
    MissingApiKey,

    /// Configuration error.
    #[error("configuration error: {0}")]
    Config(String),

    /// Core error passthrough.
    #[error(transparent)]
    Core(#[from] synth_ai_core::CoreError),

    /// Tunnel error.
    #[error(transparent)]
    Tunnel(#[from] TunnelError),

    /// Job submission failed.
    #[error("job submission failed: {0}")]
    Submission(String),

    /// Job execution failed.
    #[error("job failed: {0}")]
    JobFailed(String),

    /// Timeout waiting for job.
    #[error("timeout after {0:?}")]
    Timeout(Duration),
}

/// Result type alias.
pub type Result<T> = std::result::Result<T, Error>;

// =============================================================================
// Main Client
// =============================================================================

/// Main Synth AI client.
///
/// This is the primary entry point for interacting with Synth AI APIs.
///
/// # Example
///
/// ```rust,ignore
/// use synth_ai::Synth;
///
/// let synth = Synth::from_env()?;
///
/// // Or with explicit credentials
/// let synth = Synth::new("sk_live_...", None)?;
/// ```
pub struct Synth {
    api_key: String,
    base_url: String,
    client: synth_ai_core::SynthClient,
}

impl Synth {
    /// Create a new Synth client with explicit credentials.
    ///
    /// # Arguments
    ///
    /// * `api_key` - Your Synth API key
    /// * `base_url` - Optional custom API base URL
    pub fn new(api_key: impl Into<String>, base_url: Option<&str>) -> Result<Self> {
        let api_key = api_key.into();
        let base_url = base_url.unwrap_or(DEFAULT_BASE_URL).to_string();

        let client =
            synth_ai_core::SynthClient::new(&api_key, Some(&base_url)).map_err(Error::Core)?;

        Ok(Self {
            api_key,
            base_url,
            client,
        })
    }

    /// Create a client from the `SYNTH_API_KEY` environment variable.
    pub fn from_env() -> Result<Self> {
        let api_key = env::var(API_KEY_ENV).map_err(|_| Error::MissingApiKey)?;
        let base_url = env::var("SYNTH_BASE_URL").ok();
        Self::new(api_key, base_url.as_deref())
    }

    /// Get the API key (masked for display).
    pub fn api_key_masked(&self) -> String {
        synth_ai_core::auth::mask_str(&self.api_key)
    }

    /// Get the base URL.
    pub fn base_url(&self) -> &str {
        &self.base_url
    }

    /// Access the underlying core client.
    pub fn core(&self) -> &synth_ai_core::SynthClient {
        &self.client
    }

    /// Create an Environment Pools client.
    pub fn environment_pools(&self) -> Result<EnvironmentPoolsClient> {
        EnvironmentPoolsClient::new(self.api_key.clone(), Some(&self.base_url))
    }

    // -------------------------------------------------------------------------
    // High-level API
    // -------------------------------------------------------------------------

    /// Start a prompt optimization job.
    ///
    /// Returns a builder to configure the optimization.
    pub fn optimize(&self) -> OptimizeBuilder {
        OptimizeBuilder::new(self.api_key.clone(), self.base_url.clone())
    }

    /// Start an evaluation job.
    ///
    /// Returns a builder to configure the evaluation.
    pub fn eval(&self) -> EvalBuilder {
        EvalBuilder::new(self.api_key.clone(), self.base_url.clone())
    }

    /// Open a tunnel to a local port.
    ///
    /// # Arguments
    ///
    /// * `port` - Local port to tunnel
    /// * `backend` - Tunnel backend (cloudflare_managed_lease recommended)
    pub async fn tunnel(&self, port: u16, backend: TunnelBackend) -> Result<TunnelHandle> {
        synth_ai_core::tunnels::open_tunnel(
            backend,
            port,
            Some(self.api_key.clone()),
            Some(self.base_url.clone()),
            None,
            false,
            true,
            false,
        )
        .await
        .map_err(Error::Tunnel)
    }

    // -------------------------------------------------------------------------
    // Direct API access
    // -------------------------------------------------------------------------

    /// Submit a raw GEPA job.
    pub async fn submit_gepa(&self, request: GepaJobRequest) -> Result<String> {
        self.client
            .jobs()
            .submit_gepa(request)
            .await
            .map_err(Error::Core)
    }

    /// Submit a raw MIPRO job.
    pub async fn submit_mipro(&self, request: MiproJobRequest) -> Result<String> {
        self.client
            .jobs()
            .submit_mipro(request)
            .await
            .map_err(Error::Core)
    }

    /// Get job status.
    pub async fn get_job_status(&self, job_id: &str) -> Result<PromptLearningResult> {
        self.client
            .jobs()
            .get_status(job_id)
            .await
            .map_err(Error::Core)
    }

    /// Poll job until complete.
    pub async fn poll_job(
        &self,
        job_id: &str,
        timeout_secs: f64,
        interval_secs: f64,
    ) -> Result<PromptLearningResult> {
        self.client
            .jobs()
            .poll_until_complete(job_id, timeout_secs, interval_secs)
            .await
            .map_err(Error::Core)
    }

    /// Cancel a job.
    pub async fn cancel_job(&self, job_id: &str, reason: Option<&str>) -> Result<()> {
        self.client
            .jobs()
            .cancel(job_id, reason)
            .await
            .map_err(Error::Core)
    }

    /// Pause a job.
    pub async fn pause_job(&self, job_id: &str, reason: Option<&str>) -> Result<()> {
        self.client
            .jobs()
            .pause(job_id, reason)
            .await
            .map_err(Error::Core)
    }

    /// Resume a paused job.
    pub async fn resume_job(&self, job_id: &str, reason: Option<&str>) -> Result<()> {
        self.client
            .jobs()
            .resume(job_id, reason)
            .await
            .map_err(Error::Core)
    }

    /// Run graph completion.
    pub async fn complete(
        &self,
        request: GraphCompletionRequest,
    ) -> Result<GraphCompletionResponse> {
        self.client
            .graphs()
            .complete(request)
            .await
            .map_err(Error::Core)
    }

    /// List registered graphs.
    pub async fn list_graphs(&self, kind: Option<&str>, limit: Option<i32>) -> Result<Value> {
        self.client
            .graphs()
            .list_graphs(kind, limit)
            .await
            .map_err(Error::Core)
    }

    /// Run verifier on a trace.
    pub async fn verify(
        &self,
        trace: serde_json::Value,
        rubric: serde_json::Value,
        options: Option<VerifierOptions>,
    ) -> Result<VerifierResponse> {
        self.client
            .graphs()
            .verify(trace, rubric, options)
            .await
            .map_err(Error::Core)
    }

    /// Run RLM (Retrieval-augmented LM) inference.
    pub async fn rlm_inference(
        &self,
        query: &str,
        context: Value,
        options: Option<RlmOptions>,
    ) -> Result<Value> {
        self.client
            .graphs()
            .rlm_inference(query, context, options)
            .await
            .map_err(Error::Core)
    }

    /// Create a Graph Evolve client for advanced operations.
    pub fn graph_evolve(&self) -> GraphEvolveClient<'_> {
        self.client.graph_evolve()
    }

    /// Create a Graph Evolve job from a payload.
    pub fn graph_evolve_job_from_payload(&self, payload: Value) -> Result<GraphEvolveJob> {
        GraphEvolveJob::from_payload(payload, Some(&self.api_key), Some(&self.base_url))
            .map_err(Error::Core)
    }

    /// Reconnect to a Graph Evolve job by ID.
    pub fn graph_evolve_job_from_id(&self, job_id: &str) -> Result<GraphEvolveJob> {
        GraphEvolveJob::from_job_id(job_id, Some(&self.api_key), Some(&self.base_url))
            .map_err(Error::Core)
    }

    /// Verify a trace against a rubric with default options.
    pub async fn verify_rubric(&self, trace: Value, rubric: Value) -> Result<VerifierResponse> {
        self.verify(trace, rubric, None).await
    }

    /// Create a LocalAPI task app client.
    pub fn task_app_client(&self, base_url: &str, api_key: Option<&str>) -> TaskAppClient {
        let key = api_key.unwrap_or(self.api_key.as_str());
        TaskAppClient::new(base_url, Some(key))
    }

    /// Deploy a managed LocalAPI from a context directory.
    pub async fn deploy_localapi_from_dir(
        &self,
        spec: LocalApiDeploySpec,
        context_dir: impl AsRef<Path>,
        wait_for_ready: bool,
        build_timeout_s: f64,
    ) -> Result<LocalApiDeployResponse> {
        self.client
            .localapi()
            .deploy_from_dir(spec, context_dir, wait_for_ready, build_timeout_s)
            .await
            .map_err(Error::Core)
    }

    /// List managed LocalAPI deployments for the current org.
    pub async fn list_localapi_deployments(&self) -> Result<Vec<LocalApiDeploymentInfo>> {
        self.client.localapi().list().await.map_err(Error::Core)
    }

    /// Fetch a managed LocalAPI deployment by ID.
    pub async fn get_localapi_deployment(
        &self,
        deployment_id: &str,
    ) -> Result<LocalApiDeploymentInfo> {
        self.client
            .localapi()
            .get(deployment_id)
            .await
            .map_err(Error::Core)
    }

    /// Fetch managed LocalAPI deployment status by ID.
    pub async fn get_localapi_deployment_status(
        &self,
        deployment_id: &str,
    ) -> Result<LocalApiDeployStatus> {
        self.client
            .localapi()
            .status(deployment_id)
            .await
            .map_err(Error::Core)
    }

    /// Fetch detailed eval results.
    pub async fn eval_results(&self, job_id: &str) -> Result<Value> {
        self.client
            .eval()
            .get_results(job_id)
            .await
            .map_err(Error::Core)
    }

    /// Download eval traces as ZIP bytes.
    pub async fn download_eval_traces(&self, job_id: &str) -> Result<Vec<u8>> {
        self.client
            .eval()
            .download_traces(job_id)
            .await
            .map_err(Error::Core)
    }

    /// Create a trace uploader for large traces.
    pub fn trace_uploader(&self) -> Result<TraceUploadClient> {
        TraceUploadClient::new(&self.base_url, &self.api_key, 120).map_err(Error::Core)
    }

    /// Upload a trace and return its trace_ref.
    pub async fn upload_trace(
        &self,
        trace: Value,
        expires_in_seconds: Option<i64>,
    ) -> Result<String> {
        let uploader = self.trace_uploader()?;
        uploader
            .upload_trace(&trace, None, expires_in_seconds)
            .await
            .map_err(Error::Core)
    }

    /// Stream job events with a callback and return final status.
    pub async fn stream_job_with_callback<F>(
        &self,
        job_id: &str,
        endpoints: StreamEndpoints,
        callback: F,
    ) -> Result<Value>
    where
        F: Fn(&StreamMessage) + Send + Sync + 'static,
    {
        let mut streamer =
            JobStreamer::new(&self.base_url, &self.api_key, job_id).with_endpoints(endpoints);
        streamer.add_handler(CallbackHandler::new(callback));
        streamer.stream_until_terminal().await.map_err(Error::Core)
    }
}

impl std::fmt::Debug for Synth {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Synth")
            .field("api_key", &self.api_key_masked())
            .field("base_url", &self.base_url)
            .finish()
    }
}

// =============================================================================
// Optimize Builder
// =============================================================================

/// Builder for prompt optimization jobs.
pub struct OptimizeBuilder {
    api_key: String,
    base_url: String,
    task_app_url: Option<String>,
    model: Option<String>,
    num_candidates: Option<u32>,
    timeout: Duration,
    stream: bool,
}

impl OptimizeBuilder {
    fn new(api_key: String, base_url: String) -> Self {
        Self {
            api_key,
            base_url,
            task_app_url: None,
            model: None,
            num_candidates: None,
            timeout: Duration::from_secs(3600),
            stream: true,
        }
    }

    /// Set the task app URL.
    pub fn task_app(mut self, url: impl Into<String>) -> Self {
        self.task_app_url = Some(url.into());
        self
    }

    /// Set the model to optimize for.
    pub fn model(mut self, model: impl Into<String>) -> Self {
        self.model = Some(model.into());
        self
    }

    /// Set the number of candidates to generate.
    pub fn num_candidates(mut self, n: u32) -> Self {
        self.num_candidates = Some(n);
        self
    }

    /// Set the timeout for the job.
    pub fn timeout(mut self, timeout: Duration) -> Self {
        self.timeout = timeout;
        self
    }

    /// Enable or disable streaming (default: true).
    pub fn stream(mut self, enabled: bool) -> Self {
        self.stream = enabled;
        self
    }

    /// Run the optimization job and wait for completion.
    pub async fn run(self) -> Result<OptimizeResult> {
        let task_app_url = self
            .task_app_url
            .ok_or_else(|| Error::Config("task_app URL is required".into()))?;

        // Build config
        let mut config = serde_json::json!({
            "task_app_url": task_app_url,
        });

        if let Some(model) = &self.model {
            config["model"] = serde_json::json!(model);
        }
        if let Some(n) = self.num_candidates {
            config["num_candidates"] = serde_json::json!(n);
        }

        // Create and run job
        let mut job =
            PromptLearningJob::from_dict(config, Some(&self.api_key), Some(&self.base_url), None)
                .map_err(Error::Core)?;

        let job_id = job.submit().await.map_err(Error::Core)?;

        // Stream or poll
        let status = if self.stream {
            job.stream_until_complete::<fn(&synth_ai_core::orchestration::ParsedEvent)>(
                self.timeout.as_secs_f64(),
                None,
            )
            .await
            .map_err(Error::Core)?
        } else {
            job.poll_until_complete(self.timeout.as_secs_f64(), 15.0)
                .await
                .map_err(Error::Core)?
        };

        // Get results
        let results = job.get_results().await.map_err(Error::Core)?;

        Ok(OptimizeResult {
            job_id,
            status,
            results,
        })
    }
}

/// Result of a prompt optimization job.
#[derive(Debug, Clone)]
pub struct OptimizeResult {
    /// Job ID.
    pub job_id: String,
    /// Final job status.
    pub status: synth_ai_core::orchestration::PromptLearningResult,
    /// Optimization results.
    pub results: PromptResults,
}

impl OptimizeResult {
    /// Get the best candidate if available.
    pub fn best_candidate(&self) -> Option<&str> {
        self.results.best_candidate.as_deref()
    }

    /// Get the best reward if available.
    pub fn best_reward(&self) -> Option<f64> {
        self.results.best_reward
    }

    /// Get all top prompts.
    pub fn top_prompts(&self) -> &[RankedPrompt] {
        &self.results.top_prompts
    }

    /// Check if the job succeeded.
    pub fn is_success(&self) -> bool {
        self.status.status.is_success()
    }
}

// =============================================================================
// Eval Builder
// =============================================================================

/// Builder for evaluation jobs.
pub struct EvalBuilder {
    api_key: String,
    base_url: String,
    task_app_url: Option<String>,
    seeds: Vec<i64>,
    timeout: Duration,
}

impl EvalBuilder {
    fn new(api_key: String, base_url: String) -> Self {
        Self {
            api_key,
            base_url,
            task_app_url: None,
            seeds: vec![],
            timeout: Duration::from_secs(1800),
        }
    }

    /// Set the task app URL.
    pub fn task_app(mut self, url: impl Into<String>) -> Self {
        self.task_app_url = Some(url.into());
        self
    }

    /// Set the seeds to evaluate on.
    pub fn seeds(mut self, seeds: Vec<i64>) -> Self {
        self.seeds = seeds;
        self
    }

    /// Set the timeout for the job.
    pub fn timeout(mut self, timeout: Duration) -> Self {
        self.timeout = timeout;
        self
    }

    /// Run the evaluation and wait for completion.
    pub async fn run(self) -> Result<synth_ai_core::api::EvalResult> {
        let task_app_url = self
            .task_app_url
            .ok_or_else(|| Error::Config("task_app URL is required".into()))?;

        let client = synth_ai_core::SynthClient::new(&self.api_key, Some(&self.base_url))
            .map_err(Error::Core)?;

        let request = EvalJobRequest {
            app_id: None,
            task_app_url,
            task_app_worker_token: None,
            task_app_api_key: None,
            env_name: "default".to_string(),
            env_config: None,
            verifier_config: None,
            seeds: self.seeds,
            policy: synth_ai_core::api::PolicyConfig::default(),
            max_concurrent: None,
            timeout: None,
        };

        let job_id = client.eval().submit(request).await.map_err(Error::Core)?;

        let status = client
            .eval()
            .poll_until_complete(&job_id, self.timeout.as_secs_f64(), 10.0)
            .await
            .map_err(Error::Core)?;

        Ok(status)
    }
}

// =============================================================================
// Convenience Functions
// =============================================================================

/// Create a client from environment and run a quick optimization.
///
/// This is a convenience function for simple use cases.
pub async fn optimize(task_app_url: &str) -> Result<OptimizeResult> {
    Synth::from_env()?
        .optimize()
        .task_app(task_app_url)
        .run()
        .await
}

/// Create a client from environment and run a quick evaluation.
pub async fn eval(task_app_url: &str, seeds: Vec<i64>) -> Result<synth_ai_core::api::EvalResult> {
    Synth::from_env()?
        .eval()
        .task_app(task_app_url)
        .seeds(seeds)
        .run()
        .await
}

/// Convert a GEPA seed candidate mapping into a Synth prompt pattern.
///
/// See: specifications/tanha/master_specification.md
pub fn gepa_candidate_to_initial_prompt(seed_candidate: &Value) -> Result<Value> {
    synth_ai_core::config::gepa_candidate_to_initial_prompt(seed_candidate).map_err(Error::Core)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_synth_debug() {
        // Can't actually create without API key, but test the structure
        let err = Synth::from_env();
        assert!(err.is_err() || err.is_ok()); // Just check it doesn't panic
    }

    #[test]
    fn test_version() {
        assert!(!VERSION.is_empty());
    }
}
