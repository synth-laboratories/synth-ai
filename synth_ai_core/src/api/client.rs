//! Main Synth API client.
//!
//! The `SynthClient` is the primary entry point for interacting with the Synth API.

use crate::auth;
use crate::http::HttpClient;
use crate::CoreError;

use super::eval::EvalClient;
use super::graph_evolve::GraphEvolveClient;
use super::graphs::GraphsClient;
use super::inference::InferenceClient;
use super::jobs::JobsClient;
use super::localapi::LocalApiDeployClient;

/// Default backend URL.
pub const DEFAULT_BACKEND_URL: &str = "https://api.usesynth.ai";

/// Default request timeout in seconds.
pub const DEFAULT_TIMEOUT_SECS: u64 = 120;

/// Synth API client.
///
/// This is the main entry point for interacting with the Synth API.
/// It provides access to sub-clients for different API endpoints.
///
/// # Example
///
/// ```ignore
/// use synth_ai_core::api::SynthClient;
///
/// // Create from environment variable
/// let client = SynthClient::from_env()?;
///
/// // Or with explicit API key
/// let client = SynthClient::new("sk_live_...", None)?;
///
/// // Access sub-clients
/// let jobs = client.jobs();
/// let eval = client.eval();
/// let graphs = client.graphs();
/// ```
pub struct SynthClient {
    pub(crate) http: HttpClient,
    pub(crate) base_url: String,
}

impl SynthClient {
    /// Create a new Synth client with an API key.
    ///
    /// # Arguments
    ///
    /// * `api_key` - Your Synth API key
    /// * `base_url` - Optional base URL (defaults to https://api.usesynth.ai)
    ///
    /// # Example
    ///
    /// ```ignore
    /// let client = SynthClient::new("sk_live_...", None)?;
    /// ```
    pub fn new(api_key: &str, base_url: Option<&str>) -> Result<Self, CoreError> {
        let base_url = base_url.unwrap_or(DEFAULT_BACKEND_URL).to_string();
        let http = HttpClient::new(&base_url, api_key, DEFAULT_TIMEOUT_SECS)
            .map_err(|e| CoreError::Internal(format!("failed to create HTTP client: {}", e)))?;

        Ok(Self { http, base_url })
    }

    /// Create a new Synth client with custom timeout.
    ///
    /// # Arguments
    ///
    /// * `api_key` - Your Synth API key
    /// * `base_url` - Optional base URL
    /// * `timeout_secs` - Request timeout in seconds
    pub fn with_timeout(
        api_key: &str,
        base_url: Option<&str>,
        timeout_secs: u64,
    ) -> Result<Self, CoreError> {
        let base_url = base_url.unwrap_or(DEFAULT_BACKEND_URL).to_string();
        let http = HttpClient::new(&base_url, api_key, timeout_secs)
            .map_err(|e| CoreError::Internal(format!("failed to create HTTP client: {}", e)))?;

        Ok(Self { http, base_url })
    }

    /// Create a client from environment variables.
    ///
    /// Reads the API key from `SYNTH_API_KEY` environment variable.
    /// Optionally reads base URL from `SYNTH_BACKEND_URL`.
    ///
    /// # Example
    ///
    /// ```ignore
    /// std::env::set_var("SYNTH_API_KEY", "sk_live_...");
    /// let client = SynthClient::from_env()?;
    /// ```
    pub fn from_env() -> Result<Self, CoreError> {
        let api_key = auth::get_api_key(None).ok_or_else(|| {
            CoreError::Authentication("SYNTH_API_KEY environment variable not set".to_string())
        })?;

        let base_url = std::env::var("SYNTH_BACKEND_URL").ok();
        Self::new(&api_key, base_url.as_deref())
    }

    /// Create a client, minting a demo key if needed.
    ///
    /// This will:
    /// 1. Try to get an API key from environment
    /// 2. If not found and `allow_mint` is true, mint a demo key
    ///
    /// # Arguments
    ///
    /// * `allow_mint` - Whether to mint a demo key if no key is found
    /// * `base_url` - Optional base URL
    pub async fn from_env_or_mint(
        allow_mint: bool,
        base_url: Option<&str>,
    ) -> Result<Self, CoreError> {
        let api_key = auth::get_or_mint_api_key(base_url, allow_mint).await?;
        Self::new(&api_key, base_url)
    }

    /// Get the base URL for this client.
    pub fn base_url(&self) -> &str {
        &self.base_url
    }

    /// Get a reference to the HTTP client.
    pub fn http(&self) -> &HttpClient {
        &self.http
    }

    /// Get a Jobs API client.
    ///
    /// Use this to submit, poll, and cancel optimization jobs.
    ///
    /// # Example
    ///
    /// ```ignore
    /// let job_id = client.jobs().submit_gepa(request).await?;
    /// let result = client.jobs().poll_until_complete(&job_id, 3600.0, 15.0).await?;
    /// ```
    pub fn jobs(&self) -> JobsClient<'_> {
        JobsClient::new(self)
    }

    /// Get an Eval API client.
    ///
    /// Use this to run evaluation jobs.
    ///
    /// # Example
    ///
    /// ```ignore
    /// let job_id = client.eval().submit(request).await?;
    /// let result = client.eval().poll_until_complete(&job_id, 3600.0, 15.0).await?;
    /// ```
    pub fn eval(&self) -> EvalClient<'_> {
        EvalClient::new(self)
    }

    /// Get a Graphs API client.
    ///
    /// Use this for graph completions and verifier inference.
    ///
    /// # Example
    ///
    /// ```ignore
    /// let result = client.graphs().verify(trace, rubric, None).await?;
    /// ```
    pub fn graphs(&self) -> GraphsClient<'_> {
        GraphsClient::new(self)
    }

    /// Get a Graph Evolve API client.
    ///
    /// Use this for Graph Evolve / GraphGen optimization endpoints.
    pub fn graph_evolve(&self) -> GraphEvolveClient<'_> {
        GraphEvolveClient::new(self)
    }

    /// Get an Inference API client.
    ///
    /// Use this for chat completions via the inference proxy.
    pub fn inference(&self) -> InferenceClient<'_> {
        InferenceClient::new(&self.http)
    }

    /// Get a LocalAPI Deployments client.
    ///
    /// Use this to deploy and manage managed LocalAPI deployments.
    pub fn localapi(&self) -> LocalApiDeployClient<'_> {
        LocalApiDeployClient::new(self)
    }
}

impl std::fmt::Debug for SynthClient {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SynthClient")
            .field("base_url", &self.base_url)
            .finish_non_exhaustive()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_backend_url() {
        assert_eq!(DEFAULT_BACKEND_URL, "https://api.usesynth.ai");
    }
}
