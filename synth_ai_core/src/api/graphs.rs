//! Graphs API client.
//!
//! This module provides methods for graph completions and verifier inference.

use serde_json::{json, Value};
use std::time::Duration;

use crate::http::HttpError;
use crate::sse::stream_sse_request;
use reqwest::header::{HeaderMap, HeaderName, HeaderValue};
use reqwest::Method;
use crate::CoreError;

use super::client::SynthClient;
use super::types::{
    GraphCompletionRequest, GraphCompletionResponse, ListGraphsResponse, RlmOptions,
    VerifierOptions, VerifierResponse,
};

/// API endpoint for graph completions.
const GRAPHS_ENDPOINT: &str = "/api/graphs/completions";
/// API endpoint for listing registered graphs.
const GRAPH_LIST_ENDPOINT: &str = "/api/graph-evolve/graphs";

/// Default verifier graph ID.
pub const DEFAULT_VERIFIER: &str = "zero_shot_verifier_rubric_single";

/// RLM v1 verifier for large contexts.
pub const RLM_VERIFIER_V1: &str = "zero_shot_verifier_rubric_rlm";

/// RLM v2 verifier (multi-agent).
pub const RLM_VERIFIER_V2: &str = "zero_shot_verifier_rubric_rlm_v2";

/// Graphs API client.
///
/// Use this for graph completions and verifier inference.
pub struct GraphsClient<'a> {
    client: &'a SynthClient,
}

impl<'a> GraphsClient<'a> {
    /// Create a new Graphs client.
    pub(crate) fn new(client: &'a SynthClient) -> Self {
        Self { client }
    }

    /// Execute a graph completion.
    ///
    /// # Arguments
    ///
    /// * `request` - The graph completion request
    ///
    /// # Returns
    ///
    /// The graph output.
    ///
    /// # Example
    ///
    /// ```ignore
    /// let response = client.graphs().complete(GraphCompletionRequest {
    ///     job_id: "my-graph".into(),
    ///     input: json!({"prompt": "Hello"}),
    ///     model: None,
    ///     stream: None,
    /// }).await?;
    /// ```
    pub async fn complete(
        &self,
        request: GraphCompletionRequest,
    ) -> Result<GraphCompletionResponse, CoreError> {
        let body = serde_json::to_value(&request)
            .map_err(|e| CoreError::Validation(format!("failed to serialize request: {}", e)))?;

        self.client
            .http
            .post_json(GRAPHS_ENDPOINT, &body)
            .await
            .map_err(map_http_error)
    }

    /// List registered graphs for the organization.
    pub async fn list_graphs(
        &self,
        kind: Option<&str>,
        limit: Option<i32>,
    ) -> Result<ListGraphsResponse, CoreError> {
        let mut path = GRAPH_LIST_ENDPOINT.to_string();
        let mut params: Vec<String> = Vec::new();

        if let Some(kind) = kind {
            params.push(format!("kind={}", kind));
        }
        if let Some(limit) = limit {
            params.push(format!("limit={}", limit));
        }
        if !params.is_empty() {
            path.push('?');
            path.push_str(&params.join("&"));
        }

        self.client
            .http
            .get::<ListGraphsResponse>(&path, None)
            .await
            .map_err(map_http_error)
    }

    /// Stream a graph completion via SSE.
    pub async fn stream_completion(
        &self,
        mut request: GraphCompletionRequest,
        timeout: Option<Duration>,
    ) -> Result<crate::sse::SseStream, CoreError> {
        request.stream = Some(true);
        let body = serde_json::to_value(&request)
            .map_err(|e| CoreError::Validation(format!("failed to serialize request: {}", e)))?;

        let base = self.client.base_url.trim_end_matches('/');
        let api_base = if base.ends_with("/api") {
            base.to_string()
        } else {
            format!("{}/api", base)
        };
        let url = format!("{}/graphs/completions?stream=true", api_base);
        let api_key = self.client.http.api_key().to_string();
        
        let mut header_map = HeaderMap::new();
        header_map.insert(
            HeaderName::from_static("authorization"),
            HeaderValue::try_from(format!("Bearer {}", api_key)).unwrap(),
        );
        header_map.insert(
            HeaderName::from_static("x-api-key"),
            HeaderValue::try_from(api_key).unwrap(),
        );
        header_map.insert(
            HeaderName::from_static("content-type"),
            HeaderValue::from_static("application/json"),
        );
        header_map.insert(
            HeaderName::from_static("accept"),
            HeaderValue::from_static("text/event-stream"),
        );

        stream_sse_request(url, Method::POST, header_map, Some(body), timeout).await
    }

    /// Execute a raw graph completion from a JSON value.
    pub async fn complete_raw(&self, request: Value) -> Result<Value, CoreError> {
        self.client
            .http
            .post_json(GRAPHS_ENDPOINT, &request)
            .await
            .map_err(map_http_error)
    }

    /// Run verifier inference on a trace.
    ///
    /// This evaluates a trace against a rubric using the verifier graph.
    ///
    /// # Arguments
    ///
    /// * `trace` - The trace to verify (JSON object with events)
    /// * `rubric` - The rubric to evaluate against
    /// * `options` - Optional verifier configuration
    ///
    /// # Returns
    ///
    /// The verification result with scores and reviews.
    ///
    /// # Example
    ///
    /// ```ignore
    /// let result = client.graphs().verify(
    ///     json!({
    ///         "events": [
    ///             {"type": "user_message", "content": "Hello"},
    ///             {"type": "assistant_message", "content": "Hi there!"}
    ///         ]
    ///     }),
    ///     json!({
    ///         "objectives": [
    ///             {"name": "helpfulness", "description": "Be helpful"}
    ///         ]
    ///     }),
    ///     None,
    /// ).await?;
    /// println!("Objectives: {:?}", result.objectives);
    /// ```
    pub async fn verify(
        &self,
        trace: Value,
        rubric: Value,
        options: Option<VerifierOptions>,
    ) -> Result<VerifierResponse, CoreError> {
        let options = options.unwrap_or_default();
        let verifier_id = options
            .verifier_id
            .as_deref()
            .unwrap_or(DEFAULT_VERIFIER);

        let mut input = json!({
            "trace": trace,
            "rubric": rubric,
        });

        if let Some(model) = &options.model {
            input["model"] = json!(model);
        }

        let request = GraphCompletionRequest {
            job_id: verifier_id.to_string(),
            input,
            model: options.model.clone(),
            prompt_snapshot_id: None,
            stream: Some(false),
        };

        let body = serde_json::to_value(&request)
            .map_err(|e| CoreError::Validation(format!("failed to serialize request: {}", e)))?;

        let response: GraphCompletionResponse = self
            .client
            .http
            .post_json(GRAPHS_ENDPOINT, &body)
            .await
            .map_err(map_http_error)?;

        // Parse the output as VerifierResponse
        serde_json::from_value(response.output.clone()).map_err(|e| {
            CoreError::Validation(format!(
                "failed to parse verifier response: {} (output: {:?})",
                e, response.output
            ))
        })
    }

    /// Run RLM (Retrieval-augmented LM) inference.
    ///
    /// This is useful for large context scenarios where the full trace
    /// doesn't fit in a single context window.
    ///
    /// # Arguments
    ///
    /// * `query` - The query/question to answer
    /// * `context` - The context to search through
    /// * `options` - Optional RLM configuration
    ///
    /// # Returns
    ///
    /// The RLM output as a JSON value.
    pub async fn rlm_inference(
        &self,
        query: &str,
        context: Value,
        options: Option<RlmOptions>,
    ) -> Result<Value, CoreError> {
        let options = options.unwrap_or_default();
        let rlm_id = options.rlm_id.as_deref().unwrap_or(RLM_VERIFIER_V1);

        let mut input = json!({
            "query": query,
            "context": context,
        });

        if let Some(max_tokens) = options.max_context_tokens {
            input["max_context_tokens"] = json!(max_tokens);
        }

        let request = GraphCompletionRequest {
            job_id: rlm_id.to_string(),
            input,
            model: options.model,
            prompt_snapshot_id: None,
            stream: Some(false),
        };

        let body = serde_json::to_value(&request)
            .map_err(|e| CoreError::Validation(format!("failed to serialize request: {}", e)))?;

        let response: GraphCompletionResponse = self
            .client
            .http
            .post_json(GRAPHS_ENDPOINT, &body)
            .await
            .map_err(map_http_error)?;

        Ok(response.output)
    }

    /// Execute a policy/prompt from a job.
    ///
    /// This runs inference using a trained policy from a completed
    /// optimization job.
    ///
    /// # Arguments
    ///
    /// * `job_id` - The optimization job ID
    /// * `input` - The input to the policy
    /// * `model` - Optional model override
    ///
    /// # Returns
    ///
    /// The policy output.
    pub async fn policy_inference(
        &self,
        job_id: &str,
        input: Value,
        model: Option<&str>,
    ) -> Result<Value, CoreError> {
        let request = GraphCompletionRequest {
            job_id: job_id.to_string(),
            input,
            model: model.map(|s| s.to_string()),
            prompt_snapshot_id: None,
            stream: Some(false),
        };

        let body = serde_json::to_value(&request)
            .map_err(|e| CoreError::Validation(format!("failed to serialize request: {}", e)))?;

        let response: GraphCompletionResponse = self
            .client
            .http
            .post_json(GRAPHS_ENDPOINT, &body)
            .await
            .map_err(map_http_error)?;

        Ok(response.output)
    }
}

/// Map HTTP errors to CoreError.
fn map_http_error(e: HttpError) -> CoreError {
    match e {
        HttpError::Response(detail) => {
            if detail.status == 401 || detail.status == 403 {
                CoreError::Authentication(format!("authentication failed: {}", detail))
            } else if detail.status == 429 {
                CoreError::UsageLimit(crate::UsageLimitInfo {
                    limit_type: "rate_limit".to_string(),
                    api: "graphs".to_string(),
                    current: 0.0,
                    limit: 0.0,
                    tier: "unknown".to_string(),
                    retry_after_seconds: None,
                    upgrade_url: "https://usesynth.ai/pricing".to_string(),
                })
            } else {
                CoreError::HttpResponse(crate::HttpErrorInfo {
                    status: detail.status,
                    url: detail.url,
                    message: detail.message,
                    body_snippet: detail.body_snippet,
                })
            }
        }
        HttpError::Request(e) => CoreError::Http(e),
        _ => CoreError::Internal(format!("{}", e)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_graphs_endpoint() {
        assert_eq!(GRAPHS_ENDPOINT, "/api/graphs/completions");
    }

    #[test]
    fn test_verifier_constants() {
        assert_eq!(DEFAULT_VERIFIER, "zero_shot_verifier_rubric_single");
        assert_eq!(RLM_VERIFIER_V1, "zero_shot_verifier_rubric_rlm");
        assert_eq!(RLM_VERIFIER_V2, "zero_shot_verifier_rubric_rlm_v2");
    }
}
