//! Graphs API client.
//!
//! This module provides methods for graph completions and verifier inference.

use serde_json::Map;
use serde_json::{json, Value};

use crate::http::HttpError;
use crate::CoreError;

use super::client::SynthClient;
use super::types::{
    GraphCompletionRequest, GraphCompletionResponse, RlmOptions, VerifierOptions, VerifierResponse,
};

/// API endpoint for graph completions.
const GRAPHS_ENDPOINT: &str = "/api/graphs/completions";

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

    /// List graphs registered to the org.
    pub async fn list_graphs(
        &self,
        kind: Option<&str>,
        limit: Option<i32>,
    ) -> Result<Value, CoreError> {
        let mut params = Vec::new();
        let limit_str;
        if let Some(limit_val) = limit {
            limit_str = limit_val.to_string();
            params.push(("limit", limit_str.as_str()));
        }

        let kind_val;
        if let Some(kind_val_raw) = kind {
            kind_val = kind_val_raw.to_string();
            params.push(("kind", kind_val.as_str()));
        }

        let params_ref: Option<&[(&str, &str)]> = if params.is_empty() {
            None
        } else {
            Some(&params)
        };

        self.client
            .http
            .get_json("/graph-evolve/graphs", params_ref)
            .await
            .map_err(map_http_error)
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
        let verifier_id = options.verifier_id.as_deref().unwrap_or(DEFAULT_VERIFIER);

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

/// Build a verifier graph completion request from SDK inputs.
///
/// This matches the Python SDK behavior for `verify_with_rubric`:
/// - Uses trace_ref if provided, otherwise trace_content
/// - Defaults to RLM for trace_ref, otherwise chooses by estimated size
/// - Supports optional system/user prompts and options payload
pub fn build_verifier_request(
    trace_content: Option<Value>,
    trace_ref: Option<String>,
    rubric: Value,
    system_prompt: Option<String>,
    user_prompt: Option<String>,
    options: Option<Value>,
    model: Option<String>,
    verifier_shape: Option<String>,
    rlm_impl: Option<String>,
) -> Result<GraphCompletionRequest, CoreError> {
    let has_ref = trace_ref.as_ref().map(|s| !s.is_empty()).unwrap_or(false);
    let has_content = trace_content.is_some();

    if !has_ref && !has_content {
        return Err(CoreError::Validation(
            "trace_content or trace_ref is required".to_string(),
        ));
    }

    let shape = match verifier_shape.as_deref() {
        Some("single") => "single",
        Some("rlm") => "rlm",
        Some(other) => {
            return Err(CoreError::Validation(format!(
                "unsupported verifier_shape: {}",
                other
            )))
        }
        None => {
            if has_ref {
                "rlm"
            } else {
                let tokens = estimate_trace_tokens(trace_content.as_ref().unwrap())?;
                if tokens < 50_000 {
                    "single"
                } else {
                    "rlm"
                }
            }
        }
    };

    let verifier_id = if shape == "single" {
        if rlm_impl
            .as_deref()
            .map(|value| !value.is_empty())
            .unwrap_or(false)
        {
            return Err(CoreError::Validation(
                "rlm_impl is only valid when verifier_shape is 'rlm'".to_string(),
            ));
        }
        DEFAULT_VERIFIER
    } else if matches!(rlm_impl.as_deref(), Some("v2")) {
        RLM_VERIFIER_V2
    } else {
        RLM_VERIFIER_V1
    };

    let mut input = Map::new();
    input.insert("rubric".to_string(), rubric);
    input.insert(
        "options".to_string(),
        options.unwrap_or_else(|| Value::Object(Map::new())),
    );

    if let Some(trace_ref) = trace_ref {
        if !trace_ref.is_empty() {
            input.insert("trace_ref".to_string(), Value::String(trace_ref));
        }
    }

    if !input.contains_key("trace_ref") {
        if let Some(trace_content) = trace_content {
            input.insert("trace_content".to_string(), trace_content);
        }
    }

    if let Some(system_prompt) = system_prompt {
        input.insert("system_prompt".to_string(), Value::String(system_prompt));
    }
    if let Some(user_prompt) = user_prompt {
        input.insert("user_prompt".to_string(), Value::String(user_prompt));
    }

    Ok(GraphCompletionRequest {
        job_id: verifier_id.to_string(),
        input: Value::Object(input),
        model,
        prompt_snapshot_id: None,
        stream: Some(false),
    })
}

fn estimate_trace_tokens(trace: &Value) -> Result<usize, CoreError> {
    let payload = serde_json::to_string(trace)
        .map_err(|e| CoreError::Validation(format!("failed to serialize trace: {}", e)))?;
    Ok(payload.len() / 4)
}

/// Resolve a graph job ID from explicit job_id or graph target spec.
///
/// Mirrors the Python SDK logic for graph target resolution.
pub fn resolve_graph_job_id(
    job_id: Option<String>,
    graph: Option<Value>,
) -> Result<String, CoreError> {
    if let Some(job_id) = job_id {
        if !job_id.trim().is_empty() {
            return Ok(job_id);
        }
    }

    let graph = graph
        .ok_or_else(|| CoreError::Validation("graph_completions_missing_job_id".to_string()))?;

    let graph_obj = graph
        .as_object()
        .ok_or_else(|| CoreError::Validation("graph target must be an object".to_string()))?;

    if let Some(Value::String(job_id)) = graph_obj.get("job_id") {
        if !job_id.trim().is_empty() {
            return Ok(job_id.clone());
        }
    }

    let kind = graph_obj.get("kind").and_then(|v| v.as_str()).unwrap_or("");
    if kind == "zero_shot" {
        if let Some(shape) = graph_obj
            .get("verifier_shape")
            .and_then(|v| v.as_str())
            .or_else(|| graph_obj.get("graph_name").and_then(|v| v.as_str()))
        {
            return Ok(shape.to_string());
        }
        return Err(CoreError::Validation(
            "graph_completions_missing_verifier_shape".to_string(),
        ));
    }

    if kind == "graphgen" {
        if let Some(graphgen_job_id) = graph_obj.get("graphgen_job_id").and_then(|v| v.as_str()) {
            return Ok(graphgen_job_id.to_string());
        }
        return Err(CoreError::Validation(
            "graph_completions_missing_graphgen_job_id".to_string(),
        ));
    }

    if let Some(graph_name) = graph_obj.get("graph_name").and_then(|v| v.as_str()) {
        return Ok(graph_name.to_string());
    }

    Err(CoreError::Validation(
        "graph_completions_missing_graph_target".to_string(),
    ))
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
