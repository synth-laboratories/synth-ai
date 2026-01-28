//! Graph Evolve API client.
//!
//! This module provides methods for submitting and monitoring Graph Evolve jobs,
//! with fallbacks to legacy GraphGen endpoints where applicable.

use serde_json::Value;

use crate::http::HttpError;
use crate::CoreError;

use super::client::SynthClient;

/// Graph Evolve API client.
pub struct GraphEvolveClient<'a> {
    client: &'a SynthClient,
}

impl<'a> GraphEvolveClient<'a> {
    /// Create a new Graph Evolve client.
    pub(crate) fn new(client: &'a SynthClient) -> Self {
        Self { client }
    }

    async fn get_with_fallback(
        &self,
        primary: &str,
        legacy: Option<&str>,
    ) -> Result<Value, CoreError> {
        match self.client.http.get::<Value>(primary, None).await {
            Ok(value) => Ok(value),
            Err(e) => {
                if let Some(404) = e.status() {
                    if let Some(legacy_path) = legacy {
                        return self
                            .client
                            .http
                            .get::<Value>(legacy_path, None)
                            .await
                            .map_err(map_http_error);
                    }
                }
                Err(map_http_error(e))
            }
        }
    }

    async fn post_with_fallback(
        &self,
        primary: &str,
        legacy: Option<&str>,
        payload: &Value,
    ) -> Result<Value, CoreError> {
        match self.client.http.post_json(primary, payload).await {
            Ok(value) => Ok(value),
            Err(e) => {
                if let Some(404) = e.status() {
                    if let Some(legacy_path) = legacy {
                        return self
                            .client
                            .http
                            .post_json(legacy_path, payload)
                            .await
                            .map_err(map_http_error);
                    }
                }
                Err(map_http_error(e))
            }
        }
    }

    /// Submit a Graph Evolve job.
    pub async fn submit_job(&self, payload: Value) -> Result<Value, CoreError> {
        self.post_with_fallback(
            "/api/graph-evolve/jobs",
            Some("/api/graphgen/jobs"),
            &payload,
        )
        .await
    }

    /// Get job status.
    pub async fn get_status(&self, job_id: &str) -> Result<Value, CoreError> {
        let primary = format!("/api/graph-evolve/jobs/{job_id}");
        let legacy = format!("/api/graphgen/jobs/{job_id}");
        self.get_with_fallback(&primary, Some(&legacy)).await
    }

    /// Start a queued job.
    pub async fn start_job(&self, job_id: &str) -> Result<Value, CoreError> {
        let primary = format!("/api/graph-evolve/jobs/{job_id}/start");
        let legacy = format!("/api/graphgen/jobs/{job_id}/start");
        self.post_with_fallback(&primary, Some(&legacy), &Value::Null)
            .await
    }

    /// Get job events.
    pub async fn get_events(
        &self,
        job_id: &str,
        since_seq: i64,
        limit: i64,
    ) -> Result<Value, CoreError> {
        let primary = format!(
            "/api/graph-evolve/jobs/{job_id}/events?since_seq={since_seq}&limit={limit}"
        );
        let legacy = format!(
            "/api/graphgen/jobs/{job_id}/events?since_seq={since_seq}&limit={limit}"
        );
        self.get_with_fallback(&primary, Some(&legacy)).await
    }

    /// Get job metrics.
    pub async fn get_metrics(&self, job_id: &str, query: &str) -> Result<Value, CoreError> {
        let primary = format!("/api/graph-evolve/jobs/{job_id}/metrics?{query}");
        let legacy = format!("/api/graphgen/jobs/{job_id}/metrics?{query}");
        self.get_with_fallback(&primary, Some(&legacy)).await
    }

    /// Download prompt artifacts.
    pub async fn download_prompt(&self, job_id: &str) -> Result<Value, CoreError> {
        let primary = format!("/api/graph-evolve/jobs/{job_id}/download");
        let legacy = format!("/api/graphgen/jobs/{job_id}/download");
        self.get_with_fallback(&primary, Some(&legacy)).await
    }

    /// Download graph text export.
    pub async fn download_graph_txt(&self, job_id: &str) -> Result<String, CoreError> {
        let primary = format!("/api/graph-evolve/jobs/{job_id}/graph.txt");
        let legacy = format!("/api/graphgen/jobs/{job_id}/graph.txt");

        let bytes = match self.client.http.get_bytes(&primary, None).await {
            Ok(data) => data,
            Err(e) => {
                if let Some(404) = e.status() {
                    self.client
                        .http
                        .get_bytes(&legacy, None)
                        .await
                        .map_err(map_http_error)?
                } else {
                    return Err(map_http_error(e));
                }
            }
        };

        String::from_utf8(bytes)
            .map_err(|e| CoreError::Internal(format!("invalid utf-8 in graph export: {e}")))
    }

    /// Run inference against a graph.
    pub async fn run_inference(&self, payload: Value) -> Result<Value, CoreError> {
        self.client
            .http
            .post_json("/api/graphgen/graph/completions", &payload)
            .await
            .map_err(map_http_error)
    }

    /// Fetch graph record.
    pub async fn get_graph_record(&self, payload: Value) -> Result<Value, CoreError> {
        self.client
            .http
            .post_json("/api/graphgen/graph/record", &payload)
            .await
            .map_err(map_http_error)
    }

    /// Cancel a running job.
    pub async fn cancel_job(&self, job_id: &str, payload: Value) -> Result<Value, CoreError> {
        let path = format!("/api/jobs/{job_id}/cancel");
        self.client
            .http
            .post_json(&path, &payload)
            .await
            .map_err(map_http_error)
    }

    /// Query workflow state.
    pub async fn query_workflow_state(&self, job_id: &str) -> Result<Value, CoreError> {
        let path = format!("/api/jobs/{job_id}/workflow-state");
        self.client
            .http
            .get(&path, None)
            .await
            .map_err(map_http_error)
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
                    api: "graph_evolve".to_string(),
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

