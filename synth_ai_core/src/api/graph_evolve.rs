//! Graph Evolve API client.
//!
//! Provides helpers for the Graph Evolve / GraphGen optimization endpoints.

use serde_json::{json, Value};

use crate::http::HttpError;
use crate::CoreError;

use super::client::SynthClient;

const GRAPH_EVOLVE_ENDPOINT: &str = "/api/graph-evolve/jobs";
const GRAPHGEN_canonical_ENDPOINT: &str = "/api/graphgen/jobs";
const GRAPHGEN_COMPLETIONS_ENDPOINT: &str = "/api/graphgen/graph/completions";
const GRAPHGEN_RECORD_ENDPOINT: &str = "/api/graphgen/graph/record";

/// Graph Evolve API client.
pub struct GraphEvolveClient<'a> {
    client: &'a SynthClient,
}

impl<'a> GraphEvolveClient<'a> {
    pub(crate) fn new(client: &'a SynthClient) -> Self {
        Self { client }
    }

    async fn get_with_strict(
        &self,
        primary: &str,
        canonical: Option<&str>,
        params: Option<&[(&str, &str)]>,
    ) -> Result<Value, CoreError> {
        match self.client.http.get_json(primary, params).await {
            Ok(value) => Ok(value),
            Err(err) => {
                if err.status() == Some(404) {
                    if let Some(canonical) = canonical {
                        return self
                            .client
                            .http
                            .get_json(canonical, params)
                            .await
                            .map_err(map_http_error);
                    }
                }
                Err(map_http_error(err))
            }
        }
    }

    async fn post_with_strict(
        &self,
        primary: &str,
        canonical: Option<&str>,
        payload: &Value,
    ) -> Result<Value, CoreError> {
        match self.client.http.post_json(primary, payload).await {
            Ok(value) => Ok(value),
            Err(err) => {
                if err.status() == Some(404) {
                    if let Some(canonical) = canonical {
                        return self
                            .client
                            .http
                            .post_json(canonical, payload)
                            .await
                            .map_err(map_http_error);
                    }
                }
                Err(map_http_error(err))
            }
        }
    }

    async fn get_bytes_with_strict(
        &self,
        primary: &str,
        canonical: Option<&str>,
    ) -> Result<Vec<u8>, CoreError> {
        match self.client.http.get_bytes(primary, None).await {
            Ok(bytes) => Ok(bytes),
            Err(err) => {
                if err.status() == Some(404) {
                    if let Some(canonical) = canonical {
                        return self
                            .client
                            .http
                            .get_bytes(canonical, None)
                            .await
                            .map_err(map_http_error);
                    }
                }
                Err(map_http_error(err))
            }
        }
    }

    /// Submit a Graph Evolve job.
    pub async fn submit_job(&self, payload: Value) -> Result<Value, CoreError> {
        self.post_with_strict(
            GRAPH_EVOLVE_ENDPOINT,
            Some(GRAPHGEN_canonical_ENDPOINT),
            &payload,
        )
        .await
    }

    /// Get job status.
    pub async fn get_status(&self, job_id: &str) -> Result<Value, CoreError> {
        let primary = format!("{}/{}", GRAPH_EVOLVE_ENDPOINT, job_id);
        let canonical = format!("{}/{}", GRAPHGEN_canonical_ENDPOINT, job_id);
        self.get_with_strict(&primary, Some(&canonical), None).await
    }

    /// Start a job.
    pub async fn start_job(&self, job_id: &str) -> Result<Value, CoreError> {
        let primary = format!("{}/{}/start", GRAPH_EVOLVE_ENDPOINT, job_id);
        let canonical = format!("{}/{}/start", GRAPHGEN_canonical_ENDPOINT, job_id);
        self.post_with_strict(&primary, Some(&canonical), &Value::Null)
            .await
    }

    /// Get job events.
    pub async fn get_events(
        &self,
        job_id: &str,
        since_seq: i64,
        limit: i64,
    ) -> Result<Value, CoreError> {
        let primary = format!("{}/{}/events", GRAPH_EVOLVE_ENDPOINT, job_id);
        let canonical = format!("{}/{}/events", GRAPHGEN_canonical_ENDPOINT, job_id);
        let since = since_seq.to_string();
        let limit_s = limit.to_string();
        let params = [("since_seq", since.as_str()), ("limit", limit_s.as_str())];
        self.get_with_strict(&primary, Some(&canonical), Some(&params))
            .await
    }

    /// Get job metrics.
    pub async fn get_metrics(&self, job_id: &str, query_string: &str) -> Result<Value, CoreError> {
        let query = query_string.trim_start_matches('?');
        let primary = if query.is_empty() {
            format!("{}/{}/metrics", GRAPH_EVOLVE_ENDPOINT, job_id)
        } else {
            format!("{}/{}/metrics?{}", GRAPH_EVOLVE_ENDPOINT, job_id, query)
        };
        let canonical = if query.is_empty() {
            format!("{}/{}/metrics", GRAPHGEN_canonical_ENDPOINT, job_id)
        } else {
            format!("{}/{}/metrics?{}", GRAPHGEN_canonical_ENDPOINT, job_id, query)
        };
        self.get_with_strict(&primary, Some(&canonical), None).await
    }

    /// Download best prompt (JSON).
    pub async fn download_prompt(&self, job_id: &str) -> Result<Value, CoreError> {
        let primary = format!("{}/{}/download", GRAPH_EVOLVE_ENDPOINT, job_id);
        let canonical = format!("{}/{}/download", GRAPHGEN_canonical_ENDPOINT, job_id);
        self.get_with_strict(&primary, Some(&canonical), None).await
    }

    /// Download redacted graph export (text).
    pub async fn download_graph_txt(&self, job_id: &str) -> Result<String, CoreError> {
        let primary = format!("{}/{}/graph.txt", GRAPH_EVOLVE_ENDPOINT, job_id);
        let canonical = format!("{}/{}/graph.txt", GRAPHGEN_canonical_ENDPOINT, job_id);
        let bytes = self
            .get_bytes_with_strict(&primary, Some(&canonical))
            .await?;
        Ok(String::from_utf8_lossy(&bytes).to_string())
    }

    /// Run inference using a graph.
    pub async fn run_inference(&self, payload: Value) -> Result<Value, CoreError> {
        self.client
            .http
            .post_json(GRAPHGEN_COMPLETIONS_ENDPOINT, &payload)
            .await
            .map_err(map_http_error)
    }

    /// Fetch a graph record snapshot.
    pub async fn get_graph_record(&self, payload: Value) -> Result<Value, CoreError> {
        self.client
            .http
            .post_json(GRAPHGEN_RECORD_ENDPOINT, &payload)
            .await
            .map_err(map_http_error)
    }

    /// Cancel a job.
    pub async fn cancel_job(&self, job_id: &str, payload: Value) -> Result<Value, CoreError> {
        let path = format!("/api/v1/offline/jobs/{}", job_id);
        let body = json!({
            "state": "cancelled",
            "reason": payload.get("reason").cloned().unwrap_or(Value::Null),
        });
        self.client
            .http
            .patch_json(&path, &body)
            .await
            .map_err(map_http_error)
    }

    /// Query workflow state directly.
    pub async fn query_workflow_state(&self, job_id: &str) -> Result<Value, CoreError> {
        let path = format!("/api/v1/offline/jobs/{}", job_id);
        match self.client.http.get_json(&path, None).await {
            Ok(value) => Ok(value),
            Err(err) => {
                let status = err
                    .status()
                    .map(|s| s.to_string())
                    .unwrap_or_else(|| "unknown".to_string());
                Ok(json!({
                    "job_id": job_id,
                    "workflow_state": Value::Null,
                    "error": format!("HTTP {}: {}", status, err),
                }))
            }
        }
    }
}

/// Map HTTP errors to CoreError.
fn map_http_error(e: HttpError) -> CoreError {
    match e {
        HttpError::Response(detail) => {
            if detail.status == 401 || detail.status == 403 {
                CoreError::Authentication(format!("authentication failed: {}", detail))
            } else if detail.status == 429 {
                CoreError::UsageLimit(crate::UsageLimitInfo::from_http_429(
                    "graph_evolve",
                    &detail,
                ))
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
