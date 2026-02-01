//! Eval API client.
//!
//! This module provides methods for submitting and managing evaluation jobs.

use std::time::{Duration, Instant};

use reqwest::header::{HeaderMap, HeaderValue};
use serde_json::Value;

use crate::http::HttpError;
use crate::polling::{calculate_backoff, BackoffConfig};
use crate::CoreError;

use super::client::SynthClient;
use super::types::{CancelRequest, EvalJobRequest, EvalJobStatus, EvalResult, JobSubmitResponse};

/// Canonical API endpoint root for job status/events.
const JOBS_ROOT: &str = "/api/jobs";

/// Canonical create endpoint for eval jobs.
const EVAL_CREATE_ENDPOINT: &str = "/api/jobs/eval";

/// Legacy API endpoint for eval-specific operations (status, results, traces, list).
const EVAL_ENDPOINT: &str = "/api/eval/jobs";

/// Eval API client.
///
/// Use this to submit, poll, and cancel evaluation jobs.
pub struct EvalClient<'a> {
    client: &'a SynthClient,
}

impl<'a> EvalClient<'a> {
    /// Create a new Eval client.
    pub(crate) fn new(client: &'a SynthClient) -> Self {
        Self { client }
    }

    /// Submit an evaluation job.
    ///
    /// # Arguments
    ///
    /// * `request` - The eval job configuration
    ///
    /// # Returns
    ///
    /// The job ID on success.
    ///
    /// # Example
    ///
    /// ```ignore
    /// let job_id = client.eval().submit(EvalJobRequest {
    ///     task_app_url: "http://localhost:8000".into(),
    ///     env_name: "default".into(),
    ///     seeds: vec![1, 2, 3, 4, 5],
    ///     policy: PolicyConfig::default(),
    ///     ..Default::default()
    /// }).await?;
    /// ```
    pub async fn submit(&self, request: EvalJobRequest) -> Result<String, CoreError> {
        let worker_token = request.task_app_worker_token.clone();
        let body = serde_json::to_value(&request)
            .map_err(|e| CoreError::Validation(format!("failed to serialize request: {}", e)))?;
        let response: JobSubmitResponse = if let Some(token) = worker_token {
            let mut headers = HeaderMap::new();
            headers.insert(
                "X-SynthTunnel-Worker-Token",
                HeaderValue::from_str(&token).map_err(|_| {
                    CoreError::Validation("invalid SynthTunnel worker token".to_string())
                })?,
            );
            self.client
                .http
                .post_json_with_headers(EVAL_CREATE_ENDPOINT, &body, Some(headers))
                .await
                .map_err(map_http_error)?
        } else {
            self.client
                .http
                .post_json(EVAL_CREATE_ENDPOINT, &body)
                .await
                .map_err(map_http_error)?
        };

        Ok(response.job_id)
    }

    /// Submit a raw evaluation job from a JSON value.
    pub async fn submit_raw(&self, request: Value) -> Result<String, CoreError> {
        let response: JobSubmitResponse = self
            .client
            .http
            .post_json(EVAL_CREATE_ENDPOINT, &request)
            .await
            .map_err(map_http_error)?;

        Ok(response.job_id)
    }

    /// Get the current status of an eval job.
    ///
    /// # Arguments
    ///
    /// * `job_id` - The job ID to check
    ///
    /// # Returns
    ///
    /// The current eval result including status, mean reward, etc.
    pub async fn get_status(&self, job_id: &str) -> Result<EvalResult, CoreError> {
        // TODO: migrate to /api/jobs/{id} once backend response shape is unified
        let path = format!("{}/{}", EVAL_ENDPOINT, job_id);
        self.client
            .http
            .get(&path, None)
            .await
            .map_err(map_http_error)
    }

    /// Get detailed eval results for a job.
    ///
    /// This includes per-seed metrics, tokens, costs, and errors.
    pub async fn get_results(&self, job_id: &str) -> Result<Value, CoreError> {
        let path = format!("{}/{}/results", EVAL_ENDPOINT, job_id);
        self.client
            .http
            .get(&path, None)
            .await
            .map_err(map_http_error)
    }

    /// Download traces for an eval job as a ZIP archive.
    pub async fn download_traces(&self, job_id: &str) -> Result<Vec<u8>, CoreError> {
        let path = format!("{}/{}/traces", EVAL_ENDPOINT, job_id);
        self.client
            .http
            .get_bytes(&path, None)
            .await
            .map_err(map_http_error)
    }

    /// Poll an eval job until it reaches a terminal state.
    ///
    /// # Arguments
    ///
    /// * `job_id` - The job ID to poll
    /// * `timeout_secs` - Maximum time to wait (in seconds)
    /// * `interval_secs` - Initial polling interval (in seconds)
    ///
    /// # Returns
    ///
    /// The final eval result.
    ///
    /// # Example
    ///
    /// ```ignore
    /// let result = client.eval().poll_until_complete(&job_id, 1800.0, 10.0).await?;
    /// println!("Mean reward: {:?}", result.mean_reward);
    /// ```
    pub async fn poll_until_complete(
        &self,
        job_id: &str,
        timeout_secs: f64,
        interval_secs: f64,
    ) -> Result<EvalResult, CoreError> {
        let start = Instant::now();
        let timeout = Duration::from_secs_f64(timeout_secs);
        let base_interval_ms = (interval_secs * 1000.0) as u64;

        let config = BackoffConfig::new(base_interval_ms, 60000, 4);

        let mut consecutive_errors = 0u32;
        let max_errors = 10u32;

        loop {
            // Check timeout
            if start.elapsed() > timeout {
                return Err(CoreError::Timeout(format!(
                    "eval job {} did not complete within {:.0} seconds",
                    job_id, timeout_secs
                )));
            }

            // Get status
            match self.get_status(job_id).await {
                Ok(result) => {
                    consecutive_errors = 0;

                    if result.status.is_terminal() {
                        return Ok(result);
                    }

                    // Wait before next poll
                    let delay = calculate_backoff(&config, 0);
                    tokio::time::sleep(delay).await;
                }
                Err(e) => {
                    consecutive_errors += 1;

                    if consecutive_errors >= max_errors {
                        return Err(CoreError::Internal(format!(
                            "too many consecutive errors polling eval job {}: {}",
                            job_id, e
                        )));
                    }

                    // Backoff on errors
                    let delay = calculate_backoff(&config, consecutive_errors);
                    tokio::time::sleep(delay).await;
                }
            }
        }
    }

    /// Cancel a running eval job.
    ///
    /// # Arguments
    ///
    /// * `job_id` - The job ID to cancel
    /// * `reason` - Optional cancellation reason
    pub async fn cancel(&self, job_id: &str, reason: Option<String>) -> Result<Value, CoreError> {
        let path = format!("{}/{}/cancel", JOBS_ROOT, job_id);
        let body = serde_json::to_value(&CancelRequest { reason })
            .unwrap_or(Value::Object(serde_json::Map::new()));

        self.client
            .http
            .post_json(&path, &body)
            .await
            .map_err(map_http_error)
    }

    /// Query workflow state for an eval job.
    pub async fn query_workflow_state(&self, job_id: &str) -> Result<Value, CoreError> {
        let path = format!("/api/jobs/{}/workflow-state", job_id);
        self.client
            .http
            .get_json(&path, None)
            .await
            .map_err(map_http_error)
    }

    /// List recent eval jobs.
    ///
    /// # Arguments
    ///
    /// * `limit` - Maximum number of jobs to return
    /// * `status` - Optional status filter
    pub async fn list(
        &self,
        limit: Option<i32>,
        status: Option<EvalJobStatus>,
    ) -> Result<Vec<EvalResult>, CoreError> {
        let mut params = vec![];

        let limit_str;
        if let Some(l) = limit {
            limit_str = l.to_string();
            params.push(("limit", limit_str.as_str()));
        }

        let status_str;
        if let Some(s) = status {
            status_str = s.as_str().to_string();
            params.push(("status", status_str.as_str()));
        }

        let params_ref: Option<&[(&str, &str)]> = if params.is_empty() {
            None
        } else {
            Some(&params)
        };

        self.client
            .http
            .get(EVAL_ENDPOINT, params_ref)
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
                    api: "eval".to_string(),
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
    fn test_eval_endpoint() {
        assert_eq!(EVAL_CREATE_ENDPOINT, "/api/jobs/eval");
        assert_eq!(EVAL_ENDPOINT, "/api/eval/jobs");
        assert_eq!(JOBS_ROOT, "/api/jobs");
    }
}
