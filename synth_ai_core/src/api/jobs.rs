//! Jobs API client.
//!
//! This module provides methods for submitting, polling, and canceling
//! optimization jobs (GEPA, MIPRO).

use std::time::{Duration, Instant};

use reqwest::header::{HeaderMap, HeaderValue};
use serde_json::Value;

use crate::http::HttpError;
use crate::polling::{calculate_backoff, BackoffConfig};
use crate::CoreError;

use super::client::SynthClient;
use super::types::{
    CancelRequest, GepaJobRequest, JobSubmitResponse, MiproJobRequest, PolicyJobStatus,
    PromptLearningResult,
};

/// Canonical API endpoint root for job status/events.
const JOBS_ENDPOINT: &str = "/api/jobs";

/// Canonical create endpoint for GEPA jobs.
const GEPA_CREATE_ENDPOINT: &str = "/api/jobs/gepa";

/// Canonical create endpoint for MIPRO jobs.
const MIPRO_CREATE_ENDPOINT: &str = "/api/jobs/mipro";

/// Legacy API endpoint for prompt learning job submission (fallback).
const LEGACY_SUBMIT_ENDPOINT: &str = "/api/prompt-learning/online/jobs";

/// Jobs API client.
///
/// Use this to submit, poll, and cancel optimization jobs (GEPA, MIPRO).
pub struct JobsClient<'a> {
    client: &'a SynthClient,
}

impl<'a> JobsClient<'a> {
    /// Create a new Jobs client.
    pub(crate) fn new(client: &'a SynthClient) -> Self {
        Self { client }
    }

    /// Submit a GEPA optimization job.
    ///
    /// # Arguments
    ///
    /// * `request` - The GEPA job configuration
    ///
    /// # Returns
    ///
    /// The job ID on success.
    ///
    /// # Example
    ///
    /// ```ignore
    /// let job_id = client.jobs().submit_gepa(GepaJobRequest {
    ///     task_app_url: "http://localhost:8000".into(),
    ///     env_name: "default".into(),
    ///     policy: PolicyConfig::default(),
    ///     gepa: GepaConfig::default(),
    ///     ..Default::default()
    /// }).await?;
    /// ```
    pub async fn submit_gepa(&self, request: GepaJobRequest) -> Result<String, CoreError> {
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
                .post_json_with_headers(GEPA_CREATE_ENDPOINT, &body, Some(headers))
                .await
                .map_err(map_http_error)?
        } else {
            self.client
                .http
                .post_json(GEPA_CREATE_ENDPOINT, &body)
                .await
                .map_err(map_http_error)?
        };

        Ok(response.job_id)
    }

    /// Submit a MIPRO optimization job.
    ///
    /// # Arguments
    ///
    /// * `request` - The MIPRO job configuration
    ///
    /// # Returns
    ///
    /// The job ID on success.
    pub async fn submit_mipro(&self, request: MiproJobRequest) -> Result<String, CoreError> {
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
                .post_json_with_headers(MIPRO_CREATE_ENDPOINT, &body, Some(headers))
                .await
                .map_err(map_http_error)?
        } else {
            self.client
                .http
                .post_json(MIPRO_CREATE_ENDPOINT, &body)
                .await
                .map_err(map_http_error)?
        };

        Ok(response.job_id)
    }

    /// Submit a generic optimization job from a JSON value.
    ///
    /// Use this when you have a pre-built job configuration.
    /// Submit a generic optimization job from a JSON value.
    ///
    /// Use this when you have a pre-built job configuration.
    /// Falls back to the legacy endpoint since the job type is unknown.
    pub async fn submit_raw(&self, request: Value) -> Result<String, CoreError> {
        self.submit_raw_with_worker_token(request, None).await
    }

    /// Submit a generic optimization job with optional SynthTunnel worker token.
    pub async fn submit_raw_with_worker_token(
        &self,
        request: Value,
        worker_token: Option<String>,
    ) -> Result<String, CoreError> {
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
                .post_json_with_headers(LEGACY_SUBMIT_ENDPOINT, &request, Some(headers))
                .await
                .map_err(map_http_error)?
        } else {
            self.client
                .http
                .post_json(LEGACY_SUBMIT_ENDPOINT, &request)
                .await
                .map_err(map_http_error)?
        };

        Ok(response.job_id)
    }

    /// Get the current status of a job.
    ///
    /// # Arguments
    ///
    /// * `job_id` - The job ID to check
    ///
    /// # Returns
    ///
    /// The current job result including status, best score, etc.
    pub async fn get_status(&self, job_id: &str) -> Result<PromptLearningResult, CoreError> {
        let path = format!("{}/{}", JOBS_ENDPOINT, job_id);

        self.client
            .http
            .get(&path, None)
            .await
            .map_err(map_http_error)
    }

    /// Poll a job until it reaches a terminal state.
    ///
    /// This will repeatedly check the job status until it succeeds, fails,
    /// or the timeout is reached.
    ///
    /// # Arguments
    ///
    /// * `job_id` - The job ID to poll
    /// * `timeout_secs` - Maximum time to wait (in seconds)
    /// * `interval_secs` - Initial polling interval (in seconds)
    ///
    /// # Returns
    ///
    /// The final job result.
    ///
    /// # Example
    ///
    /// ```ignore
    /// // Poll for up to 1 hour, checking every 15 seconds
    /// let result = client.jobs().poll_until_complete(&job_id, 3600.0, 15.0).await?;
    /// if result.status.is_success() {
    ///     println!("Best score: {:?}", result.best_score);
    /// }
    /// ```
    pub async fn poll_until_complete(
        &self,
        job_id: &str,
        timeout_secs: f64,
        interval_secs: f64,
    ) -> Result<PromptLearningResult, CoreError> {
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
                    "job {} did not complete within {:.0} seconds",
                    job_id, timeout_secs
                )));
            }

            // Get status
            match self.get_status(job_id).await {
                Ok(result) => {
                    consecutive_errors = 0;

                    if result.status.is_terminal() {
                        match result.status {
                            PolicyJobStatus::Failed => {
                                let error = result.error.as_deref().unwrap_or("unknown");
                                eprintln!("[synth_ai_core] Job {} FAILED: {}", job_id, error);
                            }
                            PolicyJobStatus::Cancelled => {
                                eprintln!("[synth_ai_core] Job {} was cancelled", job_id);
                            }
                            _ => {}
                        }
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
                            "too many consecutive errors polling job {}: {}",
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

    /// Cancel a running job.
    ///
    /// # Arguments
    ///
    /// * `job_id` - The job ID to cancel
    /// * `reason` - Optional cancellation reason
    pub async fn cancel(&self, job_id: &str, reason: Option<&str>) -> Result<(), CoreError> {
        let path = format!("{}/{}/cancel", JOBS_ENDPOINT, job_id);
        let body = serde_json::to_value(&CancelRequest {
            reason: reason.map(|s| s.to_string()),
        })
        .unwrap_or(Value::Object(serde_json::Map::new()));

        let _: Value = self
            .client
            .http
            .post_json(&path, &body)
            .await
            .map_err(map_http_error)?;

        Ok(())
    }

    /// List recent jobs.
    ///
    /// # Arguments
    ///
    /// * `limit` - Maximum number of jobs to return
    /// * `status` - Optional status filter
    pub async fn list(
        &self,
        limit: Option<i32>,
        status: Option<PolicyJobStatus>,
    ) -> Result<Vec<PromptLearningResult>, CoreError> {
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
            .get(JOBS_ENDPOINT, params_ref)
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
                    api: "jobs".to_string(),
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
    fn test_jobs_endpoint() {
        assert_eq!(JOBS_ENDPOINT, "/api/jobs");
        assert_eq!(GEPA_CREATE_ENDPOINT, "/api/jobs/gepa");
        assert_eq!(MIPRO_CREATE_ENDPOINT, "/api/jobs/mipro");
        assert_eq!(LEGACY_SUBMIT_ENDPOINT, "/api/prompt-learning/online/jobs");
    }
}
