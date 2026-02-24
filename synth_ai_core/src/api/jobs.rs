//! Jobs API client.
//!
//! This module provides methods for submitting, polling, and canceling
//! optimization jobs (GEPA, MIPRO).

use std::collections::HashMap;
use std::time::{Duration, Instant};

use reqwest::header::{HeaderMap, HeaderValue};
use serde::Deserialize;
use serde_json::Value;

use crate::http::HttpError;
use crate::polling::{calculate_backoff, BackoffConfig};
use crate::CoreError;

use super::client::SynthClient;
use super::types::{
    CancelRequest, GepaJobRequest, JobSubmitResponse, MiproJobRequest, PauseRequest,
    PolicyJobStatus, PromptLearningResult,
};

/// Canonical API endpoint root for job status/events.
const JOBS_ENDPOINT: &str = "/api/v1/offline/jobs";
/// Canonical generic create endpoint.
const JOBS_CREATE_ENDPOINT: &str = "/api/v1/offline/jobs";

const GEPA_KIND: &str = "gepa_offline";
const MIPRO_KIND: &str = "mipro_offline";

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
    ///     container_url: "http://localhost:8000".into(),
    ///     env_name: "default".into(),
    ///     policy: PolicyConfig::default(),
    ///     gepa: GepaConfig::default(),
    ///     ..Default::default()
    /// }).await?;
    /// ```
    pub async fn submit_gepa(&self, request: GepaJobRequest) -> Result<String, CoreError> {
        let worker_token = request.container_worker_token.clone();
        let raw_request = serde_json::to_value(&request)
            .map_err(|e| CoreError::Validation(format!("failed to serialize request: {}", e)))?;
        let body = canonicalize_offline_create_payload(raw_request, GEPA_KIND)?;
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
                .post_json_with_headers(JOBS_CREATE_ENDPOINT, &body, Some(headers))
                .await
                .map_err(map_http_error)?
        } else {
            self.client
                .http
                .post_json(JOBS_CREATE_ENDPOINT, &body)
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
        let worker_token = request.container_worker_token.clone();
        let raw_request = serde_json::to_value(&request)
            .map_err(|e| CoreError::Validation(format!("failed to serialize request: {}", e)))?;
        let body = canonicalize_offline_create_payload(raw_request, MIPRO_KIND)?;
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
                .post_json_with_headers(JOBS_CREATE_ENDPOINT, &body, Some(headers))
                .await
                .map_err(map_http_error)?
        } else {
            self.client
                .http
                .post_json(JOBS_CREATE_ENDPOINT, &body)
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
    pub async fn submit_raw(&self, request: Value) -> Result<String, CoreError> {
        self.submit_raw_with_worker_token(request, None).await
    }

    /// Submit a generic optimization job with optional SynthTunnel worker token.
    pub async fn submit_raw_with_worker_token(
        &self,
        request: Value,
        worker_token: Option<String>,
    ) -> Result<String, CoreError> {
        let canonical_request = canonicalize_raw_offline_request(request)?;
        let headers_opt: Option<HeaderMap> = if let Some(token) = worker_token {
            let mut headers = HeaderMap::new();
            headers.insert(
                "X-SynthTunnel-Worker-Token",
                HeaderValue::from_str(&token).map_err(|_| {
                    CoreError::Validation("invalid SynthTunnel worker token".to_string())
                })?,
            );
            Some(headers)
        } else {
            None
        };

        let generic_result: Result<JobSubmitResponse, HttpError> =
            if let Some(headers) = headers_opt.clone() {
                self.client
                    .http
                    .post_json_with_headers(JOBS_CREATE_ENDPOINT, &canonical_request, Some(headers))
                    .await
            } else {
                self.client
                    .http
                    .post_json(JOBS_CREATE_ENDPOINT, &canonical_request)
                    .await
            };
        match generic_result {
            Ok(response) => return Ok(response.job_id),
            Err(HttpError::Response(detail)) if detail.status == 404 || detail.status == 405 => {
                return Err(CoreError::HttpResponse(crate::HttpErrorInfo {
                    status: detail.status,
                    url: detail.url,
                    message: detail.message,
                    body_snippet: detail.body_snippet,
                }));
            }
            Err(HttpError::JsonParse(err)) => {
                return Err(CoreError::Internal(format!(
                    "invalid JSON response: {}",
                    err
                )));
            }
            Err(err) => return Err(map_http_error(err)),
        }
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
        let canonical = format!("{}/{}", JOBS_ENDPOINT, job_id);
        let payload = self
            .client
            .http
            .get(&canonical, None)
            .await
            .map_err(map_http_error)?;
        parse_prompt_learning_result(payload, job_id)
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

                    if result.status == PolicyJobStatus::Paused {
                        eprintln!("[synth_ai_core] Job {} is paused", job_id);
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
        let path = format!("{}/{}", JOBS_ENDPOINT, job_id);
        let body = serde_json::to_value(&CancelRequest {
            reason: reason.map(|s| s.to_string()),
        })
        .unwrap_or(Value::Object(serde_json::Map::new()));

        let _: Value = self
            .client
            .http
            .patch_json(&path, &serde_json::json!({"state": "cancelled", "reason": body.get("reason").cloned().unwrap_or(Value::Null)}))
            .await
            .map_err(map_http_error)?;

        Ok(())
    }

    /// Pause a running job.
    ///
    /// # Arguments
    ///
    /// * `job_id` - The job ID to pause
    /// * `reason` - Optional pause reason
    pub async fn pause(&self, job_id: &str, reason: Option<&str>) -> Result<(), CoreError> {
        let path = format!("{}/{}", JOBS_ENDPOINT, job_id);
        let body = serde_json::to_value(&PauseRequest {
            reason: reason.map(|s| s.to_string()),
        })
        .unwrap_or(Value::Object(serde_json::Map::new()));

        let _: Value = self
            .client
            .http
            .patch_json(&path, &serde_json::json!({"state": "paused", "reason": body.get("reason").cloned().unwrap_or(Value::Null)}))
            .await
            .map_err(map_http_error)?;

        Ok(())
    }

    /// Resume a paused job.
    ///
    /// # Arguments
    ///
    /// * `job_id` - The job ID to resume
    /// * `reason` - Optional resume reason
    pub async fn resume(&self, job_id: &str, reason: Option<&str>) -> Result<(), CoreError> {
        let path = format!("{}/{}", JOBS_ENDPOINT, job_id);
        let body = serde_json::to_value(&PauseRequest {
            reason: reason.map(|s| s.to_string()),
        })
        .unwrap_or(Value::Object(serde_json::Map::new()));

        let _: Value = self
            .client
            .http
            .patch_json(&path, &serde_json::json!({"state": "running", "reason": body.get("reason").cloned().unwrap_or(Value::Null)}))
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
            params.push(("state", status_str.as_str()));
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

#[derive(Debug, Deserialize)]
struct PromptLearningResultWire {
    #[serde(default)]
    job_id: Option<String>,
    #[serde(default)]
    status: Option<String>,
    #[serde(default)]
    best_reward: Option<f64>,
    #[serde(default)]
    best_score: Option<f64>,
    #[serde(default)]
    best_candidate: Option<Value>,
    #[serde(default)]
    best_prompt: Option<Value>,
    #[serde(default)]
    lever_summary: Option<Value>,
    #[serde(default)]
    sensor_frames: Vec<Value>,
    #[serde(default)]
    lever_versions: HashMap<String, i64>,
    #[serde(default)]
    best_lever_version: Option<i64>,
    #[serde(default)]
    error: Option<String>,
    #[serde(default)]
    generations_completed: Option<i32>,
    #[serde(default)]
    candidates_evaluated: Option<i32>,
}

fn parse_prompt_learning_result(
    payload: Value,
    strict_job_id: &str,
) -> Result<PromptLearningResult, CoreError> {
    let wire: PromptLearningResultWire = serde_json::from_value(payload).map_err(|err| {
        CoreError::Protocol(format!("invalid prompt learning status payload: {err}"))
    })?;
    let status = wire
        .status
        .as_deref()
        .and_then(PolicyJobStatus::from_str)
        .unwrap_or(PolicyJobStatus::Pending);
    Ok(PromptLearningResult {
        job_id: wire.job_id.unwrap_or_else(|| strict_job_id.to_string()),
        status,
        best_reward: wire.best_reward.or(wire.best_score),
        best_candidate: wire.best_candidate.or(wire.best_prompt),
        lever_summary: wire.lever_summary,
        sensor_frames: wire.sensor_frames,
        lever_versions: wire.lever_versions,
        best_lever_version: wire.best_lever_version,
        error: wire.error,
        generations_completed: wire.generations_completed,
        candidates_evaluated: wire.candidates_evaluated,
    })
}

fn infer_offline_kind(request: &Value) -> Option<&'static str> {
    let from_section = |name: &str| {
        request
            .get(name)
            .and_then(|value| value.get("algorithm"))
            .and_then(|value| value.as_str())
    };
    let algorithm = from_section("prompt_learning")
        .or_else(|| from_section("policy_optimization"))
        .or_else(|| request.get("algorithm").and_then(|value| value.as_str()))?
        .trim()
        .to_ascii_lowercase();
    match algorithm.as_str() {
        "gepa" => Some(GEPA_KIND),
        "mipro" | "voyager" => Some(MIPRO_KIND),
        _ => None,
    }
}

fn default_system_name(kind: &str) -> &'static str {
    match kind {
        GEPA_KIND => "sdk-gepa-offline",
        MIPRO_KIND => "sdk-mipro-offline",
        _ => "sdk-offline-job",
    }
}

fn derive_system_name(request: &Value, kind: &str) -> String {
    request
        .get("system")
        .and_then(|value| value.get("name"))
        .and_then(Value::as_str)
        .map(|value| value.trim().to_string())
        .filter(|value| !value.is_empty())
        .or_else(|| {
            request
                .get("env_name")
                .and_then(Value::as_str)
                .map(|value| value.trim().to_string())
                .filter(|value| !value.is_empty())
        })
        .unwrap_or_else(|| default_system_name(kind).to_string())
}

fn canonicalize_offline_create_payload(request: Value, kind: &str) -> Result<Value, CoreError> {
    let mut map = request
        .as_object()
        .cloned()
        .ok_or_else(|| CoreError::Validation("request payload must be a JSON object".to_string()))?;
    map.insert("kind".to_string(), Value::String(kind.to_string()));
    map.entry("technique".to_string())
        .or_insert(Value::String("discrete_optimization".to_string()));
    map.entry("config_mode".to_string())
        .or_insert(Value::String("DEFAULT".to_string()));
    map.entry("system".to_string()).or_insert_with(|| {
        serde_json::json!({
            "name": derive_system_name(&request, kind),
            "reuse": true,
        })
    });
    if !map.contains_key("config") {
        map.insert(
            "config".to_string(),
            serde_json::json!({
                "prompt_learning": request,
            }),
        );
    }
    Ok(Value::Object(map))
}

fn canonicalize_raw_offline_request(request: Value) -> Result<Value, CoreError> {
    if request
        .get("kind")
        .and_then(Value::as_str)
        .map(|value| !value.trim().is_empty())
        .unwrap_or(false)
    {
        return Ok(request);
    }
    let kind = infer_offline_kind(&request).ok_or_else(|| {
        CoreError::Validation(
            "request kind is required; expected kind in {gepa_offline,mipro_offline}".to_string(),
        )
    })?;
    canonicalize_offline_create_payload(request, kind)
}

/// Map HTTP errors to CoreError.
fn map_http_error(e: HttpError) -> CoreError {
    match e {
        HttpError::Response(detail) => {
            if detail.status == 401 || detail.status == 403 {
                CoreError::Authentication(format!("authentication failed: {}", detail))
            } else if detail.status == 429 {
                CoreError::UsageLimit(crate::UsageLimitInfo::from_http_429("jobs", &detail))
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
        assert_eq!(JOBS_ENDPOINT, "/api/v1/offline/jobs");
        assert_eq!(JOBS_CREATE_ENDPOINT, "/api/v1/offline/jobs");
        assert_eq!(GEPA_KIND, "gepa_offline");
        assert_eq!(MIPRO_KIND, "mipro_offline");
    }

    #[test]
    fn test_infer_offline_kind() {
        assert_eq!(
            infer_offline_kind(&serde_json::json!({
                "prompt_learning": {"algorithm": "mipro"}
            })),
            Some(MIPRO_KIND)
        );
        assert_eq!(
            infer_offline_kind(&serde_json::json!({
                "policy_optimization": {"algorithm": "GEPA"}
            })),
            Some(GEPA_KIND)
        );
        assert_eq!(
            infer_offline_kind(&serde_json::json!({
                "algorithm": "unknown"
            })),
            None
        );
    }

    #[test]
    fn test_parse_prompt_learning_result_accepts_both_best_candidate_and_best_prompt() {
        let payload = serde_json::json!({
            "job_id": "pl_test",
            "status": "running",
            "best_score": 0.42,
            "best_candidate": {"instruction": "new"},
            "best_prompt": {"instruction": "canonical"},
            "lever_versions": {"mipro.prompt.sys": 3},
        });
        let parsed =
            parse_prompt_learning_result(payload, "strict").expect("parse should succeed");
        assert_eq!(parsed.job_id, "pl_test");
        assert_eq!(parsed.status, PolicyJobStatus::Running);
        assert_eq!(parsed.best_reward, Some(0.42));
        assert_eq!(
            parsed
                .best_candidate
                .as_ref()
                .and_then(|v| v.get("instruction"))
                .and_then(|v| v.as_str()),
            Some("new")
        );
        assert_eq!(parsed.lever_versions.get("mipro.prompt.sys"), Some(&3));
    }
}
