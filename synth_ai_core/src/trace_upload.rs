//! Trace upload utilities for large trace payloads.
//!
//! Provides helpers to request presigned upload URLs and upload trace data
//! directly to blob storage.

use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::api::SynthClient;
use crate::errors::CoreError;
use crate::http::HttpError;
use crate::shared_client::build_pooled_client;

/// Size threshold for automatic upload (100 KB).
pub const AUTO_UPLOAD_THRESHOLD_BYTES: usize = 100 * 1024;

/// Maximum trace size allowed (50 MB).
pub const MAX_TRACE_SIZE_BYTES: usize = 50 * 1024 * 1024;

/// Response from the upload URL creation endpoint.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TraceUploadInfo {
    pub trace_id: String,
    pub trace_ref: String,
    pub upload_url: String,
    pub expires_in_seconds: i64,
    pub storage_key: String,
    #[serde(default)]
    pub max_size_bytes: Option<usize>,
}

/// Trace uploader for presigned URL workflows.
pub struct TraceUploader {
    client: SynthClient,
    auto_upload_threshold: usize,
}

impl TraceUploader {
    /// Create a new uploader with API credentials.
    pub fn new(api_key: &str, base_url: Option<&str>) -> Result<Self, CoreError> {
        let client = SynthClient::new(api_key, base_url)?;
        Ok(Self {
            client,
            auto_upload_threshold: AUTO_UPLOAD_THRESHOLD_BYTES,
        })
    }

    /// Set the auto-upload threshold in bytes.
    pub fn with_threshold(mut self, bytes: usize) -> Self {
        self.auto_upload_threshold = bytes;
        self
    }

    /// Check if a trace should be uploaded based on size.
    pub fn should_upload(&self, trace: &Value) -> bool {
        self.trace_size(trace)
            .map(|size| size > self.auto_upload_threshold)
            .unwrap_or(true)
    }

    /// Compute serialized trace size in bytes.
    pub fn trace_size(&self, trace: &Value) -> Result<usize, CoreError> {
        let bytes = serde_json::to_vec(trace)
            .map_err(|e| CoreError::Validation(format!("failed to serialize trace: {}", e)))?;
        Ok(bytes.len())
    }

    /// Request a presigned upload URL.
    pub async fn create_upload_url(
        &self,
        content_type: Option<&str>,
        expires_in_seconds: Option<i64>,
    ) -> Result<TraceUploadInfo, CoreError> {
        let mut payload = serde_json::Map::new();
        if let Some(ct) = content_type {
            payload.insert("content_type".to_string(), Value::String(ct.to_string()));
        }
        if let Some(exp) = expires_in_seconds {
            payload.insert("expires_in_seconds".to_string(), Value::Number(exp.into()));
        }

        self.client
            .http
            .post_json("/v1/traces/upload-url", &Value::Object(payload))
            .await
            .map_err(map_http_error)
    }

    /// Upload a trace and return its trace_ref.
    pub async fn upload_trace(
        &self,
        trace: &Value,
        expires_in_seconds: Option<i64>,
    ) -> Result<String, CoreError> {
        let bytes = serde_json::to_vec(trace)
            .map_err(|e| CoreError::Validation(format!("failed to serialize trace: {}", e)))?;

        if bytes.len() > MAX_TRACE_SIZE_BYTES {
            return Err(CoreError::Validation(format!(
                "trace too large: {} bytes (max: {})",
                bytes.len(),
                MAX_TRACE_SIZE_BYTES
            )));
        }

        let info = self
            .create_upload_url(Some("application/json"), expires_in_seconds)
            .await?;

        let client = build_pooled_client(Some(120)); // 2 min timeout for large uploads
        let resp = client
            .put(&info.upload_url)
            .header(reqwest::header::CONTENT_TYPE, "application/json")
            .body(bytes)
            .send()
            .await?;

        if !resp.status().is_success() {
            let status = resp.status().as_u16();
            let body = resp.text().await.unwrap_or_default();
            return Err(CoreError::http_response(
                status,
                &info.upload_url,
                "trace_upload_failed",
                if body.is_empty() { None } else { Some(body.as_str()) },
            ));
        }

        Ok(info.trace_ref)
    }
}

fn map_http_error(e: HttpError) -> CoreError {
    match e {
        HttpError::Response(detail) => {
            if detail.status == 401 || detail.status == 403 {
                CoreError::Authentication(format!("authentication failed: {}", detail))
            } else if detail.status == 429 {
                CoreError::UsageLimit(crate::UsageLimitInfo {
                    limit_type: "rate_limit".to_string(),
                    api: "trace_upload".to_string(),
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
