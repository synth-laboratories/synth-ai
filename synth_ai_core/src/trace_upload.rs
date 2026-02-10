//! Trace upload helpers for large traces via presigned URLs.

use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::time::Duration;

use crate::http::{HttpClient, HttpError};
use crate::CoreError;

/// Response from creating an upload URL.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UploadUrlResponse {
    pub trace_id: String,
    pub trace_ref: String,
    pub upload_url: String,
    pub expires_in_seconds: i64,
    pub storage_key: String,
    #[serde(default)]
    pub max_size_bytes: Option<i64>,
}

/// Client for trace upload endpoints.
pub struct TraceUploadClient {
    http: HttpClient,
    base_url: String,
    timeout_secs: u64,
}

impl TraceUploadClient {
    pub fn new(base_url: &str, api_key: &str, timeout_secs: u64) -> Result<Self, CoreError> {
        let http = HttpClient::new(base_url, api_key, timeout_secs)
            .map_err(|e| CoreError::Internal(format!("failed to create http client: {}", e)))?;
        Ok(Self {
            http,
            base_url: base_url.trim_end_matches('/').to_string(),
            timeout_secs,
        })
    }

    pub fn trace_size(trace: &Value) -> Result<usize, CoreError> {
        let payload = serde_json::to_string(trace)
            .map_err(|e| CoreError::Validation(format!("failed to serialize trace: {}", e)))?;
        Ok(payload.len())
    }

    pub fn should_upload(trace: &Value, threshold_bytes: usize) -> Result<bool, CoreError> {
        Ok(Self::trace_size(trace)? > threshold_bytes)
    }

    pub async fn create_upload_url(
        &self,
        content_type: Option<&str>,
        expires_in_seconds: Option<i64>,
    ) -> Result<UploadUrlResponse, CoreError> {
        let mut payload = serde_json::Map::new();
        payload.insert(
            "content_type".to_string(),
            Value::String(content_type.unwrap_or("application/json").to_string()),
        );
        if let Some(expires) = expires_in_seconds {
            payload.insert(
                "expires_in_seconds".to_string(),
                Value::Number(expires.into()),
            );
        }

        let path = "/v1/traces/upload-url";
        let response: UploadUrlResponse = self
            .http
            .post_json(path, &Value::Object(payload))
            .await
            .map_err(map_http_error)?;

        Ok(response)
    }

    pub async fn upload_trace(
        &self,
        trace: &Value,
        content_type: Option<&str>,
        expires_in_seconds: Option<i64>,
    ) -> Result<String, CoreError> {
        let content_type = content_type.unwrap_or("application/json");
        let payload = serde_json::to_vec(trace)
            .map_err(|e| CoreError::Validation(format!("failed to serialize trace: {}", e)))?;

        let upload = self
            .create_upload_url(Some(content_type), expires_in_seconds)
            .await?;

        if let Some(max_size) = upload.max_size_bytes {
            if payload.len() as i64 > max_size {
                return Err(CoreError::Validation(format!(
                    "trace exceeds max size {} bytes",
                    max_size
                )));
            }
        }

        let client = reqwest::Client::builder()
            .timeout(Duration::from_secs(self.timeout_secs))
            .build()
            .map_err(|e| CoreError::Internal(format!("failed to build upload client: {}", e)))?;

        let resp = client
            .put(&upload.upload_url)
            .header("Content-Type", content_type)
            .body(payload)
            .send()
            .await
            .map_err(|e| CoreError::Http(e))?;

        if !resp.status().is_success() {
            let status = resp.status().as_u16();
            let body = resp.text().await.unwrap_or_default();
            return Err(CoreError::HttpResponse(crate::HttpErrorInfo {
                status,
                url: upload.upload_url,
                message: "trace_upload_failed".to_string(),
                body_snippet: Some(body.chars().take(200).collect()),
            }));
        }

        Ok(upload.trace_ref)
    }

    pub fn base_url(&self) -> &str {
        &self.base_url
    }
}

fn map_http_error(err: HttpError) -> CoreError {
    match err {
        HttpError::Response(detail) => CoreError::HttpResponse(crate::HttpErrorInfo {
            status: detail.status,
            url: detail.url,
            message: detail.message,
            body_snippet: detail.body_snippet,
        }),
        HttpError::Request(e) => CoreError::Http(e),
        _ => CoreError::Internal(format!("{}", err)),
    }
}
