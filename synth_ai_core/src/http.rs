//! HTTP client for Synth API calls.
//!
//! This module provides an async HTTP client with Bearer authentication,
//! optional dev headers (X-User-ID, X-Org-ID), and proper error handling.

use reqwest::header::{HeaderMap, HeaderValue, AUTHORIZATION};
use reqwest::multipart::{Form, Part};
use serde::de::DeserializeOwned;
use serde_json::Value;
use std::env;
use std::time::Duration;
use thiserror::Error;

use crate::shared_client::{DEFAULT_CONNECT_TIMEOUT_SECS, DEFAULT_POOL_SIZE};

/// HTTP error details.
#[derive(Debug, Clone)]
pub struct HttpErrorDetail {
    pub status: u16,
    pub url: String,
    pub message: String,
    pub body_snippet: Option<String>,
}

impl std::fmt::Display for HttpErrorDetail {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "HTTP {} for {}: {}", self.status, self.url, self.message)?;
        if let Some(ref snippet) = self.body_snippet {
            let truncated: String = snippet.chars().take(200).collect();
            write!(f, " | body[0:200]={}", truncated)?;
        }
        Ok(())
    }
}

/// HTTP client errors.
#[derive(Debug, Error)]
pub enum HttpError {
    #[error("request failed: {0} (is_connect={}, is_timeout={})", .0.is_connect(), .0.is_timeout())]
    Request(#[from] reqwest::Error),

    #[error("{0}")]
    Response(HttpErrorDetail),

    #[error("invalid url: {0}")]
    InvalidUrl(String),

    #[error("json parse error: {0}")]
    JsonParse(String),
}

/// Multipart file payload.
#[derive(Debug, Clone)]
pub struct MultipartFile {
    pub field: String,
    pub filename: String,
    pub bytes: Vec<u8>,
    pub content_type: Option<String>,
}

impl MultipartFile {
    pub fn new(
        field: impl Into<String>,
        filename: impl Into<String>,
        bytes: Vec<u8>,
        content_type: Option<String>,
    ) -> Self {
        Self {
            field: field.into(),
            filename: filename.into(),
            bytes,
            content_type,
        }
    }
}

impl HttpError {
    /// Create an HTTP error from a response.
    pub fn from_response(status: u16, url: &str, body: Option<&str>) -> Self {
        let body_snippet = body.map(|s| s.chars().take(200).collect());
        HttpError::Response(HttpErrorDetail {
            status,
            url: url.to_string(),
            message: "request_failed".to_string(),
            body_snippet,
        })
    }

    /// Get the HTTP status code, if available.
    pub fn status(&self) -> Option<u16> {
        match self {
            HttpError::Response(detail) => Some(detail.status),
            HttpError::Request(e) => e.status().map(|s| s.as_u16()),
            _ => None,
        }
    }
}

/// Async HTTP client for Synth API.
///
/// Provides Bearer token authentication and automatic JSON handling.
///
/// # Example
///
/// ```ignore
/// let client = HttpClient::new("https://api.usesynth.ai", "sk_live_...", 30)?;
/// let result: Value = client.get("/api/v1/jobs", None).await?;
/// ```
#[derive(Clone)]
pub struct HttpClient {
    client: reqwest::Client,
    base_url: String,
    #[allow(dead_code)]
    api_key: String,
}

impl HttpClient {
    /// Create a new HTTP client.
    ///
    /// # Arguments
    ///
    /// * `base_url` - Base URL for the API (without trailing slash)
    /// * `api_key` - API key for Bearer authentication
    /// * `timeout_secs` - Request timeout in seconds
    ///
    /// # Environment Variables
    ///
    /// Optional headers from environment:
    /// - `SYNTH_USER_ID` or `X_USER_ID` → `X-User-ID` header
    /// - `SYNTH_ORG_ID` or `X_ORG_ID` → `X-Org-ID` header
    pub fn new(base_url: &str, api_key: &str, timeout_secs: u64) -> Result<Self, HttpError> {
        let mut headers = HeaderMap::new();

        // Only add auth headers if api_key is non-empty
        if !api_key.is_empty() {
            let auth_value = format!("Bearer {}", api_key);
            headers.insert(
                AUTHORIZATION,
                HeaderValue::from_str(&auth_value)
                    .map_err(|_| HttpError::InvalidUrl("invalid api key characters".to_string()))?,
            );
            headers.insert(
                "X-API-Key",
                HeaderValue::from_str(api_key)
                    .map_err(|_| HttpError::InvalidUrl("invalid api key characters".to_string()))?,
            );
        }

        // Optional dev headers
        if let Some(user_id) = env::var("SYNTH_USER_ID")
            .ok()
            .or_else(|| env::var("X_USER_ID").ok())
        {
            if let Ok(val) = HeaderValue::from_str(&user_id) {
                headers.insert("X-User-ID", val);
            }
        }

        if let Some(org_id) = env::var("SYNTH_ORG_ID")
            .ok()
            .or_else(|| env::var("X_ORG_ID").ok())
        {
            if let Ok(val) = HeaderValue::from_str(&org_id) {
                headers.insert("X-Org-ID", val);
            }
        }

        let client = reqwest::Client::builder()
            .default_headers(headers)
            .timeout(Duration::from_secs(timeout_secs))
            .pool_max_idle_per_host(DEFAULT_POOL_SIZE)
            .pool_idle_timeout(Duration::from_secs(90))
            .connect_timeout(Duration::from_secs(DEFAULT_CONNECT_TIMEOUT_SECS))
            .tcp_keepalive(Duration::from_secs(60))
            .tcp_nodelay(true)
            .build()
            .map_err(HttpError::Request)?;

        Ok(Self {
            client,
            base_url: base_url.trim_end_matches('/').to_string(),
            api_key: api_key.to_string(),
        })
    }

    /// Get the API key used by this client.
    pub(crate) fn api_key(&self) -> &str {
        &self.api_key
    }

    /// Convert a relative path to an absolute URL.
    fn abs_url(&self, path: &str) -> String {
        if path.starts_with("http://") || path.starts_with("https://") {
            return path.to_string();
        }

        let path = path.trim_start_matches('/');

        // Handle /api prefix duplication
        if self.base_url.ends_with("/api") && path.starts_with("api/") {
            return format!("{}/{}", self.base_url, &path[4..]);
        }

        format!("{}/{}", self.base_url, path)
    }

    /// Make a GET request.
    ///
    /// # Arguments
    ///
    /// * `path` - API path (relative or absolute)
    /// * `params` - Optional query parameters
    pub async fn get<T: DeserializeOwned>(
        &self,
        path: &str,
        params: Option<&[(&str, &str)]>,
    ) -> Result<T, HttpError> {
        let url = self.abs_url(path);
        let mut req = self.client.get(&url);

        if let Some(p) = params {
            req = req.query(p);
        }

        let resp = req.send().await?;
        self.handle_response(resp, &url).await
    }

    /// Make a GET request and return raw bytes.
    pub async fn get_bytes(
        &self,
        path: &str,
        params: Option<&[(&str, &str)]>,
    ) -> Result<Vec<u8>, HttpError> {
        let url = self.abs_url(path);
        let mut request = self.client.get(&url);
        if let Some(params) = params {
            request = request.query(params);
        }
        let resp = request.send().await?;
        let status = resp.status();
        if status.is_success() {
            let bytes = resp.bytes().await?;
            return Ok(bytes.to_vec());
        }
        let body = resp.text().await.unwrap_or_default();
        Err(HttpError::from_response(
            status.as_u16(),
            &url,
            if body.is_empty() { None } else { Some(&body) },
        ))
    }

    /// Make a GET request returning raw JSON Value.
    pub async fn get_json(
        &self,
        path: &str,
        params: Option<&[(&str, &str)]>,
    ) -> Result<Value, HttpError> {
        self.get(path, params).await
    }

    /// Make a POST request with JSON body.
    ///
    /// # Arguments
    ///
    /// * `path` - API path
    /// * `body` - JSON body to send
    pub async fn post_json<T: DeserializeOwned>(
        &self,
        path: &str,
        body: &Value,
    ) -> Result<T, HttpError> {
        let url = self.abs_url(path);
        let resp = self.client.post(&url).json(body).send().await?;
        self.handle_response(resp, &url).await
    }

    /// Make a POST request with JSON body and extra headers.
    pub async fn post_json_with_headers<T: DeserializeOwned>(
        &self,
        path: &str,
        body: &Value,
        extra_headers: Option<HeaderMap>,
    ) -> Result<T, HttpError> {
        let url = self.abs_url(path);
        let mut request = self.client.post(&url).json(body);
        if let Some(headers) = extra_headers {
            request = request.headers(headers);
        }
        let resp = request.send().await?;
        self.handle_response(resp, &url).await
    }

    /// Make a POST request with multipart form data.
    ///
    /// # Arguments
    ///
    /// * `path` - API path
    /// * `data` - Form fields
    /// * `files` - File parts
    pub async fn post_multipart<T: DeserializeOwned>(
        &self,
        path: &str,
        data: &[(String, String)],
        files: &[MultipartFile],
    ) -> Result<T, HttpError> {
        let url = self.abs_url(path);
        let mut form = Form::new();
        for (key, value) in data {
            form = form.text(key.clone(), value.clone());
        }
        for file in files {
            let part = Part::bytes(file.bytes.clone()).file_name(file.filename.clone());
            let part = match &file.content_type {
                Some(ct) => part.mime_str(ct).unwrap_or_else(|_| {
                    Part::bytes(file.bytes.clone()).file_name(file.filename.clone())
                }),
                None => part,
            };
            form = form.part(file.field.clone(), part);
        }
        let resp = self.client.post(&url).multipart(form).send().await?;
        self.handle_response(resp, &url).await
    }

    /// Make a DELETE request.
    ///
    /// # Arguments
    ///
    /// * `path` - API path
    pub async fn delete(&self, path: &str) -> Result<(), HttpError> {
        let url = self.abs_url(path);
        let resp = self.client.delete(&url).send().await?;

        let status = resp.status().as_u16();
        if (200..300).contains(&status) {
            return Ok(());
        }

        let body = resp.text().await.ok();
        Err(HttpError::from_response(status, &url, body.as_deref()))
    }

    /// Handle HTTP response, returning parsed JSON or error.
    async fn handle_response<T: DeserializeOwned>(
        &self,
        resp: reqwest::Response,
        url: &str,
    ) -> Result<T, HttpError> {
        let status = resp.status().as_u16();
        let text = resp.text().await.unwrap_or_default();

        if (200..300).contains(&status) {
            serde_json::from_str(&text).map_err(|e| {
                HttpError::JsonParse(format!("{}: {}", e, &text[..text.len().min(100)]))
            })
        } else {
            Err(HttpError::from_response(status, url, Some(&text)))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_abs_url_relative() {
        let client = HttpClient::new("https://api.usesynth.ai", "test_key", 30).unwrap();
        assert_eq!(
            client.abs_url("/api/v1/jobs"),
            "https://api.usesynth.ai/api/v1/jobs"
        );
        assert_eq!(
            client.abs_url("api/v1/jobs"),
            "https://api.usesynth.ai/api/v1/jobs"
        );
    }

    #[test]
    fn test_abs_url_absolute() {
        let client = HttpClient::new("https://api.usesynth.ai", "test_key", 30).unwrap();
        assert_eq!(
            client.abs_url("https://other.com/path"),
            "https://other.com/path"
        );
    }

    #[test]
    fn test_abs_url_api_prefix_dedup() {
        let client = HttpClient::new("https://api.usesynth.ai/api", "test_key", 30).unwrap();
        assert_eq!(
            client.abs_url("api/v1/jobs"),
            "https://api.usesynth.ai/api/v1/jobs"
        );
    }

    #[test]
    fn test_http_error_display() {
        let err = HttpError::from_response(404, "https://api.example.com/test", Some("not found"));
        let msg = format!("{}", err);
        assert!(msg.contains("404"));
        assert!(msg.contains("api.example.com"));
    }
}
