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
use crate::x402::X402Payer;

const PAYMENT_REQUIRED_HEADER: &str = "PAYMENT-REQUIRED";
const X_PAYMENT_REQUIRED_HEADER: &str = "X-PAYMENT-REQUIRED";
const PAYMENT_SIGNATURE_HEADER: &str = "PAYMENT-SIGNATURE";

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
        // Keep enough body to preserve structured JSON error payloads.
        // Display paths still truncate to 200 chars, but parsers (e.g. rate-limit
        // detail extraction) need the full object to avoid fallback placeholders.
        let body_snippet = body.map(|s| s.chars().take(4096).collect());
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
    x402_payer: Option<X402Payer>,
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

        let x402_payer = X402Payer::from_env();

        Ok(Self {
            client,
            base_url: base_url.trim_end_matches('/').to_string(),
            api_key: api_key.to_string(),
            x402_payer,
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

        let request = req.build().map_err(HttpError::Request)?;
        let (status, _headers, body) = self.send_with_x402_retry(request).await?;
        self.parse_json(status, &url, &body)
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
        let request = request.build().map_err(HttpError::Request)?;
        let (status, _headers, body) = self.send_with_x402_retry(request).await?;
        if (200..300).contains(&status) {
            return Ok(body.to_vec());
        }
        let text = String::from_utf8_lossy(&body);
        Err(HttpError::from_response(
            status,
            &url,
            if text.trim().is_empty() { None } else { Some(&text) },
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
        let request = self.client.post(&url).json(body).build().map_err(HttpError::Request)?;
        let (status, _headers, body_bytes) = self.send_with_x402_retry(request).await?;
        self.parse_json(status, &url, &body_bytes)
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
        let request = request.build().map_err(HttpError::Request)?;
        let (status, _headers, body_bytes) = self.send_with_x402_retry(request).await?;
        self.parse_json(status, &url, &body_bytes)
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
        let request = self
            .client
            .post(&url)
            .multipart(form)
            .build()
            .map_err(HttpError::Request)?;
        let (status, _headers, body_bytes) = self.send_with_x402_retry(request).await?;
        self.parse_json(status, &url, &body_bytes)
    }

    /// Make a DELETE request.
    ///
    /// # Arguments
    ///
    /// * `path` - API path
    pub async fn delete(&self, path: &str) -> Result<(), HttpError> {
        let url = self.abs_url(path);
        let request = self.client.delete(&url).build().map_err(HttpError::Request)?;
        let (status, _headers, body_bytes) = self.send_with_x402_retry(request).await?;
        if (200..300).contains(&status) {
            return Ok(());
        }
        let text = String::from_utf8_lossy(&body_bytes);
        Err(HttpError::from_response(
            status,
            &url,
            if text.trim().is_empty() { None } else { Some(&text) },
        ))
    }

    fn parse_json<T: DeserializeOwned>(&self, status: u16, url: &str, body: &[u8]) -> Result<T, HttpError> {
        if !(200..300).contains(&status) {
            let text = String::from_utf8_lossy(body);
            return Err(HttpError::from_response(status, url, Some(&text)));
        }

        serde_json::from_slice(body).map_err(|e| {
            let text = String::from_utf8_lossy(body);
            HttpError::JsonParse(format!("{}: {}", e, &text[..text.len().min(100)]))
        })
    }

    async fn send_with_x402_retry(
        &self,
        request: reqwest::Request,
    ) -> Result<(u16, HeaderMap, bytes::Bytes), HttpError> {
        let Some(first) = request.try_clone() else {
            // Can't retry x402 anyway if we can't clone.
            let resp = self.client.execute(request).await?;
            let status = resp.status().as_u16();
            let headers = resp.headers().clone();
            let body = resp.bytes().await?;
            return Ok((status, headers, body));
        };

        let resp = self.client.execute(first).await?;
        let status = resp.status().as_u16();
        let headers = resp.headers().clone();
        let body = resp.bytes().await?;

        if status != 402 {
            return Ok((status, headers, body));
        }

        let Some(payer) = self.x402_payer.as_ref() else {
            return Ok((status, headers, body));
        };

        let Some(payment_required_header) = extract_payment_required_header(&headers, &body) else {
            return Ok((status, headers, body));
        };

        let Ok(payment_signature_header) = payer.build_payment_signature_header(&payment_required_header) else {
            return Ok((status, headers, body));
        };

        let Some(mut retry) = request.try_clone() else {
            return Ok((status, headers, body));
        };

        retry.headers_mut().insert(
            PAYMENT_SIGNATURE_HEADER,
            HeaderValue::from_str(&payment_signature_header)
                .map_err(|_| HttpError::InvalidUrl("invalid x402 payment signature header".to_string()))?,
        );

        let resp2 = self.client.execute(retry).await?;
        let status2 = resp2.status().as_u16();
        let headers2 = resp2.headers().clone();
        let body2 = resp2.bytes().await?;
        Ok((status2, headers2, body2))
    }
}

fn extract_payment_required_header(headers: &HeaderMap, body: &[u8]) -> Option<String> {
    let direct = headers
        .get(PAYMENT_REQUIRED_HEADER)
        .or_else(|| headers.get(X_PAYMENT_REQUIRED_HEADER))
        .and_then(|v| v.to_str().ok())
        .map(|v| v.to_string());
    if direct.is_some() {
        return direct;
    }

    // Fallback: some servers also place it in the JSON body under detail.x402.payment_required_header.
    let parsed = serde_json::from_slice::<serde_json::Value>(body).ok()?;
    let detail = parsed.get("detail").unwrap_or(&parsed);
    let x402 = detail.get("x402")?;
    x402.get("payment_required_header")
        .and_then(|v| v.as_str())
        .map(|v| v.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::x402::{
        decode_payment_signature_header, recover_payer_address_from_payment_payload, PaymentRequired,
        PaymentRequirements, ResourceInfo,
    };
    use base64::Engine as _;
    use bytes::Bytes;
    use http_body_util::Full;
    use hyper::service::service_fn;
    use hyper::{Request, Response, StatusCode};
    use hyper_util::rt::{TokioExecutor, TokioIo};
    use hyper_util::server::conn::auto::Builder;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Arc;
    use tokio::net::TcpListener;

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

    #[tokio::test]
    async fn test_http_client_x402_auto_retry() {
        // Configure a local payer (client-side) private key.
        std::env::set_var(
            "SYNTH_X402_PRIVATE_KEY",
            "0x0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef",
        );
        let payer = X402Payer::from_env().unwrap();

        // Start a tiny HTTP server that 402s once, then expects PAYMENT-SIGNATURE.
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let base_url = format!("http://{}", addr);

        // Build PAYMENT-REQUIRED header payload.
        let pay_to = "0x1111111111111111111111111111111111111111".to_string();
        let asset = "0x036CbD53842c5426634e7929541eC2318f3dCF7e".to_string();
        let payment_required = PaymentRequired {
            x402_version: 2,
            error: Some("Payment required".to_string()),
            resource: Some(ResourceInfo {
                url: format!("{}/test", base_url),
                description: Some("unit test".to_string()),
                mime_type: Some("application/json".to_string()),
            }),
            accepts: vec![PaymentRequirements {
                scheme: "exact".to_string(),
                network: "eip155:84532".to_string(),
                asset,
                amount: "250000".to_string(),
                pay_to,
                max_timeout_seconds: 300,
                extra: Some(serde_json::json!({"name": "USDC", "version": "2"})),
            }],
            extensions: None,
        };
        let required_header = base64::engine::general_purpose::STANDARD
            .encode(serde_json::to_vec(&payment_required).unwrap());

        let request_count = Arc::new(AtomicUsize::new(0));
        let expected_payer = payer.address().to_string();

        let required_header_clone = required_header.clone();
        let request_count_clone = request_count.clone();
        let expected_payer_clone = expected_payer.clone();

        tokio::spawn(async move {
            loop {
                let (stream, _) = match listener.accept().await {
                    Ok(value) => value,
                    Err(_) => break,
                };
                let io = TokioIo::new(stream);
                let required = required_header_clone.clone();
                let count = request_count_clone.clone();
                let expected = expected_payer_clone.clone();

                tokio::spawn(async move {
                    let svc = service_fn(move |req: Request<hyper::body::Incoming>| {
                        let required = required.clone();
                        let count = count.clone();
                        let expected = expected.clone();

                        async move {
                            let n = count.fetch_add(1, Ordering::SeqCst);
                            if n == 0 {
                                let mut resp = Response::new(Full::new(Bytes::from_static(b"")));
                                *resp.status_mut() = StatusCode::PAYMENT_REQUIRED;
                                resp.headers_mut()
                                    .insert(PAYMENT_REQUIRED_HEADER, HeaderValue::from_str(&required).unwrap());
                                return Ok::<_, hyper::Error>(resp);
                            }

                            let sig_header = req
                                .headers()
                                .get(PAYMENT_SIGNATURE_HEADER)
                                .and_then(|v| v.to_str().ok())
                                .unwrap_or("");
                            assert!(!sig_header.trim().is_empty());

                            let payment_payload = decode_payment_signature_header(sig_header).unwrap();
                            let recovered = recover_payer_address_from_payment_payload(&payment_payload).unwrap();
                            assert_eq!(recovered, expected);

                            let body = serde_json::to_vec(&serde_json::json!({"ok": true})).unwrap();
                            let mut resp = Response::new(Full::new(Bytes::from(body)));
                            *resp.status_mut() = StatusCode::OK;
                            Ok::<_, hyper::Error>(resp)
                        }
                    });

                    let _ = Builder::new(TokioExecutor::new())
                        .serve_connection(io, svc)
                        .await;
                });
            }
        });

        // Client should transparently retry and succeed.
        let client = HttpClient::new(&base_url, "test_key", 30).unwrap();

        // Remove env var immediately to avoid leaking into other tests.
        std::env::remove_var("SYNTH_X402_PRIVATE_KEY");

        let result: Value = client.get_json("/test", None).await.unwrap();
        assert_eq!(result.get("ok").and_then(|v| v.as_bool()), Some(true));
        assert_eq!(request_count.load(Ordering::SeqCst), 2);
    }
}
