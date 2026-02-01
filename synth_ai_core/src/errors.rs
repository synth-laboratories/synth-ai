//! Core error types for Synth SDK.
//!
//! This module provides shared error types that can be used across
//! TypeScript, Python, and Go SDKs via bindings.

use crate::http::HttpError;
use serde::{Deserialize, Serialize};
use thiserror::Error;

/// HTTP error details for API responses.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HttpErrorInfo {
    /// HTTP status code (e.g., 404, 500)
    pub status: u16,
    /// Request URL
    pub url: String,
    /// Error message
    pub message: String,
    /// First 200 chars of response body (for debugging)
    pub body_snippet: Option<String>,
}

impl std::fmt::Display for HttpErrorInfo {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "HTTP {} for {}: {}", self.status, self.url, self.message)?;
        if let Some(ref snippet) = self.body_snippet {
            let truncated: String = snippet.chars().take(200).collect();
            write!(f, " | body[0:200]={}", truncated)?;
        }
        Ok(())
    }
}

/// Usage limit error details.
///
/// Raised when an organization's rate limit is exceeded.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UsageLimitInfo {
    /// Type of limit exceeded (e.g., "inference_tokens_per_day")
    pub limit_type: String,
    /// API that hit the limit (e.g., "inference", "verifiers", "prompt_opt")
    pub api: String,
    /// Current usage value
    pub current: f64,
    /// The limit value
    pub limit: f64,
    /// Organization's tier (e.g., "free", "starter", "growth")
    pub tier: String,
    /// Seconds until the limit resets (if available)
    pub retry_after_seconds: Option<i64>,
    /// URL to upgrade tier
    pub upgrade_url: String,
}

impl Default for UsageLimitInfo {
    fn default() -> Self {
        Self {
            limit_type: String::new(),
            api: String::new(),
            current: 0.0,
            limit: 0.0,
            tier: "free".to_string(),
            retry_after_seconds: None,
            upgrade_url: "https://usesynth.ai/pricing".to_string(),
        }
    }
}

impl std::fmt::Display for UsageLimitInfo {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Rate limit exceeded: {} ({}/{}) for tier '{}'. Upgrade at {}",
            self.limit_type, self.current, self.limit, self.tier, self.upgrade_url
        )
    }
}

/// Job error details.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JobErrorInfo {
    /// Job ID that failed
    pub job_id: String,
    /// Error message
    pub message: String,
    /// Optional error code
    pub code: Option<String>,
}

impl std::fmt::Display for JobErrorInfo {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Job {} failed: {}", self.job_id, self.message)?;
        if let Some(ref code) = self.code {
            write!(f, " (code: {})", code)?;
        }
        Ok(())
    }
}

/// Unified error enum for all Synth core errors.
///
/// This provides a single error type that can be mapped to language-specific
/// exceptions in Python, TypeScript, etc.
#[derive(Debug, Error)]
pub enum CoreError {
    /// Invalid input provided
    #[error("invalid input: {0}")]
    InvalidInput(String),

    /// URL parsing failed
    #[error("url parse error: {0}")]
    UrlParse(#[from] url::ParseError),

    /// HTTP request failed (network layer)
    #[error("http error: {0}")]
    Http(#[from] reqwest::Error),

    /// HTTP response error (4xx/5xx)
    #[error("{0}")]
    HttpResponse(HttpErrorInfo),

    /// Authentication failed
    #[error("authentication failed: {0}")]
    Authentication(String),

    /// Validation error
    #[error("validation error: {0}")]
    Validation(String),

    /// Usage/rate limit exceeded
    #[error("{0}")]
    UsageLimit(UsageLimitInfo),

    /// Job operation failed
    #[error("{0}")]
    Job(JobErrorInfo),

    /// Configuration error
    #[error("config error: {0}")]
    Config(String),

    /// Timeout error
    #[error("timeout: {0}")]
    Timeout(String),

    /// Protocol/wire format error
    #[error("protocol error: {0}")]
    Protocol(String),

    /// Generic internal error
    #[error("internal error: {0}")]
    Internal(String),
}

impl CoreError {
    /// Create an HTTP response error.
    pub fn http_response(status: u16, url: &str, message: &str, body: Option<&str>) -> Self {
        CoreError::HttpResponse(HttpErrorInfo {
            status,
            url: url.to_string(),
            message: message.to_string(),
            body_snippet: body.map(|s| s.chars().take(200).collect()),
        })
    }

    /// Create an authentication error.
    pub fn auth(message: impl Into<String>) -> Self {
        CoreError::Authentication(message.into())
    }

    /// Create a validation error.
    pub fn validation(message: impl Into<String>) -> Self {
        CoreError::Validation(message.into())
    }

    /// Create a usage limit error.
    pub fn usage_limit(
        limit_type: &str,
        api: &str,
        current: f64,
        limit: f64,
        tier: &str,
        retry_after: Option<i64>,
    ) -> Self {
        CoreError::UsageLimit(UsageLimitInfo {
            limit_type: limit_type.to_string(),
            api: api.to_string(),
            current,
            limit,
            tier: tier.to_string(),
            retry_after_seconds: retry_after,
            upgrade_url: "https://usesynth.ai/pricing".to_string(),
        })
    }

    /// Create a job error.
    pub fn job(job_id: &str, message: &str, code: Option<&str>) -> Self {
        CoreError::Job(JobErrorInfo {
            job_id: job_id.to_string(),
            message: message.to_string(),
            code: code.map(String::from),
        })
    }

    /// Create a timeout error.
    pub fn timeout(message: impl Into<String>) -> Self {
        CoreError::Timeout(message.into())
    }

    /// Create a config error.
    pub fn config(message: impl Into<String>) -> Self {
        CoreError::Config(message.into())
    }

    /// Check if this is an authentication error.
    pub fn is_auth_error(&self) -> bool {
        matches!(self, CoreError::Authentication(_))
    }

    /// Check if this is a rate limit error.
    pub fn is_rate_limit(&self) -> bool {
        matches!(self, CoreError::UsageLimit(_))
    }

    /// Check if this is a retryable error (5xx, timeout, network).
    pub fn is_retryable(&self) -> bool {
        match self {
            CoreError::HttpResponse(info) => info.status >= 500,
            CoreError::Http(_) => true,
            CoreError::Timeout(_) => true,
            _ => false,
        }
    }

    /// Get HTTP status code if this is an HTTP error.
    pub fn http_status(&self) -> Option<u16> {
        match self {
            CoreError::HttpResponse(info) => Some(info.status),
            CoreError::Http(e) => e.status().map(|s| s.as_u16()),
            _ => None,
        }
    }
}

impl From<HttpError> for CoreError {
    fn from(err: HttpError) -> Self {
        match err {
            HttpError::Request(e) => CoreError::Http(e),
            HttpError::Response(detail) => CoreError::HttpResponse(HttpErrorInfo {
                status: detail.status,
                url: detail.url,
                message: detail.message,
                body_snippet: detail.body_snippet,
            }),
            HttpError::InvalidUrl(msg) => CoreError::InvalidInput(msg),
            HttpError::JsonParse(msg) => CoreError::Protocol(msg),
        }
    }
}

/// Result type alias using CoreError.
pub type CoreResult<T> = Result<T, CoreError>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_http_error_display() {
        let err = CoreError::http_response(404, "https://api.example.com/test", "not found", None);
        let msg = format!("{}", err);
        assert!(msg.contains("404"));
        assert!(msg.contains("api.example.com"));
    }

    #[test]
    fn test_usage_limit_display() {
        let err = CoreError::usage_limit(
            "inference_tokens_per_day",
            "inference",
            10000.0,
            5000.0,
            "free",
            Some(3600),
        );
        let msg = format!("{}", err);
        assert!(msg.contains("inference_tokens_per_day"));
        assert!(msg.contains("free"));
    }

    #[test]
    fn test_retryable() {
        let err_500 =
            CoreError::http_response(500, "https://api.example.com", "server error", None);
        assert!(err_500.is_retryable());

        let err_404 = CoreError::http_response(404, "https://api.example.com", "not found", None);
        assert!(!err_404.is_retryable());

        let err_auth = CoreError::auth("invalid key");
        assert!(!err_auth.is_retryable());
    }

    #[test]
    fn test_http_status() {
        let err = CoreError::http_response(403, "https://api.example.com", "forbidden", None);
        assert_eq!(err.http_status(), Some(403));

        let err_auth = CoreError::auth("invalid key");
        assert_eq!(err_auth.http_status(), None);
    }
}
