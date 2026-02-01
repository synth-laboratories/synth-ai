//! Shared HTTP client utilities for connection pooling.
//!
//! This module provides shared, properly-configured HTTP clients that
//! enable connection reuse across concurrent requests. Using these utilities
//! prevents TLS handshake overhead when making many parallel API calls.
//!
//! # Why This Matters
//!
//! Without connection pooling:
//! - 20 concurrent requests → 20 separate TLS handshakes
//! - Each handshake adds ~50-200ms latency
//! - Requests are effectively serialized under load
//!
//! With connection pooling:
//! - 20 concurrent requests → connections reused from pool
//! - TLS handshake only on first connection
//! - True parallel execution
//!
//! # Usage
//!
//! For most cases, use the `SHARED_CLIENT`:
//!
//! ```ignore
//! use synth_ai_core::shared_client::SHARED_CLIENT;
//!
//! let resp = SHARED_CLIENT.get("https://api.usesynth.ai/health").send().await?;
//! ```
//!
//! When you need custom configuration (e.g., different timeout):
//!
//! ```ignore
//! use synth_ai_core::shared_client::build_pooled_client;
//!
//! let client = build_pooled_client(Some(60)); // 60 second timeout
//! ```

use once_cell::sync::Lazy;
use reqwest::Client;
use std::time::Duration;

/// Default pool size for idle connections per host.
pub const DEFAULT_POOL_SIZE: usize = 200;

/// Default connection timeout in seconds.
pub const DEFAULT_CONNECT_TIMEOUT_SECS: u64 = 30;

/// Default request timeout in seconds (5 minutes for slow LLM responses).
pub const DEFAULT_TIMEOUT_SECS: u64 = 300;

/// Shared pooled HTTP client for high-concurrency workloads.
///
/// This client is configured with:
/// - 200 max idle connections per host
/// - 30 second connection timeout
/// - 300 second (5 min) request timeout
///
/// Use this for API calls, SSE streams, and any HTTP operations that
/// may run concurrently.
pub static SHARED_CLIENT: Lazy<Client> = Lazy::new(|| build_pooled_client(None));

/// Build a new pooled HTTP client.
///
/// # Arguments
///
/// * `timeout_secs` - Request timeout in seconds (default: 300)
///
/// # Returns
///
/// A reqwest::Client configured with connection pooling.
pub fn build_pooled_client(timeout_secs: Option<u64>) -> Client {
    let timeout = Duration::from_secs(timeout_secs.unwrap_or(DEFAULT_TIMEOUT_SECS));

    Client::builder()
        .pool_max_idle_per_host(DEFAULT_POOL_SIZE)
        .pool_idle_timeout(Duration::from_secs(90))
        .connect_timeout(Duration::from_secs(DEFAULT_CONNECT_TIMEOUT_SECS))
        .timeout(timeout)
        .tcp_keepalive(Duration::from_secs(60))
        .tcp_nodelay(true)
        .build()
        .unwrap_or_else(|_| Client::new())
}

/// Build a pooled client with custom pool size.
///
/// # Arguments
///
/// * `pool_size` - Max idle connections per host
/// * `timeout_secs` - Request timeout in seconds (default: 300)
///
/// # Returns
///
/// A reqwest::Client configured with the specified pool size.
pub fn build_pooled_client_with_size(pool_size: usize, timeout_secs: Option<u64>) -> Client {
    let timeout = Duration::from_secs(timeout_secs.unwrap_or(DEFAULT_TIMEOUT_SECS));

    Client::builder()
        .pool_max_idle_per_host(pool_size)
        .pool_idle_timeout(Duration::from_secs(90))
        .connect_timeout(Duration::from_secs(DEFAULT_CONNECT_TIMEOUT_SECS))
        .timeout(timeout)
        .tcp_keepalive(Duration::from_secs(60))
        .tcp_nodelay(true)
        .build()
        .unwrap_or_else(|_| Client::new())
}

/// Build a pooled client with headers.
///
/// # Arguments
///
/// * `headers` - Default headers to include with every request
/// * `timeout_secs` - Request timeout in seconds (default: 300)
///
/// # Returns
///
/// A reqwest::Client configured with connection pooling and default headers.
pub fn build_pooled_client_with_headers(
    headers: reqwest::header::HeaderMap,
    timeout_secs: Option<u64>,
) -> Result<Client, reqwest::Error> {
    let timeout = Duration::from_secs(timeout_secs.unwrap_or(DEFAULT_TIMEOUT_SECS));

    Client::builder()
        .pool_max_idle_per_host(DEFAULT_POOL_SIZE)
        .pool_idle_timeout(Duration::from_secs(90))
        .connect_timeout(Duration::from_secs(DEFAULT_CONNECT_TIMEOUT_SECS))
        .timeout(timeout)
        .tcp_keepalive(Duration::from_secs(60))
        .tcp_nodelay(true)
        .default_headers(headers)
        .build()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_shared_client_exists() {
        // Just verify it can be accessed without panic
        let _ = &*SHARED_CLIENT;
    }

    #[test]
    fn test_build_pooled_client_default() {
        let client = build_pooled_client(None);
        // Client should be created successfully
        drop(client);
    }

    #[test]
    fn test_build_pooled_client_custom_timeout() {
        let client = build_pooled_client(Some(60));
        drop(client);
    }

    #[test]
    fn test_build_pooled_client_with_size() {
        let client = build_pooled_client_with_size(50, Some(120));
        drop(client);
    }
}
