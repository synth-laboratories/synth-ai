//! Canonical Rust SDK for Synth.
//!
//! This crate mirrors the canonical Python SDK namespace model with a single
//! front-door `SynthClient`.

use std::env;
use std::sync::Arc;

pub mod containers_api;
pub mod models;
pub mod openapi_paths;
pub mod optimization_api;
pub mod pools_api;
mod transport;
pub mod tunnels_api;

pub use types::SynthError as Error;
pub use types::{Result, SynthError};

pub use containers_api::ContainersClient;
pub use optimization_api::OptimizationClient;
pub use pools_api::PoolsClient;
pub use tunnels_api::TunnelsClient;

mod types;

/// SDK version.
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// Default Synth API base URL.
pub const DEFAULT_BASE_URL: &str = synth_ai_core::urls::DEFAULT_BACKEND_URL;

/// Environment variable for API key.
pub const API_KEY_ENV: &str = "SYNTH_API_KEY";

/// Canonical front-door client.
#[derive(Clone)]
pub struct SynthClient {
    transport: Arc<transport::Transport>,
}

impl SynthClient {
    /// Create a new client with explicit credentials.
    pub fn new(api_key: impl Into<String>, base_url: Option<&str>) -> Result<Self> {
        let base_url = base_url.unwrap_or(DEFAULT_BASE_URL).to_string();
        let transport = transport::Transport::new(api_key.into(), base_url)?;
        Ok(Self {
            transport: Arc::new(transport),
        })
    }

    /// Create a client from environment variables.
    pub fn from_env() -> Result<Self> {
        let api_key = env::var(API_KEY_ENV).map_err(|_| SynthError::MissingApiKey)?;
        let base_url = env::var("SYNTH_BASE_URL")
            .ok()
            .or_else(|| env::var("SYNTH_BACKEND_URL").ok());
        Self::new(api_key, base_url.as_deref())
    }

    pub fn base_url(&self) -> &str {
        self.transport.base_url()
    }

    pub fn optimization(&self) -> OptimizationClient {
        OptimizationClient::new(self.transport.clone())
    }

    pub fn pools(&self) -> PoolsClient {
        PoolsClient::new(self.transport.clone())
    }

    pub fn containers(&self) -> ContainersClient {
        ContainersClient::new(self.transport.clone())
    }

    pub fn tunnels(&self) -> TunnelsClient {
        TunnelsClient::new(self.transport.clone())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_base_url_is_set() {
        assert!(!DEFAULT_BASE_URL.is_empty());
    }

    #[test]
    fn from_env_without_key_fails() {
        env::remove_var(API_KEY_ENV);
        let result = SynthClient::from_env();
        assert!(result.is_err());
    }
}
