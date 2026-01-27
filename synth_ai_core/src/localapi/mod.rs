//! Local API client for task app communication.
//!
//! This module provides the client for communicating with task apps:
//! - HTTP client with authentication
//! - Data contracts (request/response types)
//! - Health checking and waiting
//! - Rollout execution

pub mod client;
pub mod types;

pub use client::{EnvClient, TaskAppClient};
pub use types::{
    AuthInfo, DatasetInfo, HealthResponse, InfoResponse, InferenceInfo, LimitsInfo,
    RolloutEnvSpec, RolloutMetrics, RolloutPolicySpec, RolloutRequest, RolloutResponse,
    RolloutSafetyConfig, TaskDescriptor, TaskInfo,
};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_module_exports() {
        // Verify all types are accessible
        let _ = TaskAppClient::new("http://localhost:8000", None);
        let _ = RolloutRequest::new("test");
        let _ = TaskDescriptor::new("id", "name");
    }
}
