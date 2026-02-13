//! Data contracts for task app communication.
//!
//! These types match the Python SDK's task app API contract.

use crate::data::{Artifact, ContextOverride, ContextOverrideStatus, SuccessStatus};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::HashMap;

/// Task descriptor with basic info.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TaskDescriptor {
    /// Unique task identifier.
    pub id: String,
    /// Human-readable task name.
    pub name: String,
    /// Task description.
    #[serde(default)]
    pub description: Option<String>,
    /// Task version.
    #[serde(default)]
    pub version: Option<String>,
}

impl TaskDescriptor {
    /// Create a new task descriptor.
    pub fn new(id: impl Into<String>, name: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            name: name.into(),
            description: None,
            version: None,
        }
    }
}

/// Dataset information.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct DatasetInfo {
    /// Dataset identifier.
    #[serde(default)]
    pub id: Option<String>,
    /// Dataset name.
    #[serde(default)]
    pub name: Option<String>,
    /// Dataset version.
    #[serde(default)]
    pub version: Option<String>,
    /// Available splits (e.g., ["train", "test"]).
    #[serde(default)]
    pub splits: Vec<String>,
    /// Default split to use.
    #[serde(default)]
    pub default_split: Option<String>,
    /// Description.
    #[serde(default)]
    pub description: Option<String>,
}

/// Inference configuration.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct InferenceInfo {
    /// Model identifier.
    #[serde(default)]
    pub model: Option<String>,
    /// Inference URL (proxy endpoint).
    #[serde(default)]
    pub inference_url: Option<String>,
    /// Provider name.
    #[serde(default)]
    pub provider: Option<String>,
}

/// Limits for rollout execution.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct LimitsInfo {
    /// Maximum turns/steps.
    #[serde(default)]
    pub max_turns: Option<i32>,
    /// Maximum response tokens per turn.
    #[serde(default)]
    pub max_response_tokens: Option<i32>,
    /// Timeout in seconds.
    #[serde(default)]
    pub timeout_seconds: Option<i32>,
    /// Maximum total tokens.
    #[serde(default)]
    pub max_total_tokens: Option<i32>,
}

/// Complete task information returned by /task_info.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TaskInfo {
    /// Task descriptor.
    pub task: TaskDescriptor,
    /// Dataset information.
    #[serde(default)]
    pub dataset: DatasetInfo,
    /// Inference configuration.
    #[serde(default)]
    pub inference: InferenceInfo,
    /// Execution limits.
    #[serde(default)]
    pub limits: LimitsInfo,
    /// Additional task metadata.
    #[serde(default)]
    pub task_metadata: HashMap<String, Value>,
}

/// Environment specification for rollout.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct RolloutEnvSpec {
    /// Environment identifier.
    #[serde(default)]
    pub env_id: Option<String>,
    /// Environment name.
    #[serde(default)]
    pub env_name: Option<String>,
    /// Environment configuration.
    #[serde(default)]
    pub config: HashMap<String, Value>,
    /// Random seed.
    #[serde(default)]
    pub seed: Option<i64>,
}

/// Policy specification for rollout.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct RolloutPolicySpec {
    /// Policy identifier.
    #[serde(default)]
    pub policy_id: Option<String>,
    /// Policy name.
    #[serde(default)]
    pub policy_name: Option<String>,
    /// Policy configuration.
    #[serde(default)]
    pub config: HashMap<String, Value>,
    /// Output mode (text, tool_calls, structured).
    #[serde(default)]
    pub output_mode: Option<String>,
    /// Structured output schema.
    #[serde(default)]
    pub structured_config: Option<Value>,
}

/// Safety configuration for rollout.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RolloutSafetyConfig {
    /// Maximum execution time in seconds.
    #[serde(default = "default_max_time")]
    pub max_time_s: i32,
    /// Maximum memory in MB.
    #[serde(default)]
    pub max_memory_mb: Option<i32>,
    /// Allow network access.
    #[serde(default = "default_true")]
    pub allow_network: bool,
}

fn default_max_time() -> i32 {
    3600
}

fn default_true() -> bool {
    true
}

impl Default for RolloutSafetyConfig {
    fn default() -> Self {
        Self {
            max_time_s: 3600,
            max_memory_mb: None,
            allow_network: true,
        }
    }
}

/// Request body for POST /rollout.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RolloutRequest {
    /// Correlation ID for tracing.
    pub trace_correlation_id: String,
    /// Environment specification.
    pub env: RolloutEnvSpec,
    /// Policy specification.
    pub policy: RolloutPolicySpec,
    /// What to do when done (reset, terminate).
    #[serde(default = "default_on_done")]
    pub on_done: String,
    /// Safety configuration.
    #[serde(default)]
    pub safety: RolloutSafetyConfig,
    /// Training session ID.
    #[serde(default)]
    pub training_session_id: Option<String>,
    /// Synth backend URL.
    #[serde(default)]
    pub synth_base_url: Option<String>,
    /// Context overrides to apply.
    #[serde(default)]
    pub context_overrides: Option<Vec<ContextOverride>>,
    /// Override bundle ID for tracking.
    #[serde(default)]
    pub override_bundle_id: Option<String>,
}

fn default_on_done() -> String {
    "reset".to_string()
}

impl RolloutRequest {
    /// Create a new rollout request.
    pub fn new(trace_correlation_id: impl Into<String>) -> Self {
        Self {
            trace_correlation_id: trace_correlation_id.into(),
            env: RolloutEnvSpec::default(),
            policy: RolloutPolicySpec::default(),
            on_done: "reset".to_string(),
            safety: RolloutSafetyConfig::default(),
            training_session_id: None,
            synth_base_url: None,
            context_overrides: None,
            override_bundle_id: None,
        }
    }

    /// Set the environment spec.
    pub fn with_env(mut self, env: RolloutEnvSpec) -> Self {
        self.env = env;
        self
    }

    /// Set the policy spec.
    pub fn with_policy(mut self, policy: RolloutPolicySpec) -> Self {
        self.policy = policy;
        self
    }

    /// Set context overrides.
    pub fn with_overrides(mut self, overrides: Vec<ContextOverride>) -> Self {
        self.context_overrides = Some(overrides);
        self
    }
}

/// Metrics from a rollout.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RolloutMetrics {
    /// Total outcome reward.
    pub outcome_reward: f64,
    /// Per-event rewards.
    #[serde(default)]
    pub event_rewards: Option<Vec<f64>>,
    /// Objective values.
    #[serde(default)]
    pub outcome_objectives: Option<HashMap<String, f64>>,
    /// Event-level objective values.
    #[serde(default)]
    pub event_objectives: Option<Vec<HashMap<String, f64>>>,
    /// Additional details.
    #[serde(default)]
    pub details: HashMap<String, Value>,
}

impl RolloutMetrics {
    /// Create new metrics with outcome reward.
    ///
    /// Auto-derives `outcome_objectives` as `{"reward": outcome_reward}` so that
    /// objectives are always available for multi-objective aggregation.
    pub fn new(outcome_reward: f64) -> Self {
        let mut objectives = HashMap::new();
        objectives.insert("reward".to_string(), outcome_reward);
        Self {
            outcome_reward,
            event_rewards: None,
            outcome_objectives: Some(objectives),
            event_objectives: None,
            details: HashMap::new(),
        }
    }

    /// Create new metrics with explicit outcome_objectives (multi-objective).
    ///
    /// Use this when you have objectives beyond just reward (e.g., latency, cost).
    pub fn with_objectives(outcome_reward: f64, outcome_objectives: HashMap<String, f64>) -> Self {
        Self {
            outcome_reward,
            event_rewards: None,
            outcome_objectives: Some(outcome_objectives),
            event_objectives: None,
            details: HashMap::new(),
        }
    }
}

impl Default for RolloutMetrics {
    fn default() -> Self {
        Self::new(0.0)
    }
}

/// Response from POST /rollout.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RolloutResponse {
    /// Echo of the correlation ID.
    pub trace_correlation_id: String,
    /// Rollout metrics.
    #[serde(alias = "metrics")]
    pub reward_info: RolloutMetrics,
    /// Full trace (optional).
    #[serde(default)]
    pub trace: Option<Value>,
    /// Artifacts produced.
    #[serde(default)]
    pub artifact: Option<Vec<Artifact>>,
    /// Inference URL used.
    #[serde(default)]
    pub inference_url: Option<String>,
    /// Success/failure status.
    #[serde(default)]
    pub success_status: Option<SuccessStatus>,
    /// Status detail message.
    #[serde(default)]
    pub status_detail: Option<String>,
    /// Override application results.
    #[serde(default)]
    pub override_application_results: Option<Vec<ContextOverrideStatus>>,
}

impl RolloutResponse {
    /// Check if the rollout succeeded.
    pub fn is_success(&self) -> bool {
        self.success_status.map(|s| s.is_success()).unwrap_or(true)
    }

    /// Get the outcome reward.
    pub fn outcome_reward(&self) -> f64 {
        self.reward_info.outcome_reward
    }
}

/// Health check response.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HealthResponse {
    /// Is the service healthy.
    pub healthy: bool,
    /// Authentication info.
    #[serde(default)]
    pub auth: Option<AuthInfo>,
    /// Service version.
    #[serde(default)]
    pub version: Option<String>,
}

/// Authentication info from health check.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuthInfo {
    /// Is authentication required.
    pub required: bool,
    /// Expected key prefix (first 6 chars).
    #[serde(default)]
    pub expected_prefix: Option<String>,
}

/// Service info response.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InfoResponse {
    /// Service details.
    #[serde(default)]
    pub service: Option<Value>,
    /// Dataset info.
    #[serde(default)]
    pub dataset: Option<DatasetInfo>,
    /// Rubrics.
    #[serde(default)]
    pub rubrics: Option<Value>,
    /// Inference info.
    #[serde(default)]
    pub inference: Option<InferenceInfo>,
    /// Limits.
    #[serde(default)]
    pub limits: Option<LimitsInfo>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_task_descriptor() {
        let desc = TaskDescriptor::new("task-1", "My Task");
        assert_eq!(desc.id, "task-1");
        assert_eq!(desc.name, "My Task");
    }

    #[test]
    fn test_rollout_request() {
        let request = RolloutRequest::new("trace-123").with_env(RolloutEnvSpec {
            seed: Some(42),
            ..Default::default()
        });

        assert_eq!(request.trace_correlation_id, "trace-123");
        assert_eq!(request.env.seed, Some(42));
        assert_eq!(request.on_done, "reset");
    }

    #[test]
    fn test_rollout_metrics() {
        let metrics = RolloutMetrics::new(0.85);
        assert_eq!(metrics.outcome_reward, 0.85);
    }

    #[test]
    fn test_rollout_response_success() {
        let response = RolloutResponse {
            trace_correlation_id: "trace-1".to_string(),
            reward_info: RolloutMetrics::new(1.0),
            trace: None,
            artifact: None,
            inference_url: None,
            success_status: Some(SuccessStatus::Success),
            status_detail: None,
            override_application_results: None,
        };

        assert!(response.is_success());
        assert_eq!(response.outcome_reward(), 1.0);
    }

    #[test]
    fn test_serde_roundtrip() {
        let request = RolloutRequest::new("test");
        let json = serde_json::to_string(&request).unwrap();
        let parsed: RolloutRequest = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.trace_correlation_id, "test");
    }
}
