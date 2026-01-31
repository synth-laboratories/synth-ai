//! API request and response types.
//!
//! This module contains all the types used for communicating with the Synth API.

use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::HashMap;

// =============================================================================
// Job Status Enums
// =============================================================================

/// Status for policy optimization jobs (GEPA, MIPRO).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PolicyJobStatus {
    Pending,
    Queued,
    Running,
    Succeeded,
    Failed,
    Cancelled,
}

impl PolicyJobStatus {
    /// Check if this status is terminal (job won't change state).
    pub fn is_terminal(&self) -> bool {
        matches!(self, Self::Succeeded | Self::Failed | Self::Cancelled)
    }

    /// Check if this status indicates success.
    pub fn is_success(&self) -> bool {
        matches!(self, Self::Succeeded)
    }

    /// Parse from string (case-insensitive).
    pub fn from_str(s: &str) -> Option<Self> {
        match s.to_lowercase().as_str() {
            "pending" => Some(Self::Pending),
            "queued" => Some(Self::Queued),
            "running" => Some(Self::Running),
            "succeeded" => Some(Self::Succeeded),
            "failed" => Some(Self::Failed),
            "cancelled" | "canceled" => Some(Self::Cancelled),
            _ => None,
        }
    }

    /// Convert to string representation.
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Pending => "pending",
            Self::Queued => "queued",
            Self::Running => "running",
            Self::Succeeded => "succeeded",
            Self::Failed => "failed",
            Self::Cancelled => "cancelled",
        }
    }
}

/// Status for evaluation jobs.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EvalJobStatus {
    Pending,
    Running,
    Completed,
    Failed,
    Cancelled,
}

impl EvalJobStatus {
    pub fn is_terminal(&self) -> bool {
        matches!(self, Self::Completed | Self::Failed | Self::Cancelled)
    }

    pub fn is_success(&self) -> bool {
        matches!(self, Self::Completed)
    }

    pub fn from_str(s: &str) -> Option<Self> {
        match s.to_lowercase().as_str() {
            "pending" => Some(Self::Pending),
            "running" => Some(Self::Running),
            "completed" => Some(Self::Completed),
            "failed" => Some(Self::Failed),
            "cancelled" | "canceled" => Some(Self::Cancelled),
            _ => None,
        }
    }

    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Pending => "pending",
            Self::Running => "running",
            Self::Completed => "completed",
            Self::Failed => "failed",
            Self::Cancelled => "cancelled",
        }
    }
}

// =============================================================================
// Policy Configuration
// =============================================================================

/// Policy model configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PolicyConfig {
    /// Model name (e.g., "gpt-4o-mini", "claude-3-5-sonnet").
    pub model: String,
    /// Provider name (e.g., "openai", "anthropic").
    pub provider: String,
    /// Optional temperature setting.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f64>,
    /// Optional max tokens.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<i32>,
}

impl Default for PolicyConfig {
    fn default() -> Self {
        Self {
            model: "gpt-4o-mini".to_string(),
            provider: "openai".to_string(),
            temperature: None,
            max_tokens: None,
        }
    }
}

// =============================================================================
// GEPA Configuration
// =============================================================================

/// GEPA algorithm configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GepaConfig {
    /// Number of generations to run.
    #[serde(default = "default_generations")]
    pub generations: i32,
    /// Population size per generation.
    #[serde(default = "default_population_size")]
    pub population_size: i32,
    /// Mutation rate (0.0 to 1.0).
    #[serde(default = "default_mutation_rate")]
    pub mutation_rate: f64,
    /// Crossover rate (0.0 to 1.0).
    #[serde(default = "default_crossover_rate")]
    pub crossover_rate: f64,
    /// Number of elite candidates to preserve.
    #[serde(default = "default_elite_count")]
    pub elite_count: i32,
    /// Training seeds.
    #[serde(default)]
    pub train_seeds: Vec<i64>,
    /// Validation seeds.
    #[serde(default)]
    pub validation_seeds: Vec<i64>,
}

fn default_generations() -> i32 {
    3
}
fn default_population_size() -> i32 {
    10
}
fn default_mutation_rate() -> f64 {
    0.1
}
fn default_crossover_rate() -> f64 {
    0.5
}
fn default_elite_count() -> i32 {
    2
}

impl Default for GepaConfig {
    fn default() -> Self {
        Self {
            generations: default_generations(),
            population_size: default_population_size(),
            mutation_rate: default_mutation_rate(),
            crossover_rate: default_crossover_rate(),
            elite_count: default_elite_count(),
            train_seeds: vec![],
            validation_seeds: vec![],
        }
    }
}

// =============================================================================
// MIPRO Configuration
// =============================================================================

/// MIPRO algorithm configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MiproConfig {
    /// Number of iterations.
    #[serde(default = "default_mipro_iterations")]
    pub iterations: i32,
    /// Number of candidates per iteration.
    #[serde(default = "default_mipro_candidates")]
    pub num_candidates: i32,
    /// Training seeds.
    #[serde(default)]
    pub train_seeds: Vec<i64>,
    /// Validation seeds.
    #[serde(default)]
    pub validation_seeds: Vec<i64>,
}

fn default_mipro_iterations() -> i32 {
    5
}
fn default_mipro_candidates() -> i32 {
    10
}

impl Default for MiproConfig {
    fn default() -> Self {
        Self {
            iterations: default_mipro_iterations(),
            num_candidates: default_mipro_candidates(),
            train_seeds: vec![],
            validation_seeds: vec![],
        }
    }
}

// =============================================================================
// Job Submit Requests
// =============================================================================

/// Request to submit a GEPA optimization job.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GepaJobRequest {
    /// Algorithm name (always "gepa").
    #[serde(default = "default_gepa_algorithm")]
    pub algorithm: String,
    /// Task app URL (where the task app is running).
    pub task_app_url: String,
    /// Optional SynthTunnel worker token (sent via header, not body).
    #[serde(default, skip_serializing)]
    pub task_app_worker_token: Option<String>,
    /// Optional API key for the task app.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub task_app_api_key: Option<String>,
    /// Environment name within the task app.
    pub env_name: String,
    /// Policy configuration.
    pub policy: PolicyConfig,
    /// GEPA-specific configuration.
    pub gepa: GepaConfig,
}

fn default_gepa_algorithm() -> String {
    "gepa".to_string()
}

impl Default for GepaJobRequest {
    fn default() -> Self {
        Self {
            algorithm: default_gepa_algorithm(),
            task_app_url: String::new(),
            task_app_worker_token: None,
            task_app_api_key: None,
            env_name: "default".to_string(),
            policy: PolicyConfig::default(),
            gepa: GepaConfig::default(),
        }
    }
}

/// Request to submit a MIPRO optimization job.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MiproJobRequest {
    /// Algorithm name (always "mipro").
    #[serde(default = "default_mipro_algorithm")]
    pub algorithm: String,
    /// Task app URL.
    pub task_app_url: String,
    /// Optional SynthTunnel worker token (sent via header, not body).
    #[serde(default, skip_serializing)]
    pub task_app_worker_token: Option<String>,
    /// Optional API key for the task app.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub task_app_api_key: Option<String>,
    /// Environment name.
    pub env_name: String,
    /// Policy configuration.
    pub policy: PolicyConfig,
    /// MIPRO-specific configuration.
    pub mipro: MiproConfig,
}

fn default_mipro_algorithm() -> String {
    "mipro".to_string()
}

impl Default for MiproJobRequest {
    fn default() -> Self {
        Self {
            algorithm: default_mipro_algorithm(),
            task_app_url: String::new(),
            task_app_worker_token: None,
            task_app_api_key: None,
            env_name: "default".to_string(),
            policy: PolicyConfig::default(),
            mipro: MiproConfig::default(),
        }
    }
}

/// Request to submit an evaluation job.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvalJobRequest {
    /// Optional app identifier.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub app_id: Option<String>,
    /// Task app URL.
    pub task_app_url: String,
    /// Optional SynthTunnel worker token (sent via header, not body).
    #[serde(default, skip_serializing)]
    pub task_app_worker_token: Option<String>,
    /// Optional API key for the task app.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub task_app_api_key: Option<String>,
    /// Environment name.
    pub env_name: String,
    /// Optional environment configuration payload.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub env_config: Option<Value>,
    /// Optional verifier configuration.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub verifier_config: Option<Value>,
    /// Seeds to evaluate.
    pub seeds: Vec<i64>,
    /// Policy configuration.
    pub policy: PolicyConfig,
    /// Maximum concurrent evaluations.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_concurrent: Option<i32>,
    /// Optional timeout in seconds.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub timeout: Option<f64>,
}

impl Default for EvalJobRequest {
    fn default() -> Self {
        Self {
            app_id: None,
            task_app_url: String::new(),
            task_app_worker_token: None,
            task_app_api_key: None,
            env_name: "default".to_string(),
            env_config: None,
            verifier_config: None,
            seeds: vec![],
            policy: PolicyConfig::default(),
            max_concurrent: None,
            timeout: None,
        }
    }
}

// =============================================================================
// Job Results
// =============================================================================

/// Response from submitting a job.
#[derive(Debug, Clone, Deserialize)]
pub struct JobSubmitResponse {
    /// The job ID.
    pub job_id: String,
}

/// Result of a policy optimization job (GEPA/MIPRO).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PromptLearningResult {
    /// Job ID.
    pub job_id: String,
    /// Current status.
    pub status: PolicyJobStatus,
    /// Best reward achieved (if available).
    #[serde(skip_serializing_if = "Option::is_none", alias = "best_score")]
    pub best_reward: Option<f64>,
    /// Best prompt found (if available).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub best_prompt: Option<Value>,
    /// Error message (if failed).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
    /// Number of generations completed.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub generations_completed: Option<i32>,
    /// Total candidates evaluated.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub candidates_evaluated: Option<i32>,
}

/// Result of an evaluation job.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvalResult {
    /// Job ID.
    pub job_id: String,
    /// Current status.
    pub status: EvalJobStatus,
    /// Mean reward across all seeds.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub mean_reward: Option<f64>,
    /// Total tokens used.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub total_tokens: Option<i64>,
    /// Number of seeds completed.
    #[serde(default)]
    pub num_completed: i32,
    /// Total number of seeds.
    #[serde(default)]
    pub num_total: i32,
    /// Error message (if failed).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
}

// =============================================================================
// Graph/Verifier Types
// =============================================================================

/// Request for a graph completion.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphCompletionRequest {
    /// Graph/job ID (e.g., "zero_shot_verifier_rubric_single").
    pub job_id: String,
    /// Input to the graph.
    pub input: Value,
    /// Optional model override.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,
    /// Optional prompt snapshot ID.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prompt_snapshot_id: Option<String>,
    /// Whether to stream the response.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stream: Option<bool>,
}

/// Usage statistics from a graph completion.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Usage {
    /// Prompt tokens used.
    #[serde(default)]
    pub prompt_tokens: i64,
    /// Completion tokens generated.
    #[serde(default)]
    pub completion_tokens: i64,
    /// Total tokens.
    #[serde(default)]
    pub total_tokens: i64,
}

/// Response from a graph completion.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphCompletionResponse {
    /// Output from the graph.
    pub output: Value,
    /// Token usage (if available).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub usage: Option<Usage>,
    /// Cache status (hit/miss).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cache_status: Option<String>,
    /// Optional latency in milliseconds.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub latency_ms: Option<f64>,
}

/// Options for verifier inference.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct VerifierOptions {
    /// Model to use for verification.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,
    /// Verifier graph ID (default: zero_shot_verifier_rubric_single).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub verifier_id: Option<String>,
    /// Whether to save evidence locally.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub save_evidence: Option<bool>,
}

/// A single review payload from the verifier.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReviewPayload {
    /// Score (0.0 to 1.0).
    #[serde(default)]
    pub score: f64,
    /// Reasoning for the score.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning: Option<String>,
    /// Objective being evaluated.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub objective: Option<String>,
}

/// Evidence item from verifier.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvidenceItem {
    /// Type of evidence.
    #[serde(rename = "type")]
    pub evidence_type: String,
    /// Evidence content.
    pub content: Value,
}

/// Response from verifier inference.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VerifierResponse {
    /// Status of the verification.
    pub status: String,
    /// Reviews for individual events.
    #[serde(default)]
    pub event_reviews: Vec<ReviewPayload>,
    /// Overall outcome review.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub outcome_review: Option<ReviewPayload>,
    /// Objective scores.
    #[serde(default)]
    pub objectives: HashMap<String, f64>,
    /// Evidence collected.
    #[serde(default)]
    pub evidence: Vec<EvidenceItem>,
}

/// Options for RLM (Retrieval-augmented LM) inference.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct RlmOptions {
    /// Model to use.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,
    /// RLM graph ID.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub rlm_id: Option<String>,
    /// Maximum context tokens.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_context_tokens: Option<i32>,
}

// =============================================================================
// Cancel Request
// =============================================================================

/// Request to cancel a job.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CancelRequest {
    /// Optional reason for cancellation.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reason: Option<String>,
}

// =============================================================================
// LocalAPI Deployments
// =============================================================================

fn default_localapi_timeout_s() -> i32 {
    600
}

fn default_localapi_cpu_cores() -> i32 {
    2
}

fn default_localapi_memory_mb() -> i32 {
    4096
}

fn default_localapi_dockerfile_path() -> String {
    "Dockerfile".to_string()
}

fn default_localapi_entrypoint_mode() -> String {
    "stdio".to_string()
}

fn default_localapi_port() -> i32 {
    8000
}

/// Resource limits for managed LocalAPI deployments.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LocalApiLimits {
    /// Timeout in seconds for a rollout request.
    #[serde(default = "default_localapi_timeout_s")]
    pub timeout_s: i32,
    /// CPU cores allocated to the deployment.
    #[serde(default = "default_localapi_cpu_cores")]
    pub cpu_cores: i32,
    /// Memory allocation in MB.
    #[serde(default = "default_localapi_memory_mb")]
    pub memory_mb: i32,
}

impl Default for LocalApiLimits {
    fn default() -> Self {
        Self {
            timeout_s: default_localapi_timeout_s(),
            cpu_cores: default_localapi_cpu_cores(),
            memory_mb: default_localapi_memory_mb(),
        }
    }
}

/// Deployment specification for managed LocalAPI builds.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LocalApiDeploySpec {
    /// Deployment name (org-unique).
    pub name: String,
    /// Dockerfile path inside the build context.
    #[serde(default = "default_localapi_dockerfile_path")]
    pub dockerfile_path: String,
    /// Command to start the LocalAPI server.
    pub entrypoint: String,
    /// Entry point mode ("stdio" or "command").
    #[serde(default = "default_localapi_entrypoint_mode")]
    pub entrypoint_mode: String,
    /// Port exposed by the LocalAPI server.
    #[serde(default = "default_localapi_port")]
    pub port: i32,
    /// Optional deployment description.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    /// Environment variables to set in the deployment.
    #[serde(default)]
    pub env_vars: HashMap<String, String>,
    /// Resource limits for the deployment.
    #[serde(default)]
    pub limits: LocalApiLimits,
    /// Additional metadata.
    #[serde(default)]
    pub metadata: HashMap<String, Value>,
}

/// Response for a LocalAPI deployment request.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LocalApiDeployResponse {
    pub deployment_id: String,
    pub status: String,
    pub task_app_url: String,
    #[serde(default)]
    pub task_app_api_key_env: Option<String>,
}

/// Status response for LocalAPI deployments.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LocalApiDeployStatus {
    pub deployment_id: String,
    pub status: String,
    pub provider: String,
    #[serde(default)]
    pub error: Option<String>,
}

/// Deployment list entry for LocalAPI deployments.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LocalApiDeploymentInfo {
    pub deployment_id: String,
    pub name: String,
    pub status: String,
    pub provider: String,
    pub task_app_url: String,
    #[serde(default)]
    pub created_at: Option<String>,
    #[serde(default)]
    pub updated_at: Option<String>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_policy_job_status_terminal() {
        assert!(!PolicyJobStatus::Pending.is_terminal());
        assert!(!PolicyJobStatus::Running.is_terminal());
        assert!(PolicyJobStatus::Succeeded.is_terminal());
        assert!(PolicyJobStatus::Failed.is_terminal());
        assert!(PolicyJobStatus::Cancelled.is_terminal());
    }

    #[test]
    fn test_policy_job_status_from_str() {
        assert_eq!(
            PolicyJobStatus::from_str("pending"),
            Some(PolicyJobStatus::Pending)
        );
        assert_eq!(
            PolicyJobStatus::from_str("RUNNING"),
            Some(PolicyJobStatus::Running)
        );
        assert_eq!(
            PolicyJobStatus::from_str("cancelled"),
            Some(PolicyJobStatus::Cancelled)
        );
        assert_eq!(
            PolicyJobStatus::from_str("canceled"),
            Some(PolicyJobStatus::Cancelled)
        );
        assert_eq!(PolicyJobStatus::from_str("invalid"), None);
    }

    #[test]
    fn test_gepa_request_serialization() {
        let req = GepaJobRequest {
            task_app_url: "http://localhost:8000".to_string(),
            env_name: "test".to_string(),
            ..Default::default()
        };
        let json = serde_json::to_string(&req).unwrap();
        assert!(json.contains("\"algorithm\":\"gepa\""));
        assert!(json.contains("\"task_app_url\":\"http://localhost:8000\""));
    }
}
