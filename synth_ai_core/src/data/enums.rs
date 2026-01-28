//! Core enumeration types for the Synth SDK.
//!
//! These enums match the Python SDK's `synth_ai/data/enums.py`.

use serde::{Deserialize, Serialize};

/// Type of optimization/training job.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum JobType {
    PromptLearning,
    Sft,
    Rl,
    Gspo,
    Eval,
    ResearchAgent,
}

impl Default for JobType {
    fn default() -> Self {
        Self::PromptLearning
    }
}

/// Status of a job.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum JobStatus {
    Pending,
    Queued,
    Running,
    Succeeded,
    Failed,
    Cancelled,
}

impl JobStatus {
    /// Returns true if the job is in a terminal state.
    pub fn is_terminal(&self) -> bool {
        matches!(self, Self::Succeeded | Self::Failed | Self::Cancelled)
    }

    /// Returns true if the job completed successfully.
    pub fn is_success(&self) -> bool {
        matches!(self, Self::Succeeded)
    }
}

impl Default for JobStatus {
    fn default() -> Self {
        Self::Pending
    }
}

/// LLM provider name.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ProviderName {
    Openai,
    Groq,
    Google,
    Anthropic,
    Together,
    Fireworks,
    Bedrock,
    Azure,
}

impl Default for ProviderName {
    fn default() -> Self {
        Self::Openai
    }
}

/// Inference mode for model calls.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum InferenceMode {
    Standard,
    Batched,
    Streaming,
    SynthHosted,
}

impl Default for InferenceMode {
    fn default() -> Self {
        Self::Standard
    }
}

/// Source of reward signal.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RewardSource {
    TaskApp,
    Verifier,
    Fused,
    Environment,
    Runner,
    Evaluator,
    Human,
}

impl Default for RewardSource {
    fn default() -> Self {
        Self::TaskApp
    }
}

/// Type of reward signal.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RewardType {
    Shaped,
    Sparse,
    Penalty,
    Evaluator,
    Human,
    Achievement,
    AchievementDelta,
    UniqueAchievementDelta,
}

impl Default for RewardType {
    fn default() -> Self {
        Self::Sparse
    }
}

/// Scope of reward (event-level or outcome-level).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RewardScope {
    Event,
    Outcome,
}

impl Default for RewardScope {
    fn default() -> Self {
        Self::Outcome
    }
}

/// Key identifying an objective metric.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ObjectiveKey {
    Reward,
    LatencyMs,
    CostUsd,
    TokensTotal,
    TurnsCount,
}

impl Default for ObjectiveKey {
    fn default() -> Self {
        Self::Reward
    }
}

/// Direction for objective optimization.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ObjectiveDirection {
    Maximize,
    Minimize,
}

impl Default for ObjectiveDirection {
    fn default() -> Self {
        Self::Maximize
    }
}

/// Output mode for LLM responses.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum OutputMode {
    ToolCalls,
    Text,
    Structured,
}

impl Default for OutputMode {
    fn default() -> Self {
        Self::Text
    }
}

/// Success/failure status for rollout execution.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SuccessStatus {
    Success,
    Timeout,
    NetworkError,
    ApplyFailed,
    RuntimeError,
    Failure,
}

impl SuccessStatus {
    /// Returns true if the status indicates success.
    pub fn is_success(&self) -> bool {
        matches!(self, Self::Success)
    }
}

impl Default for SuccessStatus {
    fn default() -> Self {
        Self::Success
    }
}

/// Type of graph in graph optimization.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum GraphType {
    Sequential,
    Parallel,
    Conditional,
    Loop,
    Policy,
    Verifier,
    Rlm,
}

impl Default for GraphType {
    fn default() -> Self {
        Self::Sequential
    }
}

/// Optimization mode for learning jobs.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum OptimizationMode {
    Online,
    Offline,
    Hybrid,
    Auto,
    GraphOnly,
    PromptOnly,
}

impl Default for OptimizationMode {
    fn default() -> Self {
        Self::Online
    }
}

/// Verifier evaluation mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum VerifierMode {
    Binary,
    Rubric,
    Criteria,
    Custom,
    Contrastive,
    GoldExamples,
}

impl Default for VerifierMode {
    fn default() -> Self {
        Self::Binary
    }
}

/// Training type for SFT/RL jobs.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TrainingType {
    Sft,
    Rl,
    Dpo,
    Ppo,
    Grpo,
    Gepa,
    GraphEvolve,
    Graphgen,
    Gspo,
}

impl Default for TrainingType {
    fn default() -> Self {
        Self::Sft
    }
}

/// Adaptive curriculum difficulty level.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AdaptiveCurriculumLevel {
    Easy,
    Medium,
    Hard,
    Expert,
    #[serde(alias = "NONE")]
    None,
    #[serde(alias = "LOW")]
    Low,
    #[serde(alias = "MODERATE")]
    Moderate,
    #[serde(alias = "HIGH")]
    High,
}

impl Default for AdaptiveCurriculumLevel {
    fn default() -> Self {
        Self::Medium
    }
}

/// Adaptive batch sizing level.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AdaptiveBatchLevel {
    Small,
    Medium,
    Large,
    Auto,
    #[serde(alias = "NONE")]
    None,
    #[serde(alias = "LOW")]
    Low,
    #[serde(alias = "MODERATE")]
    Moderate,
    #[serde(alias = "HIGH")]
    High,
}

/// Synth-hosted model names.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SynthModelName {
    SynthSmall,
    SynthMedium,
}

impl Default for AdaptiveBatchLevel {
    fn default() -> Self {
        Self::Auto
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_job_status_terminal() {
        assert!(!JobStatus::Pending.is_terminal());
        assert!(!JobStatus::Running.is_terminal());
        assert!(JobStatus::Succeeded.is_terminal());
        assert!(JobStatus::Failed.is_terminal());
        assert!(JobStatus::Cancelled.is_terminal());
    }

    #[test]
    fn test_serde_roundtrip() {
        let status = JobStatus::Running;
        let json = serde_json::to_string(&status).unwrap();
        assert_eq!(json, "\"running\"");

        let parsed: JobStatus = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed, status);
    }

    #[test]
    fn test_success_status() {
        assert!(SuccessStatus::Success.is_success());
        assert!(!SuccessStatus::Failure.is_success());
        assert!(!SuccessStatus::Timeout.is_success());
    }
}
