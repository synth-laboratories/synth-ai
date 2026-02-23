use crate::{MiproError, Result};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use ulid::Ulid;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MiproConfig {
    pub num_candidates: usize,
    /// In (0, 1). Example: 0.2 means 80% train, 20% holdout.
    pub holdout_ratio: f32,
    pub max_iterations: usize,
    /// Stop after this many consecutive iterations without improvement.
    pub early_stop_rounds: usize,
    /// Minimum delta to count as improvement (avoid flapping on noise).
    pub min_improvement: f32,
    /// RNG seed for deterministic splits and sampling.
    pub seed: u64,
}

impl Default for MiproConfig {
    fn default() -> Self {
        Self {
            num_candidates: 16,
            holdout_ratio: 0.2,
            max_iterations: 10,
            early_stop_rounds: 3,
            min_improvement: 1e-6,
            seed: 0,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Policy {
    /// Prompt template. If it contains `{input}`, the example input is substituted there.
    /// Otherwise the input is appended after two newlines.
    pub template: String,
    #[serde(default)]
    pub metadata: serde_json::Value,
}

impl Policy {
    #[tracing::instrument(skip_all)]
    pub fn render(&self, input: &serde_json::Value) -> Result<String> {
        let input_text = match input {
            serde_json::Value::String(s) => s.clone(),
            other => other.to_string(),
        };

        if self.template.contains("{input}") {
            Ok(self.template.replace("{input}", &input_text))
        } else {
            Ok(format!("{}\n\n{}", self.template, input_text))
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PromptVariant {
    pub id: Ulid,
    pub policy: Policy,
    #[serde(default)]
    pub parent_id: Option<Ulid>,
    #[serde(default)]
    pub description: String,
    pub created_at: DateTime<Utc>,
}

impl PromptVariant {
    #[tracing::instrument(skip_all)]
    pub fn new(policy: Policy, parent_id: Option<Ulid>, description: impl Into<String>) -> Self {
        Self {
            id: Ulid::new(),
            policy,
            parent_id,
            description: description.into(),
            created_at: Utc::now(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvalMetricResult {
    pub metric_name: String,
    pub score: f32,
    #[serde(default)]
    pub details: serde_json::Value,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizationHistoryEntry {
    pub iteration: usize,
    pub candidate_scores: Vec<(Ulid, f32)>,
    pub best_candidate: Ulid,
    pub best_score: f32,
    pub improved: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizationResult {
    pub run_id: Ulid,
    pub started_at: DateTime<Utc>,
    pub finished_at: DateTime<Utc>,
    pub initial_policy: Policy,
    pub best_policy: Policy,
    pub best_score: f32,
    pub history: Vec<OptimizationHistoryEntry>,
}

impl OptimizationResult {
    #[tracing::instrument(skip_all)]
    pub fn new(initial_policy: Policy) -> Self {
        Self {
            run_id: Ulid::new(),
            started_at: Utc::now(),
            finished_at: Utc::now(),
            initial_policy: initial_policy.clone(),
            best_policy: initial_policy,
            best_score: f32::NEG_INFINITY,
            history: vec![],
        }
    }

    #[tracing::instrument(skip_all)]
    pub fn finish(mut self) -> Self {
        self.finished_at = Utc::now();
        self
    }
}

#[tracing::instrument(skip_all)]
pub fn require_finite_score(score: f32) -> Result<f32> {
    if !score.is_finite() {
        return Err(MiproError::Unexpected(format!(
            "non-finite score produced: {score}"
        )));
    }
    Ok(score)
}
