use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::HashMap;

pub use super::progress::{SeedInfo, StageInfo, TokenUsage};

pub const MAX_INSTRUCTION_LENGTH: usize = 4000;
pub const MAX_ROLLOUT_SAMPLES: usize = 5;
pub const MAX_SEED_INFO_COUNT: usize = 50;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MutationTypeStats {
    pub attempts: i64,
    pub acceptances: i64,
    pub acceptance_rate: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MutationSummary {
    pub by_mutation_type: HashMap<String, MutationTypeStats>,
    pub total_attempts: i64,
    pub total_acceptances: i64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SeedAnalysis {
    pub hard_seeds: Vec<i64>,
    pub easy_seeds: Vec<i64>,
    pub baseline_failures: Vec<i64>,
    pub total_seeds_evaluated: i64,
    pub total_candidates: i64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PhaseSummary {
    pub phase: String,
    #[serde(default)]
    pub duration_seconds: Option<f64>,
    #[serde(default)]
    pub rollouts_completed: Option<i64>,
    #[serde(default)]
    pub candidates_evaluated: Option<i64>,
    #[serde(default)]
    #[serde(alias = "best_score")]
    pub best_reward: Option<f64>,
    #[serde(default)]
    pub extra: HashMap<String, Value>,
}

fn default_mutation_type() -> String {
    "unknown".to_string()
}

fn default_status() -> String {
    "evaluated".to_string()
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProgramCandidate {
    pub candidate_id: String,
    pub generation: i64,
    #[serde(default)]
    pub stages: HashMap<String, StageInfo>,
    #[serde(default)]
    pub parent_id: Option<String>,
    #[serde(default = "default_mutation_type")]
    pub mutation_type: String,
    #[serde(default)]
    pub mutation_params: Option<Value>,
    #[serde(default, alias = "accuracy")]
    pub reward: f64,
    #[serde(default, alias = "val_accuracy")]
    pub val_reward: Option<f64>,
    #[serde(default, alias = "minibatch_score")]
    pub minibatch_reward: Option<f64>,
    #[serde(default, alias = "seed_scores")]
    pub seed_rewards: Option<Vec<Value>>,
    #[serde(default)]
    pub seed_info: Option<Vec<SeedInfo>>,
    #[serde(default, alias = "instance_scores")]
    pub instance_rewards: Option<Vec<Value>>,
    #[serde(default)]
    pub objectives: Option<Value>,
    #[serde(default)]
    pub instance_objectives: Option<Value>,
    #[serde(default)]
    pub newly_solved_seeds: Option<Vec<i64>>,
    #[serde(default)]
    pub artifact_refs: Option<Vec<Value>>,
    #[serde(default)]
    pub success_statuses: Option<Vec<Value>>,
    #[serde(default)]
    pub token_usage: Option<TokenUsage>,
    #[serde(default)]
    pub cost_usd: Option<f64>,
    #[serde(default)]
    pub timestamp_ms: Option<i64>,
    #[serde(default)]
    pub evaluation_duration_ms: Option<i64>,
    #[serde(default)]
    pub transformation: Option<Value>,
    #[serde(default)]
    pub prompt_length: Option<i64>,
    #[serde(default = "default_status")]
    pub status: String,
    #[serde(default)]
    pub context_override_bundle_id: Option<String>,
    #[serde(default)]
    pub context_overrides: Option<Vec<Value>>,
    #[serde(default)]
    pub override_application_status: Option<String>,
    #[serde(default)]
    pub override_application_errors: Option<Vec<Value>>,
    #[serde(default)]
    pub context_snapshot_ref: Option<String>,
}

impl ProgramCandidate {
    pub fn prompt_summary(&self, max_length: usize) -> String {
        if self.stages.is_empty() {
            return String::new();
        }

        let mut keys: Vec<&String> = self.stages.keys().collect();
        keys.sort();

        let mut parts: Vec<String> = Vec::new();
        for key in keys {
            if let Some(stage) = self.stages.get(key) {
                let mut instruction = stage.instruction.clone();
                if instruction.len() > MAX_INSTRUCTION_LENGTH {
                    instruction.truncate(MAX_INSTRUCTION_LENGTH);
                    instruction.push_str("...");
                }
                if !instruction.is_empty() {
                    parts.push(format!("[{}]: {}", key.to_uppercase(), instruction));
                }
            }
        }

        let mut summary = parts.join("\n\n");
        if summary.len() > max_length {
            summary.truncate(max_length);
            summary.push_str("...");
        }
        summary
    }
}
