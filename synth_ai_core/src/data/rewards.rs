//! Reward data structures.
//!
//! These mirror the Python `synth_ai.data.rewards` module and provide
//! pure data types for reward annotations and aggregates.

use super::enums::{RewardSource, RewardType};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::HashMap;

/// Episode-level reward summary.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OutcomeRewardRecord {
    /// Session ID this reward corresponds to.
    pub session_id: String,
    /// Aggregate reward value.
    pub total_reward: f64,
    /// Objective key (default "reward").
    #[serde(default = "default_objective_key")]
    pub objective_key: String,
    /// Number of achievements.
    #[serde(default)]
    pub achievements_count: i32,
    /// Total steps in the episode.
    #[serde(default)]
    pub total_steps: i32,
    /// Optional metadata.
    #[serde(default)]
    pub metadata: HashMap<String, Value>,
    /// Optional annotation.
    #[serde(default)]
    pub annotation: HashMap<String, Value>,
    /// Creation timestamp (ISO string).
    #[serde(default)]
    pub created_at: Option<String>,
}

fn default_objective_key() -> String {
    "reward".to_string()
}

/// Event-level reward annotation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EventRewardRecord {
    /// Event ID being rewarded.
    pub event_id: String,
    /// Session ID containing the event.
    pub session_id: String,
    /// Reward value.
    pub reward_value: f64,
    /// Objective key (default "reward").
    #[serde(default = "default_objective_key")]
    pub objective_key: String,
    /// Optional reward type.
    #[serde(default)]
    pub reward_type: Option<RewardType>,
    /// Optional rubric criterion key.
    #[serde(default)]
    pub key: Option<String>,
    /// Optional turn number.
    #[serde(default)]
    pub turn_number: Option<i32>,
    /// Optional reward source.
    #[serde(default)]
    pub source: Option<RewardSource>,
    /// Optional annotation.
    #[serde(default)]
    pub annotation: HashMap<String, Value>,
    /// Creation timestamp (ISO string).
    #[serde(default)]
    pub created_at: Option<String>,
}

/// Aggregated statistics for rewards.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct RewardAggregates {
    pub mean: f64,
    #[serde(default)]
    pub median: f64,
    #[serde(default)]
    pub std: f64,
    #[serde(default)]
    pub n: i32,
    #[serde(default)]
    pub min_value: Option<f64>,
    #[serde(default)]
    pub max_value: Option<f64>,
}

/// Calibration example for verifier evaluation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CalibrationExample {
    /// Full session trace (V3/V4 format).
    pub session_trace: Value,
    /// Rewards per event.
    pub event_rewards: Vec<f64>,
    /// Overall outcome reward.
    pub outcome_reward: f64,
    /// Optional metadata.
    #[serde(default)]
    pub metadata: HashMap<String, Value>,
}

/// Gold-standard example for contrastive evaluation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GoldExample {
    /// Summary of the trace.
    pub summary: String,
    /// Gold score.
    pub gold_score: f64,
    /// Gold reasoning/explanation.
    pub gold_reasoning: String,
    /// Optional full session trace.
    #[serde(default)]
    pub session_trace: Option<Value>,
    /// Optional metadata.
    #[serde(default)]
    pub metadata: HashMap<String, Value>,
}
