//! Reward data structures.
//!
//! Types for event-level and outcome-level rewards plus aggregates.

use super::enums::{RewardSource, RewardType};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::HashMap;

/// Episode-level reward summary.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OutcomeRewardRecord {
    /// Session ID this reward belongs to.
    pub session_id: String,
    /// Total reward value.
    pub total_reward: f64,
    /// Objective key (default: "reward").
    #[serde(default = "default_objective_key")]
    pub objective_key: String,
    /// Count of achievements.
    #[serde(default)]
    pub achievements_count: i32,
    /// Total steps in session.
    #[serde(default)]
    pub total_steps: i32,
    /// Additional metadata.
    #[serde(default)]
    pub metadata: HashMap<String, Value>,
    /// Annotation or notes.
    #[serde(default)]
    pub annotation: HashMap<String, Value>,
    /// Creation timestamp (ISO string).
    #[serde(default)]
    pub created_at: Option<String>,
}

/// Event-level reward annotation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EventRewardRecord {
    /// Event ID this reward annotates.
    pub event_id: String,
    /// Session ID this event belongs to.
    pub session_id: String,
    /// Reward value for the event.
    pub reward_value: f64,
    /// Objective key (default: "reward").
    #[serde(default = "default_objective_key")]
    pub objective_key: String,
    /// Reward type (shaped, sparse, etc.).
    #[serde(default)]
    pub reward_type: Option<RewardType>,
    /// Optional key (e.g., achievement name).
    #[serde(default)]
    pub key: Option<String>,
    /// Turn number in the session.
    #[serde(default)]
    pub turn_number: Option<i32>,
    /// Source of the reward.
    #[serde(default)]
    pub source: Option<RewardSource>,
    /// Annotation or notes.
    #[serde(default)]
    pub annotation: HashMap<String, Value>,
    /// Creation timestamp (ISO string).
    #[serde(default)]
    pub created_at: Option<String>,
}

/// Aggregated reward statistics.
#[derive(Debug, Clone, Serialize, Deserialize)]
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

/// Few-shot calibration example for verifier evaluation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CalibrationExample {
    pub session_trace: HashMap<String, Value>,
    pub event_rewards: Vec<f64>,
    pub outcome_reward: f64,
    #[serde(default)]
    pub metadata: HashMap<String, Value>,
}

/// Gold-standard example for contrastive verifier evaluation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GoldExample {
    pub summary: String,
    pub gold_score: f64,
    pub gold_reasoning: String,
    #[serde(default)]
    pub session_trace: Option<HashMap<String, Value>>,
    #[serde(default)]
    pub metadata: HashMap<String, Value>,
}

fn default_objective_key() -> String {
    "reward".to_string()
}
