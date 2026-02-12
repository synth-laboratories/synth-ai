use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::HashMap;

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq, Hash)]
#[serde(rename_all = "snake_case")]
pub enum ScopeKind {
    Org,
    System,
    Run,
    Branch,
    Iteration,
    Candidate,
    Stage,
    Seed,
    Rollout,
    Evaluation,
    Custom,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub struct ScopeKey {
    pub kind: ScopeKind,
    pub id: String,
}

impl ScopeKey {
    pub fn new(kind: ScopeKind, id: impl Into<String>) -> Self {
        Self {
            kind,
            id: id.into(),
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum LeverKind {
    Prompt,
    Context,
    Code,
    Constraint,
    Note,
    Spec,
    GraphYaml,
    Variable,
    Experiment,
    Custom,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum LeverFormat {
    Text,
    Json,
    Yaml,
    File,
    Custom,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum LeverMutability {
    Optimizer,
    Human,
    System,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct LeverConstraints {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub allowed_values: Option<Vec<Value>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub min_value: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_value: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub regex: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_bytes: Option<usize>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<usize>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub mime_type: Option<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum LeverActor {
    Optimizer,
    Human,
    System,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct LeverProvenance {
    pub actor: LeverActor,
    pub reason: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source_event_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source_trace_id: Option<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Lever {
    pub lever_id: String,
    pub kind: LeverKind,
    pub scope: Vec<ScopeKey>,
    pub value: Value,
    pub value_format: LeverFormat,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub constraints: Option<LeverConstraints>,
    pub mutability: LeverMutability,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub provenance: Option<LeverProvenance>,
    #[serde(default)]
    pub version: i64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub created_at: Option<DateTime<Utc>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub parent_version: Option<i64>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct LeverSnapshot {
    pub lever_id: String,
    pub resolved_scope: Vec<ScopeKey>,
    pub version: i64,
    pub value: Value,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub applied_at: Option<DateTime<Utc>>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct LeverMutation {
    pub lever_id: String,
    pub parent_version: i64,
    pub new_version: i64,
    pub mutation_type: String,
    #[serde(default)]
    pub delta: Value,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub created_at: Option<DateTime<Utc>>,
    #[serde(default)]
    pub optimizer_id: String,
}

/// MIPRO-specific lever summary payload (as returned in prompt-learning results).
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct MiproLeverSummary {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub prompt_lever_id: Option<String>,
    #[serde(default)]
    pub candidate_lever_versions: HashMap<String, i64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub best_candidate_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub selected_candidate_id: Option<String>,
    #[serde(default)]
    pub baseline_candidate_id: String,
    #[serde(default)]
    pub lever_count: usize,
    #[serde(default)]
    pub mutation_count: usize,
    #[serde(default)]
    pub latest_version: i64,
}
