use crate::data::levers::ScopeKey;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::HashMap;

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum SensorKind {
    Reward,
    Timing,
    Rollout,
    Resource,
    Safety,
    Quality,
    Trace,
    ContextApply,
    Experiment,
    Custom,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Sensor {
    pub sensor_id: String,
    pub kind: SensorKind,
    pub scope: Vec<ScopeKey>,
    pub value: Value,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub units: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub timestamp: Option<DateTime<Utc>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub trace_id: Option<String>,
    #[serde(default, skip_serializing_if = "HashMap::is_empty")]
    pub metadata: HashMap<String, Value>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SensorFrame {
    pub scope: Vec<ScopeKey>,
    pub sensors: Vec<Sensor>,
    #[serde(default, skip_serializing_if = "HashMap::is_empty")]
    pub lever_versions: HashMap<String, i64>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub trace_ids: Vec<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub created_at: Option<DateTime<Utc>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub frame_id: Option<String>,
    #[serde(default, skip_serializing_if = "HashMap::is_empty")]
    pub metadata: HashMap<String, Value>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SensorFrameSummary {
    pub frame_id: String,
    pub created_at: DateTime<Utc>,
    pub sensor_count: usize,
    #[serde(default)]
    pub sensor_kinds: Vec<String>,
    #[serde(default)]
    pub trace_ids: Vec<String>,
    #[serde(default, skip_serializing_if = "HashMap::is_empty")]
    pub lever_versions: HashMap<String, i64>,
}

