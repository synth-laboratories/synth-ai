//! Objective specifications and reward observations.
//!
//! Types for defining optimization objectives and recording reward signals.

use super::enums::{ObjectiveDirection, ObjectiveKey, RewardScope, RewardSource, RewardType};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::HashMap;

/// Specification for an optimization objective.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ObjectiveSpec {
    /// Key identifying the objective metric.
    pub key: ObjectiveKey,
    /// Whether to maximize or minimize this objective.
    pub direction: ObjectiveDirection,
    /// Optional units (e.g., "ms", "usd", "tokens").
    #[serde(default)]
    pub units: Option<String>,
    /// Human-readable description.
    #[serde(default)]
    pub description: Option<String>,
    /// Target value (for constrained optimization).
    #[serde(default)]
    pub target: Option<f64>,
    /// Minimum acceptable value.
    #[serde(default)]
    pub min_value: Option<f64>,
    /// Maximum acceptable value.
    #[serde(default)]
    pub max_value: Option<f64>,
}

impl ObjectiveSpec {
    /// Create a new objective spec.
    pub fn new(key: ObjectiveKey, direction: ObjectiveDirection) -> Self {
        Self {
            key,
            direction,
            units: None,
            description: None,
            target: None,
            min_value: None,
            max_value: None,
        }
    }

    /// Create a reward maximization objective.
    pub fn maximize_reward() -> Self {
        Self::new(ObjectiveKey::Reward, ObjectiveDirection::Maximize)
    }

    /// Create a cost minimization objective.
    pub fn minimize_cost() -> Self {
        Self::new(ObjectiveKey::CostUsd, ObjectiveDirection::Minimize).with_units("usd")
    }

    /// Create a latency minimization objective.
    pub fn minimize_latency() -> Self {
        Self::new(ObjectiveKey::LatencyMs, ObjectiveDirection::Minimize).with_units("ms")
    }

    /// Set units for this objective.
    pub fn with_units(mut self, units: impl Into<String>) -> Self {
        self.units = Some(units.into());
        self
    }

    /// Set description for this objective.
    pub fn with_description(mut self, desc: impl Into<String>) -> Self {
        self.description = Some(desc.into());
        self
    }

    /// Check if a value satisfies the objective's constraints.
    pub fn satisfies_constraints(&self, value: f64) -> bool {
        if let Some(min) = self.min_value {
            if value < min {
                return false;
            }
        }
        if let Some(max) = self.max_value {
            if value > max {
                return false;
            }
        }
        true
    }
}

/// A single reward observation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RewardObservation {
    /// The reward value.
    pub value: f64,
    /// Type of reward signal.
    #[serde(default)]
    pub reward_type: RewardType,
    /// Whether this is event-level or outcome-level.
    #[serde(default)]
    pub scope: RewardScope,
    /// Source of the reward signal.
    #[serde(default)]
    pub source: RewardSource,
    /// Which objective this reward corresponds to.
    #[serde(default)]
    pub objective_key: ObjectiveKey,
    /// Optional event ID (for event-level rewards).
    #[serde(default)]
    pub event_id: Option<String>,
    /// Optional turn number.
    #[serde(default)]
    pub turn_number: Option<i32>,
    /// Additional metadata.
    #[serde(default)]
    pub metadata: HashMap<String, Value>,
}

impl RewardObservation {
    /// Create a new reward observation.
    pub fn new(value: f64) -> Self {
        Self {
            value,
            reward_type: RewardType::default(),
            scope: RewardScope::default(),
            source: RewardSource::default(),
            objective_key: ObjectiveKey::default(),
            event_id: None,
            turn_number: None,
            metadata: HashMap::new(),
        }
    }

    /// Create an outcome-level reward.
    pub fn outcome(value: f64) -> Self {
        Self::new(value).with_scope(RewardScope::Outcome)
    }

    /// Create an event-level reward.
    pub fn event(value: f64, event_id: impl Into<String>) -> Self {
        let mut obs = Self::new(value).with_scope(RewardScope::Event);
        obs.event_id = Some(event_id.into());
        obs
    }

    /// Set the reward type.
    pub fn with_type(mut self, reward_type: RewardType) -> Self {
        self.reward_type = reward_type;
        self
    }

    /// Set the scope.
    pub fn with_scope(mut self, scope: RewardScope) -> Self {
        self.scope = scope;
        self
    }

    /// Set the source.
    pub fn with_source(mut self, source: RewardSource) -> Self {
        self.source = source;
        self
    }
}

/// Assignment of objectives to an outcome (session-level).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OutcomeObjectiveAssignment {
    /// Map of objective key to value.
    pub objectives: HashMap<String, f64>,
    /// Session ID this assignment belongs to.
    #[serde(default)]
    pub session_id: Option<String>,
    /// Trace correlation ID.
    #[serde(default)]
    pub trace_id: Option<String>,
    /// Additional metadata.
    #[serde(default)]
    pub metadata: HashMap<String, Value>,
}

impl OutcomeObjectiveAssignment {
    /// Create a new outcome objective assignment.
    pub fn new() -> Self {
        Self {
            objectives: HashMap::new(),
            session_id: None,
            trace_id: None,
            metadata: HashMap::new(),
        }
    }

    /// Add an objective value.
    pub fn with_objective(mut self, key: impl Into<String>, value: f64) -> Self {
        self.objectives.insert(key.into(), value);
        self
    }

    /// Set the session ID.
    pub fn with_session(mut self, session_id: impl Into<String>) -> Self {
        self.session_id = Some(session_id.into());
        self
    }
}

impl Default for OutcomeObjectiveAssignment {
    fn default() -> Self {
        Self::new()
    }
}

/// Assignment of objectives to an event.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EventObjectiveAssignment {
    /// Event ID this assignment belongs to.
    pub event_id: String,
    /// Map of objective key to value.
    pub objectives: HashMap<String, f64>,
    /// Turn number.
    #[serde(default)]
    pub turn_number: Option<i32>,
    /// Additional metadata.
    #[serde(default)]
    pub metadata: HashMap<String, Value>,
}

impl EventObjectiveAssignment {
    /// Create a new event objective assignment.
    pub fn new(event_id: impl Into<String>) -> Self {
        Self {
            event_id: event_id.into(),
            objectives: HashMap::new(),
            turn_number: None,
            metadata: HashMap::new(),
        }
    }

    /// Add an objective value.
    pub fn with_objective(mut self, key: impl Into<String>, value: f64) -> Self {
        self.objectives.insert(key.into(), value);
        self
    }
}

/// Assignment of objectives to a specific instance (dataset example).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InstanceObjectiveAssignment {
    /// Instance ID or seed.
    pub instance_id: String,
    /// Map of objective key to value.
    pub objectives: HashMap<String, f64>,
    /// Optional dataset split.
    #[serde(default)]
    pub split: Option<String>,
    /// Additional metadata.
    #[serde(default)]
    pub metadata: HashMap<String, Value>,
}

impl InstanceObjectiveAssignment {
    /// Create a new instance objective assignment.
    pub fn new(instance_id: impl Into<String>) -> Self {
        Self {
            instance_id: instance_id.into(),
            objectives: HashMap::new(),
            split: None,
            metadata: HashMap::new(),
        }
    }

    /// Add an objective value.
    pub fn with_objective(mut self, key: impl Into<String>, value: f64) -> Self {
        self.objectives.insert(key.into(), value);
        self
    }

    /// Set the dataset split.
    pub fn with_split(mut self, split: impl Into<String>) -> Self {
        self.split = Some(split.into());
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_objective_spec() {
        let spec = ObjectiveSpec::maximize_reward();
        assert_eq!(spec.key, ObjectiveKey::Reward);
        assert_eq!(spec.direction, ObjectiveDirection::Maximize);
    }

    #[test]
    fn test_objective_constraints() {
        let mut spec = ObjectiveSpec::minimize_latency();
        spec.max_value = Some(1000.0);

        assert!(spec.satisfies_constraints(500.0));
        assert!(!spec.satisfies_constraints(1500.0));
    }

    #[test]
    fn test_reward_observation() {
        let obs = RewardObservation::outcome(0.95)
            .with_type(RewardType::Sparse)
            .with_source(RewardSource::Verifier);

        assert_eq!(obs.value, 0.95);
        assert_eq!(obs.scope, RewardScope::Outcome);
        assert_eq!(obs.source, RewardSource::Verifier);
    }

    #[test]
    fn test_outcome_assignment() {
        let assignment = OutcomeObjectiveAssignment::new()
            .with_objective("reward", 0.85)
            .with_objective("cost_usd", 0.002)
            .with_session("session-123");

        assert_eq!(assignment.objectives.get("reward"), Some(&0.85));
        assert_eq!(assignment.session_id, Some("session-123".to_string()));
    }

    #[test]
    fn test_serde() {
        let obs = RewardObservation::outcome(1.0);
        let json = serde_json::to_string(&obs).unwrap();
        let parsed: RewardObservation = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.value, 1.0);
    }
}
