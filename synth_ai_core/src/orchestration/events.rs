//! Event types and parsing for optimization job events.
//!
//! This module provides event parsing and categorization for SSE events
//! from optimization jobs.

use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::HashMap;

/// Event category for classification.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EventCategory {
    /// Baseline evaluation event
    Baseline,
    /// Candidate evaluation event
    Candidate,
    /// Pareto frontier update
    Frontier,
    /// Progress update (rollouts, trials)
    Progress,
    /// Generation complete
    Generation,
    /// Throughput metrics
    Throughput,
    /// Early termination triggered
    Termination,
    /// Job complete
    Complete,
    /// Validation phase event
    Validation,
    /// Usage/billing event
    Usage,
    /// Unknown event type
    Unknown,
}

/// Terminal status derived from terminal events.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum TerminalStatus {
    Succeeded,
    Failed,
    Cancelled,
}

/// Parsed event path segments for logging/debugging.
#[derive(Debug, Clone, Default)]
pub struct EventPath {
    pub entity: Option<String>,
    pub action: Option<String>,
    pub algorithm: Option<String>,
    pub detail: Option<String>,
}

impl EventCategory {
    /// Check if this is a terminal event category.
    pub fn is_terminal(&self) -> bool {
        matches!(self, EventCategory::Complete | EventCategory::Termination)
    }

    /// Get the category as a string.
    pub fn as_str(&self) -> &'static str {
        match self {
            EventCategory::Baseline => "baseline",
            EventCategory::Candidate => "candidate",
            EventCategory::Frontier => "frontier",
            EventCategory::Progress => "progress",
            EventCategory::Generation => "generation",
            EventCategory::Throughput => "throughput",
            EventCategory::Termination => "termination",
            EventCategory::Complete => "complete",
            EventCategory::Validation => "validation",
            EventCategory::Usage => "usage",
            EventCategory::Unknown => "unknown",
        }
    }
}

/// Parsed event with category and typed data.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParsedEvent {
    /// Original event type string
    pub event_type: String,
    /// Classified category
    pub category: EventCategory,
    /// Event data payload
    pub data: Value,
    /// Sequence number for ordering
    #[serde(default)]
    pub seq: Option<i64>,
    /// Timestamp in milliseconds
    #[serde(default)]
    pub timestamp_ms: Option<i64>,
}

/// Baseline evaluation event data.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct BaselineEvent {
    /// Baseline reward
    #[serde(default, alias = "accuracy")]
    pub reward: Option<f64>,
    /// Objective scores
    #[serde(default)]
    pub objectives: Option<HashMap<String, f64>>,
    /// Per-instance rewards
    #[serde(default, alias = "instance_scores")]
    pub instance_rewards: Option<Vec<f64>>,
    /// Per-instance objectives
    #[serde(default)]
    pub instance_objectives: Option<Vec<HashMap<String, f64>>>,
    /// Prompt configuration
    #[serde(default)]
    pub prompt: Option<Value>,
}

/// Candidate evaluation event data.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct CandidateEvent {
    /// Candidate ID
    #[serde(default)]
    pub candidate_id: String,
    /// Candidate reward
    #[serde(default, alias = "accuracy")]
    pub reward: Option<f64>,
    /// Objective scores
    #[serde(default)]
    pub objectives: Option<HashMap<String, f64>>,
    /// Whether candidate was accepted
    #[serde(default)]
    pub accepted: bool,
    /// Generation number
    #[serde(default)]
    pub generation: Option<i32>,
    /// Parent candidate ID
    #[serde(default)]
    pub parent_id: Option<String>,
    /// Whether on Pareto frontier
    #[serde(default)]
    pub is_pareto: bool,
    /// Mutation type used
    #[serde(default)]
    pub mutation_type: Option<String>,
    /// Per-instance rewards
    #[serde(default, alias = "instance_scores")]
    pub instance_rewards: Option<Vec<f64>>,
    /// Per-instance objectives
    #[serde(default)]
    pub instance_objectives: Option<Vec<HashMap<String, f64>>>,
}

/// Frontier update event data.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct FrontierEvent {
    /// Current frontier candidate IDs
    #[serde(default)]
    pub frontier: Vec<String>,
    /// Candidates added to frontier
    #[serde(default)]
    pub added: Vec<String>,
    /// Candidates removed from frontier
    #[serde(default)]
    pub removed: Vec<String>,
    /// Frontier size
    #[serde(default)]
    pub frontier_size: i32,
    /// Best reward on frontier
    #[serde(default, alias = "best_score")]
    pub best_reward: Option<f64>,
    /// Rewards by candidate ID
    #[serde(default, alias = "frontier_scores")]
    pub frontier_rewards: Option<HashMap<String, f64>>,
    /// Objective scores by candidate (if provided)
    #[serde(default)]
    pub frontier_objectives: Option<Vec<HashMap<String, f64>>>,
}

/// Progress event data.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ProgressEvent {
    /// Rollouts completed
    #[serde(default)]
    pub rollouts_completed: i32,
    /// Total rollouts planned
    #[serde(default)]
    pub rollouts_total: Option<i32>,
    /// Trials completed (MIPRO)
    #[serde(default)]
    pub trials_completed: i32,
    /// Current best reward
    #[serde(default, alias = "best_score")]
    pub best_reward: Option<f64>,
    /// Baseline reward for comparison
    #[serde(default, alias = "baseline_score")]
    pub baseline_reward: Option<f64>,
}

/// Generation complete event data.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct GenerationEvent {
    /// Generation number
    #[serde(default)]
    pub generation: i32,
    /// Best reward in this generation
    #[serde(default, alias = "best_accuracy")]
    pub best_reward: f64,
    /// Candidates proposed
    #[serde(default)]
    pub candidates_proposed: i32,
    /// Candidates accepted
    #[serde(default)]
    pub candidates_accepted: i32,
}

/// Job complete event data.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct CompleteEvent {
    /// Final best reward
    #[serde(default, alias = "best_score")]
    pub best_reward: Option<f64>,
    /// Baseline reward
    #[serde(default, alias = "baseline_score")]
    pub baseline_reward: Option<f64>,
    /// Finish reason
    #[serde(default)]
    pub finish_reason: Option<String>,
    /// Total candidates evaluated
    #[serde(default)]
    pub total_candidates: i32,
}

/// Termination event data.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct TerminationEvent {
    /// Termination reason
    #[serde(default)]
    pub reason: String,
}

/// Usage event data.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct UsageEvent {
    /// Total cost in USD
    #[serde(default)]
    pub total_usd: f64,
    /// Token cost in USD
    #[serde(default)]
    pub tokens_usd: f64,
    /// Sandbox cost in USD
    #[serde(default)]
    pub sandbox_usd: f64,
}

/// Event parser for categorizing and parsing job events.
pub struct EventParser;

impl EventParser {
    /// Patterns for baseline events
    const BASELINE_PATTERNS: &'static [&'static str] = &[".baseline"];

    /// Patterns for candidate events
    const CANDIDATE_PATTERNS: &'static [&'static str] = &[
        ".candidate.evaluated",
        ".candidate.new_best",
        ".proposal.scored",
        ".optimized.scored",
        ".candidate_scored",
    ];

    /// Patterns for frontier events
    const FRONTIER_PATTERNS: &'static [&'static str] = &[".frontier_updated", ".frontier.updated"];

    /// Patterns for progress events
    const PROGRESS_PATTERNS: &'static [&'static str] = &[
        ".progress",
        ".rollouts_limit_progress",
        ".rollouts.progress",
        ".job.started",
        ".trial.started",
        ".trial.completed",
        ".iteration.started",
        ".iteration.completed",
    ];

    /// Patterns for generation events
    const GENERATION_PATTERNS: &'static [&'static str] = &[
        ".generation.complete",
        ".generation.completed",
        ".generation.started",
    ];

    /// Patterns for throughput events
    const THROUGHPUT_PATTERNS: &'static [&'static str] = &[
        ".throughput",
        ".rollout.concurrency",
        ".rollout_concurrency",
    ];

    /// Patterns for termination events
    const TERMINATION_PATTERNS: &'static [&'static str] =
        &[".termination.triggered", ".termination"];

    /// Patterns for complete events
    const COMPLETE_PATTERNS: &'static [&'static str] =
        &[".complete", ".completed", ".job.completed"];

    /// Patterns for validation events
    const VALIDATION_PATTERNS: &'static [&'static str] =
        &[".validation.scored", ".validation.completed"];

    /// Patterns for usage events
    const USAGE_PATTERNS: &'static [&'static str] =
        &[".usage.recorded", ".billing.sandboxes", ".billing.updated"];

    /// Normalize event type by replacing [MASKED] with "gepa".
    pub fn normalize_type(event_type: &str) -> String {
        event_type.replace("[MASKED]", "gepa")
    }

    /// Parse an event type into path segments for logging.
    pub fn parse_path(event_type: &str) -> EventPath {
        let normalized = Self::normalize_type(event_type);
        let parts: Vec<&str> = normalized.split('.').collect();
        let entity = parts.get(0).map(|s| s.to_string());
        let action = parts.get(1).map(|s| s.to_string());
        let algorithm = parts.get(2).map(|s| s.to_string());
        let detail = if parts.len() > 3 {
            Some(parts[3..].join("."))
        } else {
            None
        };
        EventPath {
            entity,
            action,
            algorithm,
            detail,
        }
    }

    /// Map a terminal event type to a terminal status.
    pub fn terminal_status(event_type: &str) -> Option<TerminalStatus> {
        let normalized = Self::normalize_type(event_type).to_lowercase();
        if normalized.contains("cancel") {
            return Some(TerminalStatus::Cancelled);
        }
        if normalized.contains("fail") || normalized.contains("error") {
            return Some(TerminalStatus::Failed);
        }
        if normalized.contains("complete") || normalized.contains("succeed") {
            return Some(TerminalStatus::Succeeded);
        }
        None
    }

    fn coerce_f64(value: Option<&Value>) -> Option<f64> {
        match value {
            Some(Value::Number(num)) => num.as_f64(),
            Some(Value::String(s)) => s.parse::<f64>().ok(),
            _ => None,
        }
    }

    fn coerce_i32(value: Option<&Value>) -> Option<i32> {
        match value {
            Some(Value::Number(num)) => num.as_i64().and_then(|v| i32::try_from(v).ok()),
            Some(Value::String(s)) => s.parse::<i32>().ok(),
            _ => None,
        }
    }

    fn coerce_bool(value: Option<&Value>) -> Option<bool> {
        match value {
            Some(Value::Bool(b)) => Some(*b),
            Some(Value::String(s)) => s.parse::<bool>().ok(),
            _ => None,
        }
    }

    fn coerce_string(value: Option<&Value>) -> Option<String> {
        match value {
            Some(Value::String(s)) => Some(s.clone()),
            Some(Value::Number(num)) => Some(num.to_string()),
            _ => None,
        }
    }

    fn parse_f64_map(value: Option<&Value>) -> Option<HashMap<String, f64>> {
        let obj = value?.as_object()?;
        if obj.is_empty() {
            return None;
        }
        let mut map = HashMap::new();
        for (k, v) in obj {
            if let Some(val) = Self::coerce_f64(Some(v)) {
                map.insert(k.clone(), val);
            } else {
                return None;
            }
        }
        Some(map)
    }

    fn parse_vec_f64_map(value: Option<&Value>) -> Option<Vec<HashMap<String, f64>>> {
        let arr = value?.as_array()?;
        if arr.is_empty() {
            return None;
        }
        let mut out = Vec::with_capacity(arr.len());
        for item in arr {
            let map = Self::parse_f64_map(Some(item))?;
            out.push(map);
        }
        Some(out)
    }

    fn parse_vec_string(value: Option<&Value>) -> Option<Vec<String>> {
        let arr = value?.as_array()?;
        if arr.is_empty() {
            return None;
        }
        let mut out = Vec::with_capacity(arr.len());
        for item in arr {
            let s = match item {
                Value::String(s) => s.clone(),
                Value::Number(n) => n.to_string(),
                _ => return None,
            };
            out.push(s);
        }
        Some(out)
    }

    fn extract_reward_from_value(value: Option<&Value>) -> Option<f64> {
        let value = value?;
        match value {
            Value::Number(num) => num.as_f64(),
            Value::String(s) => s.parse::<f64>().ok(),
            Value::Object(obj) => obj
                .get("reward")
                .and_then(|v| Self::coerce_f64(Some(v)))
                .or_else(|| {
                    obj.get("mean_reward")
                        .and_then(|v| Self::coerce_f64(Some(v)))
                })
                .or_else(|| {
                    obj.get("outcome_reward")
                        .and_then(|v| Self::coerce_f64(Some(v)))
                })
                .or_else(|| obj.get("accuracy").and_then(|v| Self::coerce_f64(Some(v))))
                .or_else(|| obj.get("score").and_then(|v| Self::coerce_f64(Some(v)))),
            _ => None,
        }
    }

    fn extract_instance_rewards(data: &Value) -> Option<Vec<f64>> {
        let instance_objectives = data.get("instance_objectives")?.as_array()?;
        if instance_objectives.is_empty() {
            return None;
        }
        let mut values = Vec::with_capacity(instance_objectives.len());
        for item in instance_objectives {
            let reward_val = if let Some(obj) = item.as_object() {
                if let Some(objectives) = obj.get("objectives").and_then(|v| v.as_object()) {
                    Self::coerce_f64(objectives.get("reward"))
                } else {
                    Self::coerce_f64(obj.get("reward"))
                }
            } else {
                None
            };
            let reward_val = reward_val?;
            values.push(reward_val);
        }
        if values.is_empty() {
            None
        } else {
            Some(values)
        }
    }

    /// Get the category for an event type string.
    pub fn get_category(event_type: &str) -> EventCategory {
        let normalized = Self::normalize_type(event_type);
        let lower = normalized.to_lowercase();

        // Generation.complete should be matched before generic .complete
        for pattern in Self::GENERATION_PATTERNS {
            if lower.contains(pattern) {
                return EventCategory::Generation;
            }
        }

        for pattern in Self::BASELINE_PATTERNS {
            if lower.contains(pattern) {
                return EventCategory::Baseline;
            }
        }

        for pattern in Self::CANDIDATE_PATTERNS {
            if lower.contains(pattern) {
                return EventCategory::Candidate;
            }
        }

        for pattern in Self::FRONTIER_PATTERNS {
            if lower.contains(pattern) {
                return EventCategory::Frontier;
            }
        }

        for pattern in Self::PROGRESS_PATTERNS {
            if lower.contains(pattern) {
                return EventCategory::Progress;
            }
        }

        for pattern in Self::THROUGHPUT_PATTERNS {
            if lower.contains(pattern) {
                return EventCategory::Throughput;
            }
        }

        for pattern in Self::TERMINATION_PATTERNS {
            if lower.contains(pattern) {
                return EventCategory::Termination;
            }
        }

        for pattern in Self::COMPLETE_PATTERNS {
            if lower.contains(pattern) {
                return EventCategory::Complete;
            }
        }

        for pattern in Self::VALIDATION_PATTERNS {
            if lower.contains(pattern) {
                return EventCategory::Validation;
            }
        }

        for pattern in Self::USAGE_PATTERNS {
            if lower.contains(pattern) {
                return EventCategory::Usage;
            }
        }

        EventCategory::Unknown
    }

    /// Parse a raw event JSON into a ParsedEvent.
    pub fn parse(event: &Value) -> ParsedEvent {
        let raw_type = event.get("type").and_then(|v| v.as_str()).unwrap_or("");
        let event_type = Self::normalize_type(raw_type);
        let category = Self::get_category(&event_type);

        let seq = event.get("seq").and_then(|v| v.as_i64());

        let timestamp_ms = event
            .get("timestamp_ms")
            .and_then(|v| v.as_i64())
            .or_else(|| {
                event.get("ts").and_then(|v| v.as_str()).and_then(|ts| {
                    chrono::DateTime::parse_from_rfc3339(ts)
                        .ok()
                        .map(|dt| dt.timestamp_millis())
                })
            });

        let data = event
            .get("data")
            .and_then(|v| v.as_object())
            .cloned()
            .unwrap_or_default();

        ParsedEvent {
            event_type,
            category,
            data: Value::Object(data),
            seq,
            timestamp_ms,
        }
    }

    /// Parse baseline event data from a ParsedEvent.
    pub fn parse_baseline(event: &ParsedEvent) -> BaselineEvent {
        let data = event.data.as_object().cloned().unwrap_or_default();
        let data_value = Value::Object(data.clone());

        let objectives = Self::parse_f64_map(data.get("objectives"));
        let reward_value = objectives.as_ref().and_then(|m| m.get("reward").copied());
        let accuracy = reward_value
            .or_else(|| Self::coerce_f64(data.get("accuracy")))
            .or_else(|| Self::coerce_f64(data.get("baseline_score")))
            .or_else(|| Self::coerce_f64(data.get("baseline_accuracy")))
            .or_else(|| {
                data.get("outcome_objectives")
                    .and_then(|v| Self::extract_reward_from_value(Some(v)))
            })
            .or_else(|| {
                data.get("outcome_reward")
                    .and_then(|v| Self::coerce_f64(Some(v)))
            })
            .or_else(|| {
                data.get("score")
                    .and_then(|v| Self::extract_reward_from_value(Some(v)))
            });

        let instance_objectives = Self::parse_vec_f64_map(data.get("instance_objectives"));
        let instance_rewards = Self::extract_instance_rewards(&data_value);
        let prompt = data.get("prompt").cloned();

        BaselineEvent {
            reward: accuracy,
            objectives,
            instance_rewards,
            instance_objectives,
            prompt,
        }
    }

    /// Parse candidate event data from a ParsedEvent.
    pub fn parse_candidate(event: &ParsedEvent) -> CandidateEvent {
        let data = event.data.as_object().cloned().unwrap_or_default();
        let candidate_data = data
            .get("program_candidate")
            .and_then(|v| v.as_object())
            .cloned();
        let candidate_view = candidate_data.as_ref().unwrap_or(&data);
        let candidate_value = Value::Object(candidate_view.clone());

        // Extract objectives: try top-level, then score.objectives
        let objectives = Self::parse_f64_map(candidate_view.get("objectives")).or_else(|| {
            candidate_view
                .get("score")
                .and_then(|v| v.as_object())
                .and_then(|score| Self::parse_f64_map(score.get("objectives")))
        });

        let reward_value = objectives.as_ref().and_then(|m| m.get("reward").copied());

        // Extract accuracy/reward: try objectives.reward, then direct fields
        // Backend emits `reward` (top-level), `score.mean_reward`, `score.objectives.reward`
        let accuracy = reward_value
            .or_else(|| Self::coerce_f64(candidate_view.get("reward")))
            .or_else(|| Self::coerce_f64(candidate_view.get("accuracy")))
            .or_else(|| Self::coerce_f64(candidate_view.get("score")))
            .or_else(|| Self::extract_reward_from_value(candidate_view.get("score")))
            .or_else(|| {
                candidate_view
                    .get("outcome_objectives")
                    .and_then(|v| Self::extract_reward_from_value(Some(v)))
            })
            .or_else(|| {
                candidate_view
                    .get("outcome_reward")
                    .and_then(|v| Self::coerce_f64(Some(v)))
            });

        // If we found a reward but no objectives dict, construct one
        let objectives = objectives.or_else(|| {
            accuracy.map(|r| {
                let mut m = std::collections::HashMap::new();
                m.insert("reward".to_string(), r);
                m
            })
        });

        let instance_objectives =
            Self::parse_vec_f64_map(candidate_view.get("instance_objectives"));
        let instance_rewards = Self::extract_instance_rewards(&candidate_value);

        CandidateEvent {
            candidate_id: Self::coerce_string(data.get("version_id"))
                .or_else(|| Self::coerce_string(data.get("candidate_id")))
                .unwrap_or_default(),
            reward: accuracy,
            objectives,
            accepted: Self::coerce_bool(data.get("accepted")).unwrap_or(false),
            generation: Self::coerce_i32(data.get("generation")),
            parent_id: Self::coerce_string(data.get("parent_id")),
            is_pareto: Self::coerce_bool(data.get("is_pareto")).unwrap_or(false),
            mutation_type: Self::coerce_string(data.get("mutation_type"))
                .or_else(|| Self::coerce_string(data.get("operator"))),
            instance_rewards,
            instance_objectives,
        }
    }

    /// Parse frontier event data from a ParsedEvent.
    pub fn parse_frontier(event: &ParsedEvent) -> FrontierEvent {
        let data = event.data.as_object().cloned().unwrap_or_default();
        let frontier = Self::parse_vec_string(data.get("frontier"));

        FrontierEvent {
            frontier: frontier.clone().unwrap_or_default(),
            added: Self::parse_vec_string(data.get("added")).unwrap_or_default(),
            removed: Self::parse_vec_string(data.get("removed")).unwrap_or_default(),
            frontier_size: Self::coerce_i32(data.get("frontier_size"))
                .unwrap_or_else(|| frontier.as_ref().map(|v| v.len() as i32).unwrap_or(0)),
            best_reward: Self::coerce_f64(data.get("best_score")),
            frontier_rewards: Self::parse_f64_map(data.get("frontier_scores")),
            frontier_objectives: Self::parse_vec_f64_map(data.get("frontier_objectives")),
        }
    }

    /// Parse progress event data from a ParsedEvent.
    pub fn parse_progress(event: &ParsedEvent) -> ProgressEvent {
        let data = event.data.as_object().cloned().unwrap_or_default();

        ProgressEvent {
            rollouts_completed: Self::coerce_i32(data.get("rollouts_completed"))
                .or_else(|| Self::coerce_i32(data.get("rollouts_executed")))
                .unwrap_or(0),
            rollouts_total: Self::coerce_i32(data.get("rollouts_total"))
                .or_else(|| Self::coerce_i32(data.get("total_rollouts"))),
            trials_completed: Self::coerce_i32(data.get("trials_completed")).unwrap_or(0),
            best_reward: Self::coerce_f64(data.get("best_score")),
            baseline_reward: Self::coerce_f64(data.get("baseline_score")),
        }
    }

    /// Parse generation event data from a ParsedEvent.
    pub fn parse_generation(event: &ParsedEvent) -> GenerationEvent {
        let data = event.data.as_object().cloned().unwrap_or_default();

        GenerationEvent {
            generation: Self::coerce_i32(data.get("generation")).unwrap_or(0),
            best_reward: Self::coerce_f64(data.get("best_accuracy")).unwrap_or(0.0),
            candidates_proposed: Self::coerce_i32(data.get("candidates_proposed")).unwrap_or(0),
            candidates_accepted: Self::coerce_i32(data.get("candidates_accepted")).unwrap_or(0),
        }
    }

    /// Parse complete event data from a ParsedEvent.
    pub fn parse_complete(event: &ParsedEvent) -> CompleteEvent {
        let data = event.data.as_object().cloned().unwrap_or_default();

        CompleteEvent {
            best_reward: Self::coerce_f64(data.get("best_score")),
            baseline_reward: Self::coerce_f64(data.get("baseline_score")),
            finish_reason: Self::coerce_string(data.get("finish_reason"))
                .or_else(|| Self::coerce_string(data.get("reason_terminated"))),
            total_candidates: Self::coerce_i32(data.get("total_candidates")).unwrap_or(0),
        }
    }

    /// Parse termination event data from a ParsedEvent.
    pub fn parse_termination(event: &ParsedEvent) -> TerminationEvent {
        let data = event.data.as_object().cloned().unwrap_or_default();
        TerminationEvent {
            reason: Self::coerce_string(data.get("reason"))
                .unwrap_or_else(|| "unknown".to_string()),
        }
    }

    /// Parse usage event data from a ParsedEvent.
    pub fn parse_usage(event: &ParsedEvent) -> UsageEvent {
        let data = event.data.as_object().cloned().unwrap_or_default();

        UsageEvent {
            total_usd: Self::coerce_f64(data.get("total_usd")).unwrap_or(0.0),
            tokens_usd: Self::coerce_f64(data.get("usd_tokens")).unwrap_or(0.0),
            sandbox_usd: Self::coerce_f64(data.get("sandbox_usd")).unwrap_or(0.0),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn test_event_category_terminal() {
        assert!(EventCategory::Complete.is_terminal());
        assert!(EventCategory::Termination.is_terminal());
        assert!(!EventCategory::Progress.is_terminal());
        assert!(!EventCategory::Candidate.is_terminal());
    }

    #[test]
    fn test_normalize_type() {
        assert_eq!(
            EventParser::normalize_type("learning.policy.[MASKED].candidate.evaluated"),
            "learning.policy.gepa.candidate.evaluated"
        );
    }

    #[test]
    fn test_get_category() {
        assert_eq!(
            EventParser::get_category("learning.policy.gepa.baseline"),
            EventCategory::Baseline
        );
        assert_eq!(
            EventParser::get_category("learning.policy.gepa.candidate.evaluated"),
            EventCategory::Candidate
        );
        assert_eq!(
            EventParser::get_category("learning.policy.gepa.frontier_updated"),
            EventCategory::Frontier
        );
        assert_eq!(
            EventParser::get_category("learning.policy.gepa.progress"),
            EventCategory::Progress
        );
        assert_eq!(
            EventParser::get_category("learning.policy.gepa.job.completed"),
            EventCategory::Complete
        );
        assert_eq!(
            EventParser::get_category("unknown.event.type"),
            EventCategory::Unknown
        );
        assert_eq!(
            EventParser::get_category("learning.policy.gepa.rollout.concurrency"),
            EventCategory::Throughput
        );
        assert_eq!(
            EventParser::get_category("learning.policy.gepa.candidate.new_best"),
            EventCategory::Candidate
        );
        assert_eq!(
            EventParser::get_category("learning.policy.gepa.job.started"),
            EventCategory::Progress
        );
        assert_eq!(
            EventParser::get_category("learning.policy.gepa.generation.started"),
            EventCategory::Generation
        );
    }

    #[test]
    fn test_parse_event() {
        let raw = json!({
            "type": "learning.policy.gepa.candidate.evaluated",
            "seq": 42,
            "data": {
                "candidate_id": "cand_123",
                "accuracy": 0.85,
                "accepted": true,
                "generation": 2
            }
        });

        let parsed = EventParser::parse(&raw);
        assert_eq!(parsed.category, EventCategory::Candidate);
        assert_eq!(parsed.seq, Some(42));

        let candidate = EventParser::parse_candidate(&parsed);
        assert_eq!(candidate.candidate_id, "cand_123");
        assert_eq!(candidate.reward, Some(0.85));
        assert!(candidate.accepted);
        assert_eq!(candidate.generation, Some(2));
    }

    #[test]
    fn test_parse_baseline() {
        let raw = json!({
            "type": "learning.policy.gepa.baseline",
            "data": {
                "accuracy": 0.72,
                "objectives": {"score": 0.72, "cost": 0.01}
            }
        });

        let parsed = EventParser::parse(&raw);
        let baseline = EventParser::parse_baseline(&parsed);
        assert_eq!(baseline.reward, Some(0.72));
        assert!(baseline.objectives.is_some());
    }

    #[test]
    fn test_parse_frontier() {
        let raw = json!({
            "type": "learning.policy.gepa.frontier_updated",
            "data": {
                "frontier": ["cand_1", "cand_2"],
                "added": ["cand_2"],
                "removed": [],
                "best_score": 0.88
            }
        });

        let parsed = EventParser::parse(&raw);
        let frontier = EventParser::parse_frontier(&parsed);
        assert_eq!(frontier.frontier.len(), 2);
        assert_eq!(frontier.best_reward, Some(0.88));
    }
}
