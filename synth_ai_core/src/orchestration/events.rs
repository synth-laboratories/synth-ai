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

impl EventCategory {
    /// Check if this is a terminal event category.
    pub fn is_terminal(&self) -> bool {
        matches!(self, EventCategory::Complete | EventCategory::Termination)
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
    /// Baseline accuracy/score
    #[serde(default)]
    pub accuracy: Option<f64>,
    /// Objective scores
    #[serde(default)]
    pub objectives: Option<HashMap<String, f64>>,
    /// Per-instance scores
    #[serde(default)]
    pub instance_scores: Option<Vec<f64>>,
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
    /// Candidate accuracy/score
    #[serde(default)]
    pub accuracy: Option<f64>,
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
    /// Per-instance scores
    #[serde(default)]
    pub instance_scores: Option<Vec<f64>>,
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
    /// Best score on frontier
    #[serde(default)]
    pub best_score: Option<f64>,
    /// Scores by candidate ID
    #[serde(default)]
    pub frontier_scores: Option<HashMap<String, f64>>,
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
    /// Current best score
    #[serde(default)]
    pub best_score: Option<f64>,
    /// Baseline score for comparison
    #[serde(default)]
    pub baseline_score: Option<f64>,
}

/// Generation complete event data.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct GenerationEvent {
    /// Generation number
    #[serde(default)]
    pub generation: i32,
    /// Best accuracy in this generation
    #[serde(default)]
    pub best_accuracy: f64,
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
    /// Final best score
    #[serde(default)]
    pub best_score: Option<f64>,
    /// Baseline score
    #[serde(default)]
    pub baseline_score: Option<f64>,
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
        ".proposal.scored",
        ".optimized.scored",
        ".candidate_scored",
    ];

    /// Patterns for frontier events
    const FRONTIER_PATTERNS: &'static [&'static str] = &[".frontier_updated", ".frontier.updated"];

    /// Patterns for progress events
    const PROGRESS_PATTERNS: &'static [&'static str] =
        &[".progress", ".rollouts_limit_progress", ".rollouts.progress"];

    /// Patterns for generation events
    const GENERATION_PATTERNS: &'static [&'static str] =
        &[".generation.complete", ".generation.completed"];

    /// Patterns for throughput events
    const THROUGHPUT_PATTERNS: &'static [&'static str] = &[".throughput"];

    /// Patterns for termination events
    const TERMINATION_PATTERNS: &'static [&'static str] =
        &[".termination.triggered", ".termination"];

    /// Patterns for complete events
    const COMPLETE_PATTERNS: &'static [&'static str] = &[".complete", ".completed", ".job.completed"];

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

    /// Get the category for an event type string.
    pub fn get_category(event_type: &str) -> EventCategory {
        let normalized = Self::normalize_type(event_type);
        let lower = normalized.to_lowercase();

        // Check patterns in order of specificity
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

        for pattern in Self::GENERATION_PATTERNS {
            if lower.contains(pattern) {
                return EventCategory::Generation;
            }
        }

        for pattern in Self::COMPLETE_PATTERNS {
            if lower.contains(pattern) {
                return EventCategory::Complete;
            }
        }

        for pattern in Self::TERMINATION_PATTERNS {
            if lower.contains(pattern) {
                return EventCategory::Termination;
            }
        }

        for pattern in Self::VALIDATION_PATTERNS {
            if lower.contains(pattern) {
                return EventCategory::Validation;
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

        for pattern in Self::USAGE_PATTERNS {
            if lower.contains(pattern) {
                return EventCategory::Usage;
            }
        }

        EventCategory::Unknown
    }

    /// Parse a raw event JSON into a ParsedEvent.
    pub fn parse(event: &Value) -> ParsedEvent {
        let event_type = event
            .get("type")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string();

        let category = Self::get_category(&event_type);

        let seq = event.get("seq").and_then(|v| v.as_i64());

        let timestamp_ms = event
            .get("timestamp_ms")
            .and_then(|v| v.as_i64())
            .or_else(|| {
                // Try parsing ISO timestamp
                event.get("ts").and_then(|v| v.as_str()).and_then(|ts| {
                    chrono::DateTime::parse_from_rfc3339(ts)
                        .ok()
                        .map(|dt| dt.timestamp_millis())
                })
            });

        let data = event.get("data").cloned().unwrap_or(Value::Null);

        ParsedEvent {
            event_type,
            category,
            data,
            seq,
            timestamp_ms,
        }
    }

    /// Parse baseline event data from a ParsedEvent.
    pub fn parse_baseline(event: &ParsedEvent) -> BaselineEvent {
        serde_json::from_value(event.data.clone()).unwrap_or_default()
    }

    /// Parse candidate event data from a ParsedEvent.
    pub fn parse_candidate(event: &ParsedEvent) -> CandidateEvent {
        serde_json::from_value(event.data.clone()).unwrap_or_default()
    }

    /// Parse frontier event data from a ParsedEvent.
    pub fn parse_frontier(event: &ParsedEvent) -> FrontierEvent {
        serde_json::from_value(event.data.clone()).unwrap_or_default()
    }

    /// Parse progress event data from a ParsedEvent.
    pub fn parse_progress(event: &ParsedEvent) -> ProgressEvent {
        serde_json::from_value(event.data.clone()).unwrap_or_default()
    }

    /// Parse generation event data from a ParsedEvent.
    pub fn parse_generation(event: &ParsedEvent) -> GenerationEvent {
        serde_json::from_value(event.data.clone()).unwrap_or_default()
    }

    /// Parse complete event data from a ParsedEvent.
    pub fn parse_complete(event: &ParsedEvent) -> CompleteEvent {
        serde_json::from_value(event.data.clone()).unwrap_or_default()
    }

    /// Parse termination event data from a ParsedEvent.
    pub fn parse_termination(event: &ParsedEvent) -> TerminationEvent {
        serde_json::from_value(event.data.clone()).unwrap_or_default()
    }

    /// Parse usage event data from a ParsedEvent.
    pub fn parse_usage(event: &ParsedEvent) -> UsageEvent {
        serde_json::from_value(event.data.clone()).unwrap_or_default()
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
        assert_eq!(candidate.accuracy, Some(0.85));
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
        assert_eq!(baseline.accuracy, Some(0.72));
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
        assert_eq!(frontier.best_score, Some(0.88));
    }
}
