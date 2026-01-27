//! Progress tracking for optimization jobs.
//!
//! This module provides progress tracking and aggregation for GEPA/MIPRO
//! optimization jobs.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::Instant;

use super::events::{EventCategory, EventParser, ParsedEvent};

/// Token usage statistics.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct TokenUsage {
    /// Input/prompt tokens
    pub prompt_tokens: i64,
    /// Output/completion tokens
    pub completion_tokens: i64,
    /// Total tokens
    pub total_tokens: i64,
    /// Reasoning tokens (for o1-style models)
    #[serde(default)]
    pub reasoning_tokens: i64,
    /// Cached tokens
    #[serde(default)]
    pub cached_tokens: i64,
}

impl TokenUsage {
    /// Create from prompt and completion counts.
    pub fn new(prompt: i64, completion: i64) -> Self {
        Self {
            prompt_tokens: prompt,
            completion_tokens: completion,
            total_tokens: prompt + completion,
            reasoning_tokens: 0,
            cached_tokens: 0,
        }
    }
}

/// Information about a single candidate.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CandidateInfo {
    /// Unique candidate ID
    pub candidate_id: String,
    /// Accuracy/score on training set
    #[serde(default)]
    pub accuracy: Option<f64>,
    /// Multi-objective scores
    #[serde(default)]
    pub objectives: Option<HashMap<String, f64>>,
    /// Validation accuracy (if validation phase completed)
    #[serde(default)]
    pub val_accuracy: Option<f64>,
    /// Training accuracy (alias for accuracy)
    #[serde(default)]
    pub train_accuracy: Option<f64>,
    /// Generation number
    #[serde(default)]
    pub generation: Option<i32>,
    /// Parent candidate ID (for mutations)
    #[serde(default)]
    pub parent_id: Option<String>,
    /// Whether on Pareto frontier
    #[serde(default)]
    pub is_pareto: bool,
    /// Whether accepted into population
    #[serde(default)]
    pub accepted: bool,
    /// Type of mutation used
    #[serde(default)]
    pub mutation_type: Option<String>,
    /// Token usage for this candidate
    #[serde(default)]
    pub token_usage: Option<TokenUsage>,
    /// Cost in USD
    #[serde(default)]
    pub cost_usd: Option<f64>,
    /// Unix timestamp when evaluated
    #[serde(default)]
    pub timestamp: f64,
    /// Timestamp in milliseconds
    #[serde(default)]
    pub timestamp_ms: Option<i64>,
}

impl Default for CandidateInfo {
    fn default() -> Self {
        Self {
            candidate_id: String::new(),
            accuracy: None,
            objectives: None,
            val_accuracy: None,
            train_accuracy: None,
            generation: None,
            parent_id: None,
            is_pareto: false,
            accepted: false,
            mutation_type: None,
            token_usage: None,
            cost_usd: None,
            timestamp: 0.0,
            timestamp_ms: None,
        }
    }
}

/// Baseline information.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct BaselineInfo {
    /// Baseline accuracy/score
    pub accuracy: Option<f64>,
    /// Multi-objective scores
    #[serde(default)]
    pub objectives: Option<HashMap<String, f64>>,
    /// Validation accuracy
    #[serde(default)]
    pub val_accuracy: Option<f64>,
    /// Per-instance scores
    #[serde(default)]
    pub instance_scores: Vec<f64>,
}

/// Frontier update record.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct FrontierUpdate {
    /// Update timestamp
    pub timestamp: f64,
    /// Candidates added
    #[serde(default)]
    pub added: Vec<String>,
    /// Candidates removed
    #[serde(default)]
    pub removed: Vec<String>,
    /// Current frontier
    #[serde(default)]
    pub frontier: Vec<String>,
    /// Scores by candidate
    #[serde(default)]
    pub frontier_scores: HashMap<String, f64>,
    /// Frontier size
    #[serde(default)]
    pub frontier_size: i32,
    /// Best optimistic score
    #[serde(default)]
    pub optimistic_score: Option<f64>,
    /// Generation number
    #[serde(default)]
    pub generation: Option<i32>,
}

/// Overall GEPA progress state.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GEPAProgress {
    /// Current phase: "init", "optimization", "validation", "complete", "failed"
    pub phase: String,
    /// Rollouts completed
    pub rollouts_completed: i32,
    /// Total rollouts planned
    pub rollouts_total: i32,
    /// Generations completed
    pub generations_completed: i32,
    /// Candidates evaluated
    pub candidates_evaluated: i32,
    /// Current best score
    pub best_score: f64,
    /// Baseline score for lift calculation
    pub baseline_score: Option<f64>,
    /// Elapsed time in seconds
    pub elapsed_seconds: f64,
    /// Estimated time remaining
    pub eta_seconds: Option<f64>,
    /// Finish reason if complete
    pub finish_reason: Option<String>,
}

impl Default for GEPAProgress {
    fn default() -> Self {
        Self {
            phase: "init".to_string(),
            rollouts_completed: 0,
            rollouts_total: 0,
            generations_completed: 0,
            candidates_evaluated: 0,
            best_score: 0.0,
            baseline_score: None,
            elapsed_seconds: 0.0,
            eta_seconds: None,
            finish_reason: None,
        }
    }
}

impl GEPAProgress {
    /// Calculate progress percentage.
    pub fn progress_pct(&self) -> f64 {
        if self.rollouts_total > 0 {
            (self.rollouts_completed as f64 / self.rollouts_total as f64) * 100.0
        } else {
            0.0
        }
    }

    /// Calculate lift over baseline.
    pub fn lift(&self) -> Option<f64> {
        self.baseline_score.map(|b| {
            if b > 0.0 {
                (self.best_score - b) / b
            } else {
                0.0
            }
        })
    }
}

/// Progress tracker that aggregates events into state.
pub struct ProgressTracker {
    /// Overall progress
    pub progress: GEPAProgress,
    /// All evaluated candidates
    pub candidates: Vec<CandidateInfo>,
    /// Candidates indexed by ID
    candidates_by_id: HashMap<String, usize>,
    /// Baseline information
    pub baseline: Option<BaselineInfo>,
    /// Current Pareto frontier
    pub frontier: Vec<String>,
    /// Frontier update history
    pub frontier_history: Vec<FrontierUpdate>,
    /// Generation history
    pub generation_history: Vec<GenerationInfo>,
    /// Start time for elapsed calculation
    start_time: Option<Instant>,
    /// Last event sequence number
    pub last_seq: i64,
}

/// Generation summary info.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct GenerationInfo {
    /// Generation number
    pub generation: i32,
    /// Best accuracy in generation
    pub best_accuracy: f64,
    /// Candidates proposed
    pub candidates_proposed: i32,
    /// Candidates accepted
    pub candidates_accepted: i32,
}

impl Default for ProgressTracker {
    fn default() -> Self {
        Self::new()
    }
}

impl ProgressTracker {
    /// Create a new progress tracker.
    pub fn new() -> Self {
        Self {
            progress: GEPAProgress::default(),
            candidates: Vec::new(),
            candidates_by_id: HashMap::new(),
            baseline: None,
            frontier: Vec::new(),
            frontier_history: Vec::new(),
            generation_history: Vec::new(),
            start_time: None,
            last_seq: -1,
        }
    }

    /// Get the current best score.
    pub fn best_score(&self) -> f64 {
        self.progress.best_score
    }

    /// Get baseline score.
    pub fn baseline_score(&self) -> Option<f64> {
        self.progress.baseline_score
    }

    /// Get lift over baseline.
    pub fn lift(&self) -> Option<f64> {
        self.progress.lift()
    }

    /// Get current frontier candidates.
    pub fn current_frontier(&self) -> &[String] {
        &self.frontier
    }

    /// Update tracker with an event.
    pub fn update(&mut self, event: &ParsedEvent) {
        // Start timer on first event
        if self.start_time.is_none() {
            self.start_time = Some(Instant::now());
        }

        // Update elapsed time
        if let Some(start) = self.start_time {
            self.progress.elapsed_seconds = start.elapsed().as_secs_f64();
        }

        // Track sequence
        if let Some(seq) = event.seq {
            if seq > self.last_seq {
                self.last_seq = seq;
            }
        }

        // Handle by category
        match event.category {
            EventCategory::Baseline => self.handle_baseline(event),
            EventCategory::Candidate => self.handle_candidate(event),
            EventCategory::Frontier => self.handle_frontier(event),
            EventCategory::Progress => self.handle_progress(event),
            EventCategory::Generation => self.handle_generation(event),
            EventCategory::Complete => self.handle_complete(event),
            EventCategory::Termination => self.handle_termination(event),
            EventCategory::Validation => self.handle_validation(event),
            _ => {}
        }
    }

    fn handle_baseline(&mut self, event: &ParsedEvent) {
        let data = EventParser::parse_baseline(event);

        self.baseline = Some(BaselineInfo {
            accuracy: data.accuracy,
            objectives: data.objectives,
            val_accuracy: None,
            instance_scores: data.instance_scores.unwrap_or_default(),
        });

        if let Some(acc) = data.accuracy {
            self.progress.baseline_score = Some(acc);
            // Initialize best score to baseline
            if self.progress.best_score == 0.0 {
                self.progress.best_score = acc;
            }
        }

        self.progress.phase = "optimization".to_string();
    }

    fn handle_candidate(&mut self, event: &ParsedEvent) {
        let data = EventParser::parse_candidate(event);

        let candidate = CandidateInfo {
            candidate_id: data.candidate_id.clone(),
            accuracy: data.accuracy,
            objectives: data.objectives,
            val_accuracy: None,
            train_accuracy: data.accuracy,
            generation: data.generation,
            parent_id: data.parent_id,
            is_pareto: data.is_pareto,
            accepted: data.accepted,
            mutation_type: data.mutation_type,
            token_usage: None,
            cost_usd: None,
            timestamp: self.progress.elapsed_seconds,
            timestamp_ms: event.timestamp_ms,
        };

        // Update best score
        if let Some(acc) = data.accuracy {
            if acc > self.progress.best_score {
                self.progress.best_score = acc;
            }
        }

        // Store candidate
        let idx = self.candidates.len();
        self.candidates.push(candidate);
        self.candidates_by_id.insert(data.candidate_id, idx);
        self.progress.candidates_evaluated += 1;
    }

    fn handle_frontier(&mut self, event: &ParsedEvent) {
        let data = EventParser::parse_frontier(event);

        self.frontier = data.frontier.clone();

        if let Some(best) = data.best_score {
            if best > self.progress.best_score {
                self.progress.best_score = best;
            }
        }

        let update = FrontierUpdate {
            timestamp: self.progress.elapsed_seconds,
            added: data.added,
            removed: data.removed,
            frontier: data.frontier,
            frontier_scores: data.frontier_scores.unwrap_or_default(),
            frontier_size: data.frontier_size,
            optimistic_score: data.best_score,
            generation: None,
        };
        self.frontier_history.push(update);
    }

    fn handle_progress(&mut self, event: &ParsedEvent) {
        let data = EventParser::parse_progress(event);

        self.progress.rollouts_completed = data.rollouts_completed;
        if let Some(total) = data.rollouts_total {
            self.progress.rollouts_total = total;
        }

        if let Some(best) = data.best_score {
            if best > self.progress.best_score {
                self.progress.best_score = best;
            }
        }

        if let Some(baseline) = data.baseline_score {
            if self.progress.baseline_score.is_none() {
                self.progress.baseline_score = Some(baseline);
            }
        }

        // Estimate ETA
        if self.progress.rollouts_total > 0 && self.progress.rollouts_completed > 0 {
            let remaining = self.progress.rollouts_total - self.progress.rollouts_completed;
            let rate = self.progress.elapsed_seconds / self.progress.rollouts_completed as f64;
            self.progress.eta_seconds = Some(remaining as f64 * rate);
        }
    }

    fn handle_generation(&mut self, event: &ParsedEvent) {
        let data = EventParser::parse_generation(event);

        self.progress.generations_completed = data.generation;

        let info = GenerationInfo {
            generation: data.generation,
            best_accuracy: data.best_accuracy,
            candidates_proposed: data.candidates_proposed,
            candidates_accepted: data.candidates_accepted,
        };
        self.generation_history.push(info);
    }

    fn handle_complete(&mut self, event: &ParsedEvent) {
        let data = EventParser::parse_complete(event);

        self.progress.phase = "complete".to_string();
        self.progress.finish_reason = data.finish_reason;

        if let Some(best) = data.best_score {
            self.progress.best_score = best;
        }

        if let Some(baseline) = data.baseline_score {
            self.progress.baseline_score = Some(baseline);
        }
    }

    fn handle_termination(&mut self, event: &ParsedEvent) {
        let data = EventParser::parse_termination(event);

        self.progress.phase = "complete".to_string();
        self.progress.finish_reason = Some(data.reason);
    }

    fn handle_validation(&mut self, event: &ParsedEvent) {
        self.progress.phase = "validation".to_string();

        // Update candidate validation scores if provided
        if let Some(candidate_id) = event.data.get("candidate_id").and_then(|v| v.as_str()) {
            if let Some(val_score) = event.data.get("val_accuracy").and_then(|v| v.as_f64()) {
                if let Some(&idx) = self.candidates_by_id.get(candidate_id) {
                    self.candidates[idx].val_accuracy = Some(val_score);
                }
            }
        }
    }

    /// Get a summary dict for serialization.
    pub fn to_summary(&self) -> serde_json::Value {
        serde_json::json!({
            "phase": self.progress.phase,
            "rollouts_completed": self.progress.rollouts_completed,
            "rollouts_total": self.progress.rollouts_total,
            "candidates_evaluated": self.progress.candidates_evaluated,
            "generations_completed": self.progress.generations_completed,
            "best_score": self.progress.best_score,
            "baseline_score": self.progress.baseline_score,
            "lift": self.lift(),
            "elapsed_seconds": self.progress.elapsed_seconds,
            "frontier_size": self.frontier.len(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn test_progress_default() {
        let progress = GEPAProgress::default();
        assert_eq!(progress.phase, "init");
        assert_eq!(progress.progress_pct(), 0.0);
        assert!(progress.lift().is_none());
    }

    #[test]
    fn test_progress_lift() {
        let mut progress = GEPAProgress::default();
        progress.baseline_score = Some(0.5);
        progress.best_score = 0.75;

        let lift = progress.lift().unwrap();
        assert!((lift - 0.5).abs() < 0.001); // 50% lift
    }

    #[test]
    fn test_tracker_baseline() {
        let mut tracker = ProgressTracker::new();

        let event = EventParser::parse(&json!({
            "type": "learning.policy.gepa.baseline",
            "seq": 1,
            "data": { "accuracy": 0.72 }
        }));

        tracker.update(&event);

        assert!(tracker.baseline.is_some());
        assert_eq!(tracker.baseline_score(), Some(0.72));
        assert_eq!(tracker.progress.phase, "optimization");
    }

    #[test]
    fn test_tracker_candidate() {
        let mut tracker = ProgressTracker::new();

        // First baseline
        tracker.update(&EventParser::parse(&json!({
            "type": "learning.policy.gepa.baseline",
            "data": { "accuracy": 0.72 }
        })));

        // Then candidate
        tracker.update(&EventParser::parse(&json!({
            "type": "learning.policy.gepa.candidate.evaluated",
            "seq": 2,
            "data": {
                "candidate_id": "cand_1",
                "accuracy": 0.85,
                "accepted": true,
                "generation": 1
            }
        })));

        assert_eq!(tracker.candidates.len(), 1);
        assert_eq!(tracker.best_score(), 0.85);
        assert_eq!(tracker.progress.candidates_evaluated, 1);
    }

    #[test]
    fn test_tracker_frontier() {
        let mut tracker = ProgressTracker::new();

        tracker.update(&EventParser::parse(&json!({
            "type": "learning.policy.gepa.frontier_updated",
            "data": {
                "frontier": ["cand_1", "cand_2"],
                "best_score": 0.88
            }
        })));

        assert_eq!(tracker.frontier.len(), 2);
        assert_eq!(tracker.frontier_history.len(), 1);
        assert_eq!(tracker.best_score(), 0.88);
    }

    #[test]
    fn test_tracker_complete() {
        let mut tracker = ProgressTracker::new();

        tracker.update(&EventParser::parse(&json!({
            "type": "learning.policy.gepa.job.completed",
            "data": {
                "best_score": 0.92,
                "baseline_score": 0.72,
                "finish_reason": "budget_exhausted"
            }
        })));

        assert_eq!(tracker.progress.phase, "complete");
        assert_eq!(tracker.progress.finish_reason, Some("budget_exhausted".to_string()));
        assert_eq!(tracker.best_score(), 0.92);
    }
}
