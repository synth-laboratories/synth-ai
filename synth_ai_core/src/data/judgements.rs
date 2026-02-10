//! Judgement and rubric assignment types.
//!
//! Types for recording evaluation results and criterion scores.

use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::HashMap;

/// Score data for a single criterion.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CriterionScoreData {
    /// The numeric score.
    pub score: f64,
    /// Explanation/reasoning for the score.
    #[serde(default)]
    pub reason: Option<String>,
    /// Weight used in aggregation.
    #[serde(default = "default_weight")]
    pub weight: f64,
    /// Normalized score (0-1 range).
    #[serde(default)]
    pub normalized_score: Option<f64>,
    /// Whether this criterion passed (for required criteria).
    #[serde(default)]
    pub passed: Option<bool>,
}

fn default_weight() -> f64 {
    1.0
}

impl CriterionScoreData {
    /// Create a new criterion score.
    pub fn new(score: f64) -> Self {
        Self {
            score,
            reason: None,
            weight: 1.0,
            normalized_score: None,
            passed: None,
        }
    }

    /// Create a score with reason.
    pub fn with_reason(mut self, reason: impl Into<String>) -> Self {
        self.reason = Some(reason.into());
        self
    }

    /// Set the weight.
    pub fn with_weight(mut self, weight: f64) -> Self {
        self.weight = weight;
        self
    }

    /// Mark as passed/failed.
    pub fn with_passed(mut self, passed: bool) -> Self {
        self.passed = Some(passed);
        self
    }

    /// Calculate weighted score.
    pub fn weighted_score(&self) -> f64 {
        self.score * self.weight
    }
}

impl Default for CriterionScoreData {
    fn default() -> Self {
        Self::new(0.0)
    }
}

/// Assignment of scores to a rubric's criteria.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct RubricAssignment {
    /// Map of criterion ID to score data.
    #[serde(default)]
    pub criterion_scores: HashMap<String, CriterionScoreData>,
    /// Aggregated total score.
    #[serde(default)]
    pub total: f64,
    /// Reference to the rubric used.
    #[serde(default)]
    pub rubric_ref: Option<String>,
    /// Summary of the evaluation.
    #[serde(default)]
    pub summary: Option<String>,
    /// Whether all required criteria passed.
    #[serde(default)]
    pub all_required_passed: Option<bool>,
    /// Normalized total (0-1 range).
    #[serde(default)]
    pub normalized_total: Option<f64>,
}

impl RubricAssignment {
    /// Create a new rubric assignment.
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a criterion score.
    pub fn with_score(
        mut self,
        criterion_id: impl Into<String>,
        score: CriterionScoreData,
    ) -> Self {
        self.criterion_scores.insert(criterion_id.into(), score);
        self
    }

    /// Set the total score.
    pub fn with_total(mut self, total: f64) -> Self {
        self.total = total;
        self
    }

    /// Set the rubric reference.
    pub fn with_rubric_ref(mut self, rubric_ref: impl Into<String>) -> Self {
        self.rubric_ref = Some(rubric_ref.into());
        self
    }

    /// Set the summary.
    pub fn with_summary(mut self, summary: impl Into<String>) -> Self {
        self.summary = Some(summary.into());
        self
    }

    /// Calculate total from criterion scores using weighted sum.
    pub fn calculate_weighted_total(&mut self) {
        let total_weight: f64 = self.criterion_scores.values().map(|s| s.weight).sum();
        if total_weight > 0.0 {
            let weighted_sum: f64 = self
                .criterion_scores
                .values()
                .map(|s| s.weighted_score())
                .sum();
            self.total = weighted_sum / total_weight;
        }
    }

    /// Get score for a criterion.
    pub fn get_score(&self, criterion_id: &str) -> Option<f64> {
        self.criterion_scores.get(criterion_id).map(|s| s.score)
    }
}

/// A complete judgement including rubric assignment and annotations.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Judgement {
    /// The rubric-based evaluation.
    #[serde(default)]
    pub rubric_assignment: Option<RubricAssignment>,
    /// Free-form annotations.
    #[serde(default)]
    pub annotation: HashMap<String, Value>,
    /// Overall pass/fail determination.
    #[serde(default)]
    pub passed: Option<bool>,
    /// Confidence in the judgement (0-1).
    #[serde(default)]
    pub confidence: Option<f64>,
    /// Source of the judgement (e.g., "verifier", "human", "model").
    #[serde(default)]
    pub source: Option<String>,
    /// Timestamp of when judgement was made.
    #[serde(default)]
    pub judged_at: Option<String>,
}

impl Judgement {
    /// Create a new judgement.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the rubric assignment.
    pub fn with_rubric_assignment(mut self, assignment: RubricAssignment) -> Self {
        self.rubric_assignment = Some(assignment);
        self
    }

    /// Add an annotation.
    pub fn with_annotation(mut self, key: impl Into<String>, value: Value) -> Self {
        self.annotation.insert(key.into(), value);
        self
    }

    /// Set passed status.
    pub fn with_passed(mut self, passed: bool) -> Self {
        self.passed = Some(passed);
        self
    }

    /// Set confidence.
    pub fn with_confidence(mut self, confidence: f64) -> Self {
        self.confidence = Some(confidence.clamp(0.0, 1.0));
        self
    }

    /// Set source.
    pub fn with_source(mut self, source: impl Into<String>) -> Self {
        self.source = Some(source.into());
        self
    }

    /// Get the total score from the rubric assignment.
    pub fn total_score(&self) -> Option<f64> {
        self.rubric_assignment.as_ref().map(|a| a.total)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_criterion_score() {
        let score = CriterionScoreData::new(8.5)
            .with_reason("Good explanation")
            .with_weight(2.0);

        assert_eq!(score.score, 8.5);
        assert_eq!(score.weighted_score(), 17.0);
    }

    #[test]
    fn test_rubric_assignment() {
        let mut assignment = RubricAssignment::new()
            .with_score("clarity", CriterionScoreData::new(9.0).with_weight(1.0))
            .with_score("accuracy", CriterionScoreData::new(7.0).with_weight(2.0))
            .with_rubric_ref("eval_v1");

        assignment.calculate_weighted_total();

        // Weighted average: (9*1 + 7*2) / (1+2) = 23/3 ≈ 7.67
        assert!((assignment.total - 7.666).abs() < 0.01);
    }

    #[test]
    fn test_judgement() {
        let assignment = RubricAssignment::new()
            .with_total(8.5)
            .with_summary("Good overall performance");

        let judgement = Judgement::new()
            .with_rubric_assignment(assignment)
            .with_passed(true)
            .with_confidence(0.95)
            .with_source("verifier");

        assert_eq!(judgement.total_score(), Some(8.5));
        assert_eq!(judgement.passed, Some(true));
        assert_eq!(judgement.confidence, Some(0.95));
    }

    #[test]
    fn test_serde() {
        let judgement = Judgement::new()
            .with_passed(true)
            .with_annotation("note", serde_json::json!("test"));

        let json = serde_json::to_string(&judgement).unwrap();
        let parsed: Judgement = serde_json::from_str(&json).unwrap();

        assert_eq!(parsed.passed, Some(true));
        assert_eq!(
            parsed.annotation.get("note"),
            Some(&serde_json::json!("test"))
        );
    }
}
