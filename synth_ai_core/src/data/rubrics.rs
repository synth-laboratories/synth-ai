//! Rubric and criterion types for evaluation.
//!
//! Rubrics define evaluation criteria with weights and descriptions.

use serde::{Deserialize, Serialize};
use std::collections::HashSet;

/// A single evaluation criterion.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Criterion {
    /// Unique identifier for this criterion.
    pub id: String,
    /// Human-readable description of what this criterion evaluates.
    pub description: String,
    /// Weight for aggregation (must be positive).
    #[serde(default = "default_weight")]
    pub weight: f64,
    /// Whether this criterion must be satisfied.
    #[serde(default)]
    pub required: bool,
    /// Optional scoring scale (e.g., 0-10, 0-100).
    #[serde(default)]
    pub scale_max: Option<f64>,
    /// Optional examples of good/bad responses.
    #[serde(default)]
    pub examples: Vec<CriterionExample>,
}

fn default_weight() -> f64 {
    1.0
}

/// Example for a criterion showing expected scores.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CriterionExample {
    /// The input/output being evaluated.
    pub content: String,
    /// Expected score for this example.
    pub expected_score: f64,
    /// Explanation of why this score is appropriate.
    #[serde(default)]
    pub explanation: Option<String>,
}

impl Criterion {
    /// Create a new criterion with the given ID and description.
    pub fn new(id: impl Into<String>, description: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            description: description.into(),
            weight: 1.0,
            required: false,
            scale_max: None,
            examples: Vec::new(),
        }
    }

    /// Set the weight for this criterion.
    pub fn with_weight(mut self, weight: f64) -> Self {
        self.weight = weight;
        self
    }

    /// Mark this criterion as required.
    pub fn required(mut self) -> Self {
        self.required = true;
        self
    }

    /// Validate this criterion's configuration.
    pub fn validate(&self) -> Result<(), String> {
        if self.id.is_empty() {
            return Err("Criterion ID cannot be empty".to_string());
        }
        if self.weight <= 0.0 {
            return Err(format!(
                "Criterion '{}' weight must be positive, got {}",
                self.id, self.weight
            ));
        }
        if let Some(scale_max) = self.scale_max {
            if scale_max <= 0.0 {
                return Err(format!(
                    "Criterion '{}' scale_max must be positive, got {}",
                    self.id, scale_max
                ));
            }
        }
        Ok(())
    }
}

/// A rubric containing multiple evaluation criteria.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Rubric {
    /// Version identifier for this rubric.
    pub version: String,
    /// High-level goal or purpose of the evaluation.
    #[serde(default)]
    pub goal_text: Option<String>,
    /// List of criteria to evaluate.
    #[serde(default)]
    pub criteria: Vec<Criterion>,
    /// How to aggregate criterion scores.
    #[serde(default = "default_aggregation")]
    pub aggregation: String,
    /// Optional metadata.
    #[serde(default)]
    pub metadata: std::collections::HashMap<String, serde_json::Value>,
}

fn default_aggregation() -> String {
    "weighted_sum".to_string()
}

impl Rubric {
    /// Create a new rubric with the given version.
    pub fn new(version: impl Into<String>) -> Self {
        Self {
            version: version.into(),
            goal_text: None,
            criteria: Vec::new(),
            aggregation: "weighted_sum".to_string(),
            metadata: std::collections::HashMap::new(),
        }
    }

    /// Set the goal text for this rubric.
    pub fn with_goal(mut self, goal: impl Into<String>) -> Self {
        self.goal_text = Some(goal.into());
        self
    }

    /// Add a criterion to this rubric.
    pub fn with_criterion(mut self, criterion: Criterion) -> Self {
        self.criteria.push(criterion);
        self
    }

    /// Set the aggregation method.
    pub fn with_aggregation(mut self, aggregation: impl Into<String>) -> Self {
        self.aggregation = aggregation.into();
        self
    }

    /// Validate this rubric's configuration.
    pub fn validate(&self) -> Result<(), String> {
        // Check aggregation method
        const VALID_AGGREGATIONS: &[&str] = &[
            "sum",
            "weighted_sum",
            "mean",
            "weighted_mean",
            "custom",
            "inherit",
        ];
        if !VALID_AGGREGATIONS.contains(&self.aggregation.as_str()) {
            return Err(format!(
                "Invalid aggregation '{}'. Valid options: {:?}",
                self.aggregation, VALID_AGGREGATIONS
            ));
        }

        // Check for duplicate criterion IDs
        let mut seen = HashSet::new();
        for criterion in &self.criteria {
            if !seen.insert(&criterion.id) {
                return Err(format!("Duplicate criterion ID: {}", criterion.id));
            }
            criterion.validate()?;
        }

        // Check that at least one criterion exists (unless inheriting)
        if self.criteria.is_empty() && self.aggregation != "inherit" {
            return Err("Rubric must have at least one criterion".to_string());
        }

        Ok(())
    }

    /// Get total weight of all criteria.
    pub fn total_weight(&self) -> f64 {
        self.criteria.iter().map(|c| c.weight).sum()
    }

    /// Get a criterion by ID.
    pub fn get_criterion(&self, id: &str) -> Option<&Criterion> {
        self.criteria.iter().find(|c| c.id == id)
    }
}

impl Default for Rubric {
    fn default() -> Self {
        Self::new("1.0")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_criterion_creation() {
        let criterion = Criterion::new("accuracy", "Response is factually correct")
            .with_weight(2.0)
            .required();

        assert_eq!(criterion.id, "accuracy");
        assert_eq!(criterion.weight, 2.0);
        assert!(criterion.required);
        assert!(criterion.validate().is_ok());
    }

    #[test]
    fn test_criterion_validation() {
        let invalid = Criterion::new("test", "desc").with_weight(-1.0);
        assert!(invalid.validate().is_err());
    }

    #[test]
    fn test_rubric_creation() {
        let rubric = Rubric::new("1.0")
            .with_goal("Evaluate response quality")
            .with_criterion(Criterion::new("clarity", "Response is clear"))
            .with_criterion(Criterion::new("accuracy", "Response is accurate"));

        assert_eq!(rubric.criteria.len(), 2);
        assert!(rubric.validate().is_ok());
    }

    #[test]
    fn test_rubric_duplicate_ids() {
        let rubric = Rubric::new("1.0")
            .with_criterion(Criterion::new("test", "First"))
            .with_criterion(Criterion::new("test", "Duplicate"));

        assert!(rubric.validate().is_err());
    }

    #[test]
    fn test_rubric_serde() {
        let rubric = Rubric::new("1.0").with_criterion(Criterion::new("test", "Test criterion"));

        let json = serde_json::to_string(&rubric).unwrap();
        let parsed: Rubric = serde_json::from_str(&json).unwrap();

        assert_eq!(parsed.version, rubric.version);
        assert_eq!(parsed.criteria.len(), 1);
    }
}
