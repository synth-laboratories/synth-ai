use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use ulid::Ulid;

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct SignalWeight(pub f32);

impl SignalWeight {
    #[tracing::instrument]
    pub fn new(weight: f32) -> Result<Self, crate::RlmError> {
        if !weight.is_finite() || weight < 0.0 {
            return Err(crate::RlmError::InvalidArgument(
                "signal weight must be finite and >= 0".to_string(),
            ));
        }
        Ok(Self(weight))
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VerifierConfig {
    /// Pass/fail threshold on the aggregated score in [0,1].
    pub pass_threshold: f32,
}

impl Default for VerifierConfig {
    fn default() -> Self {
        Self {
            pass_threshold: 0.8,
        }
    }
}

impl VerifierConfig {
    #[tracing::instrument]
    pub fn validate(&self) -> crate::Result<()> {
        if !self.pass_threshold.is_finite() || !(0.0..=1.0).contains(&self.pass_threshold) {
            return Err(crate::RlmError::InvalidConfig(
                "pass_threshold must be finite and in [0,1]".to_string(),
            ));
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SignalKind {
    /// Score 1.0 if output exactly matches expected (string), else 0.0.
    ExactMatch,
    /// Score 1.0 if output contains the given substring, else 0.0.
    Contains { needle: String },
    /// Ask an LLM to grade the output against a rubric, producing a score in [0,1].
    LlmRubric { rubric: String },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RewardSignal {
    pub name: String,
    pub weight: SignalWeight,
    pub kind: SignalKind,
    #[serde(default)]
    pub description: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VerificationCase {
    pub id: Ulid,
    pub input: String,
    pub output: String,
    pub expected: Option<String>,
    #[serde(default)]
    pub metadata: serde_json::Value,
}

impl VerificationCase {
    #[tracing::instrument(skip_all)]
    pub fn new(
        input: impl Into<String>,
        output: impl Into<String>,
        expected: Option<String>,
    ) -> Self {
        Self {
            id: Ulid::new(),
            input: input.into(),
            output: output.into(),
            expected,
            metadata: serde_json::json!({}),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SignalScore {
    pub name: String,
    pub weight: SignalWeight,
    pub score_0_to_1: f32,
    #[serde(default)]
    pub reasoning: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RewardOutcome {
    pub total_score_0_to_1: f32,
    pub passed: bool,
    pub signal_scores: Vec<SignalScore>,
    #[serde(default)]
    pub reasoning: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvalReport {
    pub id: Ulid,
    pub created_at: DateTime<Utc>,
    pub config: VerifierConfig,
    pub case: VerificationCase,
    pub outcome: RewardOutcome,
}
