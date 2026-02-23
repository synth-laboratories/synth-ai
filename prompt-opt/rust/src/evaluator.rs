use crate::dataset::{Dataset, Example};
use crate::models::{Policy, require_finite_score};
use crate::{LlmClient, MiproError, Result};
use async_trait::async_trait;

#[async_trait]
pub trait EvalMetric: Send + Sync {
    async fn score(&self, example: &Example, output: &str) -> Result<f32>;
    fn name(&self) -> &'static str;
}

#[derive(Debug, Default)]
pub struct ExactMatchMetric;

#[async_trait]
impl EvalMetric for ExactMatchMetric {
    #[tracing::instrument(skip_all)]
    async fn score(&self, example: &Example, output: &str) -> Result<f32> {
        let expected = match &example.expected {
            serde_json::Value::String(s) => s.as_str(),
            _ => {
                return Err(MiproError::InvalidArgument(
                    "expected must be a string for ExactMatchMetric".to_string(),
                ));
            }
        };
        Ok(if output.trim() == expected.trim() {
            1.0
        } else {
            0.0
        })
    }

    fn name(&self) -> &'static str {
        "exact_match"
    }
}

#[derive(Debug, Clone)]
pub struct EvalSummary {
    pub mean_score: f32,
    pub per_example: Vec<f32>,
}

pub struct Evaluator {
    llm: std::sync::Arc<dyn LlmClient>,
    metric: std::sync::Arc<dyn EvalMetric>,
}

impl Evaluator {
    #[tracing::instrument(skip_all)]
    pub fn new(llm: std::sync::Arc<dyn LlmClient>, metric: std::sync::Arc<dyn EvalMetric>) -> Self {
        Self { llm, metric }
    }

    #[tracing::instrument(skip_all)]
    pub fn llm(&self) -> &std::sync::Arc<dyn LlmClient> {
        &self.llm
    }

    #[tracing::instrument(skip_all)]
    pub fn metric(&self) -> &std::sync::Arc<dyn EvalMetric> {
        &self.metric
    }

    #[tracing::instrument(skip_all)]
    pub async fn evaluate_policy(&self, policy: &Policy, dataset: &Dataset) -> Result<EvalSummary> {
        if dataset.examples.is_empty() {
            return Err(MiproError::InvalidArgument(
                "dataset has no examples".to_string(),
            ));
        }

        let mut scores = Vec::with_capacity(dataset.examples.len());
        for ex in &dataset.examples {
            let prompt = policy.render(&ex.input)?;
            let output = self.llm.complete(&prompt).await?;
            let s = self.metric.score(ex, &output).await?;
            scores.push(require_finite_score(s)?);
        }

        let mean = scores.iter().sum::<f32>() / (scores.len() as f32);
        Ok(EvalSummary {
            mean_score: require_finite_score(mean)?,
            per_example: scores,
        })
    }
}
