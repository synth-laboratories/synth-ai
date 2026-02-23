use crate::dataset::Dataset;
use crate::evaluator::{EvalSummary, Evaluator};
use crate::models::{MiproConfig, OptimizationHistoryEntry, OptimizationResult, Policy};
use crate::sampler::VariantSampler;
use crate::{MiproError, Result};
use std::sync::Arc;

pub struct Optimizer {
    sampler: Arc<dyn VariantSampler>,
    evaluator: Evaluator,
}

impl Optimizer {
    #[tracing::instrument(skip_all)]
    pub fn new(sampler: Arc<dyn VariantSampler>, evaluator: Evaluator) -> Self {
        Self { sampler, evaluator }
    }

    #[tracing::instrument(skip_all)]
    pub fn sampler(&self) -> &Arc<dyn VariantSampler> {
        &self.sampler
    }

    #[tracing::instrument(skip_all)]
    pub fn evaluator(&self) -> &Evaluator {
        &self.evaluator
    }

    #[tracing::instrument(skip_all)]
    pub async fn evaluate(&self, policy: &Policy, dataset: &Dataset) -> Result<EvalSummary> {
        self.evaluator.evaluate_policy(policy, dataset).await
    }

    #[tracing::instrument(skip_all)]
    pub async fn run_batch(
        &self,
        cfg: MiproConfig,
        initial_policy: Policy,
        dataset: Dataset,
    ) -> Result<OptimizationResult> {
        cfg.validate()?;
        let (train, holdout) = dataset.split_train_holdout(cfg.holdout_ratio, cfg.seed)?;
        let _ = train;

        let mut result = OptimizationResult::new(initial_policy.clone());

        let baseline = self.evaluate(&initial_policy, &holdout).await?;
        let mut best_policy = initial_policy;
        let mut best_score = baseline.mean_score;
        let mut no_improve_rounds = 0usize;

        result.best_score = best_score;
        result.best_policy = best_policy.clone();

        for iter in 0..cfg.max_iterations {
            let candidates = self
                .sampler
                .generate_variants(
                    &best_policy,
                    Some(self.evaluator.llm().as_ref()),
                    cfg.num_candidates,
                    cfg.seed.wrapping_add(iter as u64),
                )
                .await?;
            if candidates.is_empty() {
                return Err(MiproError::Unexpected(
                    "sampler returned no candidates".to_string(),
                ));
            }

            let mut scored = Vec::with_capacity(candidates.len());
            for c in &candidates {
                let s = self.evaluate(&c.policy, &holdout).await?.mean_score;
                scored.push((c, s));
            }

            let (best_cand, best_cand_score) = scored
                .iter()
                .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(c, s)| (*c, *s))
                .ok_or_else(|| MiproError::Unexpected("no scored candidates".to_string()))?;

            let improved = best_cand_score > best_score + cfg.min_improvement;
            if improved {
                best_score = best_cand_score;
                best_policy = best_cand.policy.clone();
                no_improve_rounds = 0;
            } else {
                no_improve_rounds += 1;
            }

            result.history.push(OptimizationHistoryEntry {
                iteration: iter,
                candidate_scores: scored.iter().map(|(c, s)| (c.id, *s)).collect(),
                best_candidate: best_cand.id,
                best_score: best_cand_score,
                improved,
            });

            result.best_score = best_score;
            result.best_policy = best_policy.clone();

            if no_improve_rounds >= cfg.early_stop_rounds {
                break;
            }
        }

        Ok(result.finish())
    }
}
