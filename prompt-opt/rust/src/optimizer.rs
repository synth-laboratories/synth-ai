use crate::dataset::Dataset;
use crate::evaluator::{EvalSummary, Evaluator};
use crate::models::{MiproConfig, OptimizationHistoryEntry, OptimizationResult, Policy};
use crate::sampler::VariantSampler;
use crate::{MiproError, Result};
use rlm_rs::{
    RewardSignal, RlmVerifier, SignalKind, SignalWeight, VerificationCase, VerifierConfig,
};
use std::sync::Arc;
use ulid::Ulid;

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
        let _ = train; // reserved for future use (e.g., sampler conditioning). Still split for determinism.

        let mut result = OptimizationResult::new(initial_policy.clone());

        let baseline = self.evaluate(&initial_policy, &holdout).await?;
        let mut best_policy = initial_policy;
        let mut best_score = baseline.mean_score;
        let mut no_improve_rounds = 0usize;

        // If the baseline is the only thing evaluated, still set best_score.
        result.best_score = best_score;
        result.best_policy = best_policy.clone();

        for iter in 0..cfg.max_iterations {
            let candidates = self
                .sampler
                .generate_variants(
                    &best_policy,
                    Some(self.evaluator.llm().as_ref()),
                    cfg.proposer_backend,
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
                let task_score = self.evaluate(&c.policy, &holdout).await?.mean_score;
                let adjusted_score = if cfg.proposer_backend == crate::models::ProposerBackend::Rlm {
                    self.rlm_adjusted_score(&c.policy, &holdout, task_score).await?
                } else {
                    task_score
                };
                scored.push((c, adjusted_score));
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

    async fn rlm_adjusted_score(
        &self,
        policy: &Policy,
        holdout: &Dataset,
        base_score: f32,
    ) -> Result<f32> {
        let verifier = RlmVerifier::new(
            VerifierConfig::default(),
            vec![RewardSignal {
                name: "exact_match".to_string(),
                weight: SignalWeight(1.0),
                kind: SignalKind::ExactMatch,
                description: "Exact match signal over holdout examples".to_string(),
            }],
            None,
        )
        .map_err(|e| MiproError::Unexpected(format!("rlm verifier build failed: {e}")))?;

        let mut rlm_scores: Vec<f32> = Vec::with_capacity(holdout.examples.len());
        for ex in &holdout.examples {
            let prompt = policy.render(&ex.input)?;
            let output = self.evaluator.llm().complete(&prompt).await?;
            let expected = ex.expected.as_str().map(ToString::to_string);
            let case = VerificationCase {
                id: Ulid::new(),
                input: match &ex.input {
                    serde_json::Value::String(s) => s.clone(),
                    other => other.to_string(),
                },
                output,
                expected,
                metadata: serde_json::json!({}),
            };
            let outcome = verifier
                .verify(case)
                .await
                .map_err(|e| MiproError::Unexpected(format!("rlm verify failed: {e}")))?;
            rlm_scores.push(outcome.total_score_0_to_1);
        }

        if rlm_scores.is_empty() {
            return Ok(base_score);
        }
        let rlm_mean = rlm_scores.iter().sum::<f32>() / (rlm_scores.len() as f32);
        Ok((base_score + rlm_mean) / 2.0)
    }
}
