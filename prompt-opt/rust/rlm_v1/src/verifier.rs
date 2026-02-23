use crate::models::{
    EvalReport, RewardOutcome, RewardSignal, SignalScore, VerificationCase, VerifierConfig,
};
use crate::signals::Signals;
use crate::{LlmClient, Result, RlmError};
use chrono::Utc;
use std::sync::Arc;
use ulid::Ulid;

pub struct RlmVerifier {
    cfg: VerifierConfig,
    signals: Signals,
    llm: Option<Arc<dyn LlmClient>>,
}

impl RlmVerifier {
    #[tracing::instrument(skip_all)]
    pub fn new(
        cfg: VerifierConfig,
        signals: Vec<RewardSignal>,
        llm: Option<Arc<dyn LlmClient>>,
    ) -> Result<Self> {
        cfg.validate()?;
        let signals = Signals::new(signals)?;
        signals.validate()?;
        Ok(Self { cfg, signals, llm })
    }

    #[tracing::instrument(skip_all)]
    pub fn config(&self) -> &VerifierConfig {
        &self.cfg
    }

    #[tracing::instrument(skip_all)]
    pub fn signals(&self) -> &[RewardSignal] {
        self.signals.signals()
    }

    #[tracing::instrument(skip_all)]
    pub async fn verify(&self, case: VerificationCase) -> Result<RewardOutcome> {
        let scores = self.signals.score_all(&case, self.llm.as_deref()).await?;
        aggregate(&self.cfg, scores)
    }

    #[tracing::instrument(skip_all)]
    pub async fn verify_report(&self, case: VerificationCase) -> Result<EvalReport> {
        let outcome = self.verify(case.clone()).await?;
        Ok(EvalReport {
            id: Ulid::new(),
            created_at: Utc::now(),
            config: self.cfg.clone(),
            case,
            outcome,
        })
    }
}

#[tracing::instrument(skip_all)]
fn aggregate(cfg: &VerifierConfig, signal_scores: Vec<SignalScore>) -> Result<RewardOutcome> {
    let mut sum_w = 0.0f32;
    let mut sum_ws = 0.0f32;
    let mut reasoning = String::new();

    for s in &signal_scores {
        if s.weight.0 == 0.0 {
            continue;
        }
        if !s.score_0_to_1.is_finite() {
            return Err(RlmError::Unexpected(format!(
                "non-finite signal score for {}",
                s.name
            )));
        }
        let w = s.weight.0;
        sum_w += w;
        sum_ws += w * s.score_0_to_1.clamp(0.0, 1.0);
        if !s.reasoning.trim().is_empty() {
            if !reasoning.is_empty() {
                reasoning.push('\n');
            }
            reasoning.push_str(&format!("{}: {}", s.name, s.reasoning.trim()));
        }
    }

    if sum_w == 0.0 {
        return Err(RlmError::InvalidArgument(
            "no non-zero-weight signals in outcome".to_string(),
        ));
    }
    let total = (sum_ws / sum_w).clamp(0.0, 1.0);
    let passed = total >= cfg.pass_threshold;
    Ok(RewardOutcome {
        total_score_0_to_1: total,
        passed,
        signal_scores,
        reasoning,
    })
}
