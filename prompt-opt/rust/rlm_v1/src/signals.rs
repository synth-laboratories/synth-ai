use crate::models::{RewardSignal, SignalKind, SignalScore, VerificationCase};
use crate::{LlmClient, Result, RlmError};

pub struct Signals {
    signals: Vec<RewardSignal>,
}

impl Signals {
    #[tracing::instrument]
    pub fn new(signals: Vec<RewardSignal>) -> Result<Self> {
        if signals.is_empty() {
            return Err(RlmError::InvalidArgument(
                "signals must be non-empty".to_string(),
            ));
        }
        Ok(Self { signals })
    }

    #[tracing::instrument(skip_all)]
    pub fn signals(&self) -> &[RewardSignal] {
        &self.signals
    }

    #[tracing::instrument(skip_all)]
    pub fn validate(&self) -> Result<()> {
        let mut sum = 0.0f32;
        for s in &self.signals {
            if s.name.trim().is_empty() {
                return Err(RlmError::InvalidArgument(
                    "signal name must be non-empty".to_string(),
                ));
            }
            if !s.weight.0.is_finite() || s.weight.0 < 0.0 {
                return Err(RlmError::InvalidArgument(
                    "signal weight must be finite and >= 0".to_string(),
                ));
            }
            if s.weight.0 > 0.0 {
                sum += s.weight.0;
            }
        }
        if sum == 0.0 {
            return Err(RlmError::InvalidArgument(
                "at least one signal must have non-zero weight".to_string(),
            ));
        }
        Ok(())
    }

    #[tracing::instrument(skip_all)]
    pub async fn score_all(
        &self,
        case: &VerificationCase,
        llm: Option<&dyn LlmClient>,
    ) -> Result<Vec<SignalScore>> {
        let mut out = vec![];
        for s in &self.signals {
            if s.weight.0 == 0.0 {
                continue;
            }
            let (score, reasoning) = score_one(s, case, llm).await?;
            out.push(SignalScore {
                name: s.name.clone(),
                weight: s.weight,
                score_0_to_1: score,
                reasoning,
            });
        }
        Ok(out)
    }
}

#[tracing::instrument(skip_all)]
async fn score_one(
    signal: &RewardSignal,
    case: &VerificationCase,
    llm: Option<&dyn LlmClient>,
) -> Result<(f32, String)> {
    match &signal.kind {
        SignalKind::ExactMatch => {
            let Some(expected) = case.expected.as_ref() else {
                return Err(RlmError::InvalidArgument(
                    "ExactMatch signal requires case.expected".to_string(),
                ));
            };
            let ok = case.output.trim() == expected.trim();
            Ok((if ok { 1.0 } else { 0.0 }, String::new()))
        }
        SignalKind::Contains { needle } => {
            let ok = case.output.contains(needle);
            Ok((if ok { 1.0 } else { 0.0 }, String::new()))
        }
        SignalKind::LlmRubric { rubric } => {
            let Some(llm) = llm else {
                return Err(RlmError::InvalidArgument(
                    "LlmRubric signal requires an LlmClient".to_string(),
                ));
            };
            let grade = llm.grade(rubric, &case.input, &case.output).await?;
            let score = grade.score_0_to_1.clamp(0.0, 1.0);
            Ok((score, grade.reasoning))
        }
    }
}
