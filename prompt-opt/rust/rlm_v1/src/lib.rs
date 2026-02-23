//! rlm: Reward Language Model verifier (standalone).
//!
//! Defines reward signals, computes weighted outcomes, and generates reports.

#![forbid(unsafe_code)]

pub mod models;
pub mod report;
pub mod signals;
pub mod verifier;

pub type Result<T> = std::result::Result<T, RlmError>;

#[derive(thiserror::Error, Debug)]
pub enum RlmError {
    #[error("invalid config: {0}")]
    InvalidConfig(String),

    #[error("invalid argument: {0}")]
    InvalidArgument(String),

    #[error("llm error: {0}")]
    Llm(String),

    #[error("serialization error: {0}")]
    Serde(#[from] serde_json::Error),

    #[error("unexpected error: {0}")]
    Unexpected(String),
}

#[async_trait::async_trait]
pub trait LlmClient: Send + Sync {
    async fn grade(&self, rubric: &str, input: &str, output: &str) -> Result<LlmGrade>;
    fn name(&self) -> &'static str;
}

#[derive(Debug, Clone)]
pub struct LlmGrade {
    pub score_0_to_1: f32,
    pub reasoning: String,
}

pub use models::{
    EvalReport, RewardOutcome, RewardSignal, SignalKind, SignalScore, SignalWeight,
    VerificationCase, VerifierConfig,
};
pub use report::{ReportFormat, render_markdown_report};
pub use signals::Signals;
pub use verifier::RlmVerifier;
