//! mipro_v2: general-purpose batch prompt/policy optimization.
//!
//! This crate is standalone (no Horizons dependency).

#![forbid(unsafe_code)]

pub mod config;
pub mod dataset;
pub mod evaluator;
pub mod models;
pub mod optimizer;
pub mod sampler;

pub type Result<T> = std::result::Result<T, MiproError>;

#[derive(thiserror::Error, Debug)]
pub enum MiproError {
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

pub use dataset::{Dataset, Example};
pub use evaluator::{EvalMetric, EvalSummary, Evaluator, ExactMatchMetric};
pub use models::{
    EvalMetricResult, MiproConfig, OptimizationHistoryEntry, OptimizationResult, Policy,
    PromptVariant,
};
pub use optimizer::Optimizer;
pub use sampler::{BasicSampler, VariantSampler};

#[async_trait::async_trait]
pub trait LlmClient: Send + Sync {
    async fn complete(&self, prompt: &str) -> Result<String>;
    fn name(&self) -> &'static str;
}
