use crate::models::{Policy, PromptVariant, ProposerBackend};
use crate::{LlmClient, Result};
use async_trait::async_trait;

#[async_trait]
pub trait VariantSampler: Send + Sync {
    async fn generate_variants(
        &self,
        base: &Policy,
        llm: Option<&dyn LlmClient>,
        proposer_backend: ProposerBackend,
        n: usize,
        seed: u64,
    ) -> Result<Vec<PromptVariant>>;

    fn name(&self) -> &'static str;
}

#[derive(Debug, Default)]
pub struct BasicSampler;

impl BasicSampler {
    #[tracing::instrument]
    pub fn new() -> Self {
        Self
    }
}

#[async_trait]
impl VariantSampler for BasicSampler {
    #[tracing::instrument(skip_all)]
    async fn generate_variants(
        &self,
        base: &Policy,
        llm: Option<&dyn LlmClient>,
        proposer_backend: ProposerBackend,
        n: usize,
        seed: u64,
    ) -> Result<Vec<PromptVariant>> {
        let mut out = Vec::with_capacity(n);
        if n == 0 {
            return Ok(out);
        }

        // Deterministic set of mutations first.
        for i in 0..n {
            let policy = match i % 4 {
                0 => Policy {
                    template: format!("You are precise and concise.\n\n{}", base.template),
                    metadata: base.metadata.clone(),
                },
                1 => Policy {
                    template: format!(
                        "{}\n\nConstraints:\n- Use short sentences.\n- Avoid extra commentary.",
                        base.template
                    ),
                    metadata: base.metadata.clone(),
                },
                2 => Policy {
                    template: format!(
                        "{}\n\nReturn format:\n- Answer only.\n- No preamble.",
                        base.template
                    ),
                    metadata: base.metadata.clone(),
                },
                _ => {
                    if proposer_backend == ProposerBackend::Rlm {
                        Policy {
                            template: format!(
                                "{}\n\nRLM guidance:\n- Maximize task reward and correctness.\n- Penalize verbosity.",
                                base.template
                            ),
                            metadata: base.metadata.clone(),
                        }
                    } else if let Some(llm) = llm {
                        let rewrite_prompt = format!(
                            "Rewrite this prompt to be clearer and more specific, preserving intent. Output only the rewritten prompt.\n\nPROMPT:\n{}",
                            base.template
                        );
                        let rewritten = llm.complete(&rewrite_prompt).await?;
                        // Local task-model callbacks may return task labels, not prompt rewrites.
                        // If rewrite output looks too short/degenerate, keep a deterministic mutation.
                        let rewritten_clean = rewritten.trim();
                        let rewritten_usable =
                            rewritten_clean.len() >= 20 && rewritten_clean.contains(' ');
                        Policy {
                            template: if rewritten_usable {
                                rewritten_clean.to_string()
                            } else {
                                format!("{}\n\nBe explicit.", base.template)
                            },
                            metadata: base.metadata.clone(),
                        }
                    } else {
                        Policy {
                            template: format!("{}\n\nBe explicit.", base.template),
                            metadata: base.metadata.clone(),
                        }
                    }
                }
            };
            out.push(PromptVariant::new(
                policy,
                None,
                format!("basic-mutation-{seed}-{i}"),
            ));
        }

        Ok(out)
    }

    fn name(&self) -> &'static str {
        "basic"
    }
}
