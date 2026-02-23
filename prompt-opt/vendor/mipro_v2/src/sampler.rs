use crate::models::{Policy, PromptVariant};
use crate::{LlmClient, Result};
use async_trait::async_trait;

#[async_trait]
pub trait VariantSampler: Send + Sync {
    async fn generate_variants(
        &self,
        base: &Policy,
        llm: Option<&dyn LlmClient>,
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
        n: usize,
        seed: u64,
    ) -> Result<Vec<PromptVariant>> {
        let mut out = Vec::with_capacity(n);
        if n == 0 {
            return Ok(out);
        }

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
                    if let Some(llm) = llm {
                        let rewrite_prompt = format!(
                            "Rewrite this prompt to be clearer and more specific, preserving intent. Output only the rewritten prompt.\n\nPROMPT:\n{}",
                            base.template
                        );
                        let rewritten = llm.complete(&rewrite_prompt).await?;
                        Policy {
                            template: rewritten,
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
