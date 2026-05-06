# NanoHorizon Craftax Hello World

This staged ReportBench task runs a minimal NanoHorizon Craftax baseline:

- model: `x-ai/grok-4.1-fast` through OpenRouter
- task: `craftax`
- default requested rollout count: `1`
- default requested total LLM calls: `1`
- requested LLM calls per rollout: `1`
- default rollout concurrency: `1`

The runner writes a compact reportbench bundle containing the concrete rollout
records, the aggregate evaluation summary, a result manifest, and a short
reproduction report.

The task uses the local `nanohorizon` checkout on this machine through
`uv run --directory <nanohorizon-root> ...`. If the checkout is not at the
default sibling monorepo location, set `NANOHORIZON_REPO_ROOT`.
When using OpenRouter, keep `NANOHORIZON_MODEL` provider-qualified; do not send
unqualified names such as `gpt-4.1-nano` to OpenRouter. Set
`NANOHORIZON_ROLLOUTS` and `NANOHORIZON_ROLLOUT_CONCURRENCY` explicitly for
larger non-demo evaluations.
