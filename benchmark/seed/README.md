# NanoHorizon Craftax Hello World

This staged ReportBench task runs a minimal NanoHorizon Craftax baseline:

- model: `x-ai/grok-4.3` through OpenRouter
- task: `craftax`
- launch-smoke requested rollout count: `100`
- launch-smoke requested total LLM calls: `100`
- requested LLM calls per rollout: `1`
- launch-smoke rollout concurrency: `10`

The runner writes a compact reportbench bundle containing the concrete rollout
records, the aggregate evaluation summary, scorecard, leaderboard evidence,
bounded real gameplay media, a result manifest, and a short reproduction report.

The task uses the local `nanohorizon` checkout on this machine through
`uv run --directory <nanohorizon-root> ...`. If the checkout is not at the
default sibling monorepo location, set `NANOHORIZON_REPO_ROOT`.
When using OpenRouter, keep `NANOHORIZON_MODEL` provider-qualified; do not send
unqualified names such as `gpt-4.1-nano` to OpenRouter. Set
`NANOHORIZON_ROLLOUTS` and `NANOHORIZON_ROLLOUT_CONCURRENCY` explicitly for
larger evaluations. Serial rollout execution is disabled for multi-rollout
smokes unless `NANOHORIZON_ALLOW_SERIAL_ROLLOUTS=1` is set deliberately.
Do not replay recorded traces to regenerate media after the runner completes;
the fast batched runner emits representative real gameplay media and metadata
for all requested rollouts.
