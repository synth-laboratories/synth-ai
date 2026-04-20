# NanoHorizon Craftax Hello World

This staged ReportBench task runs a minimal NanoHorizon Craftax baseline:

- model: `gpt-4.1-nano`
- task: `craftax`
- requested rollout count: `10`
- requested total LLM calls: `10`
- requested LLM calls per rollout: `1`
- rollout concurrency: `10`

The runner writes a compact reportbench bundle containing the concrete rollout
records, the aggregate evaluation summary, a result manifest, and a short
reproduction report.

The task uses the local `nanohorizon` checkout on this machine through
`uv run --directory <nanohorizon-root> ...`. If the checkout is not at the
default sibling monorepo location, set `NANOHORIZON_REPO_ROOT`.
