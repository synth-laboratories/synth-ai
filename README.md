# Synth AI SDK

<!-- CI release pins: PyPI-0.11.7-orange synth-ai==0.11.7 -->

[![PyPI version](https://img.shields.io/pypi/v/synth-ai.svg)](https://pypi.org/project/synth-ai/)
[![License](https://img.shields.io/pypi/l/synth-ai.svg)](https://pypi.org/project/synth-ai/)
[![Python versions](https://img.shields.io/pypi/pyversions/synth-ai.svg)](https://pypi.org/project/synth-ai/)

Python SDK and CLI for Synth infrastructure surfaces: tunnels, pools, and hosted containers.

**Documentation:** https://docs.usesynth.ai/sdk/overview

## Installation

```bash
uv add synth-ai
```

Or install with pip:

```bash
pip install synth-ai
```

## Authenticate

Set `SYNTH_API_KEY` before using the SDK or CLI:

```bash
export SYNTH_API_KEY="sk_..."
```

## Local Workspaces

For local multi-repo development, `synth-ai` treats workspace resolution as a
read-only overlay. `.env` and Synth home config may provide defaults and
secrets; selecting a worktree must not rewrite those defaults.

Use `SYNTH_WORKSPACE_MANIFEST` or `SYNTH_WORKSPACE_ROOT` for command-scoped
worktree resolution. The resolver in `synth_ai.core.utils.workspace` returns
repo paths and a scoped env mapping for subprocesses without mutating `.env`.

Pass `base_url` when you need to pin a production, local, staging, or private
backend explicitly:

```python
from synth_ai import SynthClient

client = SynthClient(base_url="http://127.0.0.1:8000")
```

The CLI also reads `SYNTH_BACKEND_URL` and accepts `--backend-url`.

## Quickstart

```python
from synth_ai import SynthClient

client = SynthClient()

print(client.containers.list())
print(client.tunnels.health())
print(client.pools.list())
```

## CLI

```bash
synth-ai --help
synth-ai containers list
synth-ai tunnels health
synth-ai pools list
```

## Public Surface

Use `SynthClient` as the front door:

| Surface | Client namespace | Use it for |
| --- | --- | --- |
| **Managed Research / Factory** | `client.research` | Hosted research runs, projects, evidence, MCP (`synth-ai[research]`). |
| Containers | `client.containers` | Hosted container records and lifecycle operations. |
| Tunnels | `client.tunnels` | Managed tunnel records, leases, health, and rotation. |
| Pools | `client.pools` | Container pools, tasks, rollouts, artifacts, usage, and events. |
| CLI | `synth-ai` | Terminal access to containers, tunnels, and pools. |

Use [Managed Research](https://docs.usesynth.ai/managed-research/intro) when you
want hosted research workers, repo runs, evidence, checkpoints, MCP, or final
reports.

## Managed Research Billing

Standalone SMR and Managed Factory draw from the same org-level allowance and
flex-credit wallet. Free, Standard ($20/month), and Max ($200/month) expose
premium and value usage windows with reset times, then use explicit flex credits
after included usage is exhausted. Premium models consume allowance faster;
value models stretch the same allowance further. Promo, make-good, banked, and
override grants are manual audit events rather than automatic resets.

The canonical backend surfaces are `GET /smr/billing/catalog`,
`GET /smr/billing/plan`, `GET /smr/billing/runs/{run_id}/drawdown`, and
`GET /smr/billing/factory-efforts/{factory_effort_id}/drawdown`. In the Python
SDK, use `control.billing.catalog()`, `control.billing.plan()`,
`control.billing.run_drawdown(run_id)`,
`control.billing.factory_effort_drawdown(effort_id)`, and the matching
`preflight_*` helpers. The top-level aliases `control.get_billing_catalog()`,
`control.get_billing_plan()`, `control.get_run_billing_drawdown(run_id)`, and
`control.get_factory_effort_billing_drawdown(effort_id)` call the same billing
namespace. Do not infer allowance from legacy Autumn balances or local spend
summaries.

## Links

- [Install and authenticate](https://docs.usesynth.ai/sdk/install-and-auth)
- [SynthClient guide](https://docs.usesynth.ai/sdk/synth-client)
- [Tunnels](https://docs.usesynth.ai/sdk/tunnels)
- [Pools](https://docs.usesynth.ai/sdk/pools)
- [Containers](https://docs.usesynth.ai/sdk/containers)
- [SDK reference](https://docs.usesynth.ai/reference/sdk)
- [OpenAPI contracts](https://docs.usesynth.ai/reference/openapi)

## Local Development

Use `uv run` for Python tools:

```bash
uv sync --group dev
uv run ruff format --check .
uv run ruff check .
uv run ty check
```

Optional: install [Lefthook](https://github.com/evilmartians/lefthook) and run
`lefthook install` to run formatting, linting, and type checks on staged Python
files.

[SMR Handoff X thread](https://github.com/usesynth/smr-handoff/blob/main/marketing/smr-handoff-x-thread.md) — hand agent tasks to [Managed Research](https://usesynth.ai/smr) from Cursor, Codex, or Claude Code ([repo](https://github.com/usesynth/smr-handoff)).
