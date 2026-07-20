# Synth AI SDK

<!-- CI release pins: PyPI-0.15.2-orange synth-ai==0.15.2 -->

[![PyPI version](https://img.shields.io/pypi/v/synth-ai.svg)](https://pypi.org/project/synth-ai/)
[![License](https://img.shields.io/pypi/l/synth-ai.svg)](https://pypi.org/project/synth-ai/)
[![Python versions](https://img.shields.io/pypi/pyversions/synth-ai.svg)](https://pypi.org/project/synth-ai/)

Python SDK and CLI for Managed Research, Research Factory, and the infrastructure
surfaces that support them.

**Documentation:** https://docs.usesynth.ai/sdk/overview

## Installation

```bash
uv add synth-ai
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

print(client.research.limits.get_typed().plan)
```

## Managed Research (hero SDK)

Install the research extra when you need hosted runs, projects, Factory Tag, or MCP:

```bash
uv add "synth-ai[research]"
```

Hero entrypoint — **`SynthClient().research`** only (no standalone control client in new code):

```python
import os

from synth_ai import SynthClient
from synth_ai.research import ResearchTagSessionCreateRequest, ResearchWorkMode

client = SynthClient()
research = client.research
factory_id = os.environ["SYNTH_FACTORY_ID"]
effort_id = os.environ["SYNTH_FACTORY_EFFORT_ID"]
project_id = os.environ["SYNTH_RESEARCH_PROJECT_ID"]  # An existing, prepared project.

# Org limits
limits = research.limits.get_typed()
print(limits.plan)

# Authoritative economics reads; the client does not recompute allowances or discounts.
plan = research.economics.plan()
catalog = research.economics.catalog()
entitlements = research.economics.entitlements()

# Async Research Factory: inspect the experiment floor before launching work
factory = research.factories.get(factory_id)
floor = research.factories.status(factory.factory_id)
preview = research.factories.preview_wake(factory.factory_id)
# After reviewing preview.efforts, the SDK replays the resolved request_contract
# with its opaque preview_token; callers do not reconstruct the write request.
if preview.confirmation_required:
    receipt = research.factories.wake_due(
        factory.factory_id,
        preview=preview,
    )

# Factory Tag loop
session = research.factories.tag.sessions.create(
    ResearchTagSessionCreateRequest(
        request="Improve rollout throughput",
        factory_id=factory_id,
        effort_id=effort_id,
    )
)
research.factories.tag.sessions.messages.send(session.session_id, "Status update")
scope = research.factories.tag.scopes.get_default()

# Launch against the explicitly selected pre-existing project.
work_mode = ResearchWorkMode.DIRECTED_EFFORT
preflight = research.swarms.check_preflight(project_id, work_mode=work_mode)
session = research.swarms.create(
    project_id,
    objective="Produce a bounded repository assessment and a readable report.",
    work_mode=work_mode,
)

# Swarm readouts (nested namespaces — never ``manderqueue`` on hero)
session.snapshots.get(detail="control")
progress = session.progress.get_typed()
usage = session.usage.get()
work_products = session.work_products.list()
artifacts = session.artifacts.list()
if work_products:
    report = session.work_products.content.get(work_products[0].work_product_id)
session.message_queue.messages.list()
research.projects.objectives.list(project_id, run_id=session.swarm_id)
```

CLI smoke:

```bash
synth-ai research limits get
synth-ai research tag smoke
synth-ai research smoke
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
| **Managed Research / Factory** | `client.research` | Hosted research runs, projects, Factory Tag, limits, MCP (`synth-ai[research]`). |
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
SDK, use `client.research.economics.entitlements()` for the organization snapshot
and `client.research.economics.plan()`, `.catalog()`, `.run_drawdown(run_id)`, or
`.factory_effort_drawdown(factory_effort_id)` for canonical billing reads. Use
`.project(project_id)` for project usage, budgets, and entitlements. Do not infer
allowance from legacy Autumn balances or local spend summaries, and do not
recompute discounts in the client.

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
make docs-gen   # generate Mintlify SDK reference into docs/
make docs-dev   # preview at http://localhost:3000/overview
```

Optional: install [Lefthook](https://github.com/evilmartians/lefthook) and run
`lefthook install` to run formatting, linting, and type checks on staged Python
files.

[SMR Handoff X thread](https://github.com/usesynth/smr-handoff/blob/main/marketing/smr-handoff-x-thread.md) — hand agent tasks to [Managed Research](https://usesynth.ai/smr) from Cursor, Codex, or Claude Code ([repo](https://github.com/usesynth/smr-handoff)).
