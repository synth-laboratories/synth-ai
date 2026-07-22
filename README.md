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
from synth_ai.research import SwarmSpec

with SynthClient() as client:
    swarm = client.research.swarms.create(
        SwarmSpec(objective="Assess this repository and produce a concise report.")
    )
    for event in swarm.events():
        print(event.kind, event.telemetry.sequence)
    result = swarm.wait(timeout_seconds=900)
    print(result.swarm_id, result.state)
    resolved = swarm.configuration()
    print(resolved.config_version_id, resolved.snapshot_sha256)
    usage = swarm.usage()
    print(usage.money.nominal_pico_usd, usage.tokens.totals.input_tokens)
    evidence = swarm.evidence()
    print(evidence.artifacts, evidence.work_products)
```

`create` returns a durable handle immediately. `events()` yields typed events,
including an explicit `UnknownSwarmEvent` for forward-compatible server events;
`wait()` uses a monotonic deadline and returns the terminal typed `Swarm`.
`configuration()` returns the immutable, versioned, secret-redacted launch
snapshot bound to that swarm, so replay and audit do not depend on the
project's current mutable configuration.
`usage()` returns one typed cost, token, and actor-attribution projection plus
its source, record count, observation time, and terminal-state freshness. It
does not expose the legacy raw ledger-entry dictionaries.
`evidence()` returns the complete durable artifact and WorkProduct index with
strict counts and lifecycle freshness. Artifact and WorkProduct content reads
use the same typed transport and return bytes; they do not expose storage
authority.

## Research SDK

The only customer entrypoint is `SynthClient().research`. Its stable namespaces
are `projects`, `swarms`, and `factories`.

Create a durable project when work needs reusable configuration:

```python
from synth_ai import SynthClient
from synth_ai.research import EnvironmentKind, ProjectSpec, RuntimeKind, SwarmSpec

with SynthClient() as client:
    project = client.research.projects.create(
        ProjectSpec(
            name="Repository assessment",
            pool_id="pool_default",
            runtime_kind=RuntimeKind("python"),
            environment_kind=EnvironmentKind("docker"),
            orchestrator_profile_id="profile_orchestrator",
            default_worker_profile_id="profile_worker",
        )
    )
    swarm = client.research.swarms.create(
        SwarmSpec(objective="Produce the assessment."),
        project_id=project.project_id,
    )
    print(swarm.wait().state)
```

Factories provide a typed durable optimization loop with native sync/async
parity:

```python
from synth_ai import SynthClient
from synth_ai.research import EffortSpec, FactorySpec, ProjectId

with SynthClient() as client:
    factory = client.research.factories.create(
        FactorySpec(name="Prompt optimizer")
    )
    effort = client.research.factories.efforts.create(
        EffortSpec(
            factory_id=factory.factory_id,
            project_id=ProjectId("project_existing"),
            name="Improve the system prompt",
        )
    )
    print(effort.effort_id, effort.state)
```

Limits, economics, secrets, Tag, rich evidence projections, and administrative
resource APIs remain available under `client.research.advanced` while their
contracts are stabilized. Advanced APIs are not covered by the stable surface
guarantee.

CLI discovery:

```bash
synth-ai research --help
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
| **Research / Factory** | `client.research` | Typed hosted projects, swarms, Factory lifecycles, and Efforts. |
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
SDK, use `client.research.advanced.economics` for authoritative billing reads
while the economics contract remains advanced. Do not infer
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
