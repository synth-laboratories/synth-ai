# Synth AI SDK

<!-- CI release pins: PyPI-0.22.2-orange synth-ai==0.22.2 -->

> Typed Index v0.2 APIs include QA and lifecycle operations, exact-revision rights attestations, and resealed research corrections.

[![PyPI version](https://img.shields.io/pypi/v/synth-ai.svg)](https://pypi.org/project/synth-ai/)
[![License](https://img.shields.io/pypi/l/synth-ai.svg)](https://pypi.org/project/synth-ai/)
[![Python versions](https://img.shields.io/pypi/pyversions/synth-ai.svg)](https://pypi.org/project/synth-ai/)

Python SDK, CLI and MCP servers for **Synth Index** (cited search over reviewed
research) and **Swarms** (hosted research runs under `client.research.swarms`).

**Documentation:** https://docs.usesynth.ai/sdk/overview

## Installation

```bash
uv add synth-ai
```

## Authenticate

Free anonymous public Index Search needs no key (see [Synth Index](#synth-index)).
Keyed Index Search, private collections and Swarms need `SYNTH_API_KEY`:

```bash
export SYNTH_API_KEY="sk_..."
```

A key selects the paid keyed Index route; it never makes a public request free.

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

## Quickstart: Swarms

```python
from synth_ai import SynthClient
from synth_ai.sdk.research.public import SwarmSpec

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

## Research SDK (Swarms)

`SynthClient().research` hosts Swarms. Its stable namespaces are `projects`,
`swarms`, and `factories` (the last is a supported compatibility API; it is not
part of this release's recommended path).

Create a durable project when work needs reusable configuration:

```python
from synth_ai import SynthClient
from synth_ai.sdk.research.public import EnvironmentKind, ProjectSpec, RuntimeKind, SwarmSpec

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

Limits, economics, secrets, Tag, rich evidence projections, and administrative
resource APIs remain available under `client.research.advanced` while their
contracts are stabilized. Advanced APIs are not covered by the stable surface
guarantee.

CLI discovery:

```bash
synth-ai research --help
```

Project creation also accepts the backend-owned `ProjectSpec.policy` mapping.
For example, a server-enabled fresh project can request
`policy={"host_resource_custody_mode": "horizons_docker_sessions_only"}`.
The backend validates this restricted mode and owns its immutable resource
binding; SDK serialization does not grant additional authority.

## Container pools

The optional `synth-ai[pools]` extra exposes the canonical `synth-containers`
client through `AsyncSynthClient.pools`. It uses the same configured backend
credential and keeps hosted admission, resource ownership, and recovery in the
backend. The enclosing async client closes the pool transport.

```python
from synth_ai import AsyncSynthClient

async def inspect_lease(lease_id: str, task_id: str):
    async with AsyncSynthClient() as client:
        return await client.pools.get_lease_interactive(lease_id, task_id=task_id)
```

For explicit lifetime management, `from synth_ai.pools import PoolClient`
re-exports the same implementation. Research-only installations do not import
this optional dependency. Development candidates must install the exact pinned
containers wheel; an unpublished candidate extra is not a release claim.

## CLI

```bash
synth-ai --help
synth-ai index --help      # Synth Index Search and research intake
synth-ai research --help   # Swarms and Research projects
```

## Synth Index

Index Search returns a cited `response` plus the exact Contribution revisions it
cites. There are two routes, chosen explicitly by the client you use:

| | Public route | Keyed route |
| --- | --- | --- |
| Caller | Anonymous only: no API key or session is sent | `SYNTH_API_KEY` (or a Clerk session) |
| Python | `PublicIndexClient().public_search(...)` | `SynthClient().index.search(...)` |
| CLI | `synth-ai index search --public` (0.21.1+) | `synth-ai index search --keyed` |
| MCP `index_search` | No `SYNTH_API_KEY` in the server env | `SYNTH_API_KEY` set; runs only after the org turns on wallet payments |
| Corpus | Published public Contributions | Public, or authorized private collections |
| Price | Free to the caller | Paid: Fast is 5 cents per delivered Search; Deep is 5 cents plus measured model cost, capped by your `max_charge_cents` and at $1 per Search. The backend publishes the current price. |
| Funding | None | Index promo credit, then the wallet with explicit consent (`allow_wallet`, `max_charge_cents`) |
| Limits | Per-caller and platform-wide per-minute/per-day limits, a daily service budget and a Deep concurrency cap, published in `capabilities().public_search` | Your organization's mode grants, monthly caps and per-mode concurrency |
| Retention | Published in `capabilities().public_search` (`retention`, `privacy_copy`) | Your organization's terms; public terms do not describe keyed Search |

The public route exists only where the backend enables it (`public_search.enabled`
in capabilities); otherwise it fails closed with `PublicSearchDisabledError`.
The public route refuses any credential (409, `PublicSearchAuthenticatedError`),
so a key never turns a public request into a free keyed one, and the SDK and CLI
never switch routes on their own.

```python
from synth_ai.sdk.index import PublicIndexClient

with PublicIndexClient() as index:  # anonymous: sends no key
    print(index.public_search_terms().lines)  # price, limits, privacy from capabilities
    result = index.public_search("RLVR verifier design", mode="fast", max_results=5)
    print(result.response, [item.citation for item in result.citations])
```

Keyed (paid) Search:

```python
from uuid import uuid4

from synth_ai import SynthClient
from synth_ai.sdk.index import SearchBillingConstraints

request_key = str(uuid4())  # Persist this before sending; reuse it on uncertain retry.
with SynthClient() as synth:  # Reads SYNTH_API_KEY.
    result = synth.index.search(
        query="What evidence supports the retrieval design?",
        mode="fast",
        billing=SearchBillingConstraints(allow_wallet=True, max_charge_cents=5),
        idempotency_key=request_key,
    )
    print(result.response, result.usage)
```

CLI:

```bash
synth-ai index search "RLVR verifier design" --public
synth-ai index search "RLVR verifier design" --keyed --allow-wallet --max-charge-cents 5
```

With no key and no route flag the CLI refuses and asks for one. With an inherited
`SYNTH_API_KEY`, `--public` still sends no key; without a flag the CLI uses the
keyed route and says so on stderr.

- **MCP:** `synth-ai-index-mcp` is an Index-only stdio server, read-only by
  default. Setup and per-configuration economics are in the
  [Index SDK guide](synth_ai/sdk/index/README.md#external-agents-over-stdio-mcp).
- **Errors:** public-route refusals have their own types (rate limit, budget
  exhausted, disabled route, credential conflict); see the
  [Index SDK guide](synth_ai/sdk/index/README.md#errors).

These calls require a deployed Index API; installing the SDK alone does not make
a Search available.

## Public Surface

Use `SynthClient` as the front door:

| Surface | Client namespace | Use it for |
| --- | --- | --- |
| **Index** | `client.index`, `PublicIndexClient` | Keyed (paid) and anonymous public (free) Search, exact contents. |
| **Swarms** | `client.research.swarms` | Hosted research runs, events, usage and evidence. |
| Research projects | `client.research.projects` | Reusable Swarm configuration. |
| CLI / MCP | `synth-ai`, `synth-ai-index-mcp` | Terminal commands and an Index-only coding-agent server. |

Swarms run on Synth's hosted research workers; see the
[Managed Research docs](https://docs.usesynth.ai/managed-research/intro) for
repo runs, evidence, checkpoints and final reports.

## Swarms Billing

Swarms and other Managed Research work draw from the same org-level allowance and
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
- [SDK reference](https://docs.usesynth.ai/reference/sdk)
- [OpenAPI contracts](https://docs.usesynth.ai/reference/openapi)

## Local Development

Use `uv run` for Python tools:

```bash
uv sync --group dev
uv run ruff format --check .
uv run ruff check .
uv run ty check
make docs-gen   # generate Blume SDK reference into docs/
make docs-dev   # preview at http://localhost:3000/overview
```

Optional: install [Lefthook](https://github.com/evilmartians/lefthook) and run
`lefthook install` to run formatting, linting, and type checks on staged Python
files.
