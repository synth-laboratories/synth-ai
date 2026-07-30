# `synth_ai` Package

Runtime package for the public Synth AI SDK and CLI.

The public first-mile surface is intentionally small:

- `SynthClient`
- `AsyncSynthClient`
- `client.containers`
- `client.tunnels`
- `client.pools`
- `client.research`
- `synth-ai` CLI

Public docs live at https://docs.usesynth.ai/sdk/overview.

## Package Structure

```text
synth_ai/
├── client.py       # SynthClient and AsyncSynthClient composition layer
├── sdk/            # Public client modules and request/response contracts
├── managed_research/ # Managed Research client, models, MCP, and billing SDK
├── core/           # Shared runtime helpers and errors
├── cli/            # CLI commands for containers, tunnels, and pools
└── __init__.py     # Package version and top-level exports
```

## Dependency Direction

```text
core/ -> sdk/ -> client.py -> cli/
core/ -> managed_research/ -> managed_research/mcp/
```

- `core/` owns shared runtime plumbing such as errors, environment lookup, and URL normalization.
- `sdk/` owns HTTP clients and contracts for the supported public surfaces.
- `managed_research/` owns Managed Research SDK models, clients, MCP tools, and
  typed billing helpers generated from the backend SMR contract.
- `client.py` composes those clients behind `SynthClient` and `AsyncSynthClient`.
- `cli/` wraps the SDK for terminal use.

## Supported Imports

Prefer the front-door client:

```python
from synth_ai import SynthClient

client = SynthClient()
client.containers.list()
client.tunnels.health()
client.pools.list()
```

Use specific clients only when you need lower-level control:

```python
from synth_ai.sdk.containers import ContainersClient
from synth_ai.sdk.pools import ContainerPoolsClient
from synth_ai.sdk.tunnels import TunnelsClient
```

Research callers go through the `SynthClient` front door and the Research
facade:

```python
from synth_ai import SynthClient

research = SynthClient().research
catalog = research.advanced.economics.billing_catalog()
plan = research.advanced.economics.billing_plan()
```

## Layering

One DAG, enforced by `check_sdk_layering.py` in the sibling `testing` repo.
See `unify_sdk_layering.md` for the rationale.

```text
core/      plumbing only: auth, http, errors, utils, generic contracts
  |
  v
sdk/       ALL public HTTP clients and domain contracts
             containers, tunnels, pools, managed_agents, research/
  |
  v
client.py  SynthClient / AsyncSynthClient composition
  |
  +--> cli/            thin terminal adapter
  +--> mcp/research/   thin MCP adapter
```

| From \ To | `core` | `sdk` | `client` | `cli` | `mcp` |
|-----------|--------|-------|----------|-------|-------|
| `core` | yes | **no** | no | no | no |
| `sdk` | yes | yes | no | no | no |
| `client` | yes | yes | — | no | no |
| `cli` | yes | yes | yes | — | no |
| `mcp` | yes | yes | preferred yes | no | — |

`core/` must not import `sdk/`: plumbing cannot depend on the clients built on
top of it. Peer imports inside a layer are fine.

## Guidelines for New Code

1. Put shared errors, URL handling, and environment helpers in `core/`.
2. Put public HTTP clients and request/response contracts in `sdk/`.
3. Put Research contracts, clients, and operator session surfaces in `sdk/research/`;
   the Research MCP delivery adapter lives in `mcp/research/`. Research used to
   live under `core/research/`, which made `core/` mean two things; that path is
   now a deprecated alias.
4. Put front-door composition in `client.py`.
5. Put terminal commands in `cli/`.
6. Keep unreleased or internal compatibility APIs out of public README examples and public-first docs.
