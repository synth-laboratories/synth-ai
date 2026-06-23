# `synth_ai` Package

Runtime package for the public Synth AI SDK and CLI.

The public first-mile surface is intentionally small:

- `SynthClient`
- `AsyncSynthClient`
- `client.containers`
- `client.tunnels`
- `client.pools`
- `synth_ai.managed_research`
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

Managed Research callers should use the Managed Research front door and its
co-located billing namespace:

```python
from synth_ai.managed_research import ManagedResearchClient

control = ManagedResearchClient()
catalog = control.billing.catalog()
plan = control.billing.plan()
```

## Guidelines for New Code

1. Put shared errors, URL handling, and environment helpers in `core/`.
2. Put public HTTP clients and request/response contracts in `sdk/`.
3. Put Managed Research SDK/client/model/MCP surfaces in `managed_research/`.
4. Put front-door composition in `client.py`.
5. Put terminal commands in `cli/`.
6. Keep unreleased or internal compatibility APIs out of public README examples and public-first docs.
