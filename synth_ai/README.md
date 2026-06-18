# `synth_ai` Package

Runtime package for the public Synth AI SDK, Managed Research client, Research
Factory client surface, MCP server, and CLI.

The public first-mile surface is intentionally small and product-led:

- `SynthClient`
- `AsyncSynthClient`
- `client.research`
- `synth-ai-managed-research-mcp`
- `synth-ai` CLI

Advanced infrastructure namespaces are still available when a workflow needs
direct access to hosted infrastructure records:

- `client.containers`
- `client.tunnels`
- `client.pools`

Public docs live at https://docs.usesynth.ai/managed-research/intro.

## Package Structure

```text
synth_ai/
├── client.py       # SynthClient and AsyncSynthClient composition layer
├── sdk/            # Public client modules and request/response contracts
├── managed_research/ # Managed Research SDK, models, schemas, and MCP server
├── core/           # Shared runtime helpers and errors
├── cli/            # CLI commands for containers, tunnels, and pools
└── __init__.py     # Package version and top-level exports
```

## Dependency Direction

```text
core/ -> sdk/ -> client.py -> cli/
```

- `core/` owns shared runtime plumbing such as errors, environment lookup, and URL normalization.
- `sdk/` owns HTTP clients and contracts for the supported public surfaces.
- `client.py` composes those clients behind `SynthClient` and `AsyncSynthClient`.
- `cli/` wraps the SDK for terminal use.

## Supported Imports

Prefer the front-door client:

```python
from synth_ai import SynthClient

client = SynthClient()
research = client.research
run = research.runs.start(
    "Inspect this repository and leave a reviewable report.",
    work_mode="directed_effort",
    providers=[{"provider": "openrouter"}],
    runbook="lite",
)
```

Use lower-level clients only when you need direct infrastructure control:

```python
from synth_ai.sdk.containers import ContainersClient
from synth_ai.sdk.pools import ContainerPoolsClient
from synth_ai.sdk.tunnels import TunnelsClient
```

## Guidelines for New Code

1. Put shared errors, URL handling, and environment helpers in `core/`.
2. Put public HTTP clients and request/response contracts in `sdk/`.
3. Put front-door composition in `client.py`.
4. Put terminal commands in `cli/`.
5. Keep unreleased or internal compatibility APIs out of public README examples and public-first docs.
