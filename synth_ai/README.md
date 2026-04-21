# `synth_ai` Package

Runtime package for the public Synth AI SDK and CLI.

The public first-mile surface is intentionally small:

- `SynthClient`
- `AsyncSynthClient`
- `client.containers`
- `client.tunnels`
- `client.pools`
- `synth-ai` CLI

Public docs live at https://docs.usesynth.ai/sdk/overview.

## Package Structure

```text
synth_ai/
├── client.py       # SynthClient and AsyncSynthClient composition layer
├── sdk/            # Public client modules and request/response contracts
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

## Guidelines for New Code

1. Put shared errors, URL handling, and environment helpers in `core/`.
2. Put public HTTP clients and request/response contracts in `sdk/`.
3. Put front-door composition in `client.py`.
4. Put terminal commands in `cli/`.
5. Keep unreleased or internal compatibility APIs out of public README examples and public-first docs.
