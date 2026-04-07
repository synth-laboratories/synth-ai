# Synth AI Python Package

Python-only SDK surface for Synth containers, tunnels, and container pools.

## Quick Start

```python
from synth_ai import (
    SynthClient,
    ContainersClient,
    ContainerPoolsClient,
    TunnelsClient,
)
```

## Package Structure

```
synth_ai/
├── containers.py   # Public containers client surface
├── tunnels.py      # Public tunnels client surface
├── pools.py        # Public container-pools and rollout surface
├── client.py       # Thin SynthClient composition layer
├── sdk/            # Minimal HTTP clients and pool/container contracts
├── core/           # Shared runtime helpers and errors
├── cli/            # CLI for containers, tunnels, and pools
└── __init__.py     # Package version and top-level exports
```

## Module Hierarchy

The live package follows a narrow dependency hierarchy:

```
┌─────────┐
│  core/  │  ← Shared runtime helpers and errors
└────┬────┘
     │
┌────▼────┐
│  sdk/   │  ← HTTP clients and contracts for containers/pools
└────┬────┘
     │
┌────▼────┐
│ public  │  ← containers.py, tunnels.py, pools.py, client.py
└────┬────┘
     │
┌────▼────┐
│  cli/   │  ← Thin terminal wrapper around the live clients
└─────────┘
```

### Dependency Rules

| Module | Purpose |
|--------|---------|
| `core/` | Shared runtime helpers used by the live SDK |
| `sdk/` | Low-level container and pool clients/contracts |
| `containers.py`, `tunnels.py`, `pools.py` | Stable public entry points |
| `cli/` | Terminal commands for the same three domains |

## Module Purposes

### `core/` - Infrastructure Layer
Internal shared utilities. Not user-facing.

- Logging and errors
- Environment helpers
- Shared URL resolution

**Design principle**: Keep only the runtime pieces needed by the live containers/tunnels/pools SDK.

### `sdk/` - SDK Layer
User-facing programmatic API.

- Container and pool HTTP clients
- Container auth helpers used by live eval flows
- Shared request/response contracts

**Design principle**: Small, explicit building blocks underneath the public clients.

### `cli/` - CLI Layer
Command-line interface. Thin wrapper around SDK.

- Containers commands
- Tunnels commands
- Pools and rollout commands

**Design principle**: Minimal logic; delegate to SDK.

## Top-Level Exports

```python
from synth_ai import (
    SynthClient,
    AsyncSynthClient,
    ContainersClient,
    TunnelsClient,
    ContainerPoolsClient,
)
```

For most use cases, import from the specific module:

```python
from synth_ai import SynthClient
from synth_ai.sdk.containers import ContainersClient
from synth_ai.sdk.tunnels import TunnelsClient
from synth_ai.sdk.pools import ContainerPoolsClient
```

## Entry Points

- `python -m synth_ai` → CLI (via `__main__.py`)
- `synth-ai` command → CLI (installed by package)

## Guidelines for New Code

1. **Shared runtime helpers** → `core/`
2. **HTTP clients/contracts** → `sdk/`
3. **Stable user-facing APIs** → `containers.py`, `tunnels.py`, `pools.py`, `client.py`
4. **CLI commands** → `cli/`
5. **Anything outside containers/tunnels/pools** → archive under `../research/old/synth_ai` unless it is intentionally being restored
