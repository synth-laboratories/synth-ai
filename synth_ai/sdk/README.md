# SDK Modules

Public HTTP clients and contracts for the Synth AI infrastructure SDK.

Prefer the top-level client in user-facing examples:

```python
from synth_ai import SynthClient

client = SynthClient()
client.containers.list()
client.tunnels.health()
client.pools.list()
```

Use module-level clients when implementing or testing a specific surface:

```python
from synth_ai.sdk.containers import ContainersClient
from synth_ai.sdk.managed_agents import SynthManagedAgents
from synth_ai.sdk.pools import ContainerPoolsClient
from synth_ai.sdk.tunnels import TunnelsClient
```

## Supported Public Surfaces

- hosted containers
- managed tunnels and tunnel leases
- container pools, tasks, rollouts, artifacts, usage, summaries, and events
- Anthropic-shaped managed agents via `SynthManagedAgents`

## Canonical Paths

- `/v1/containers/*`
- `/v1/tunnels/*`
- `/v1/pools/*`
- `/v1/rollouts/*`
- `/api/managed-agents/anthropic/v1/*`
- `/anthropic/v1/*` for direct local Horizons Private usage

## Ownership Rules

- Keep transport details in SDK modules.
- Keep shared error and environment helpers in `core/`.
- Keep front-door composition in `client.py`.
- Keep public examples focused on `SynthClient`, tunnels, pools, and containers.
