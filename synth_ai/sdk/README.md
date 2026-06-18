# Lower-Level SDK Modules

HTTP clients and contracts for lower-level Synth infrastructure surfaces used by
the product SDK. User-facing examples should start with Managed Research and
Research Factory through `SynthClient().research`.

Prefer the top-level client in user-facing examples:

```python
from synth_ai import SynthClient

client = SynthClient()
research = client.research
```

Use module-level infrastructure clients when implementing or testing a specific
advanced surface:

```python
from synth_ai.sdk.containers import ContainersClient
from synth_ai.sdk.managed_agents import SynthManagedAgents
from synth_ai.sdk.pools import ContainerPoolsClient
from synth_ai.sdk.tunnels import TunnelsClient
```

## Supported Advanced Surfaces

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

## Ownership Rules

- Keep transport details in SDK modules.
- Keep shared error and environment helpers in `core/`.
- Keep front-door composition in `client.py`.
- Keep public examples focused on `SynthClient`, Managed Research, Research
  Factory, and documented optimizer workflows. Infrastructure examples belong
  only where those lower-level surfaces are deliberately documented.
