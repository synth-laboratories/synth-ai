# Synth AI SDK

Python-only SDK for the containers platform.

## Stable surface

```python
from synth_ai.sdk import (
    AsyncContainerPoolsClient,
    Container,
    ContainerPoolsClient,
    ContainerSpec,
    ContainerType,
    ContainersClient,
)
```

The shipped SDK is intentionally limited to:

- hosted containers
- managed tunnels
- Rhodes/container-pool APIs

Everything else from the previous mixed SDK has been archived under `../research/old/synth_ai`
and is not part of the supported import surface.

## Canonical paths

- `/v1/containers/*`
- `/v1/tunnels/*`
- `/v1/pools/*`
- `/v1/rollouts/*`

## Example

```python
from synth_ai import SynthClient

client = SynthClient(api_key="sk_...")
client.containers.list()
client.tunnels.list()
client.pools.rollouts.list("pool_123")
```
