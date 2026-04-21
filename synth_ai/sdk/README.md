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
    OpenAIAgentsSdkClient,
)
```

The shipped SDK is intentionally limited to:

- hosted containers
- managed tunnels
- container-pool APIs
- managed-agents Anthropic compatibility APIs
- OpenAI Agents SDK compatibility APIs

Everything else from the previous mixed SDK has been archived under `../research/old/synth_ai`
and is not part of the supported import surface.

## Canonical paths

- `/v1/containers/*`
- `/v1/tunnels/*`
- `/v1/pools/*`
- `/v1/rollouts/*`
- `/api/managed-agents/anthropic/v1/*`
- `/api/managed-agents/openai/v1/*` (BFF)
- `/openai/v1/*` (direct fallback lane)

OpenAI compatibility scope in this SDK is phase1-core + phase1-adjacent
Responses/Conversations routes.

`OpenAIAgentsSdkClient` transport mode defaults to `auto`:

- BFF-first on `/api/managed-agents/openai/v1/*`
- direct fallback to `/openai/v1/*` only on `404`, `405`, `501`
- no fallback for auth/validation/runtime failures

## Example

```python
from synth_ai import SynthClient

client = SynthClient(api_key="sk_...")
client.containers.list()
client.tunnels.list()
client.pools.rollouts.list("pool_123")
client.openai_agents_sdk.create_response(
    {
        "model": "gpt-4.1-mini",
        "input": [{"role": "user", "content": "List risk items."}],
    }
)
```
