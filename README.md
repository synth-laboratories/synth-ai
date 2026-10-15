# Synth SDK 0.23 development branch

Local SDK-first Index implementation. The pinned 0.23 development wheel is an
engineering qualification candidate and must not be published as a release.

The new client targets the closed `index.v1` contract on `/api/v1`; it refuses
legacy responses and does not fall back. Backend adoption and native/cached/live
acceptance are pending. This worktree also implements the first Swarms core,
output/file delivery and unified MCP adapter. The initial caller model override
accepts `gpt-6-luna`, with `openai` as the supported provider override.
Omitted overrides preserve backend project defaults; resolved metadata is not
an override allowlist. Hosted MCP requires the backend authorization/service
adapter; its native OAuth/API-key journeys remain unqualified.

```python
from synth_ai import SynthClient
from synth_ai.sdk.index import PublicSearchOptions

with SynthClient.anonymous() as client:
    result = client.index.search("reproducible retrieval evidence", options=PublicSearchOptions())
    print(result.response)
```

This requires an index.v1 backend. Keyed calls require KeyedSearchOptions with
an exact Money charge ceiling. Durable search and mutation recovery are keyed.
Use typed content refs for current-rights retrieval; citations grant no access.
Synchronous clients own a cancellable HTTP pool; synchronous client injection is
not supported. Async clients use native async HTTP and may borrow an AsyncClient.
Local observer timeouts and close never cancel durable remote work.

For explicit HTTPX use, `synth_ai.transports.DeadlineHTTPTransport` supplies the
same owned, cancellable network implementation. Set a finite HTTPX read timeout;
it bounds the whole exchange, including headers and body. Close the HTTPX client
to release its pool and join its network thread. Arbitrary non-cooperative
synchronous transports cannot be preempted; synchronous client injection into
`SynthClient` is deliberately unsupported. Observer total, idle, reconnect and
event-count bounds remain independent of explicit remote cancellation.

The retired source remains on `legacy/sdk-pre-0230-20261013`; no broad compatibility
aliases, old CLI or split MCP are shipped.

- [Implementation and contract choices](SDK_0230_IMPLEMENTATION.md)
- [Backend delta map](SDK_0230_DESIGN.md)
- [Testing expectations](SDK_0230_TESTING.md)
- [Unified MCP implementation and auth integration](SDK_0230_MCP.md)

Tests live in the dedicated testing repository and install the selected wheel
in an owned isolated environment. SDK unit/contract passes are not backend or
end-to-end qualification. No backend services/data/deployment are changed here.
