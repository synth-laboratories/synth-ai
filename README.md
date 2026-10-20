# Synth Python SDK

A typed Python client for Synth Index search with citations, Swarms research
runs, account access, and hosted or local MCP. Python 3.11 or newer is required.

Swarms requires organization admission. An unavailable operation raises
`UnsupportedOperationError` with code `operation_unavailable`. Inspect
`client.account.entitlements()` for your organization's access conditions.

```sh
pip install synth-ai
pip install 'synth-ai[mcp]'
synth-ai login
```

Use browser login or set `SYNTH_API_KEY`. Stored login credentials are reloaded
on every authenticated call. `SYNTH_API_URL` overrides the default API origin,
`https://api.usesynth.ai`. Anonymous Index search needs no login or API key.

```python
from synth_ai import SynthClient
from synth_ai.sdk.index import PublicSearchOptions

with SynthClient.anonymous() as client:
    result = client.index.search("reproducible retrieval", options=PublicSearchOptions())
    print(result.response)
```

Authenticated search has an explicit maximum charge:

```python
from synth_ai import SynthClient
from synth_ai.sdk.index import KeyedSearchOptions
from synth_ai.types.money import Money

with SynthClient() as client:
    result = client.index.search(
        "reproducible retrieval",
        options=KeyedSearchOptions(maximum_charge=Money(coefficient=5, scale=2)),
    )
    print(result.response)
```

Create a Swarms project, launch bounded work, wait, and download its results:

```python
from synth_ai import SynthClient
from synth_ai.options import ObservationOptions
from synth_ai.sdk.swarms import Limits
from synth_ai.types.money import Money

with SynthClient() as client:
    project = client.swarms.projects.create("Research", objective="Compare retrieval methods")
    run = client.swarms.create(message="Compare retrieval methods", project_id=project.project_id,
        limits=Limits(maximum_charge=Money(coefficient=2, scale=0), wall_time_seconds=600,
                      maximum_tasks=4, maximum_parallel_tasks=2))
    result = run.wait(observation=ObservationOptions(timeout_seconds=660, idle_timeout_seconds=120,
        reconnect_deadline_seconds=30, retry_count_max=3, maximum_events=2000, poll_interval_seconds=2))
    bundle = run.download("research-results")
```

The download verifies exact repository and artifact digests before publishing a
new destination directory. Its manifest records the run reference, final commit,
artifacts, objective verdicts, and answers. Existing destinations are refused.
See [Swarms](docs/swarms.md) for input artifacts, guidance, resources, controls,
and usage. Both `SynthClient` and `AsyncSynthClient` expose the same operations;
async methods are awaited and clients support managed contexts.

Catch `SynthError` for API and transport failures. Common types are
`AuthenticationError`, `RateLimitError`, `BudgetExceededError`,
`RequestTimeoutError`, and `OutcomeUncertainError`.

**After `OutcomeUncertainError`, resolve the retained identity before deciding
whether to submit new work:** `client.resolve_mutation(error.identity)`.
`MutationCancelledError` derives from `asyncio.CancelledError` and also retains
its mutation identity. A local timeout or closing a client does not cancel a run.

Requests have total deadlines. Reads retry transient failures with bounded
jitter and honor `Retry-After`. Mutations are never automatically replayed.
Call `run.cancel()`, `run.pause()`, or `run.resume()` for remote controls.
Use a stable `MutationOptions(idempotency_key=...)` for recoverable writes.

Errors carry operation and request identifiers. Responses accept additive
fields; unknown enum values remain visible. `ClientOptions(strict_contract=True)`
requires exact schema hashes. Client options can borrow caller-owned HTTPX
clients; closing the SDK leaves a borrowed client open. File-based login requires
POSIX ownership and permissions; explicit API keys work on other platforms.

- [Documentation](https://docs.usesynth.ai)
- [Swarms guide](docs/swarms.md)
- [Changelog and migration](CHANGELOG.md)
- [Release policy](RELEASING.md)
- [Issues](https://github.com/synth-laboratories/synth-ai/issues)
