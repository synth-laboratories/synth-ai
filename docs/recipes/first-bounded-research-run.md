# First bounded Research swarm

**Prerequisites:** `uv add synth-ai` and set `SYNTH_API_KEY`.

**Guarantees:** `create` returns a durable `SwarmHandle`; `wait` uses a
monotonic client deadline; server work can outlive a client timeout; cancellation
is an explicit lifecycle request. The backend remains the authority for
admission, budget, capacity, and terminal state.

```python
from synth_ai import SynthClient
from synth_ai.research import Error, SwarmSpec

with SynthClient() as client:
    handle = None
    try:
        handle = client.research.swarms.create(
            SwarmSpec(
                objective=(
                    "Inspect the repository and publish one bounded findings report."
                ),
                timebox_seconds=900,
            )
        )
        terminal = handle.wait(
            timeout_seconds=900,
            poll_interval_seconds=2,
        )
        print(terminal.swarm_id, terminal.state)
        resolved = handle.configuration()
        print(resolved.config_version_id, resolved.snapshot_sha256)
        usage = handle.usage()
        print(usage.money.nominal_pico_usd, usage.freshness.as_of)
        evidence = handle.evidence()
        print(len(evidence.artifacts), len(evidence.work_products))
        if evidence.work_products:
            report = client.research.swarms.work_product_content(
                evidence.work_products[0].work_product_id
            )
            print(report.decode("utf-8"))
    except Error as error:
        if handle is not None:
            handle.cancel()
        failure = error.failure
        if failure is not None:
            print(
                failure.code,
                failure.operation,
                failure.request_id,
                failure.retry.retryable,
            )
        raise
```

To attach the swarm to a prepared project, pass
`project_id=ProjectId("...")` to `swarms.create`. The same `SwarmSpec` is used
for one-off and project-bound work; there is no separate overrides payload.

`handle.configuration()` reads the exact backend-owned configuration version
bound at launch. The returned snapshot is recursively immutable and
secret-redacted, and `snapshot_sha256` provides the stable replay/audit
identity. Historical runs without a configuration version fail explicitly;
the SDK never substitutes the project's current configuration.

`handle.usage()` makes evidence freshness observable: `freshness.source`
identifies the canonical usage authority, `as_of` records the newest included
fact, and `run_is_terminal` states the lifecycle observation used for this
projection. Aggregate money uses exact integer cents and pico-USD; actor money
uses the cent precision supplied by its authority. Tokens and attribution are
closed typed records rather than arbitrary mappings.

`handle.evidence()` returns the complete durable artifact and WorkProduct index
for the swarm at one observable read time. Counts are checked against the
decoded records, identifiers are opaque types, WorkProduct kind/status/
readiness and artifact roles are closed enums, and content is read as bytes
through the same typed transport and failure contract. The SDK does not probe
project-scoped legacy routes or inspect storage URIs.

Stream typed events when live progress is needed:

```python
for event in handle.events(last_event_id=saved_cursor):
    save_cursor(event.event_id)
    print(event.kind, event.telemetry.correlation_id, event.payload)
```

Known event kinds decode to `KnownSwarmEvent`. A server event introduced after
the installed SDK decodes to `UnknownSwarmEvent` with its raw JSON preserved,
rather than being misclassified or rejected.

Advanced operator timelines, traces, economics, and administrative projections
remain under `client.research.advanced` until they graduate onto the bounded
stable operation contract.
