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

Stream typed events when live progress is needed:

```python
for event in handle.events(last_event_id=saved_cursor):
    save_cursor(event.event_id)
    print(event.kind, event.telemetry.correlation_id, event.payload)
```

Known event kinds decode to `KnownSwarmEvent`. A server event introduced after
the installed SDK decodes to `UnknownSwarmEvent` with its raw JSON preserved,
rather than being misclassified or rejected.

Advanced evidence, artifact, economics, and administrative projections remain
under `client.research.advanced` until they graduate onto the bounded stable
operation contract.
