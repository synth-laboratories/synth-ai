# Swarms

A run takes a message and a budget, the maximum charge in USD. `swarms.run`
launches, waits for the run to close and returns the closed run with its outputs.

```python
from synth_ai import SynthClient

with SynthClient(project_id="proj_123") as client:
    result = client.swarms.run("Compare retrieval methods", budget="5.00")
    print(result.run.state, [output.filename for output in result.outputs])
```

`project_id` defaults to `SYNTH_PROJECT_ID`. `AsyncSynthClient` has the same
calls, awaited.

## Step by step

```python
run = client.swarms.launch("Compare retrieval methods", budget="5.00")
page = client.swarms.events(run.run_id, limit=100)
later = client.swarms.events(run.run_id, cursor=page.resume_cursor)
for event in client.swarms.iter_events(run.run_id):
    print(event.sequence, event.kind)
closed = client.swarms.wait(run.run_id, timeout_seconds=3600, poll_seconds=5)
outputs = client.swarms.outputs(run.run_id)
record = client.swarms.output(run.run_id, outputs.items[0].output_id)
receipt = client.swarms.cancel(run.run_id)
runs = client.swarms.list(limit=20, project_id="proj_123")
more = client.swarms.list(cursor=runs.next_cursor)
```

Pages carry `items`, `next_cursor` and `complete`. Event pages also carry
`resume_cursor`, which stays set after the last page so a reader can poll for
later events. Outputs are JSON records (`output_id`, `state`, `filename`,
`reference`). Runs imported from the earlier backend have no event history
(`409 run_not_orchestra_authority`, `RunNotOrchestraAuthorityError`) and no
output rows.

`wait` returns when the state is `completed`, `partial`, `failed` or `cancelled`,
or when execution closes. A wait timeout raises `WaitTimeoutError` and leaves the
run going. `run` raises `RunFailedError` for a failed run.

## Limits

`budget` sets `limits.maximum_charge`. Pass `limits` for the other bounds:

```python
from synth_ai.public.models import Limits, Money

limits = Limits(
    maximum_charge=Money.usd("5.00"),
    wall_time_seconds=3600,
    maximum_tasks=20,
    maximum_parallel_tasks=4,
)
client.swarms.launch("Compare retrieval methods", limits=limits)
```

## Idempotency

`launch`, `run` and `cancel` send an `Idempotency-Key`. Pass
`idempotency_key=` to retry the same request safely.

## Errors

Refusals raise `synth_ai.public.errors.SwarmsAPIError` subclasses carrying
`error.code`, `error.service`, `error.stage`, `error.upstream_status` and
`error.cause`, plus `outcome`: `not_accepted` means nothing was started and a
retry is safe; `unknown` means check with `get` or `list` before retrying.
