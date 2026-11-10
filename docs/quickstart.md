# Quickstart

One journey, three surfaces: upload a task, launch a run with a budget, wait for
it, read the answer and the run's usage. `tests/test_journey.py` drives this same
journey on all three.

Install with `pip install synth-ai` (`pip install "synth-ai[mcp]"` for MCP) and sign
in with `synth login`, or set `SYNTH_API_KEY`.

Only the latest release is supported; upgrade with `pip install -U synth-ai`.

## Python SDK

```python
from synth_ai import SynthClient
from synth_ai.models import InputFile, RunInputs

with SynthClient() as client:
    print(client.swarms.profiles())
    file = client.swarms.upload_file("task.md")
    run = client.swarms.run(
        "Follow the uploaded task", budget="2.00",
        inputs=RunInputs(files=[InputFile(
            file_id=file.file_id, revision_id=file.revision_id,
            digest_sha256=file.digest_sha256,
        )]),
    )  # launches, then waits; the event stream reconnects on its own
    result = client.swarms.result(run)
    print(run.state, result.content)
    print(client.account.usage_records(run_id=run.run_id).items)
```

`launch()` returns at once; `wait(run_id)`, `pause(run_id)` and `resume(run_id)`
steer it. `AsyncSynthClient` has the same methods with `await`.

## CLI

```sh
synth profiles
synth upload task.md
synth run "Follow the uploaded task" --budget 2.00 --file task.md --wait
synth pause RUN && synth resume RUN
synth wait RUN
synth result RUN            # the answer text; --json prints the RunResult
synth outputs RUN
synth download RUN -o results/
synth usage RUN
```

`synth run --wait` prints the run ID, the final state and then the answer. Add
`--json` to any command for the SDK record. Exit codes are in [cli.md](cli.md).

## MCP

Add the stdio server to your MCP client:

```json
{"mcpServers": {"synth": {"command": "synth-ai-mcp", "env": {"SYNTH_API_KEY": "..."}}}}
```

Then the agent calls, in order: `swarms_profiles_list`, `swarms_upload_file`
(`body: {name, media_type, content}`), `swarms_run_launch`
(`body: {message, limits: {maximum_charge: {amount, currency}}, inputs}`),
`swarms_run_wait` (repeat while `outcome` is `timeout`), `swarms_run_pause` /
`swarms_run_resume`, `swarms_run_outputs`, `swarms_run_output_get` and
`usage_for_run`. Mutating tools require an `idempotency_key`. The hosted `/mcp`
endpoint lists the same tools; see [mcp.md](mcp.md) and [tools](mcp-tools.md).

Errors on every surface use one set of codes: see [errors.md](errors.md).
