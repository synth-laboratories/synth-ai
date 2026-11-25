# Quickstart

Upload a task, launch a run with a budget, wait for completion, then read its
answer and usage. These examples target synth-ai 0.42.0 and OpenAPI 1.43.0.
Save the task instructions in `task.md` before running the examples.

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
    print(run.state, result.answer, result.charge)
    print(client.account.usage_records(run_id=run.run_id).items)
```

`launch()` returns at once; `wait(run_id)`, `pause(run_id)` and `resume(run_id)`
steer it. `AsyncSynthClient` has the same methods with `await`.

### Send a message during a run

Messaging requires the run's admitted manager queue. A run without that
interface refuses sends with `RunNotAcceptingMessagesError`; history reports
`messaging_unavailable`. An accepted message may still be queued.

```python
from synth_ai import SynthClient
from synth_ai.models import MessageKind

with SynthClient() as client:
    run = client.swarms.launch("Review the task", budget="2.00")
    message = client.swarms.send_message(
        run.run_id, "Which approach did you choose?",
        kind=MessageKind.QUESTION, idempotency_key="question-1",
    )
    reply = client.swarms.wait_for_reply(run.run_id, message.message_id, timeout=60)
    print(reply.body)
```

`wait_for_reply` reads history and then follows run events. Its local timeout
leaves the message and run unchanged. A retained body is `BodyUnavailable` with
reason `retention`.

## CLI

```sh
synth profiles
synth upload task.md
synth run "Follow the uploaded task" --budget 2.00 --file task.md --wait
synth result RUN
synth outputs RUN
synth download RUN -o results/
synth usage RUN
```

Replace `RUN` with the run ID printed by `synth run --wait`.

`synth run --wait` prints the run ID, the final state and the answer on stdout,
then one line with the run's charge on stderr, for example
`charge  0.0237 USD (final)`. Add
`--json` to any command for the SDK record. Exit codes are in [cli.md](cli.md).

## MCP

Add the stdio server to your MCP client:

```json
{"mcpServers": {"synth": {"command": "synth-ai-mcp", "env": {"SYNTH_API_KEY": "..."}}}}
```

The agent calls these tools in order. In the launch arguments, replace
`FILE_ID`, `REVISION_ID` and `DIGEST_SHA256` with values from the upload.
Replace `RUN` with the launch result's run ID. Repeat the wait call if its
`outcome` is `timeout`. Choose unique idempotency keys for each new upload and
launch. Reuse the original keys if the outcome is uncertain.

```json
[
  {"tool":"swarms_profiles_list","arguments":{}},
  {"tool":"swarms_upload_file","arguments":{"idempotency_key":"task-upload-1","body":{"name":"task.md","media_type":"text/markdown","content":"# Task\nFind the fault.\n"}}},
  {"tool":"swarms_run_launch","arguments":{"idempotency_key":"task-run-1","body":{"message":"Follow the uploaded task","limits":{"maximum_charge":{"amount":"2.00","currency":"USD"}},"inputs":{"files":[{"file_id":"FILE_ID","revision_id":"REVISION_ID","digest_sha256":"DIGEST_SHA256"}]}}}},
  {"tool":"swarms_run_wait","arguments":{"run_id":"RUN"}},
  {"tool":"swarms_run_result","arguments":{"run_id":"RUN"}},
  {"tool":"swarms_run_outputs","arguments":{"run_id":"RUN"}},
  {"tool":"account_usage_for_run","arguments":{"run_id":"RUN"}}
]
```

See [MCP setup](mcp.md) and the [tool reference](mcp-tools.md) for hosted access
and output retrieval.

Errors on every surface use one set of codes: see [errors.md](errors.md).

For messaging, call `swarms_run_message_send` with `run_id`, `body: {kind, text}`
and `idempotency_key`, then `swarms_run_messages` for that run's visible history.
