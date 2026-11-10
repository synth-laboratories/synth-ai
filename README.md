# synth-ai

Typed Python 3.11+ client for the Synth public API v1. Version 0.31.0 (launch contract 3).

1. Install the qualified package: `pip install synth-ai`.
2. Sign in: `synth login`.
3. Launch and wait for a run:

```python
from synth_ai import SynthClient

with SynthClient() as client:
    run = client.swarms.run("Compare retrieval methods", budget="2.00")
    print(run.run_id, run.state)
```

Set `SYNTH_API_KEY` and optionally `SYNTH_API_BASE_URL`.
`AsyncSynthClient` provides the same methods with `await`.

`pip install synth-ai[mcp]` adds the `synth-ai-mcp` stdio server.
Use `synth --help` for the CLI and `synth login` for browser authorization.

Upload a task file and include its immutable reference at launch:

```python
from synth_ai.models import InputFile, RunInputs

with SynthClient() as client:
    file = client.swarms.upload_file("task.md")
    run = client.swarms.run(
        "Follow the uploaded task", budget="2.00",
        inputs=RunInputs(files=[InputFile(
            file_id=file.file_id,
            revision_id=file.revision_id,
            digest_sha256=file.digest_sha256,
        )]),
    )
```

Uploads accept UTF-8 text, Markdown and JSON, at most 32 MiB per file and per
launch, with at most 32 references. `upload_text` and MCP `swarms_upload_file`
accept up to 32 KiB inline. `synth swarms upload task.md` uses
the full upload protocol. Pool resources use the public selector `pool`.
`update_labels(run_id, labels, expected_revision=...)` refuses stale revisions.


Choose a published run profile. Omit `profile` to let Core use its default.

```python
with SynthClient() as client:
    print(client.swarms.profiles())
    run = client.swarms.run("Solve this task", budget="2.00", profile="swarms-v2")
```

```sh
synth swarms profiles
synth swarms run "Solve this task" --budget 2.00 --profile swarms-v2
synth swarms profiles --research-task published-task
synth swarms run "Improve the policy" --budget 2.00 --profile fork-flush --research-task published-task
```

The catalog reports each profile's description, revision, supported routing and
caller-specific availability. Research profiles require a published task and
its approval. Every run retains its exact resolved profile in `run.profile`.
New configurations are published profiles; arbitrary actor overrides are not
part of the public API.

See [Swarms](docs/swarms.md) for routing, lifecycle and transfer details.
Version 0.29.0 replaces the public `program`, `method` and `profile_overrides`
launch fields with `profile` and, for research profiles, `research_task`.

Since 0.31.0, the SDK uses strict, frozen records, named nested models, and enum values.
Use `CompletionMilestone.FINISHED` for a fully settled wait, `run.finished` to
inspect it, and `run.known_state` when a known state enum is useful. Unknown server
states remain available as `run.state`. See [typing and protocol standards](docs/protocols.md)
for migration details and the `ty`/Ruff checks.

Runnable workflows are in [examples](examples/README.md); [protocol details](docs/protocols.md) define compatibility, recovery, pagination and diagnostic guarantees.
