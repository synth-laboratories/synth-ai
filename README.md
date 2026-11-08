# synth-ai

Typed Python 3.11+ client for the Synth public API v1. Version 0.28.0 (launch contract 2).

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
with SynthClient() as client:
    file = client.swarms.upload_file("task.md")
    run = client.swarms.run(
        "Follow the uploaded task", budget="2.00",
        inputs={"files": [{key: getattr(file, key) for key in
            ("file_id", "revision_id", "digest_sha256")}]},
    )
```

Uploads accept UTF-8 text, Markdown and JSON, at most 32 MiB per file and per
launch, with at most 32 references. `upload_text` and MCP `swarms_upload_file`
accept up to 32 KiB inline. `synth swarms upload task.md` uses
the full upload protocol. Pool resources use the public selector `pool`.
`update_labels(run_id, labels, expected_revision=...)` refuses stale revisions.


Choose the execution method with `--method status-quo`, `--method fork-flush`, or
`--method forgetting`. Status Quo uses the default program. The two research
methods run on a published task, named with `--method-task`, that has an approval
for the method: the evaluator, baseline, context and actor profile are pinned by
that approval. A task with no approval is refused.

```sh
synth swarms run "Improve the policy" --budget 2.00 --method status-quo
synth swarms run "Improve the policy" --budget 2.00 --method fork-flush --method-task <task> --method-overrides '{"branches": 2}'
synth swarms run "Improve the policy" --budget 2.00 --method forgetting --method-task <task> --method-overrides '{"peers": 2}'
```

The equivalent Python arguments are `method={"release": "research.fork_flush@1",
"overrides": {"branches": 2}}` and `method={"release": "research.forgetting@1",
"overrides": {"peers": 2}}`. Omit `method` for Status Quo. Read `run.method` to
inspect the immutable release and execution pins retained with the run.

Routing restrictions, mandatory budgets, completion milestones and resumable
transfers are documented in [Swarms](docs/swarms.md). Version 0.28.0 removes
public task-count controls and the old top-level provider/model mappings.
