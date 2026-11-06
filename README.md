# synth-ai

Typed Python 3.11+ client for the Synth public API v1. Version 0.26.0.

1. Install the qualified package: `pip install synth-ai`.
2. Sign in: `synth login`.
3. Launch and wait for a run:

```python
from synth_ai import SynthClient

with SynthClient() as client:
    project = client.swarms.create_project(name="First run")
    run = client.swarms.run(
        "Compare retrieval methods", project_id=project.project_id, budget="2.00"
    )
    print(run.run_id, run.state)
```

Set `SYNTH_API_KEY` and optionally `SYNTH_API_BASE_URL` and `SYNTH_PROJECT_ID`.
`AsyncSynthClient` provides the same methods with `await`.

`pip install synth-ai[mcp]` adds the `synth-ai-mcp` stdio server.
Use `synth --help` for the CLI and `synth login` for browser authorization.

Upload a task file and include its immutable reference at launch:

```python
with SynthClient() as client:
    project = client.swarms.create_project(name="File task")
    file = client.swarms.upload_file("task.md", project_id=project.project_id)
    run = client.swarms.run(
        "Follow the uploaded task", project_id=project.project_id, budget="2.00",
        inputs={"files": [{key: getattr(file, key) for key in
            ("file_id", "revision_id", "project_id", "digest_sha256")}]},
    )
```

Uploads accept UTF-8 text, Markdown and JSON, at most 32 MiB per file and per
launch, with at most 32 references. `upload_text` and MCP `swarms_upload_file`
accept up to 32 KiB inline. `synth swarms upload task.md --project-id <id>` uses
the full upload protocol. Pool resources use the public selector `pool`.
`update_labels(run_id, labels, expected_revision=...)` refuses stale revisions.


Choose the execution method with `--method status-quo`, `--method fork-flush`, or
`--method forgetting`. Status Quo uses the default program. The two research
methods require a published project pairing: the evaluator, baseline, context,
and actor profile are pinned by that pairing. An unpaired project is refused.

```sh
synth swarms run "Improve the policy" --project-id <id> --budget 2.00 --method status-quo
synth swarms run "Improve the policy" --project-id <id> --budget 2.00 --method fork-flush --method-overrides '{"branches": 2}'
synth swarms run "Improve the policy" --project-id <id> --budget 2.00 --method forgetting --method-overrides '{"peers": 2}'
```

The equivalent Python arguments are `method={"release": "research.fork_flush@1",
"overrides": {"branches": 2}}` and `method={"release": "research.forgetting@1",
"overrides": {"peers": 2}}`. Omit `method` for Status Quo. Read `run.method` to
inspect the immutable release and execution pins retained with the run.
