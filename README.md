# synth-ai

Typed Python 3.11+ client for the Synth public API v1. Version 0.26.0.

1. Install the qualified package: `pip install synth-ai` (0.26.0 publication requires owner GO).
2. Sign in: `synth login`.
3. Launch and wait for a run:

```python
from synth_ai import SynthClient

with SynthClient() as client:
    project = client.swarms.create_project(name="First run")
    run = client.swarms.run(
        "Compare retrieval methods", project_id=project.project_id, budget="0.10"
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
        "Follow the uploaded task", project_id=project.project_id, budget="0.10",
        inputs={"files": [{key: getattr(file, key) for key in
            ("file_id", "revision_id", "project_id", "digest_sha256")}]},
    )
```

Uploads accept UTF-8 text, Markdown and JSON, at most 32 MiB per file and per
launch, with at most 32 references. `upload_text` and MCP `swarms_upload_file`
accept up to 32 KiB inline. `synth swarms upload task.md --project-id <id>` uses
the full upload protocol. Pool resources use the public selector `pool`.
`update_labels(run_id, labels, expected_revision=...)` refuses stale revisions.
