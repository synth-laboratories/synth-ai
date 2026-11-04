# synth-ai

Typed Python 3.11+ client for the Synth public API v1. Version 0.26.0.

```python
from synth_ai import SynthClient

with SynthClient() as client:
    project = client.swarms.create_project(name="First run")
    run = client.swarms.run("Compare retrieval methods", project_id=project.project_id, budget="0.10")
    print(run.run_id, run.state)
```

Set `SYNTH_API_KEY` and optionally `SYNTH_API_BASE_URL` and `SYNTH_PROJECT_ID`.
`AsyncSynthClient` provides the same methods with `await`.

`pip install synth-ai[mcp]` adds the `synth-ai-mcp` stdio server.
Use `synth --help` for the CLI and `synth login` for browser authorization.
