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

Scoped inference sessions use the core's budget admission and the gateway ledger:

```python
with SynthClient() as client:
    session = client.inference.session(
        project_id="proj_AAAAAAAAAAAAAAAAAAAAAA",
        preset="hosted.openrouter_zdr_us",
        budget="2.00",
        ttl=3600,
        idempotency_key="trial-001",
    )
```

For bench and tool processes, export the two session variables:

```sh
eval "$(synth inference session --project-id "$SYNTH_PROJECT_ID" --budget 2 --shell)"
```

A session admits a shared aggregate budget, with a maximum of $50 and a TTL of
60–86,400 seconds. Tools read `SYNTH_INFERENCE_BASE_URL` and
`SYNTH_INFERENCE_SESSION_TOKEN`; model and output limits come from the session's
`/models` catalogue. The public presets are `hosted.openrouter_zdr_us` (with
same-model OpenAI fallback), `hosted.openai_direct`, and `test.codex_pool`, all
on `gpt-6-luna`. Pool sessions are available only in local and staging
environments, use a fixed account generation, and are non-chargeable with a
bounded token allowance. Other models require a matching catalogue route.
Creating another trial budget requires a fresh session and idempotency key.
