# Crafter Task App

This example is now wired through the shared Synth task-app harness. Use the
`uvx synth-ai` CLI to run it locally or deploy it to Modal without touching the
underlying FastAPI plumbing.

## Local development
```bash
uvx synth-ai serve grpo-crafter --port 8001
# Optional extras:
#   --env-file path/to/.env    # load additional environment variables
#   --reload                   # enable uvicorn auto-reload
```

Useful endpoints while the server is running:
- `GET http://localhost:8001/health`
- `GET http://localhost:8001/info`
- `GET http://localhost:8001/task_info?seed=42`
- `POST http://localhost:8001/rollout`

## Deploy to Modal
```bash
uvx synth-ai deploy grpo-crafter --name grpo-crafter-task-app
```

Requirements:
- Modal CLI installed and authenticated (`modal token new`).
- Secrets `crafter-environment-sdk`, `groq-api-key`, and `openai-api-key`
  available in your Modal account.

The CLI generates a Modal entrypoint on the fly using the shared
`TaskAppConfig`, ensuring the container matches the local FastAPI behavior.

## Compatibility note
`examples/warming_up_to_rl/task_app/grpo_crafter_task_app.py` remains as a
legacy wrapper exposing `fastapi_app()` and a `__main__` entrypoint. Behind the
scenes it proxies to the shared configuration; prefer the CLI workflow above
for new automation and tests.
