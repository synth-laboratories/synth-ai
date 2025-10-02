# Crafter Task App (Modal)

This is a copy of the Crafter task service used by RL rollouts.

- App name: `grpo-crafter-task-app`
- FastAPI entrypoint: `fastapi_app()` (decorated with `@app.function(..., @asgi_app())`)
- Wraps: `synth_envs_hosted.hosted_app.create_app(allowed_environments=["crafter"])`
- Extra endpoint: `/proxy/v1/chat/completions` (OpenAI proxy convenience)

## Prerequisites
- Python with `uv` and `modal` CLI installed
- Modal account/credentials configured (`modal token new`)
- A Modal secret named `crafter-environment-sdk` containing:
  - `ENVIRONMENT_API_KEY` (required)
  - `OPENAI_API_KEY` (required for proxy)

Create the secret (example):
```bash
modal secret create crafter-environment-sdk \
  ENVIRONMENT_API_KEY=sk_env_... \
  OPENAI_API_KEY=sk-...
```

## Run locally
```bash
cd /Users/joshpurtell/Documents/GitHub/research/testing/crafter/task_app
export ENVIRONMENT_API_KEY=sk_env_...
export OPENAI_API_KEY=sk-...
uv run python grpo_crafter_task_app.py --local --host 0.0.0.0 --port 8001
# Health:  http://localhost:8001/health
# Rollout: http://localhost:8001/rollout
```

## Deploy to Modal
From the repo root (`/Users/joshpurtell/Documents/GitHub/research`):

Option A (module path):
```bash
MODAL_ENV=dev ENVIRONMENT=dev \
uv run modal deploy -m testing.crafter.task_app.grpo_crafter_task_app --env dev
```

Option B (file path):
```bash
MODAL_ENV=dev ENVIRONMENT=dev \
uv run modal deploy testing/crafter/task_app/grpo_crafter_task_app.py --env dev
```

After deploy, the URL will look like:
```
https://<hash>.modal.run
```
Use `/health` to verify, and point backend configs (e.g., job `endpoint_base_url`) to this base URL.

## Endpoints
- `GET /health` — readiness check
- Hosted env/policy/rollout endpoints via `synth_envs_hosted`
- `POST /proxy/v1/chat/completions` — forwards to OpenAI (requires `OPENAI_API_KEY`)

## Troubleshooting
- 401/503 from proxy: Ensure `OPENAI_API_KEY` is present in the `crafter-environment-sdk` secret
- 401 from hosted endpoints: Ensure `ENVIRONMENT_API_KEY` is present and passed as `X-API-Key` by callers
- Not reachable after deploy: Confirm `MODAL_ENV`, `--env dev`, and that the function `fastapi_app` is decorated with `@asgi_app()`

## Notes
- This app is for Crafter-only flows in development. For production, deploy and configure via the monorepo.
