# Task Apps

Task apps wrap task-specific logic behind a shared FastAPI surface so the Synth training stack (evaluation, SFT, RL) can talk to any environment the same way. The reference implementation lives in `synth_ai/task/server.py` and is configured through `TaskAppConfig`.

## Core Concepts

- **Task metadata** (`TaskInfo`): static description of the environment, action space, observation schema, datasets, and rubric information (`synth_ai/task/contracts.py:109`).
- **Seeds & instances**: `provide_task_instances` receives a list of seeds and must return seed-specific `TaskInfo` objects. The Crafter example (`synth_ai/task/apps/grpo_crafter.py:228`) decorates each instance with difficulty, trait summaries, and per-seed configs.
- **Environment lifecycle**: every app exposes `/env/{env_name}/initialize`, `/step`, and `/terminate` handlers. The SDK client in `examples/warming_up_to_rl/run_eval.py:54` uses this contract.
- **Rollouts**: RL tooling may call `/rollout` with a `RolloutRequest` (`synth_ai/task/contracts.py:51`). The Crafter app adapts the request into its legacy runner inside `rollout_executor` (`synth_ai/task/apps/grpo_crafter.py:246`).
- **Rewards**: outcome and event rewards are packaged into weighted rubrics (`OUTCOME_RUBRIC`, `EVENTS_RUBRIC`) that describe how achievements and step-wise progress are scored (`synth_ai/task/apps/grpo_crafter.py:205`).
- **API keys**: every inbound request sends `X-API-Key`. Key validation is handled in `synth_ai/task/auth.py`; the CLI helpers load secrets from `.env` files before starting the server.
- **Vendor proxies**: enable `ProxyConfig` on your `TaskAppConfig` to surface `/proxy/v1/chat/completions` (OpenAI) or `/proxy/groq/v1/chat/completions` (`synth_ai/task/server.py:135`). Requests are sanitized by `synth_ai/task/proxy.py` to enforce the shared `interact` tool schema.

## Building a TaskAppConfig

The Crafter configuration (`synth_ai/task/apps/grpo_crafter.py:295`) illustrates all moving parts:

1. **Describe the task** using `TaskInfo` and optional `TaskDatasetRegistry` so downstream tooling can enumerate seeds and datasets.
2. **Wire rollouts** by converting `request.ops` into alternating agent/env steps and delegating to your environment runner.
3. **Enable tracing** by honouring `TASKAPP_TRACING_ENABLED`, `TASKAPP_SFT_OUTPUT_DIR`, and `SQLD_DB_PATH`. `build_config()` mounts tracing helpers when those env vars are set.
4. **Declare proxies** if you want the task app to host vendor-facing chat completions. The `CRAFTING_RULES_SYSTEM_HINT` constant shows how to inject domain guidance into every request.
5. **Register the app** via `register_task_app` so the CLI knows how to start/deploy it (`synth_ai/task/apps/__init__.py`).

## Running and Deploying

Use the CLI to serve or ship your app without touching uvicorn directly:

```bash
# Local development
uvx synth-ai serve grpo-crafter --port 8001 --env-file examples/warming_up_to_rl/.env

# Modal deploy
uvx synth-ai deploy grpo-crafter --name grpo-crafter-task-app

# Dry-run the Modal deployment to inspect the generated entrypoint
uvx synth-ai deploy grpo-crafter --dry-run
```

`uvx synth-ai serve` automatically loads `.env` files registered with the task app (plus any `--env-file` overrides), ensures the port is free, and starts uvicorn with optional reload (`synth_ai/cli/task_apps.py:55`). `uvx synth-ai deploy` and `uvx synth-ai modal-serve` package the same app for Modal, preflighting backend access by uploading the `ENVIRONMENT_API_KEY` if available (`synth_ai/cli/task_apps.py:295`).

## Authentication & Backend Integration

Task apps interact with two services:

- **Environment service**: The Modal or local host running the task app validates `X-API-Key` (the *environment API key*). Populate the secret before serving—`uvx synth-ai setup` can write it into your `.env`.
- **Synth backend**: Training jobs (FFT, RL) call Synth’s `/learning/files` or `/rl/jobs` endpoints using `SYNTH_API_KEY`. `uvx synth-ai train` pulls both keys from disk and verifies the task app before job submission.

Keep the secrets aligned across local and hosted environments to avoid 401s during rollouts: run `uvx synth-ai serve grpo-crafter --trace traces/v3` locally to test before deploying to Modal.

## Event & Outcome Rewards

Event rewards are emitted during the episode (`event_rewards` table in tracing) and record incremental achievements such as crafting steps. Outcome rewards summarise the final state (`outcome_rewards`). `examples/warming_up_to_rl/export_trace_sft.py:52` shows how rewards are extracted from tracing v3 SQLite files:

- Unique achievements per turn build the behavioural cloning signal.
- Outcome metadata stores final achievements so evaluators can report progress.

When designing new tasks, supply similar annotations so evaluation, tracing, and SFT export remain consistent.

