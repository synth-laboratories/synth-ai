# Archipelago Environment Pool Eval (Public Example)

This demo shows how to run a Synth **eval job** against **environment pools** that execute an **Archipelago** task. It uses a tiny LocalAPI task app that forwards `/rollout` calls to the environment-pools API, then wraps the pool rollout result back into a LocalAPI response.

## Files

- `create_archipelago_pool.py`: creates a pool with a single Archipelago task definition.
- `archipelago_env_pool_task_app.py`: LocalAPI proxy that routes rollouts into the pool.
- `run_archipelago_env_pool_eval.py`: submits an eval job that targets the proxy.

## Prereqs

- `synth-ai` installed locally (`pip install -e .` from this repo).
- Synth backend + rust backend running (local) **or** a dev/prod backend URL.
- Archipelago images for env/agent/grading available and containing the task configs at `/configs/*`.
  - The defaults align with `Rhodes/archipelago/examples/simple_task/*` if you bake those files into your images.

## 1) Configure environment

```bash
export SYNTH_API_KEY=sk_live_...
export SYNTH_BACKEND_URL=http://localhost:8000   # or https://api-dev.usesynth.ai

# Archipelago images (required)
export RHODES_APEX_ENV_IMAGE=ghcr.io/synth-labs/archipelago-env:latest
export RHODES_APEX_AGENT_IMAGE=ghcr.io/synth-labs/archipelago-agent:latest
export RHODES_APEX_GRADING_IMAGE=ghcr.io/synth-labs/archipelago-grading:latest

# Optional: override config paths (defaults point to /configs/*.json in the images)
# export ARCHIPELAGO_AGENT_CONFIG_PATH=/configs/agent_config.json
# export ARCHIPELAGO_ORCHESTRATOR_CONFIG_PATH=/configs/orchestrator_config.json
# export ARCHIPELAGO_VERIFIERS_PATH=/configs/verifiers.json
# export ARCHIPELAGO_EVAL_CONFIGS_PATH=/configs/eval_configs.json
```

## 2) Create an Archipelago pool

```bash
python demos/archipelago_env_pool_eval/create_archipelago_pool.py
```

This prints a `pool_id`. Keep it for the eval run.

## 3) Start the LocalAPI proxy

```bash
python demos/archipelago_env_pool_eval/archipelago_env_pool_task_app.py
```

The task app listens on `http://localhost:8001` by default. It will auto-mint an `ENVIRONMENT_API_KEY` if needed.

## 4) Run the eval job

```bash
python demos/archipelago_env_pool_eval/run_archipelago_env_pool_eval.py --pool-id <pool_id>
```

### Optional pool tags

If you route by tags instead of `pool_id`:

```bash
python demos/archipelago_env_pool_eval/run_archipelago_env_pool_eval.py --pool-tags archipelago,apex
```

## How routing works

- `run_archipelago_env_pool_eval.py` submits a Synth eval job to the backend.
- The eval job calls the LocalAPI proxy `/rollout` endpoint.
- The proxy creates a **pool rollout** via `synth_ai.sdk.environment_pools.create_rollout` using the Archipelago config + `pool_id`/`pool_tags`.
- The pool rollout status and reward are returned as a LocalAPI response so the eval job can complete.

## Customizing the Archipelago task

You can override the Archipelago rollout config in two ways:

1) Env vars (see the list in `archipelago_env_pool_task_app.py`).
2) Pass an override payload in `env_config`:

```python
env_config = {
    "pool_id": "<pool_id>",
    "archipelago": {
        "initial_snapshot_path": "/configs/original_snapshot.zip",
        "mcp_config_path": "/configs/mcp_config.json",
    },
}
```

## Local stack reminder

When running locally, use the local stack per `synth-ai/AGENTS.md` so both backends are healthy before running the demo.
