---
name: synth-api
version: "0.7.15"
description: Use the Synth AI SDK end-to-end for policy optimization (GEPA), graph optimization, eval, and inference
---

# Synth API (SDK v0.7.15)

This skill explains how to run Synth end-to-end with:
- a **container** (task app) exposed via **SynthTunnel** or **Cloudflare tunnel**
- **PolicyOptimizationJob** (GEPA / MIPRO prompt optimization)
- **GraphOptimizationJob** (verifier graph training)
- **GEPA compat layer** (drop-in `gepa.optimize()` interface)
- **Eval jobs** on held-out seeds
- **InProcessTaskApp** for all-in-one local development

Reference demos have moved to the sibling `Benchmarking` repo.

## Required env

- `SYNTH_API_KEY`: your API key (or mint a demo key below)
- `SYNTH_BACKEND_URL` (optional): backend base URL, default `https://api.usesynth.ai`

## Auth (three keys)

Synth uses **three** distinct keys. Do not mix them:

| Key | Purpose | Header | When needed |
|-----|---------|--------|-------------|
| `SYNTH_API_KEY` | Authenticates SDK/CLI calls to the **Synth backend** | `Authorization: Bearer <key>` | Always |
| `ENVIRONMENT_API_KEY` | Authenticates backend-to-**container** requests | `x-api-key: <key>` | Cloudflare tunnels |
| SynthTunnel `worker_token` | Authenticates tunnel **relay to container** | Passed as `task_app_worker_token` in job config | SynthTunnel (default) |

Common failures:
- `Invalid API key` on `/api/jobs/*` = wrong key sent to backend.
- `SYNTH_TUNNEL_ERROR: Invalid worker token` = wrong tunnel relay token.

### Mint a demo Synth API key (optional)

Demo keys are short-lived (default 4 hours) and are great for notebooks or quick starts.

```python
import os
from synth_ai.core.utils.env import mint_demo_api_key

SYNTH_API_BASE = os.environ.get("SYNTH_BACKEND_URL", "https://api.usesynth.ai")
SYNTH_API_KEY = os.environ.get("SYNTH_API_KEY") or mint_demo_api_key(SYNTH_API_BASE)
os.environ["SYNTH_API_KEY"] = SYNTH_API_KEY
```

### Mint + upload an Environment API key (Cloudflare tunnels only)

Only needed when using Cloudflare tunnels. SynthTunnel handles auth automatically.

```python
import os
from synth_ai.sdk.container.auth import mint_environment_api_key, setup_environment_api_key

SYNTH_API_BASE = os.environ.get("SYNTH_BACKEND_URL", "https://api.usesynth.ai")
SYNTH_API_KEY = os.environ["SYNTH_API_KEY"]

ENVIRONMENT_API_KEY = mint_environment_api_key()
os.environ["ENVIRONMENT_API_KEY"] = ENVIRONMENT_API_KEY
setup_environment_api_key(SYNTH_API_BASE, SYNTH_API_KEY, token=ENVIRONMENT_API_KEY)
```

## Core concepts

- **Container**: Your task app runs locally and exposes `/rollout` + `/task_info`.
- **Tunnel**: SynthTunnel (default) or Cloudflare Quick Tunnel makes the container reachable by Synth.
- **GEPA**: Evolutionary prompt optimizer that mutates prompts to maximize reward.
- **MIPRO**: Systematic instruction proposal optimizer.
- **Graph Optimize**: Train verifier graphs (RLM-based) on your evaluation data.
- **Eval jobs**: Formal evaluation on held-out seeds after optimization.

## Version check

```python
import synth_ai
print(synth_ai.__version__)  # "0.7.15"
```

## 1) Define a container

Minimum container shape:
- `provide_taskset_description()`
- `provide_task_instances(seeds)`
- `rollout(request) -> RolloutResponse`

```python
from synth_ai import ContainerConfig, create_task_app
from synth_ai.sdk.container._impl.contracts import (
    RolloutMetrics, RolloutRequest, RolloutResponse, TaskInfo,
)


def create_banking77_task_app(system_prompt: str):
    async def run_rollout(request: RolloutRequest, fastapi_request) -> RolloutResponse:
        reward = 1.0  # Your task logic here; return reward in [0, 1]
        return RolloutResponse(
            trace_correlation_id=request.trace_correlation_id,
            reward_info=RolloutMetrics(outcome_reward=reward),
            trace=None,
        )

    def provide_taskset_description():
        return {"splits": ["train", "test"], "sizes": {"train": 1000, "test": 1000}}

    def provide_task_instances(seeds):
        for seed in seeds:
            yield TaskInfo(
                task={"id": "banking77", "name": "Banking77 Intent Classification"},
                dataset={"id": "banking77", "split": "train", "index": seed},
                inference={"tool": "banking77_classify"},
                limits={"max_turns": 1},
                task_metadata={"seed": seed},
            )

    return create_task_app(
        ContainerConfig(
            app_id="banking77",
            name="Banking77 Intent Classification",
            description="Classify customer queries into intents.",
            provide_taskset_description=provide_taskset_description,
            provide_task_instances=provide_task_instances,
            rollout=run_rollout,
            cors_origins=["*"],
        )
    )
```

## 2) Expose with a tunnel

### SynthTunnel (recommended)

Relay-based tunnel — no external binary required, supports 128 concurrent requests:

```python
from synth_ai.core.tunnels import TunneledContainer

app = create_banking77_task_app("baseline prompt")
tunnel = await TunneledContainer.create_for_app(
    app=app,
    local_port=None,  # auto-select
    api_key=os.environ["SYNTH_API_KEY"],
)
TASK_APP_URL = tunnel.url            # https://st.usesynth.ai/s/rt_...
WORKER_TOKEN = tunnel.worker_token   # pass to job config
print("Task app URL:", TASK_APP_URL)
```

### Cloudflare Quick Tunnel (alternative)

Requires `cloudflared` installed (`brew install cloudflared`):

```python
from synth_ai.core.tunnels import TunneledContainer, TunnelBackend

app = create_banking77_task_app("baseline prompt")
tunnel = await TunneledContainer.create_for_app(
    app=app,
    local_port=None,
    backend=TunnelBackend.CloudflareQuickTunnel,
)
TASK_APP_URL = tunnel.url  # https://....trycloudflare.com
```

When using Cloudflare tunnels, pass `task_app_api_key` instead of `task_app_worker_token` in job configs.

## 3) Run GEPA (policy optimization)

GEPA mutates prompt candidates and evaluates them via rollouts. Use a config dict
or a TOML file.

```python
import os
from synth_ai import PolicyOptimizationJob

config_body = {
    "policy_optimization": {
        "algorithm": "gepa",
        "task_app_url": TASK_APP_URL,
        "env_name": "banking77",
        "initial_prompt": {
            "messages": [
                {"role": "system", "order": 0, "pattern": "Baseline system prompt"},
                {"role": "user", "order": 1, "pattern": "Customer Query: {query}\n\nAvailable Intents:\n{available_intents}"},
            ],
            "wildcards": {"query": "REQUIRED", "available_intents": "OPTIONAL"},
        },
        "policy": {
            "model": "gpt-4.1-nano",
            "provider": "openai",
            "inference_mode": "synth_hosted",
            "temperature": 0.0,
            "max_completion_tokens": 256,
        },
        "gepa": {
            "env_name": "banking77",
            "evaluation": {
                "seeds": list(range(50)),
                "validation_seeds": list(range(50, 60)),
            },
            "rollout": {"budget": 80, "max_concurrent": 8, "minibatch_size": 8},
            "mutation": {"rate": 0.3},
            "population": {"initial_size": 4, "num_generations": 3, "children_per_generation": 3},
            "archive": {"size": 5, "pareto_set_size": 10},
        },
    },
}

job = PolicyOptimizationJob.from_dict(
    config_dict=config_body,
    task_app_worker_token=WORKER_TOKEN,  # from SynthTunnel
    skip_health_check=True,
)
job_id = job.submit()
result = job.stream_until_complete(timeout=3600.0)
print(f"Best score: {result.best_score}")
```

### From a TOML config file

```python
from synth_ai import PolicyOptimizationJob

job = PolicyOptimizationJob.from_config(
    config_path="gepa_config.toml",
    task_app_worker_token=WORKER_TOKEN,
)
job.submit()
result = job.stream_until_complete(timeout=3600.0)
```

## 4) GEPA compat layer (drop-in)

For a simpler interface that handles task app + tunnel setup automatically:

```python
from synth_ai import gepa

trainset, valset, _ = gepa.examples.banking77.init_dataset()
result = gepa.optimize(
    seed_candidate={"system_prompt": "You are a helpful assistant."},
    trainset=trainset,
    valset=valset,
    task_lm="openai/gpt-4.1-mini",
    max_metric_calls=150,
    reflection_lm="openai/gpt-5",
)
print(result.best_candidate["system_prompt"])
```

Key parameters:
- `seed_candidate`: dict with a `system_prompt` (or `instruction`/`prompt`) key
- `trainset` / `valset`: lists of dicts with `input` and `answer` keys
- `task_lm`: `"provider/model"` string
- `reflection_lm`: model for GEPA proposer (controls proposer effort)
- `max_metric_calls`: budget cap

## 5) InProcessTaskApp (all-in-one)

Combines container + tunnel + lifecycle in a single async context manager:

```python
from synth_ai import InProcessTaskApp, PolicyOptimizationJob

async with InProcessTaskApp(
    app=create_banking77_task_app("baseline prompt"),
    port=8114,
    tunnel_mode="synthtunnel",  # default; also "quick", "local", "preconfigured"
    api_key=os.environ["SYNTH_API_KEY"],
) as task_app:
    print(f"Running at: {task_app.url}")

    job = PolicyOptimizationJob.from_dict(
        config_dict=config_body,
        task_app_worker_token=task_app.worker_token,
    )
    job.submit()
    result = job.stream_until_complete(timeout=3600.0)
```

## 6) Graph optimization (verifier training)

Train a verifier graph with an RLM backbone:

```python
from synth_ai import GraphOptimizationJob

job = GraphOptimizationJob.from_dataset(
    dataset="verifier_dataset.json",
    graph_type="rlm",
    policy_models=["gpt-4.1"],
    proposer_effort="medium",
    rollout_budget=200,
)
job.submit()
result = job.stream_until_complete(timeout=3600.0)
```

## 7) Run Eval jobs (held-out seeds)

Eval jobs score a fixed set of held-out seeds for a final report.

```python
import os
from synth_ai import EvalJob, EvalJobConfig

config = EvalJobConfig(
    task_app_url=TASK_APP_URL,
    backend_url=os.environ.get("SYNTH_BACKEND_URL", "https://api.usesynth.ai"),
    api_key=os.environ["SYNTH_API_KEY"],
    task_app_worker_token=WORKER_TOKEN,  # for SynthTunnel
    env_name="banking77",
    seeds=list(range(100, 150)),
    policy_config={"model": "gpt-4.1-nano", "provider": "openai"},
    env_config={"split": "test"},
    concurrency=10,
)
job = EvalJob(config)
job.submit()
result = job.poll_until_complete(timeout=600.0, interval=2.0, progress=True)
print(f"Mean reward: {result.mean_reward}")
```

## 8) Zero-shot verifiers

Run a built-in verifier graph with rubric criteria passed at runtime:

```python
import os
from synth_ai import VerifierClient

client = VerifierClient(
    base_url=os.environ.get("SYNTH_BACKEND_URL", "https://api.usesynth.ai"),
    api_key=os.environ["SYNTH_API_KEY"],
)
result = await client.evaluate(
    job_id="zero_shot_verifier_single",
    trace={"session_id": "s", "session_time_steps": []},
    rubric={
        "event": [{"id": "accuracy", "weight": 1.0, "description": "Correctness"}],
        "outcome": [{"id": "task_completion", "weight": 1.0, "description": "Completed task"}],
    },
    options={"event": True, "outcome": True, "model": "gpt-5-nano"},
    policy_name="my_policy",
    task_app_id="my_task",
)
```

## CLI

```bash
# Install
pip install synth-ai==0.7.15
# or: uv add synth-ai

# Check version
synth-ai --version

# List packaged OpenCode skills
synth-ai skill list

# Install a skill to a custom directory
synth-ai skill install synth-api --dir ~/custom/opencode/skill

# Serve a container locally
synth-ai localapi serve my_module:app --port 8114

# Deploy a container to Synth Harbor
synth-ai localapi deploy --name my-app --app my_module:app --dockerfile ./Dockerfile --context . --wait
```

## HTTP example (raw)

```python
import os
import requests

base = os.environ.get("SYNTH_BACKEND_URL", "https://api.usesynth.ai")
resp = requests.get(
    f"{base}/api/health",
    headers={"Authorization": f"Bearer {os.environ['SYNTH_API_KEY']}"},
    timeout=30,
)
resp.raise_for_status()
print(resp.json())
```

## API stability

Modules follow the [API Stability Lifecycle](../../specifications/api-stability-lifecycle.md):

| Module | Status | Import |
|--------|--------|--------|
| `PolicyOptimizationJob` | Stable | `from synth_ai import PolicyOptimizationJob` |
| `GraphOptimizationJob` | Stable | `from synth_ai import GraphOptimizationJob` |
| `EvalJob` | Stable | `from synth_ai import EvalJob` |
| `ContainerConfig` | Stable | `from synth_ai import ContainerConfig` |
| `create_task_app` | Stable | `from synth_ai import create_task_app` |
| `InProcessTaskApp` | Stable | `from synth_ai import InProcessTaskApp` |
| `TunneledContainer` | Stable | `from synth_ai.core.tunnels import TunneledContainer` |
| `VerifierClient` | Beta | `from synth_ai import VerifierClient` |
| `GraphCompletionsClient` | Beta | `from synth_ai import GraphCompletionsClient` |
| `InferenceClient` | Beta | `from synth_ai import InferenceClient` |
| `gepa.optimize` | Beta | `from synth_ai.gepa import optimize` |
| `dspy.GEPA` | Beta | `from synth_ai.dspy import GEPA` |
| `EnvironmentPoolsClient` | Alpha | `from synth_ai.sdk.environment_pools import EnvironmentPoolsClient` |
| `ManagedPools` | Alpha | `from synth_ai.sdk.managed_pools import ...` |

Legacy aliases (`PromptLearningJob`, `GraphEvolveJob`, `TunneledLocalAPI`, `LocalAPIConfig`, `create_local_api`, `synth_ai.sdk.localapi.*`) still work but are deprecated.

## Troubleshooting checklist

- **SynthTunnel**: Expect a `st.usesynth.ai` URL. Pass `tunnel.worker_token` to job configs.
- **Cloudflare tunnel**: Expect a `trycloudflare.com` URL. Requires `cloudflared` binary.
- **Inference URL**: If using hosted inference, model requests go to `https://api.usesynth.ai/api/inference/v1`.
- **Auth**: Confirm `SYNTH_API_KEY` is set and valid. Do not confuse it with `ENVIRONMENT_API_KEY` or `worker_token`.
- **Container shape**: Ensure `/task_info` and `/rollout` return valid `RolloutResponse`.
- **Streaming errors**: If `stream_until_complete()` disconnects, it auto-reconnects via SSE. Check backend logs for job status.
- **Legacy import paths**: If you see `PromptLearningJob`, update to `PolicyOptimizationJob`. If you see `prompt_learning` config keys, update to `policy_optimization`. If you see `synth_ai.sdk.localapi`, update to `synth_ai.sdk.container`.
