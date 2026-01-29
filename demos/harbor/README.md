# Harbor Demo: EngineBench Evaluations

This demo shows how to use Harbor hosted sandboxes to run EngineBench coding agent evaluations. Harbor eliminates the need for users to manage Daytona credentials or set up tunnels.

## Overview

Harbor provides:
- **Hosted sandboxes**: Run coding agents in isolated Daytona containers
- **Pre-built images**: Create deployments from Dockerfiles with agent dependencies
- **Per-seed execution**: Each rollout gets a fresh sandbox instance
- **Centralized traces**: All LLM calls go through the interceptor for training data

## Quick Start

### 1. Create a Deployment

First, create a Harbor deployment with the EngineBench runner:

```bash
export SYNTH_API_KEY=sk_live_...

# Create a single deployment
uv run python demos/harbor/create_deployment.py \
    --name engine-bench-v1 \
    --agent opencode \
    --wait

# Or create multiple deployments (one per seed variant)
uv run python demos/harbor/create_deployment.py \
    --name engine-bench \
    --count 5 \
    --wait
```

This will:
1. Package the Dockerfile and run_rollout.py script
2. Upload to the Harbor API
3. Build a Daytona snapshot (takes ~5 minutes)
4. Return the deployment ID(s)

### 2. Run Evaluations

Once deployments are ready, run evaluations:

```bash
# Run 5 seeds against a single deployment
uv run python demos/harbor/run_harbor_eval.py \
    --deployment-id <deployment-id> \
    --seeds 5 \
    --model gpt-4.1-mini \
    --output results/harbor_eval.json

# Run with multiple deployments (parallel per-seed images)
uv run python demos/harbor/run_harbor_eval.py \
    --deployment-ids <id1>,<id2>,<id3>,<id4>,<id5> \
    --seeds 5 \
    --concurrency 5
```

### 3. Review Results

Results include:
- Mean reward across all seeds
- Per-seed breakdown (compilation, tests passed)
- Trace correlation IDs for debugging

```json
{
  "summary": {
    "total_seeds": 5,
    "successful": 4,
    "failed": 1,
    "mean_reward": 0.65
  },
  "results": [
    {
      "seed": 0,
      "result": {
        "metrics": {
          "reward_mean": 0.85,
          "details": {
            "task_id": "charizard",
            "compilation": true,
            "tests_passed": 8,
            "tests_total": 10
          }
        }
      }
    }
  ]
}
```

## Architecture

```
┌─────────────────┐
│  SDK Client     │
│  (this demo)    │
└────────┬────────┘
         │ POST /api/harbor/rollout
         ▼
┌─────────────────┐
│  Harbor API     │
│  (backend)      │
└────────┬────────┘
         │ maps deployment_id → snapshot_id
         ▼
┌─────────────────┐
│  Daytona        │
│  Sandbox        │
└────────┬────────┘
         │ runs run_rollout.py
         ▼
┌─────────────────┐     LLM calls via inference_url
│  Runner Script  │ ──────────────────────────────►
│  (in sandbox)   │     (interceptor captures traces)
└─────────────────┘
```

## Runner Contract

The runner script follows the Harbor contract:

**Input** (`/tmp/rollout.json`):
```json
{
  "trace_correlation_id": "trace_abc123",
  "seed": 42,
  "prompt_template": {"sections": [...]},
  "inference_url": "https://api.usesynth.ai/api/inference/v1/...",
  "limits": {"timeout_s": 600}
}
```

**Output** (`/tmp/result.json`):
```json
{
  "trace_correlation_id": "trace_abc123",
  "metrics": {
    "reward_mean": 0.85,
    "details": {"passed": 8, "total": 10}
  },
  "success": true
}
```

## Files

- `Dockerfile` - Container image with Rust, Node.js, OpenCode/Codex
- `run_rollout.py` - Runner script that executes inside the sandbox
- `create_deployment.py` - Creates Harbor deployments
- `run_harbor_eval.py` - Runs evaluations via Harbor API

## Environment Variables

| Variable | Description |
|----------|-------------|
| `SYNTH_API_KEY` | Synth API key (required) |

Note: No Daytona or OpenAI API keys needed - Harbor handles all infrastructure.

## Comparison to Local Execution

| Aspect | Local (TunneledLocalAPI) | Harbor |
|--------|-------------------------|--------|
| Daytona keys | Required | Not needed |
| Tunnel setup | Required | Not needed |
| Sandbox lifecycle | User manages | Harbor manages |
| Parallelism | Limited by local | Scale with deployments |
| Traces | Via interceptor | Via interceptor |

## Troubleshooting

### Deployment stuck in "building"
- Check build logs via `GET /api/harbor/deployments/{id}/status`
- Dockerfile issues usually cause build failures

### Rollouts timing out
- Increase `--timeout` (default 600s)
- Check sandbox resource limits in deployment config

### 429 Rate Limited
- Harbor enforces capacity limits
- Use `--concurrency 1` or wait and retry
