# MIPRO Demos

Minimal offline and online [MIPRO](https://arxiv.org/abs/2406.11695) (Multi-prompt Instruction PRoposal Optimizer) examples using the Banking77 classification task.

MIPRO optimizes Language Model (LM) programs by improving free-form instructions and few-shot demonstrations without requiring module-level labels or gradients.

## Quick Start

### Prerequisites

1. Set your API key:
   ```bash
   export SYNTH_API_KEY=sk_live_your_key_here
   ```

2. Ensure you're in the `synth-ai` directory:
   ```bash
   cd /path/to/synth-ai
   ```

### Offline MIPRO Demo

Runs MIPRO optimization in offline/batch mode. The optimizer proposes prompt candidates, evaluates them on a fixed set of seeds, and returns the best performing prompt. The backend orchestrates all rollouts.

```bash
# Against local backend (default)
uv run python demos/mipro/offline_demo.py

# Against dev infrastructure
SYNTH_BACKEND_URL=https://api-dev.usesynth.ai uv run python demos/mipro/offline_demo.py

# With custom rollout count
uv run python demos/mipro/offline_demo.py --rollouts 10
```

**Expected output:**
```
Created job: pl_xxxxxxxxxxxxxxxx
Polling for completion...
  Status: running (elapsed: 3s)
  ...
Final job status:
  Status: succeeded
  Best score: 0.8
```

### Online MIPRO Demo

Runs MIPRO in online mode where **you drive the rollouts locally**. The backend provides a proxy URL that selects prompt candidates for each LLM call. No tunneling or ENVIRONMENT_API_KEY upload required.

```bash
# Against local backend
uv run python demos/mipro/online_demo.py

# Against dev infrastructure
SYNTH_BACKEND_URL=https://api-dev.usesynth.ai uv run python demos/mipro/online_demo.py --rollouts 3
```

**Expected output:**
```
Waiting for local API on port 8016...
Local API URL: http://localhost:8016
Online job: pl_xxxxxxxxxxxxxxxx
System ID: xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx
Proxy URL: https://infra-api-dev.usesynth.ai/api/mipro/v1/...
Rollout 0: reward=1.000 id=trace_rollout_0_abc123 candidate=baseline
Rollout 1: reward=1.000 id=trace_rollout_1_def456 candidate=baseline
...
Online state: best_candidate_id=baseline version=X candidates=N
```

## Online vs Offline MIPRO

| Feature | Online | Offline |
|---------|--------|---------|
| Who runs rollouts | You (locally) | Backend |
| Tunneling needed | No | Yes (for remote backends) |
| ENVIRONMENT_API_KEY | Not needed | Required |
| Use case | Real-time learning, interactive | Batch optimization |

## Environment Variables

### Required

| Variable | Description |
|----------|-------------|
| `SYNTH_API_KEY` | Your Synth API key |

### Backend Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `SYNTH_BACKEND_URL` | SDK default | Backend URL (e.g., `https://api-dev.usesynth.ai`) |

### Model Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `BANKING77_MODEL` | `gpt-4.1-nano` | Model for rollouts |
| `BANKING77_POLICY_MODEL` | `gpt-4.1-nano` | Policy model for evaluation |
| `BANKING77_POLICY_PROVIDER` | `openai` | Policy model provider |
| `BANKING77_PROPOSER_MODEL` | `gpt-4.1-nano` | Proposer model for generating candidates |
| `BANKING77_PROPOSER_PROVIDER` | `openai` | Proposer model provider |

### Offline Mode Only

| Variable | Default | Description |
|----------|---------|-------------|
| `MIPRO_TUNNEL_TASK_APP` | auto | `1` to force tunnel, `0` to disable |
| `MIPRO_TASK_APP_URL` | auto | Override task app URL (skips tunneling) |

## Troubleshooting

### Missing API Key
- Ensure `SYNTH_API_KEY` is exported in your shell

### Offline Mode: DNS Resolution Errors
- Cloudflare quick tunnels are ephemeral; retry if DNS fails
- For local testing: `MIPRO_TUNNEL_TASK_APP=0`
