# EngineBench Demo

EngineBench evaluates coding agents on Pokemon TCG card implementations in Rust. This demo runs evaluations using Daytona cloud sandboxes for isolated, parallel execution.

## Prerequisites

1. **Environment variables** (in `.env` or exported):
   ```bash
   OPENAI_API_KEY=sk-...          # OpenAI API key
   DAYTONA_API_KEY=dtn_...        # Daytona sandbox API key
   SYNTH_API_KEY=sk_live_...      # Synth API key
   ```

2. **Install dependencies**:
   ```bash
   cd synth-ai
   uv sync
   ```

3. **Install cloudflared** (for production mode):
   ```bash
   # macOS
   brew install cloudflare/cloudflare/cloudflared

   # Linux
   curl -L https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64 -o cloudflared
   chmod +x cloudflared && sudo mv cloudflared /usr/local/bin/
   ```

## Understanding the Architecture

The demo involves three components that need to communicate:

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│  Synth Backend  │────▶│   Task App       │────▶│ Daytona Sandbox │
│ (api.usesynth.ai)     │ (your machine)   │     │ (cloud VM)      │
└─────────────────┘     └──────────────────┘     └─────────────────┘
        │                        │                       │
        │                        │                       │
        ▼                        ▼                       ▼
   Submits job            Runs rollouts           Executes agent
   Stores results         Returns rewards         Makes LLM calls
```

**Key insight**: In production mode, the Synth backend needs to reach your local task app via HTTP. This requires a **tunnel** to expose your local server to the internet.

## Running Modes

### Mode 1: Local Mode (Recommended for Development)

Uses your local backend - no tunnel needed. The Daytona sandbox still needs to reach the interceptor, so we use `INTERCEPTOR_TUNNEL_URL`.

**Step 1**: Start the local backend
```bash
cd monorepo
./scripts/run_backend_local.sh
```

**Step 2**: Start a tunnel for the interceptor (so Daytona can reach it)
```bash
cloudflared tunnel --url http://localhost:8000
# Note the URL: https://xxx-xxx-xxx.trycloudflare.com
```

**Step 3**: Run the eval
```bash
cd synth-ai
INTERCEPTOR_TUNNEL_URL=https://xxx-xxx-xxx.trycloudflare.com \
USE_DAYTONA_SANDBOXES=1 \
uv run python demos/engine_bench/run_eval.py \
  --local --seeds 1 --model gpt-4o-mini --agent opencode --timeout 180
```

### Mode 2: Production Mode with Synth Managed Tunnel (Recommended)

Synth provides managed Cloudflare tunnels with no timeout limits. This is the most reliable approach.

**Step 1**: Check for existing tunnel or provision one
```bash
# List existing tunnels
curl -s https://api.usesynth.ai/api/v1/tunnels/ \
  -H "Authorization: Bearer $SYNTH_API_KEY" | python3 -m json.tool

# If no tunnel exists, create one:
curl -s -X POST https://api.usesynth.ai/api/v1/tunnels/ \
  -H "Authorization: Bearer $SYNTH_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"subdomain":"my-task-app","local_port":8001,"local_host":"127.0.0.1"}'
```

**Step 2**: Start cloudflared with the tunnel token
```bash
# Get the tunnel_token from the API response, then:
cloudflared tunnel run --token "YOUR_TUNNEL_TOKEN"
```

**Step 3**: Run the eval using the managed tunnel
```bash
cd synth-ai
INTERCEPTOR_TUNNEL_URL=https://api.usesynth.ai \
USE_DAYTONA_SANDBOXES=1 \
uv run python demos/engine_bench/run_eval.py \
  --seeds 1 --model gpt-4o-mini --agent opencode --timeout 180 \
  --port 8001 \
  --task-app-url https://YOUR-SUBDOMAIN.usesynth.ai
```

### Mode 3: Production Mode with Quick Tunnel (Simple but Limited)

Cloudflare quick tunnels are free and require no setup, but have a **~100 second timeout**. If your rollouts take longer, the request will fail with HTTP 524.

**Step 1**: Start quick tunnel for task app
```bash
cloudflared tunnel --url http://localhost:8019
# Note the URL: https://xxx-xxx-xxx.trycloudflare.com
```

**Step 2**: Run the eval
```bash
cd synth-ai
INTERCEPTOR_TUNNEL_URL=https://api.usesynth.ai \
USE_DAYTONA_SANDBOXES=1 \
uv run python demos/engine_bench/run_eval.py \
  --seeds 1 --model gpt-4o-mini --agent opencode --timeout 180 \
  --port 8019 \
  --task-app-url https://xxx-xxx-xxx.trycloudflare.com
```

**WARNING**: Quick tunnels timeout after ~100 seconds. EngineBench rollouts typically take 90-120 seconds, so you may see `HTTP 524: A timeout occurred` errors. Use a managed tunnel for reliability.

## Agent Options

| Agent | Description |
|-------|-------------|
| `codex` | OpenAI Codex CLI (`@openai/codex`) |
| `opencode` | OpenCode CLI (`opencode.ai`) |

## Model Options

Any OpenAI model works. Recommended:
- `gpt-4o-mini` - Fast, cheap, good for testing
- `gpt-4o` - Better quality
- `codex-5.1-mini` - Codex-optimized model

## How It Works

1. **Job submission**: `run_eval.py` submits an eval job to the Synth backend
2. **Task app**: Backend calls your local task app's `/rollout` endpoint (via tunnel)
3. **Sandbox provisioning**: Task app provisions a Daytona sandbox from snapshot (~3s)
4. **Agent execution**: Agent (codex/opencode) implements Pokemon TCG card logic in Rust
5. **LLM interception**: Agent's LLM calls route through Synth interceptor (via `INTERCEPTOR_TUNNEL_URL`)
6. **Test evaluation**: `cargo test` runs to verify the implementation
7. **Results**: Task app returns `outcome_reward` (test pass rate) to backend

## Environment Variables Reference

| Variable | Required | Description |
|----------|----------|-------------|
| `OPENAI_API_KEY` | Yes | OpenAI API key for agent LLM calls |
| `DAYTONA_API_KEY` | Yes | Daytona API key for sandbox provisioning |
| `SYNTH_API_KEY` | Yes | Synth API key for backend auth |
| `USE_DAYTONA_SANDBOXES` | Yes | Set to `1` to enable Daytona sandboxes |
| `INTERCEPTOR_TUNNEL_URL` | Prod | URL where Daytona sandbox can reach the interceptor |

## CLI Arguments Reference

| Argument | Default | Description |
|----------|---------|-------------|
| `--local` | - | Use local backend (localhost:8000) |
| `--seeds N` | 1 | Number of random instances to evaluate |
| `--model` | gpt-4o-mini | Model for agent to use |
| `--agent` | opencode | Agent type: `codex` or `opencode` |
| `--timeout` | 180 | Timeout per rollout in seconds |
| `--port` | 8017 | Port for local task app |
| `--task-app-url` | - | Public URL for task app (tunnel URL) |

## Daytona Snapshot

The snapshot `synth-engine-bench-codex-v1` includes:
- Python 3.11 + Rust/cargo
- Codex CLI and OpenCode CLI
- Pre-cloned engine-bench repo with pre-compiled deps
- synth-ai SDK

To rebuild the snapshot:
```bash
USE_DAYTONA_SANDBOXES=1 uv run python demos/engine_bench/create_daytona_snapshot.py
```

## Troubleshooting

### "DAYTONA_API_KEY required"
```bash
export DAYTONA_API_KEY=dtn_...
```

### HTTP 524: A timeout occurred
This means your cloudflare quick tunnel timed out (~100s limit). Solutions:
1. Use a **Synth managed tunnel** (no timeout limit)
2. Use **local mode** with tunneled interceptor
3. Use a faster model or simpler instances

### Trace hydration failed
The Daytona sandbox couldn't reach the interceptor. Make sure:
1. `INTERCEPTOR_TUNNEL_URL` is set correctly
2. The tunnel is still running
3. For local mode: tunnel points to localhost:8000
4. For prod mode: use `https://api.usesynth.ai`

### Sandbox provisioning slow
The snapshot should provision in ~2-3s. If slower, the snapshot may need rebuilding or Daytona may be overloaded.

### reward = 0.00 but tests passed
This typically means trace hydration failed. Check:
1. Interceptor URL is correctly configured
2. The correlation_id matches between task app and backend

## Complete Example: Production with Managed Tunnel

```bash
# 1. Set environment
export OPENAI_API_KEY=sk-...
export DAYTONA_API_KEY=dtn_...
export SYNTH_API_KEY=sk_live_...

# 2. Get your managed tunnel
curl -s https://api.usesynth.ai/api/v1/tunnels/ \
  -H "Authorization: Bearer $SYNTH_API_KEY"
# Note: hostname, tunnel_token, and local_port

# 3. Start cloudflared with your tunnel token
cloudflared tunnel run --token "eyJhI..." &

# 4. Run the eval
cd synth-ai
INTERCEPTOR_TUNNEL_URL=https://api.usesynth.ai \
USE_DAYTONA_SANDBOXES=1 \
uv run python demos/engine_bench/run_eval.py \
  --seeds 3 \
  --model gpt-4o-mini \
  --agent opencode \
  --timeout 180 \
  --port 8001 \
  --task-app-url https://task-8001-XXXXX.usesynth.ai

# Expected output:
# Job submitted: eval_xxxx
# [DaytonaRollout] Complete in 95.2s: 10/10 tests
# Status: EvalStatus.COMPLETED
# Mean reward: 1.0
```

## Related Documentation

- [EngineBench Repository](https://github.com/JoshuaPurtell/engine-bench)
- [Synth Documentation](https://docs.usesynth.ai)
- [Daytona SDK](https://www.daytona.io/docs)
