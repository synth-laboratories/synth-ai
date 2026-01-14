# EngineBench Demo

EngineBench evaluates coding agents on Pokemon TCG card implementations in Rust. This demo runs evaluations using Daytona cloud sandboxes for parallel execution.

## Prerequisites

1. **Environment variables** (in `.env` or exported):
   ```bash
   OPENAI_API_KEY=sk-...          # OpenAI API key
   DAYTONA_API_KEY=dtn_...        # Daytona sandbox API key
   SYNTH_API_KEY=sk_live_...      # Synth API key (optional for local mode)
   ```

2. **Install dependencies**:
   ```bash
   cd synth-ai
   uv sync
   ```

## Running the Demo

### Local Mode (with local backend)

Start the backend first:
```bash
cd monorepo
./scripts/run_backend_local.sh
```

Then run the eval:
```bash
cd synth-ai

# Run with Codex agent (1 seed)
USE_DAYTONA_SANDBOXES=1 uv run python demos/engine_bench/run_eval.py \
  --local --seeds 1 --model gpt-4o-mini --agent codex --timeout 180

# Run with OpenCode agent (1 seed)
USE_DAYTONA_SANDBOXES=1 uv run python demos/engine_bench/run_eval.py \
  --local --seeds 1 --model gpt-4o-mini --agent opencode --timeout 180

# Run with more seeds for better stats
USE_DAYTONA_SANDBOXES=1 uv run python demos/engine_bench/run_eval.py \
  --local --seeds 10 --model gpt-4o-mini --agent codex --timeout 180
```

### Production Mode (with api.usesynth.ai)

```bash
cd synth-ai

# Run with Codex agent
USE_DAYTONA_SANDBOXES=1 uv run python demos/engine_bench/run_eval.py \
  --backend-url https://api.usesynth.ai \
  --seeds 1 --model gpt-4o-mini --agent codex --timeout 180

# Run with OpenCode agent
USE_DAYTONA_SANDBOXES=1 uv run python demos/engine_bench/run_eval.py \
  --backend-url https://api.usesynth.ai \
  --seeds 1 --model gpt-4o-mini --agent opencode --timeout 180
```

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

1. **Sandbox provisioning**: Each rollout gets its own Daytona sandbox from a pre-built snapshot
2. **Agent execution**: The agent (codex or opencode) implements Pokemon TCG card logic in Rust
3. **Test evaluation**: `cargo test` runs to verify the implementation
4. **Trace capture**: LLM calls are routed through the Synth interceptor for trace collection

## Architecture

```
run_eval.py
    │
    ├── Starts local task app (localapi_engine_bench.py)
    │
    ├── Submits eval job to backend
    │
    └── Backend calls task app /rollout endpoint
            │
            └── DaytonaRolloutRunner (daytona_helper.py)
                    │
                    ├── Provisions sandbox from snapshot
                    ├── Sets up instance files
                    ├── Runs agent (codex/opencode)
                    ├── Injects eval tests
                    ├── Runs cargo test
                    └── Returns results
```

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

**"DAYTONA_API_KEY required"**: Export your Daytona API key
```bash
export DAYTONA_API_KEY=dtn_...
```

**Sandbox provisioning slow**: The snapshot should provision in ~2-3s. If slower, the snapshot may need rebuilding.

**Sandbox provisioning slow**: The snapshot should provision in ~2-3s. If slower, the snapshot may need rebuilding.

**Agent timeout**: Increase `--timeout` (default 180s). Complex cards may need more time.

### "opencode: command not found"

Install OpenCode:
```bash
bun install -g opencode
```

### "cargo: command not found"

Install Rust toolchain:
```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source $HOME/.cargo/env
```

### Agent timeout errors

Increase timeout in config:
```toml
[policy.config]
timeout = 600  # 10 minutes
```

## Next Steps

After optimization completes:

1. **Inspect evolved artifacts**: Check how GEPA modified the system prompt and context docs
2. **Analyze Pareto frontier**: See trade-offs between compile success and test pass rate
3. **Test generalization**: Evaluate best candidate on held-out cards (seeds 100-150)
4. **Extract insights**: What patterns did GEPA discover? Can we learn from them?

## Related Documentation

- [EngineBench Repository](https://github.com/JoshuaPurtell/engine-bench)
- [GEPA Algorithm Documentation](https://docs.usesynth.ai/algorithms/gepa)
