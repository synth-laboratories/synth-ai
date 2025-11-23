# Zig Task App Example

A minimal but complete Task App implementation in Zig for Synth prompt optimization.

**Requires Zig 0.15+** (uses new HTTP client and I/O APIs)

## Features

- Zero external dependencies (uses only Zig standard library)
- Single static binary (~1MB optimized)
- Cross-compilation to any target
- Implements `/health`, `/task_info`, and `/rollout` endpoints per OpenAPI contract
- Embedded sample dataset (Banking77 - 12 samples matching shared `banking77.json`)
- Multiple `seed` query parameter support for `/task_info`

## Quick Start

```bash
# Build (debug)
zig build

# Build (release - optimized)
zig build -Doptimize=ReleaseFast

# Run
./zig-out/bin/synth-task-app

# With authentication
ENVIRONMENT_API_KEY=your-secret ./zig-out/bin/synth-task-app

# Custom port
PORT=3000 ./zig-out/bin/synth-task-app
```

## Cross-Compilation

Zig makes cross-compilation trivial:

```bash
# Linux (static musl)
zig build -Doptimize=ReleaseFast -Dtarget=x86_64-linux-musl

# Linux ARM64
zig build -Doptimize=ReleaseFast -Dtarget=aarch64-linux-musl

# macOS ARM64
zig build -Doptimize=ReleaseFast -Dtarget=aarch64-macos

# Windows
zig build -Doptimize=ReleaseFast -Dtarget=x86_64-windows
```

## Testing

```bash
# Run unit tests
zig build test

# Health check
curl http://localhost:8001/health

# Task info (all seeds)
curl -H "X-API-Key: your-secret" http://localhost:8001/task_info

# Task info (specific seeds)
curl -H "X-API-Key: your-secret" "http://localhost:8001/task_info?seed=0&seed=1&seed=2"

# Manual rollout
curl -X POST http://localhost:8001/rollout \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your-secret" \
  -d '{
    "run_id": "test-1",
    "env": {"seed": 0},
    "policy": {
      "config": {
        "inference_url": "https://your-llm-proxy/v1?model=gpt-4o-mini"
      }
    }
  }'
```

## Critical Implementation Details

1. **URL Construction**: The `inference_url` may contain query params (e.g., `?model=x`). When appending `/chat/completions`, preserve the query:
   - Input: `https://api.example.com/v1?model=gpt-4o-mini`
   - Output: `https://api.example.com/v1/chat/completions?model=gpt-4o-mini`

2. **Multiple Seed Parameters**: The `/task_info` endpoint must handle multiple `seed` query params:
   - `/task_info?seed=0&seed=1` → returns TaskInfo for seeds 0 and 1
   - `/task_info` (no seeds) → returns TaskInfo with all available seeds

3. **Dataset Alignment**: The embedded samples must match `../data/banking77.json` exactly for MIPRO to work correctly.

## Running with Synth Optimizer

1. **Build and run:**
   ```bash
   zig build -Doptimize=ReleaseFast
   ENVIRONMENT_API_KEY=my-secret ./zig-out/bin/synth-task-app
   ```

2. **Expose via Cloudflare tunnel:**
   ```bash
   cloudflared tunnel --url http://localhost:8001
   ```

3. **Start optimization:**
   ```bash
   curl -X POST https://api.usesynth.ai/api/prompt-learning/online/jobs \
     -H "Authorization: Bearer $SYNTH_API_KEY" \
     -H "Content-Type: application/json" \
     -d '{
       "algorithm": "mipro",
       "config_body": {
         "prompt_learning": {
           "task_app_url": "https://your-tunnel.trycloudflare.com",
           "task_app_api_key": "my-secret"
         }
       }
     }'
   ```

## Why Zig?

- **No runtime dependencies** - Produces fully static binaries
- **Trivial cross-compilation** - Build for any target from any host
- **Small binaries** - ReleaseFast builds are typically 2-5MB
- **No garbage collection** - Predictable, low latency
- **C interop** - Easy to integrate existing C libraries if needed

## Contract Reference

See [`synth_ai/contracts/task_app.yaml`](../../../synth_ai/contracts/task_app.yaml) for the full OpenAPI specification.

## Project Structure

```
zig/
├── build.zig       # Build configuration
├── src/
│   └── main.zig    # Task app implementation
└── README.md
```
