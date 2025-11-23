# Go Task App Example

A minimal but complete Task App implementation in Go for Synth prompt optimization.
**Tested end-to-end with MIPRO optimizer** - achieves 100% accuracy on Banking77 classification.

## Features

- Zero external dependencies (uses only Go standard library)
- Single static binary
- Cross-compilation built-in
- Implements `/health`, `/task_info`, and `/rollout` endpoints per OpenAPI contract
- Loads dataset from shared JSON file (with embedded fallback)
- Proper URL construction with query parameter handling

## Quick Start

```bash
# Build
go build -o synth-task-app

# Run
./synth-task-app

# With authentication
ENVIRONMENT_API_KEY=your-secret ./synth-task-app

# Custom port
PORT=3000 ./synth-task-app

# Or run directly
go run main.go
```

## Cross-Compilation

Go makes cross-compilation simple:

```bash
# Linux AMD64
GOOS=linux GOARCH=amd64 go build -o synth-task-app-linux-amd64

# Linux ARM64
GOOS=linux GOARCH=arm64 go build -o synth-task-app-linux-arm64

# macOS ARM64 (Apple Silicon)
GOOS=darwin GOARCH=arm64 go build -o synth-task-app-macos-arm64

# Windows
GOOS=windows GOARCH=amd64 go build -o synth-task-app.exe
```

## Testing

```bash
# Health check
curl http://localhost:8001/health

# Manual rollout
curl -X POST http://localhost:8001/rollout \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your-secret" \
  -d '{
    "run_id": "test-1",
    "env": {"seed": 0},
    "policy": {
      "config": {
        "model": "gpt-4o-mini",
        "inference_url": "https://api.openai.com/v1"
      }
    },
    "mode": "eval"
  }'
```

## Running with Synth Optimizer

### Local Development (Recommended for Testing)

1. **Start the local backend** (from monorepo):
   ```bash
   cd monorepo && bash scripts/run_backend_local.sh
   # Starts: Redis, sqld, uvicorn on port 8000
   ```

2. **Start the task app:**
   ```bash
   go build -o synth-task-app
   ENVIRONMENT_API_KEY=test-polyglot-key ./synth-task-app
   ```

3. **Submit a job** using the example config:
   ```bash
   curl -X POST "http://localhost:8000/api/prompt-learning/online/jobs" \
     -H "Authorization: Bearer $SYNTH_API_KEY" \
     -H "Content-Type: application/json" \
     -d @../mipro_job.json
   ```

### Production (via Cloudflare Tunnel)

1. **Build and run:**
   ```bash
   go build -o synth-task-app
   ENVIRONMENT_API_KEY=my-secret ./synth-task-app
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

## Critical Implementation Details

### URL Construction with Query Parameters

The `inference_url` provided by the optimizer includes query parameters for tracing:
```
http://localhost:8000/api/interceptor/v1/trial-id?cid=trace_xxx
```

When appending `/chat/completions`, the path must come BEFORE the query string:

```go
// CORRECT: path before query
var url string
if queryIdx := strings.Index(inferenceURL, "?"); queryIdx != -1 {
    base := strings.TrimSuffix(inferenceURL[:queryIdx], "/")
    query := inferenceURL[queryIdx:]
    url = base + "/chat/completions" + query
} else {
    url = strings.TrimSuffix(inferenceURL, "/") + "/chat/completions"
}

// Result: http://host/path/chat/completions?cid=xxx
```

**Wrong approach** (causes 404):
```go
// WRONG: appends path after query string
url := inferenceURL + "/chat/completions"
// Result: http://host/path?cid=xxx/chat/completions  <-- 404!
```

### Handling Multiple Seeds in /task_info

The backend sends seed parameters as repeated keys: `?seeds=0&seeds=1&seeds=2`.
Parse both `seed` and `seeds` variants:

```go
var requestedSeeds []int
query := r.URL.RawQuery
for _, param := range strings.Split(query, "&") {
    parts := strings.SplitN(param, "=", 2)
    if len(parts) == 2 && (parts[0] == "seed" || parts[0] == "seeds") {
        if seed, err := strconv.Atoi(parts[1]); err == nil {
            requestedSeeds = append(requestedSeeds, seed)
        }
    }
}
```

## Why Go?

- **No external dependencies** - Uses only the standard library
- **Single static binary** - Easy deployment
- **Fast compilation** - Quick iteration
- **Built-in cross-compilation** - Deploy anywhere
- **Excellent concurrency** - Handles many requests efficiently

## Project Structure

```
go/
├── go.mod      # Module definition
├── main.go     # Task app implementation
└── README.md
```

## Contract Reference

See [`synth_ai/contracts/task_app.yaml`](../../../synth_ai/contracts/task_app.yaml) for the full OpenAPI specification.
