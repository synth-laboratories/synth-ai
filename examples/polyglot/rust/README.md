# Rust Task App Example

A minimal but complete Task App implementation in Rust for Synth prompt optimization.
**Tested end-to-end with MIPRO optimizer** - achieves 100% accuracy on Banking77 classification.

## Features

- Implements `/health`, `/task_info`, and `/rollout` endpoints per OpenAPI contract
- Embedded sample dataset (Banking77-style)
- Prompt template rendering with `{placeholder}` substitution
- LLM calls via `inference_url` (with proper query parameter handling)
- Tool-based classification with reward computation

## Quick Start

```bash
# Build and run
cargo run --release

# With authentication (recommended)
ENVIRONMENT_API_KEY=your-secret-key cargo run --release

# Custom port
PORT=3000 cargo run --release
```

## Testing Locally

```bash
# Health check
curl http://localhost:8001/health

# Manual rollout (requires running optimizer or mock inference)
curl -X POST http://localhost:8001/rollout \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your-secret-key" \
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
   cd monorepo && ./scripts/run_backend_local.sh
   # Starts: Redis, sqld, uvicorn on port 8000
   ```

2. **Start the task app:**
   ```bash
   ENVIRONMENT_API_KEY=test-polyglot-key cargo run --release
   ```

3. **Submit a job** using the example config:
   ```bash
   curl -X POST "http://localhost:8000/api/prompt-learning/online/jobs" \
     -H "Authorization: Bearer $SYNTH_API_KEY" \
     -H "Content-Type: application/json" \
     -d @../mipro_job.json
   ```

4. **Monitor job:**
   ```bash
   curl "http://localhost:8000/api/prompt-learning/online/jobs/{job_id}" \
     -H "Authorization: Bearer $SYNTH_API_KEY"
   ```

### Production (via Cloudflare Tunnel)

1. **Start the task app:**
   ```bash
   ENVIRONMENT_API_KEY=my-secret cargo run --release
   ```

2. **Expose via Cloudflare tunnel:**
   ```bash
   cloudflared tunnel --url http://localhost:8001
   # Note the URL: https://random-words.trycloudflare.com
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
           "task_app_url": "https://random-words.trycloudflare.com",
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

```rust
// CORRECT: path before query
let url = if let Some(query_start) = inference_url.find('?') {
    let (base, query) = inference_url.split_at(query_start);
    format!("{}/chat/completions{}", base.trim_end_matches('/'), query)
} else {
    format!("{}/chat/completions", inference_url.trim_end_matches('/'))
};

// Result: http://host/path/chat/completions?cid=xxx
```

**Wrong approach** (causes 404):
```rust
// WRONG: appends path after query string
let url = format!("{}/chat/completions", inference_url);
// Result: http://host/path?cid=xxx/chat/completions  <-- 404!
```

### Handling Multiple Seeds in /task_info

The backend sends seed parameters as repeated keys: `?seeds=0&seeds=1&seeds=2`.
Parse both `seed` and `seeds` variants:

```rust
let seeds: Vec<i32> = query_string
    .split('&')
    .filter_map(|param| {
        let mut parts = param.split('=');
        match (parts.next(), parts.next()) {
            (Some("seed"), Some(val)) | (Some("seeds"), Some(val)) => val.parse().ok(),
            _ => None,
        }
    })
    .collect();
```

## Contract Reference

See [`synth_ai/contracts/task_app.yaml`](../../../synth_ai/contracts/task_app.yaml) for the full OpenAPI specification.

## Customizing for Your Task

1. **Replace the dataset** - Load your own samples in `Dataset::new()`
2. **Update labels** - Modify the intent/label list
3. **Adjust the tool schema** - Change `classify` to match your output format
4. **Update reward logic** - Modify `extract_prediction()` and reward computation

## Dependencies

- `axum` - Web framework
- `tokio` - Async runtime
- `serde` / `serde_json` - Serialization
- `reqwest` - HTTP client for LLM calls
- `anyhow` - Error handling
- `tracing` - Logging
