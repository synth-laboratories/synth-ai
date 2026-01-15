---
name: synth-scan
description: Discover local task apps and Cloudflare tunnels
---

# Discovering Task Apps with Synth AI

Use `synth-ai scan` to find running task apps on your local machine.

## Basic Usage

```bash
# Scan default port range (8000-8100)
synth-ai scan

# Scan specific port range
synth-ai scan --port-range 8000-8200

# Output as JSON for scripting
synth-ai scan --json

# Verbose output
synth-ai scan --verbose
```

## What It Finds

The scanner looks for:
1. **Local apps**: Services running on localhost ports
2. **Cloudflare tunnels**: Apps exposed via `*.tunnel.usesynth.ai`
3. **Health status**: Whether the app responds to health checks
4. **Task metadata**: Name, version, and other info from `/info` endpoint

## Task App Requirements

For an app to be discovered, it should expose:

```
GET /health
  → Returns 200 OK if healthy

GET /info
  → Returns JSON:
    {
      "name": "my_task",
      "version": "1.0.0",
      "description": "Task description"
    }
```

## Example Output

```
Found 2 task apps:

  http://localhost:8000
    ✓ Healthy
    Name: sentiment-analyzer
    Version: 1.2.0

  https://my-app.tunnel.usesynth.ai
    ✓ Healthy (via tunnel)
    Name: code-reviewer
    Version: 0.5.0
```

## Using with Eval/Train

Once you find an app, use its URL in your config:

```toml
[task]
name = "sentiment-analyzer"
app_url = "http://localhost:8000"  # From scan results
```

Or with the `--app-url` flag:

```bash
synth-ai eval --config config.toml --app-url http://localhost:8000
```

## Tips

- Run scan before starting eval/train to verify your app is reachable
- Use JSON output (`--json`) for scripting and automation
- Tunnels provide more reliable connectivity for longer jobs

