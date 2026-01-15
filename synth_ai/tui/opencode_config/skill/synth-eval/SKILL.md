---
name: synth-eval
description: Run evaluations against task apps using Synth AI platform
---

# Running Evaluations with Synth AI

Use `synth-ai eval` to run evaluations against your task apps.

## Basic Usage

```bash
# Run eval with a config file
synth-ai eval --config eval_config.toml

# With polling for real-time results (recommended)
synth-ai eval --config eval_config.toml --poll

# Override task app URL
synth-ai eval --config eval_config.toml --app-url http://localhost:8000 --poll
```

## Config File Format

Create an `eval_config.toml` file:

```toml
[task]
name = "my_task"
app_url = "http://localhost:8000"  # Your task app endpoint

[eval]
num_samples = 100                   # Number of eval samples
dataset = "default"                 # Dataset to use
```

## Task App Requirements

Your task app must expose:
- `GET /health` - Health check (returns 200 if healthy)
- `GET /info` - Task metadata JSON
- `POST /run` - Execute the task

## Monitoring Progress

1. Use the `--poll` flag to see real-time progress
2. Check the TUI Jobs panel (press 'b' to view jobs)
3. View job events with 'e' key

## Example Workflow

```bash
# 1. Start your task app
python -m your_app serve --port 8000

# 2. Verify it's running
synth-ai scan

# 3. Run evaluation
synth-ai eval --config eval_config.toml --poll

# 4. View results in TUI or via API
```

## Tips

- Start with a small num_samples (10-20) to verify setup
- Use tunnels for more reliable connectivity
- Check job events for detailed error messages

