# Hello World Bench - GEPA Demo

This demo runs GEPA (Genetic Evolutionary Prompt Augmentation) optimization on a simple "Hello, World!" task using different AI coding agents.

## Task

The task is simple: edit `output.txt` to contain exactly `Hello, world!`

This serves as a minimal test case for verifying GEPA infrastructure works correctly with different agent types.

## Supported Agents

| Agent | API Format | Description |
|-------|------------|-------------|
| **OpenCode** | OpenAI Responses API | Open-source coding agent using `input[]` format |
| **Codex CLI** | OpenAI Responses API | OpenAI's Codex CLI agent |

## Prerequisites

1. **Backend running locally**:
   ```bash
   cd /path/to/monorepo
   bash scripts/run_backend_local.sh
   ```

2. **Redis running** (for trace storage):
   ```bash
   redis-server
   ```

3. **OpenAI API key** set in environment or backend config

## Running GEPA

### Basic Usage

```bash
cd /path/to/synth-ai

# Run with OpenCode (default)
uv run python demos/hello_world_bench/run_gepa_minimal.py --local --model gpt-4.1-mini

# Run with Codex CLI
uv run python demos/hello_world_bench/run_gepa_minimal.py --local --model gpt-5.1-codex-mini --agent codex
```

### Command Line Options

| Option | Default | Description |
|--------|---------|-------------|
| `--local` | Required | Run against local backend (localhost:8000) |
| `--model` | Required | Model to use (e.g., `gpt-4.1-mini`, `gpt-5.1-codex-mini`) |
| `--agent` | `opencode` | Agent type: `opencode` or `codex` |
| `--generations` | From config | Number of GEPA generations |
| `--budget` | From config | Rollout budget per generation |
| `--timeout` | From config | Timeout per rollout (seconds) |

### Example Commands

```bash
# OpenCode with gpt-4.1-mini (1 generation, 10 rollouts)
uv run python demos/hello_world_bench/run_gepa_minimal.py \
  --local \
  --model gpt-4.1-mini \
  --generations 1 \
  --budget 10 \
  --timeout 90

# Codex CLI with gpt-5.1-codex-mini
uv run python demos/hello_world_bench/run_gepa_minimal.py \
  --local \
  --model gpt-5.1-codex-mini \
  --generations 1 \
  --budget 10 \
  --timeout 120 \
  --agent codex
```

## Supported Models

### OpenAI Models
- `gpt-4o`, `gpt-4o-mini`
- `gpt-4.1`, `gpt-4.1-mini`, `gpt-4.1-nano`
- `gpt-5`, `gpt-5-mini`, `gpt-5-nano`
- `gpt-5.1-codex-mini`, `gpt-5.1-codex-max`
- `gpt-5.2`

## Config Files

| Agent | Config File |
|-------|-------------|
| OpenCode | `hello_world_gepa_minimal.toml` |
| Codex | `hello_world_gepa_codex.toml` |

The script auto-selects the correct config based on `--agent`.

## Expected Output

Successful run shows:
```
Job ID: pl_xxxxxxxxxxxxx
[00:00] queued | score: --
[00:30] running | score: --
...
✅ Baseline evaluation complete: minibatch=1.000 (4 seeds), pareto=1.000 (10 seeds)
[PROPOSAL] Generated in 9.00s (evaluation pending) | reasoning=0 | output=505
...
[XX:XX] succeeded | score: 1.00
Status: JobStatus.SUCCEEDED
```

## Troubleshooting

### Pattern Validation Failed
If you see `Pattern validation failed`, check:
1. The TOML config patterns match the actual message format
2. For Codex: uses 3 user messages (AGENTS.md, environment_context, task)
3. For OpenCode: uses 4 messages (system, AGENTS.md, environment, task)

### Trace Not Found
If you see `No trace registration for correlation_id`:
1. Ensure backend is running and healthy: `curl http://localhost:8000/health`
2. Check Redis is running: `redis-cli ping`
3. Verify `OPENAI_BASE_URL` is being set correctly for the agent

### stream_options Error (Codex)
The `Unknown parameter: 'stream_options.include_usage'` error was fixed - the interceptor now only adds this parameter for Chat Completions, not Responses API.

## Architecture

```
┌─────────────────┐     ┌──────────────┐     ┌─────────────────┐
│  run_gepa_      │────▶│   Backend    │────▶│  OpenAI API     │
│  minimal.py     │     │  (localhost  │     │  (/responses)   │
└─────────────────┘     │   :8000)     │     └─────────────────┘
        │               └──────────────┘
        │                      │
        ▼                      ▼
┌─────────────────┐     ┌──────────────┐
│  Task App       │     │  Interceptor │
│  (localhost     │     │  (captures   │
│   :8030)        │     │   traces)    │
└─────────────────┘     └──────────────┘
        │
        ▼
┌─────────────────┐
│  Agent          │
│  (OpenCode/     │
│   Codex CLI)    │
└─────────────────┘
```

## Context Overrides (NEW)

The task app now supports **context overrides** for unified optimization:

```python
# Example override in TOML config
[prompt_learning.policy.context_override]
agents_md = "# Custom instructions..."

[prompt_learning.policy.context_override.file_artifacts]
".codex/skills.yaml" = "style:\n  verbosity: concise"

preflight_script = """#!/bin/bash
echo "Setup complete"
"""

[prompt_learning.policy.context_override.env_vars]
STRATEGY = "read_then_write"
```

### Agent-Specific Skills Paths

| Agent | Workspace Path | Global Path (opt-in) |
|-------|----------------|---------------------|
| Codex | `.codex/skills.yaml` | `~/.codex/skills.yaml` |
| OpenCode | `.opencode/skills.yaml` | `~/.opencode/skills.yaml` |

The override applicator writes workspace-local by default (safe). Global writes require explicit opt-in.

### New SDK Types

```python
from synth_ai.data.artifacts import ContextOverride, ContextOverrideStatus
from synth_ai.sdk.task.override_helpers import apply_context_overrides, AgentType
```

## Key Fixes Made

1. **Truncation removed** - `max_string` set to 100M in `_sanitize_trace_payload` to preserve full message content for pattern matching

2. **OPENAI_BASE_URL** - Task app now sets this env var for agents to ensure LLM calls route through interceptor

3. **stream_options fix** - Only add `stream_options.include_usage` for `/chat/completions`, not `/responses`

4. **None content handling** - Interceptor now handles `content: null` in Responses API messages

5. **Codex config patterns** - Fixed to match actual Codex CLI message format (3 user messages, no "instructions" role)

6. **Context override plumbing** - `RolloutRequest.context_overrides` and `RolloutResponse.override_application_results` now properly flow through the SDK contracts