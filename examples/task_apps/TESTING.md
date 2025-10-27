# Task App Testing Guide

This document describes how to run tests for the task apps in this directory.

## Overview

Each task app has unit and integration tests following a consistent pattern inspired by the customer environment tests in `customers/`.

## Test Structure

```
examples/task_apps/<app_name>/tests/
├── __init__.py
├── integration/
│   ├── __init__.py
│   └── test_<app>_eval.py      # Server startup + eval tests
└── unit/
    ├── __init__.py
    └── test_<app>_*.py          # Environment, scoring, dataset tests
```

## Running Tests

### Prerequisites

```bash
# Install test dependencies
uv sync --dev

# Set required environment variables
export GROQ_API_KEY="your-groq-key"
export OPENAI_API_KEY="your-openai-key"  # For Sokoban
```

### Run All Tests for a Task App

```bash
# Verilog
pytest examples/task_apps/verilog/tests/ -v

# Enron
pytest examples/task_apps/enron/tests/ -v

# Sokoban
pytest examples/task_apps/sokoban/tests/ -v
```

### Run Only Unit Tests (Fast)

```bash
# Runs quickly, no server startup required
pytest examples/task_apps/verilog/tests/unit/ -v
pytest examples/task_apps/enron/tests/unit/ -v
pytest examples/task_apps/sokoban/tests/unit/ -v
```

### Run Only Integration Tests

```bash
# Slower, starts servers and runs evals
pytest examples/task_apps/verilog/tests/integration/ -v
pytest examples/task_apps/enron/tests/integration/ -v
pytest examples/task_apps/sokoban/tests/integration/ -v
```

### Run All Task App Tests

```bash
# Run everything
pytest examples/task_apps/*/tests/ -v

# Skip slow tests
pytest examples/task_apps/*/tests/ -v -m "not slow"
```

## Test Categories

### Unit Tests

**Purpose**: Test individual components in isolation
- Environment initialization
- Reward calculation
- Tool implementations
- State management

**Characteristics**:
- Fast (< 1 second each)
- No external dependencies
- No server startup
- No API calls

**Examples**:
- `test_verilog_scoring.py`: Tests reward components (compile, simulate, submit)
- `test_enron_environment.py`: Tests search, answer, reward calculation
- `test_sokoban_environment.py`: Tests actions, rewards, truncation

### Integration Tests

**Purpose**: Test the full system end-to-end
- Server startup
- Health/info endpoints
- Full evaluation runs
- **Rollout execution** (manual and policy-driven)

**Characteristics**:
- Slower (30-300 seconds)
- Requires server startup
- May require API keys
- Tests real workflows

**Examples**:
- `test_verilog_eval.py`: Starts server, runs Groq eval with Qwen3-32B
- `test_verilog_rollout.py`: **Manual & policy rollouts via /rollout endpoint**
- `test_enron_eval.py`: Starts server, runs Groq eval
- `test_enron_rollout.py`: **Manual & policy rollouts, auth testing**
- `test_sokoban_eval.py`: Starts server, tests manual rollout
- `test_sokoban_rollout.py`: **6 rollout tests (manual, policy, difficulties, limits)**

## What Each Test Validates

### Verilog Tests

**Unit Tests** (4 tests):
- ✅ Compile success gives +0.1 reward
- ✅ Simulation pass gives +1.0 reward
- ✅ Submit success gives +10.0 reward
- ✅ Submit checks last simulation output correctly

**Integration Tests** (5 tests):
- ✅ Server starts and responds to /health
- ✅ /task_info returns valid Verilog task metadata
- ✅ Full eval with Qwen3-32B completes successfully
- ✅ **Manual rollout** with explicit write/compile/simulate/submit
- ✅ **Policy rollout** using Groq/Qwen3-32B (verifies LLM integration)

### Enron Tests

**Unit Tests** (3 tests):
- ✅ search_emails tool works correctly
- ✅ answer_question tool calculates rewards
- ✅ Exact answer match gives high reward (>0.9)
- ✅ Partial answer match gives medium reward (>0.5)
- ✅ Wrong answer gives low reward (<0.5)

**Integration Tests** (6 tests):
- ✅ Server starts and responds to /health
- ✅ /task_info returns valid Enron task metadata
- ✅ Full eval with Qwen3-32B completes successfully
- ✅ **Manual rollout** with explicit search/read/answer actions
- ✅ **Policy rollout** using Groq/Qwen3-32B
- ✅ **Authentication** enforcement (rejects requests without auth header)

### Sokoban Tests

**Unit Tests** (3 tests):
- ✅ Module imports work correctly
- ✅ Reward components exist (goal achieved, step penalty)
- ✅ Engine creation with different difficulty levels

**Integration Tests** (9 tests):
- ✅ Server starts and responds to /health
- ✅ /task_info returns valid Sokoban task metadata
- ✅ **Manual rollout** with movement actions (left/right/up/down)
- ✅ **Policy rollout** with OpenAI GPT-5-mini (may skip if slow)
- ✅ **All difficulty levels** (easy/medium/hard) work correctly
- ✅ **Max steps limit** enforcement (stops at configured limit)
- ✅ **Puzzle completion detection** (terminated=True when solved)
- ✅ Truncation on max_steps
- ✅ Response structure validation

## Debugging Test Failures

### Server Won't Start

```bash
# Check if port is already in use
lsof -i :<port>

# Check logs manually
uv run -m synth_ai task-app serve <app_name> --port 8999

# Check environment variables
echo $GROQ_API_KEY
echo $OPENAI_API_KEY
```

### Tests Timeout

```bash
# Run with more verbose output
pytest <test_file> -v -s

# Skip slow tests
pytest <test_file> -v --timeout=60
```

### Import Errors

```bash
# Ensure you're in the right directory
cd /path/to/synth-ai

# Reinstall dependencies
uv sync --dev
```

## CI/CD Integration

These tests can be run in CI with:

```yaml
# .github/workflows/test-task-apps.yml
- name: Run task app tests
  env:
    GROQ_API_KEY: ${{ secrets.GROQ_API_KEY }}
    OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}
  run: |
    # Unit tests (fast, always run)
    pytest examples/task_apps/*/tests/unit/ -v
    
    # Integration tests (slower, only on main)
    if [ "$GITHUB_REF" = "refs/heads/main" ]; then
      pytest examples/task_apps/*/tests/integration/ -v --timeout=300
    fi
```

## Adding Tests for New Task Apps

When creating a new task app, follow this pattern:

1. **Create test structure**:
   ```bash
   mkdir -p examples/task_apps/<new_app>/tests/{unit,integration}
   touch examples/task_apps/<new_app>/tests/__init__.py
   touch examples/task_apps/<new_app>/tests/unit/__init__.py
   touch examples/task_apps/<new_app>/tests/integration/__init__.py
   ```

2. **Create unit tests** (`tests/unit/test_<app>_*.py`):
   - Test environment initialization
   - Test reward calculation
   - Test tool implementations
   - Test edge cases

3. **Create integration tests** (`tests/integration/test_<app>_eval.py`):
   - Copy from an existing integration test
   - Update app name, port, config path
   - Add app-specific endpoint tests

4. **Add to CI**:
   - Update CI config to include new tests
   - Ensure required env vars are set

## Test Coverage Goals

- Unit test coverage: >80%
- Integration test coverage: 100% of critical paths
- All public APIs have at least one integration test
- All reward components have unit tests

## Common Issues

### "Task app terminated immediately"
- Check that the app name is correct
- Verify the app is registered in `synth_ai/task/apps.py`
- Check recent changes to the app code

### "GROQ_API_KEY must be set"
- Set the environment variable
- Or skip the test: `pytest -k "not groq"`

### "Config file not found"
- Ensure eval config exists in task app directory
- Check the path in the test matches actual location

