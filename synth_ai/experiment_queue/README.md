# Experiment Queue

The experiment queue system allows you to submit multiple prompt learning experiments (GEPA/MIPRO) and have them processed automatically by a background Celery worker. This is useful for batch processing, local development, and managing resource usage.

## Overview

The experiment queue consists of:

- **Queue Worker**: A Celery worker that processes jobs from the queue
- **Beat Scheduler**: Periodically checks for queued jobs and dispatches them (runs every 5 seconds)
- **Database**: SQLite database stores experiment metadata and job status
- **Redis Broker**: Redis handles task queuing and result storage

## Quick Start

### 1. Prerequisites

Install and start Redis:

```bash
# macOS
brew install redis
brew services start redis

# Verify Redis is running
redis-cli ping  # Should return: PONG
```

### 2. Set Environment Variables

```bash
export EXPERIMENT_QUEUE_DB_PATH="$HOME/.experiment_queue/db.sqlite"
export EXPERIMENT_QUEUE_BROKER_URL="redis://localhost:6379/0"  # Optional
export EXPERIMENT_QUEUE_RESULT_BACKEND_URL="redis://localhost:6379/1"  # Optional
```

### 3. Start the Queue Worker

```bash
synth-ai queue start
```

### 4. Submit an Experiment

Create a JSON file with your experiment specification:

```json
{
  "name": "My GEPA Experiment",
  "description": "Testing GEPA optimization",
  "parallelism": 2,
  "jobs": [
    {
      "job_type": "gepa",
      "config_path": "path/to/config.toml",
      "config_overrides": {
        "prompt_learning.gepa.rollouts": 100
      }
    }
  ]
}
```

Submit it:

```bash
synth-ai experiment submit experiment.json
```

### 5. Monitor Progress

```bash
# Check worker status
synth-ai queue status

# View experiment dashboard
synth-ai experiments --watch

# View specific experiment
synth-ai experiment get <experiment_id>
```

### 6. Stop the Worker

```bash
synth-ai queue stop
```

## CLI Commands

### Queue Management

- `synth-ai queue start` - Start the queue worker
- `synth-ai queue stop` - Stop all workers
- `synth-ai queue status` - Show worker status (or just `synth-ai queue`)

### Experiment Management

- `synth-ai experiment submit <file>` - Submit an experiment from JSON file
- `synth-ai experiment submit --inline '{"name": "..."}'` - Submit inline JSON
- `synth-ai experiment list` - List experiments
- `synth-ai experiment get <id>` - Get experiment details
- `synth-ai experiment cancel <id>` - Cancel an experiment
- `synth-ai experiments` - Show dashboard view
- `synth-ai experiments --watch` - Watch mode with auto-refresh

## Architecture

### Components

1. **SQLite Database** (`EXPERIMENT_QUEUE_DB_PATH`)
   - Stores experiment metadata, job status, and trial results
   - Uses WAL (Write-Ahead Logging) mode for concurrent access
   - Single database path enforced across all workers

2. **Redis Broker** (`EXPERIMENT_QUEUE_BROKER_URL`)
   - Default: `redis://localhost:6379/0`
   - Handles task queuing and message passing
   - Eliminates SQLite locking issues

3. **Redis Result Backend** (`EXPERIMENT_QUEUE_RESULT_BACKEND_URL`)
   - Default: `redis://localhost:6379/1`
   - Stores task results and status

4. **Celery Worker**
   - Processes jobs from the queue
   - Runs training commands and collects results
   - Supports parallelism control per experiment

5. **Celery Beat**
   - Periodic scheduler (runs every 5 seconds)
   - Automatically dispatches queued jobs
   - Integrated into worker process by default

### Data Flow

```
1. User submits experiment → SQLite database
2. Beat scheduler checks queue → Finds queued jobs
3. Beat dispatches jobs → Redis broker
4. Worker consumes from Redis → Executes training command
5. Results collected → SQLite database + Redis backend
6. Status updated → Experiment marked COMPLETED/FAILED
```

## Configuration

### Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `EXPERIMENT_QUEUE_DB_PATH` | ✅ Yes | None | Path to SQLite database file |
| `EXPERIMENT_QUEUE_BROKER_URL` | No | `redis://localhost:6379/0` | Redis broker URL |
| `EXPERIMENT_QUEUE_RESULT_BACKEND_URL` | No | `redis://localhost:6379/1` | Redis result backend URL |
| `EXPERIMENT_QUEUE_TRAIN_CMD` | No | Auto-detected | Training command override |

### Training Command

By default, the queue uses the venv Python executable to run training commands:

```bash
<venv_python> -m synth_ai.cli train --type prompt_learning --config <config> --poll --stream-format cli
```

Override with `EXPERIMENT_QUEUE_TRAIN_CMD`:

```bash
export EXPERIMENT_QUEUE_TRAIN_CMD="python -m custom.train"
```

## Experiment Submission Format

Experiments are submitted as JSON with the following structure:

```json
{
  "name": "Experiment Name",
  "description": "Optional description",
  "parallelism": 2,
  "metadata": {},
  "jobs": [
    {
      "job_type": "gepa",
      "config_path": "/path/to/config.toml",
      "config_overrides": {
        "prompt_learning.gepa.rollouts": 100,
        "prompt_learning.termination_config.max_cost_usd": 10.0
      }
    }
  ]
}
```

### Job Types

- `gepa` - Genetic Evolution for Prompt Optimization
- `mipro` - Meta-Instruction Prompt Optimization

### Config Overrides

Config overrides are nested dictionaries that override values in the TOML config file. Use dot notation for nested keys:

```json
{
  "prompt_learning.gepa.rollouts": 100,
  "prompt_learning.termination_config.max_cost_usd": 10.0
}
```

## Troubleshooting

### Worker Won't Start

**Error**: `EXPERIMENT_QUEUE_DB_PATH environment variable must be set`

**Solution**: Set the environment variable:
```bash
export EXPERIMENT_QUEUE_DB_PATH="$HOME/.experiment_queue/db.sqlite"
synth-ai queue start
```

### Redis Connection Errors

**Error**: `Error connecting to Redis`

**Solution**: Ensure Redis is running:
```bash
redis-cli ping  # Should return: PONG
brew services start redis  # If not running
```

### Jobs Not Being Dispatched

**Symptoms**: Jobs remain in QUEUED status

**Checklist**:
1. Verify worker is running: `synth-ai queue status`
2. Check Beat scheduler: Ensure `--beat` flag is set (default)
3. Check Redis: `redis-cli ping`
4. Review logs: `tail -f logs/experiment_queue_worker.log`

### Database Path Mismatch

**Warning**: `Some workers are using different database paths!`

**Solution**: Stop all workers and restart with consistent path:
```bash
synth-ai queue stop
export EXPERIMENT_QUEUE_DB_PATH="/path/to/db.sqlite"
synth-ai queue start
```

### Silent Failures

Jobs that exit with return code 0 but produce no results are detected and marked as FAILED. Check logs for details:

```bash
tail -f logs/experiment_queue_worker.log
```

## Logs

Worker logs are written to `logs/experiment_queue_worker.log` when running in background mode:

```bash
# View logs
tail -f logs/experiment_queue_worker.log

# View last 100 lines
tail -n 100 logs/experiment_queue_worker.log
```

## Testing

Run the test suite:

```bash
# Set required environment variables
export EXPERIMENT_QUEUE_DB_PATH="/tmp/test_queue.db"
export EXPERIMENT_QUEUE_BROKER_URL="redis://localhost:6379/0"
export EXPERIMENT_QUEUE_RESULT_BACKEND_URL="redis://localhost:6379/1"

# Run tests
pytest tests/unit/experiment_queue/ tests/integration/experiment_queue/ -v
```

## Examples

### Basic GEPA Experiment

```json
{
  "name": "Banking77 GEPA",
  "parallelism": 1,
  "jobs": [
    {
      "job_type": "gepa",
      "config_path": "examples/banking77_gepa.toml",
      "config_overrides": {
        "prompt_learning.gepa.rollouts": 50
      }
    }
  ]
}
```

### Multiple Jobs with Parallelism

```json
{
  "name": "Multi-Job Experiment",
  "parallelism": 2,
  "jobs": [
    {
      "job_type": "gepa",
      "config_path": "config1.toml"
    },
    {
      "job_type": "mipro",
      "config_path": "config2.toml"
    },
    {
      "job_type": "gepa",
      "config_path": "config3.toml"
    }
  ]
}
```

### With Environment File Override

```json
{
  "name": "Experiment with Env",
  "jobs": [
    {
      "job_type": "gepa",
      "config_path": "config.toml",
      "config_overrides": {
        "prompt_learning.env_file_path": "/path/to/.env"
      }
    }
  ]
}
```

## Related Documentation

- [CLI Queue Documentation](https://docs.usesynth.ai/cli/queue)
- [CLI Experiment Documentation](https://docs.usesynth.ai/cli/experiment)
- [GEPA Algorithm Guide](https://docs.usesynth.ai/po/algorithms)




