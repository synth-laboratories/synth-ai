# Prompt Learning SDK and CLI Guide

This document describes how to use prompt optimization (MIPRO and GEPA) via both CLI and SDK.

## CLI Usage

The CLI provides a first-class interface for running prompt learning jobs:

```bash
# Basic usage
uvx synth-ai train --type prompt_learning --config my_config.toml --poll

# With custom backend
uvx synth-ai train \
    --type prompt_learning \
    --config examples/blog_posts/gepa/configs/banking77_gepa_local.toml \
    --backend https://api.usesynth.ai \
    --poll

# Without polling (submit and exit)
uvx synth-ai train \
    --type prompt_learning \
    --config my_config.toml \
    --no-poll
```

### CLI Options

- `--type prompt_learning`: Specifies prompt learning (MIPRO or GEPA)
- `--config PATH`: Path to TOML config file (required)
- `--backend URL`: Backend API URL (defaults to env or production)
- `--poll/--no-poll`: Whether to poll until completion (default: poll)
- `--poll-timeout SECONDS`: Maximum time to poll (default: 3600)
- `--poll-interval SECONDS`: Seconds between polls (default: 5.0)
- `--stream-format [cli|chart]`: Output format (default: cli)

## SDK Usage

The SDK provides programmatic access for running jobs in Python scripts:

### Basic Example

```python
from synth_ai.api.train.prompt_learning import PromptLearningJob
import os

# Create job from config
job = PromptLearningJob.from_config(
    config_path="my_config.toml",
    backend_url=os.environ.get("BACKEND_BASE_URL", "https://api.usesynth.ai"),
    api_key=os.environ["SYNTH_API_KEY"],
)

# Submit job
job_id = job.submit()
print(f"Job submitted: {job_id}")

# Poll until complete
result = job.poll_until_complete(timeout=3600.0)
print(f"Status: {result['status']}")
print(f"Best score: {result.get('best_score')}")

# Get results
results = job.get_results()
print(f"Top prompts: {len(results['top_prompts'])}")

# Get best prompt text
best_prompt = job.get_best_prompt_text(rank=1)
print(f"Best prompt:\n{best_prompt}")
```

### Resuming an Existing Job

```python
# Resume a job by ID
job = PromptLearningJob.from_job_id(
    job_id="pl_9c58b711c2644083",
    backend_url="https://api.usesynth.ai",
    api_key=os.environ["SYNTH_API_KEY"],
)

# Check status
status = job.get_status()
print(f"Status: {status['status']}")

# Continue polling if needed
if status['status'] not in ['succeeded', 'failed', 'cancelled']:
    result = job.poll_until_complete()
```

### Advanced: Status Callbacks

```python
def on_status(status: dict) -> None:
    print(f"Status: {status.get('status')}")
    if 'best_score' in status:
        print(f"Best score: {status['best_score']}")

job.poll_until_complete(
    timeout=3600.0,
    interval=5.0,
    on_status=on_status,
)
```

## SDK API Reference

### `PromptLearningJob`

Main class for running prompt learning jobs.

#### Class Methods

- `from_config(config_path, backend_url=None, api_key=None, ...) -> PromptLearningJob`
  - Create a new job from a TOML config file
  
- `from_job_id(job_id, backend_url=None, api_key=None) -> PromptLearningJob`
  - Resume an existing job by ID

#### Instance Methods

- `submit() -> str`
  - Submit the job and return job ID
  
- `get_status() -> Dict[str, Any]`
  - Get current job status
  
- `poll_until_complete(timeout=3600.0, interval=5.0, on_status=None) -> Dict[str, Any]`
  - Poll until job reaches terminal state
  
- `get_results() -> Dict[str, Any]`
  - Get job results (prompts, scores, candidates)
  
- `get_best_prompt_text(rank=1) -> Optional[str]`
  - Get prompt text by rank (1 = best)

#### Properties

- `job_id: Optional[str]`
  - Job ID (None if not yet submitted)

### `PromptLearningJobConfig`

Configuration for a prompt learning job.

### `PromptLearningJobPoller`

Low-level poller for prompt learning jobs (used internally).

## Environment Variables

- `SYNTH_API_KEY`: API key for backend authentication (required)
- `ENVIRONMENT_API_KEY`: Task app API key (required for prompt learning)
- `BACKEND_BASE_URL`: Backend API URL (optional, defaults to production)

## Examples

See `examples/sdk_prompt_learning_example.py` for a complete working example.

## Config File Format

See example configs:
- `examples/blog_posts/gepa/configs/banking77_gepa_local.toml`
- `examples/blog_posts/mipro/configs/banking77_pipeline_mipro_local.toml`

