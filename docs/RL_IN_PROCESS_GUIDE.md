# RL/GSPO Training with In-Process Task Apps

This guide demonstrates how to run RL/GSPO training with in-process task apps using the Synth AI SDK.

## Overview

The `RLJob` SDK class provides a clean API for running RL training jobs programmatically, similar to `PromptLearningJob` for prompt optimization. When combined with `InProcessTaskApp`, you can run entire RL training pipelines from a single Python script without manual process management.

## Quick Start

```python
from synth_ai.sdk.api.train.rl import RLJob
from synth_ai.sdk.task.in_process import InProcessTaskApp

async def main():
    async with InProcessTaskApp(
        task_app_path="my_task_app.py",
        port=8114,
    ) as task_app:
        job = RLJob.from_config(
            config_path="rl_config.toml",
            backend_url="https://api.usesynth.ai",
            api_key=os.environ["SYNTH_API_KEY"],
            task_app_url=task_app.url,
        )
        job_id = job.submit()
        result = job.poll_until_complete(timeout=7200.0)
        print(f"Final reward: {result.get('final_reward', 'N/A')}")

asyncio.run(main())
```

## Features

- **Automatic Task App Management**: Task app starts in background thread
- **Automatic Tunnel Creation**: Cloudflare tunnel opens automatically
- **Clean Shutdown**: Everything cleans up automatically when done
- **Status Polling**: Built-in polling with progress callbacks
- **Error Handling**: Comprehensive error messages and validation

## API Reference

### `RLJob`

High-level SDK class for running RL training jobs.

#### `from_config()`

Create an RL job from a TOML config file.

```python
job = RLJob.from_config(
    config_path="rl_config.toml",
    backend_url="https://api.usesynth.ai",
    api_key=os.environ["SYNTH_API_KEY"],
    task_app_url="https://my-task-app.usesynth.ai",  # Optional, can use env var
    task_app_api_key=os.environ["ENVIRONMENT_API_KEY"],  # Optional
    allow_experimental=False,  # Optional
    overrides={"training": {"num_epochs": 5}},  # Optional config overrides
    idempotency_key="unique-key",  # Optional idempotency key
)
```

#### `from_job_id()`

Resume an existing RL job by ID.

```python
job = RLJob.from_job_id(
    job_id="rl_abc123",
    backend_url="https://api.usesynth.ai",
    api_key=os.environ["SYNTH_API_KEY"],
)
```

#### `submit()`

Submit the job to the backend. Returns job ID.

```python
job_id = job.submit()
```

#### `poll_until_complete()`

Poll job until it reaches a terminal state.

```python
result = job.poll_until_complete(
    timeout=7200.0,  # Maximum seconds to wait (default: 7200)
    interval=10.0,  # Seconds between poll attempts (default: 10)
    on_status=lambda status: print(f"Status: {status['status']}"),  # Optional callback
)
```

#### `get_status()`

Get current job status without polling.

```python
status = job.get_status()
print(f"Status: {status['status']}")
print(f"Progress: {status.get('progress', {})}")
```

#### `get_results()`

Get final job results (only works after job completes successfully).

```python
results = job.get_results()
print(f"Final metrics: {results.get('metrics', {})}")
```

## Configuration

RL config files use TOML format with the following sections:

- `[algorithm]`: Algorithm type (online/offline), method (policy_gradient/ppo/gspo), variety
- `[services]`: Task app URL (can be overridden via `task_app_url` parameter)
- `[model]`: Base model, trainer mode (lora/full), label
- `[rollout]`: Environment name, policy name, rollout parameters
- `[training]`: Training hyperparameters (epochs, learning rate, etc.)
- `[compute]`: GPU configuration
- `[topology]`: Cluster topology configuration

See `examples/rl_in_process_config.toml` for a complete example.

## Examples

### Basic Example

```python
from synth_ai.sdk.api.train.rl import RLJob

job = RLJob.from_config("rl_config.toml")
job.submit()
result = job.poll_until_complete()
```

### With In-Process Task App

```python
from synth_ai.sdk.api.train.rl import RLJob
from synth_ai.sdk.task.in_process import InProcessTaskApp

async with InProcessTaskApp(
    task_app_path="my_task_app.py",
    port=8114,
) as task_app:
    job = RLJob.from_config(
        config_path="rl_config.toml",
        task_app_url=task_app.url,
    )
    job.submit()
    result = job.poll_until_complete()
```

### With Progress Callback

```python
def on_status(status):
    state = status.get("status", "unknown")
    progress = status.get("progress", {})
    metrics = status.get("metrics", {})
    
    if progress:
        print(f"Progress: {progress.get('completed', 0)}/{progress.get('total', 0)}")
    if metrics:
        print(f"Reward: {metrics.get('reward', 'N/A')}")

job = RLJob.from_config("rl_config.toml")
job.submit()
result = job.poll_until_complete(on_status=on_status)
```

## Integration with synth-research

The `RLJob` SDK works seamlessly with task apps from `synth-research`. Simply point `task_app_path` to your task app file:

```python
from synth_ai.sdk.api.train.rl import RLJob
from synth_ai.sdk.task.in_process import InProcessTaskApp

# Task app from synth-research
task_app_path = Path("../synth-research/environments/examples/crafter/crafter_task_app.py")

async with InProcessTaskApp(
    task_app_path=task_app_path,
    port=8114,
) as task_app:
    job = RLJob.from_config(
        config_path="rl_config.toml",
        task_app_url=task_app.url,
    )
    job.submit()
    result = job.poll_until_complete()
```

## Error Handling

The SDK provides comprehensive error handling:

- **Config validation**: Validates config file format and required fields
- **Task app health checks**: Verifies task app is accessible before submission
- **Backend errors**: Clear error messages for backend failures
- **Timeout handling**: Raises `TimeoutError` if job doesn't complete in time

## See Also

- `examples/rl_in_process_example.py`: Complete working example
- `examples/rl_in_process_config.toml`: Example config file
- `synth_ai/sdk/api/train/prompt_learning.py`: Similar API for prompt optimization
- `synth_ai/sdk/task/in_process.py`: In-process task app documentation

