# In-Process GEPA Demo

This directory contains a complete demo of running GEPA (Genetic Evolution for Prompt Optimization) with **in-process task apps** - a streamlined approach that eliminates manual process management.

## Quick Start

### Synth GEPA (Recommended)

```bash
cd examples/gepa
source ../../../.env
uv run python run_synth_gepa_in_process.py
```

That's it! The script will:
1. Start the Heart Disease task app automatically
2. Open a Cloudflare tunnel automatically
3. Run Synth GEPA optimization (budget: 50 rollouts)
4. Display results
5. Clean up everything automatically

### Original Combined Script

```bash
uv run python run_in_process_gepa.py
```

## What's Included

- **`run_synth_gepa_in_process.py`**: Synth GEPA demo script (budget: 50, faster)
- **`run_in_process_gepa.py`**: Original combined demo script
- **`IN_PROCESS_GEPA_DEMO.md`**: Comprehensive guide and documentation
- **`synth_ai/task/in_process.py`**: `InProcessTaskApp` class implementation

## How It Works

The `InProcessTaskApp` class is a context manager that:

```python
from synth_ai.task import InProcessTaskApp

async with InProcessTaskApp(
    task_app_path="path/to/task_app.py",
    port=8114,
) as task_app:
    # task_app.url contains the Cloudflare tunnel URL
    # Use it for GEPA jobs
    job = PromptLearningJob.from_config(
        config_path="config.toml",
        task_app_url=task_app.url,
    )
    results = await job.poll_until_complete()
# Everything cleaned up automatically!
```

## Benefits

- ✅ **Single script**: No separate terminals needed
- ✅ **Automatic cleanup**: No manual process management
- ✅ **Reproducible**: Entire workflow in one file
- ✅ **CI/CD friendly**: Perfect for automated testing

## Requirements

- `GROQ_API_KEY` in `.env` or environment
- `cloudflared` binary (auto-installs if missing)
- synth-ai backend running (localhost:8000)

## Documentation

See [`IN_PROCESS_GEPA_DEMO.md`](./IN_PROCESS_GEPA_DEMO.md) for:
- Detailed explanation
- Advanced usage examples
- Troubleshooting guide
- Architecture diagrams

## Related Files

- Task app: `examples/task_apps/other_langprobe_benchmarks/heartdisease_task_app.py`
- GEPA config: `examples/blog_posts/gepa/configs/heartdisease_gepa_local.toml`
- Implementation: `synth_ai/task/in_process.py`

