"""First-class SDK API for evaluation jobs.

This module provides high-level abstractions for running evaluation jobs
both via CLI and programmatically in Python scripts.

Example CLI usage:
    python -m synth_ai.cli eval --config banking77_eval.toml --backend http://localhost:8000

Example SDK usage:
    from synth_ai.sdk.api.eval import EvalJob

    job = EvalJob.from_config(
        config_path="banking77_eval.toml",
        backend_url="https://api.usesynth.ai",
        api_key="sk_live_...",
    )
    job.submit()
    results = job.poll_until_complete()
    print(f"Mean score: {results['summary']['mean_score']}")

See Also:
    - `synth_ai.cli.commands.eval`: CLI implementation
    - Backend API: POST /api/eval/jobs
"""

from .job import EvalJob, EvalJobConfig

__all__ = ["EvalJob", "EvalJobConfig"]
