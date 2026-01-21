"""First-class SDK API for evaluation jobs.

This module provides high-level abstractions for running evaluation jobs
both via CLI and programmatically in Python scripts.

Example CLI usage:
    python -m synth_ai.cli eval --config banking77_eval.toml --backend http://localhost:8000

Example SDK usage:
    from synth_ai.sdk.eval import EvalJob, EvalResult

    job = EvalJob(config)
    job.submit()

    # progress=True provides built-in status printing:
    # [00:05] running | 3/10 completed
    # [00:10] running | 7/10 completed
    # [00:15] completed | mean_reward: 0.85
    result = job.poll_until_complete(progress=True)

    # Typed result access (not raw dict)
    if result.succeeded:
        print(f"Mean reward: {result.mean_reward}")
        print(f"Total cost: ${result.total_cost_usd:.4f}")

See Also:
    - `synth_ai.cli.eval`: CLI implementation
    - Backend API: POST /api/eval/jobs
"""

from .job import EvalJob, EvalJobConfig, EvalResult, EvalStatus

__all__ = ["EvalJob", "EvalJobConfig", "EvalResult", "EvalStatus"]
