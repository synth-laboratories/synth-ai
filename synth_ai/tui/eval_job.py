"""Submit an eval job to the Synth backend.

Usage:
    python -m synth_ai.tui.eval_job <task_app_url>

Outputs JSON to stdout for progress/results:
    {"status": "submitted", "job_id": "eval-abc123"}
    {"status": "progress", "completed": 5, "total": 20}
    {"status": "completed", "mean_score": 0.85, "cost_usd": 0.02}
    {"status": "error", "error": "..."}
"""

import json
import os
import sys

# Default eval configuration
DEFAULT_SEEDS = list(range(20))  # 20 seeds for quick eval
DEFAULT_MODEL = "gpt-4.1-nano"
DEFAULT_PROVIDER = "openai"
DEFAULT_INFERENCE_MODE = "synth_hosted"
DEFAULT_CONCURRENCY = 10
DEFAULT_TIMEOUT = 600.0


def _output(data: dict) -> None:
    """Output JSON to stdout."""
    print(json.dumps(data), flush=True)


def run_eval_job(task_app_url: str) -> None:
    """Submit and poll an eval job."""
    from synth_ai.sdk.api.eval import EvalJob, EvalJobConfig

    # Get config from environment
    api_key = os.environ.get("SYNTH_API_KEY")
    if not api_key:
        _output({"status": "error", "error": "SYNTH_API_KEY not set"})
        return

    backend_url = os.environ.get("SYNTH_BACKEND_URL", "https://api.usesynth.ai")

    # Build config with defaults
    config = EvalJobConfig(
        task_app_url=task_app_url,
        backend_url=backend_url,
        api_key=api_key,
        env_name="default",
        seeds=DEFAULT_SEEDS,
        policy_config={
            "model": DEFAULT_MODEL,
            "provider": DEFAULT_PROVIDER,
            "inference_mode": DEFAULT_INFERENCE_MODE,
            "api_key": api_key,
        },
        env_config={"split": "test"},
        concurrency=DEFAULT_CONCURRENCY,
        timeout=DEFAULT_TIMEOUT,
    )

    # Submit job
    try:
        job = EvalJob(config)
        job_id = job.submit()
        _output({"status": "submitted", "job_id": job_id})
    except Exception as e:
        _output({"status": "error", "error": f"Failed to submit job: {e}"})
        return

    # Poll until complete with progress updates
    def on_status(status_data: dict) -> None:
        completed = status_data.get("completed", 0)
        total = status_data.get("total", len(DEFAULT_SEEDS))
        _output({"status": "progress", "completed": completed, "total": total})

    try:
        result = job.poll_until_complete(
            timeout=DEFAULT_TIMEOUT,
            interval=2.0,
            progress=False,
            on_status=on_status,
        )

        if result.succeeded:
            _output({
                "status": "completed",
                "mean_score": result.mean_score,
                "cost_usd": result.total_cost_usd,
                "num_completed": result.num_completed,
                "num_total": result.num_total,
            })
        else:
            _output({"status": "error", "error": result.error or "Job failed"})

    except Exception as e:
        _output({"status": "error", "error": f"Polling failed: {e}"})


def main() -> None:
    if len(sys.argv) != 2:
        _output({"status": "error", "error": "Usage: python -m synth_ai.tui.eval_job <task_app_url>"})
        sys.exit(1)

    task_app_url = sys.argv[1]
    run_eval_job(task_app_url)


if __name__ == "__main__":
    main()
