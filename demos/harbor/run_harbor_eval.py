#!/usr/bin/env python3
"""
Run EngineBench evaluations via Harbor hosted sandboxes.

This script uses the SDK's EvalJob to properly route through the backend,
which ensures trace capture via the interceptor.

Usage:
    export SYNTH_API_KEY=sk_live_...

    # Run 5 seeds against a single deployment
    uv run python demos/harbor/run_harbor_eval.py \
        --deployment-id <id> \
        --seeds 5

    # Run with a specific model
    uv run python demos/harbor/run_harbor_eval.py \
        --deployment-id <id> \
        --seeds 5 \
        --model gpt-4o-mini
"""

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from synth_ai.sdk.eval.job import EvalJob, EvalJobConfig


def run_harbor_eval(
    deployment_id: str,
    seeds: List[int],
    model: str = "gpt-4o-mini",
    provider: str = "openai",
    backend_url: str = "https://api-dev.usesynth.ai",
    api_key: Optional[str] = None,
    timeout_s: int = 600,
    max_concurrent: int = 1,
) -> Dict[str, Any]:
    """Run Harbor evaluation through the SDK's EvalJob.
    
    This properly routes through the backend's interceptor for trace capture.
    
    Args:
        deployment_id: Harbor deployment ID
        seeds: List of seeds to evaluate
        model: Model name (e.g., "gpt-4o-mini")
        provider: Model provider (e.g., "openai")
        backend_url: Backend API URL
        api_key: Synth API key (defaults to SYNTH_API_KEY env var)
        timeout_s: Timeout per rollout
        max_concurrent: Max concurrent rollouts
        
    Returns:
        Evaluation results dict
    """
    api_key = api_key or os.environ.get("SYNTH_API_KEY")
    if not api_key:
        raise ValueError("SYNTH_API_KEY not set")
    
    # Build the task app URL pointing to Harbor's deployment base
    # The eval system appends "/rollout" automatically, so we provide the deployment base
    # Full endpoint: /api/harbor/deployments/{id}/rollout
    task_app_url = f"{backend_url}/api/harbor/deployments/{deployment_id}"
    
    # Create eval job config
    # NOTE: Harbor's task app endpoint uses the same Synth API key for auth,
    # NOT an environment key. Pass the api_key explicitly as task_app_api_key.
    config = EvalJobConfig(
        task_app_url=task_app_url,
        backend_url=backend_url,
        api_key=api_key,
        task_app_api_key=api_key,  # Harbor requires the actual Synth API key
        app_id=f"harbor-{deployment_id[:8]}",
        env_name="enginebench",
        seeds=seeds,
        policy_config={
            "model": model,
            "provider": provider,
        },
        concurrency=max_concurrent,
        timeout=float(timeout_s),
    )
    
    # Create and submit job
    job = EvalJob(config)
    print(f"Submitting eval job for deployment {deployment_id}...")
    job.submit()
    print(f"Job ID: {job.job_id}")
    
    # Poll until complete with progress
    print("\nRunning evaluation...")
    result = job.poll_until_complete(progress=True)
    
    return {
        "job_id": job.job_id,
        "status": result.status.value,
        "mean_reward": result.mean_reward,
        "total_tokens": result.total_tokens,
        "total_cost_usd": result.total_cost_usd,
        "num_completed": result.num_completed,
        "num_total": result.num_total,
        "seed_results": result.seed_results,
        "error": result.error,
        "succeeded": result.succeeded,
        "failed": result.failed,
    }


def main():
    parser = argparse.ArgumentParser(description="Run EngineBench eval via Harbor")
    parser.add_argument(
        "--deployment-id",
        type=str,
        required=True,
        help="Harbor deployment ID",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        default=5,
        help="Number of seeds to evaluate (default: 5)",
    )
    parser.add_argument(
        "--seed-start",
        type=int,
        default=0,
        help="Starting seed number (default: 0)",
    )
    parser.add_argument(
        "--backend-url",
        type=str,
        default="https://api-dev.usesynth.ai",
        help="Backend API URL",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-4o-mini",
        help="Model to use (default: gpt-4o-mini)",
    )
    parser.add_argument(
        "--provider",
        type=str,
        default="openai",
        help="Model provider (default: openai)",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=600,
        help="Timeout per rollout in seconds (default: 600)",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=1,
        help="Max concurrent rollouts (default: 1)",
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Output file for results (JSON)",
    )
    args = parser.parse_args()

    # Get API key
    api_key = os.environ.get("SYNTH_API_KEY")
    if not api_key:
        print("ERROR: SYNTH_API_KEY not set")
        sys.exit(1)

    # Generate seeds
    seeds = list(range(args.seed_start, args.seed_start + args.seeds))

    print("=" * 60)
    print("HARBOR ENGINEBENCH EVALUATION")
    print("=" * 60)
    print(f"Backend: {args.backend_url}")
    print(f"Deployment: {args.deployment_id}")
    print(f"Seeds: {seeds}")
    print(f"Model: {args.model}")
    print(f"Provider: {args.provider}")
    print(f"Timeout: {args.timeout}s")
    print(f"Concurrency: {args.concurrency}")
    print()

    try:
        results = run_harbor_eval(
            deployment_id=args.deployment_id,
            seeds=seeds,
            model=args.model,
            provider=args.provider,
            backend_url=args.backend_url,
            api_key=api_key,
            timeout_s=args.timeout,
            max_concurrent=args.concurrency,
        )
    except Exception as e:
        print(f"\nERROR: {e}")
        sys.exit(1)

    print()
    print("=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"Job ID: {results['job_id']}")
    print(f"Status: {results['status']}")
    print(f"Completed: {results['num_completed']}/{results['num_total']}")
    print(f"Mean reward: {results['mean_reward']:.4f}" if results['mean_reward'] else "Mean reward: N/A")
    if results['total_cost_usd']:
        print(f"Total cost: ${results['total_cost_usd']:.4f}")
    print()

    if results['error']:
        print(f"Error: {results['error']}")

    # Per-seed breakdown
    if results['seed_results']:
        print("Per-seed results:")
        for sr in results['seed_results']:
            seed = sr.get('seed', '?')
            score = sr.get('score', 0.0)
            success = sr.get('success', False)
            print(f"  Seed {seed}: score={score:.3f}, success={success}")

    # Save results
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        output_data = {
            "timestamp": datetime.utcnow().isoformat(),
            "config": {
                "deployment_id": args.deployment_id,
                "seeds": seeds,
                "model": args.model,
                "provider": args.provider,
                "timeout_s": args.timeout,
            },
            "results": results,
        }

        with open(output_path, "w") as f:
            json.dump(output_data, f, indent=2, default=str)

        print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
