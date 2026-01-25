#!/usr/bin/env python3
"""
Run EngineBench evaluations via Harbor hosted sandboxes.

This script:
1. Calls the Harbor /rollout endpoint for each seed
2. Collects metrics and traces
3. Outputs evaluation results

Usage:
    export SYNTH_API_KEY=sk_live_...

    # Run 5 seeds against a single deployment
    uv run python demos/harbor/run_harbor_eval.py \
        --deployment-id <id> \
        --seeds 5

    # Run against multiple deployments (one per seed)
    uv run python demos/harbor/run_harbor_eval.py \
        --deployment-ids <id1>,<id2>,<id3>,<id4>,<id5> \
        --seeds 5

    # Use the backend's interceptor for traces
    uv run python demos/harbor/run_harbor_eval.py \
        --deployment-id <id> \
        --seeds 5 \
        --model gpt-4.1-mini
"""

import argparse
import asyncio
import json
import os
import sys
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import httpx


def get_api_client(base_url: str, api_key: str) -> httpx.AsyncClient:
    """Create an async HTTP client with auth."""
    return httpx.AsyncClient(
        base_url=base_url,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        timeout=900.0,  # 15 minutes for long rollouts
    )


async def run_rollout(
    client: httpx.AsyncClient,
    deployment_id: str,
    seed: int,
    inference_url: str,
    timeout_s: int = 600,
    params: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Run a single rollout via the Harbor API.

    Args:
        client: HTTP client
        deployment_id: Deployment to run
        seed: Random seed for task selection
        inference_url: URL for LLM calls (interceptor endpoint)
        timeout_s: Timeout in seconds
        params: Additional parameters for the runner

    Returns:
        Rollout response
    """
    trace_correlation_id = f"harbor-eval-{uuid.uuid4().hex[:12]}"

    request_body = {
        "deployment_id": deployment_id,
        "trace_correlation_id": trace_correlation_id,
        "seed": seed,
        "prompt_template": {
            "sections": [
                {
                    "role": "user",
                    "pattern": "{task}",
                    "order": 0,
                }
            ]
        },
        "inference_url": inference_url,
        "limits": {
            "timeout_s": timeout_s,
        },
        "params": params or {},
    }

    start_time = time.time()
    response = await client.post("/api/harbor/rollout", json=request_body)
    duration_s = time.time() - start_time

    if response.status_code == 429:
        return {
            "trace_correlation_id": trace_correlation_id,
            "metrics": {"reward_mean": 0.0, "details": {}},
            "success": False,
            "error": "Rate limited (429) - try again later",
            "duration_s": duration_s,
        }

    if response.status_code != 200:
        return {
            "trace_correlation_id": trace_correlation_id,
            "metrics": {"reward_mean": 0.0, "details": {}},
            "success": False,
            "error": f"HTTP {response.status_code}: {response.text[:500]}",
            "duration_s": duration_s,
        }

    result = response.json()
    result["duration_s"] = duration_s
    return result


async def run_eval(
    client: httpx.AsyncClient,
    deployment_ids: List[str],
    seeds: List[int],
    inference_url: str,
    timeout_s: int = 600,
    concurrency: int = 1,
    agent_type: str = "opencode",
    model: str = "gpt-4o-mini",
) -> Dict[str, Any]:
    """Run evaluation across multiple seeds.

    Args:
        client: HTTP client
        deployment_ids: List of deployment IDs (cycled for seeds)
        seeds: List of seeds to evaluate
        inference_url: URL for LLM calls
        timeout_s: Timeout per rollout
        concurrency: Max concurrent rollouts
        agent_type: Agent type for params
        model: Model name for LLM calls

    Returns:
        Evaluation results
    """
    results = []
    semaphore = asyncio.Semaphore(concurrency)

    async def run_with_semaphore(seed: int, deployment_id: str):
        async with semaphore:
            print(f"  Running seed {seed} on deployment {deployment_id[:8]}...")
            result = await run_rollout(
                client=client,
                deployment_id=deployment_id,
                seed=seed,
                inference_url=inference_url,
                timeout_s=timeout_s,
                params={"agent": agent_type, "model": model},
            )
            reward = result.get("metrics", {}).get("reward_mean", 0.0)
            success = result.get("success", False)
            print(f"    Seed {seed}: reward={reward:.3f}, success={success}")
            return {"seed": seed, "deployment_id": deployment_id, "result": result}

    # Create tasks
    tasks = []
    for i, seed in enumerate(seeds):
        deployment_id = deployment_ids[i % len(deployment_ids)]
        tasks.append(run_with_semaphore(seed, deployment_id))

    # Run all tasks
    results = await asyncio.gather(*tasks, return_exceptions=True)

    # Process results
    successful_results = []
    failed_results = []

    for r in results:
        if isinstance(r, Exception):
            failed_results.append({"error": str(r)})
        elif r["result"].get("success"):
            successful_results.append(r)
        else:
            failed_results.append(r)

    # Calculate summary metrics
    rewards = [
        r["result"]["metrics"]["reward_mean"]
        for r in successful_results
        if "result" in r and "metrics" in r["result"]
    ]

    return {
        "total_seeds": len(seeds),
        "successful": len(successful_results),
        "failed": len(failed_results),
        "mean_reward": sum(rewards) / len(rewards) if rewards else 0.0,
        "results": [r for r in results if not isinstance(r, Exception)],
        "errors": [str(r) for r in results if isinstance(r, Exception)],
    }


async def main():
    parser = argparse.ArgumentParser(description="Run EngineBench eval via Harbor")
    parser.add_argument(
        "--deployment-id",
        type=str,
        help="Single deployment ID to use for all seeds",
    )
    parser.add_argument(
        "--deployment-ids",
        type=str,
        help="Comma-separated deployment IDs (cycled for seeds)",
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
        help="Model to use via interceptor (default: gpt-4o-mini)",
    )
    parser.add_argument(
        "--agent",
        type=str,
        default="opencode",
        choices=["opencode", "codex"],
        help="Agent type (default: opencode)",
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

    # Parse deployment IDs
    if args.deployment_ids:
        deployment_ids = args.deployment_ids.split(",")
    elif args.deployment_id:
        deployment_ids = [args.deployment_id]
    else:
        print("ERROR: Must specify --deployment-id or --deployment-ids")
        sys.exit(1)

    # Generate seeds
    seeds = list(range(args.seed_start, args.seed_start + args.seeds))

    # Build inference URL
    # NOTE: This uses the simple inference proxy, NOT the trace-capturing interceptor.
    # For trace capture, Harbor would need to use the interceptor URL pattern:
    #   {base}/api/interceptor/v1/{trial_id}/{correlation_id}/chat/completions
    # which requires trial registration with the prompt registry.
    inference_url = f"{args.backend_url}/api/inference/v1/chat/completions"

    print("=" * 60)
    print("HARBOR ENGINEBENCH EVALUATION")
    print("=" * 60)
    print(f"Backend: {args.backend_url}")
    print(f"Deployments: {len(deployment_ids)}")
    print(f"Seeds: {seeds}")
    print(f"Model: {args.model}")
    print(f"Agent: {args.agent}")
    print(f"Timeout: {args.timeout}s")
    print(f"Concurrency: {args.concurrency}")
    print()

    async with get_api_client(args.backend_url, api_key) as client:
        print("Running evaluations...")
        start_time = time.time()

        results = await run_eval(
            client=client,
            deployment_ids=deployment_ids,
            seeds=seeds,
            inference_url=inference_url,
            timeout_s=args.timeout,
            concurrency=args.concurrency,
            agent_type=args.agent,
            model=args.model,
        )

        total_time = time.time() - start_time

    print()
    print("=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"Total seeds: {results['total_seeds']}")
    print(f"Successful: {results['successful']}")
    print(f"Failed: {results['failed']}")
    print(f"Mean reward: {results['mean_reward']:.4f}")
    print(f"Total time: {total_time:.1f}s")
    print()

    # Per-seed breakdown
    print("Per-seed results:")
    for r in results.get("results", []):
        if isinstance(r, dict) and "result" in r:
            seed = r["seed"]
            reward = r["result"]["metrics"]["reward_mean"]
            success = r["result"]["success"]
            details = r["result"]["metrics"].get("details", {})
            task_id = details.get("task_id", "?")
            compilation = details.get("compilation", False)
            tests = f"{details.get('tests_passed', 0)}/{details.get('tests_total', 0)}"
            print(f"  Seed {seed}: task={task_id}, reward={reward:.3f}, compile={compilation}, tests={tests}")

    # Save results
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        output_data = {
            "timestamp": datetime.utcnow().isoformat(),
            "config": {
                "deployment_ids": deployment_ids,
                "seeds": seeds,
                "model": args.model,
                "agent": args.agent,
                "timeout_s": args.timeout,
            },
            "summary": {
                "total_seeds": results["total_seeds"],
                "successful": results["successful"],
                "failed": results["failed"],
                "mean_reward": results["mean_reward"],
                "total_time_s": total_time,
            },
            "results": results["results"],
        }

        with open(output_path, "w") as f:
            json.dump(output_data, f, indent=2, default=str)

        print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    asyncio.run(main())
