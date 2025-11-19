#!/usr/bin/env python3
"""Baseline script for HumanEval task app with gpt-5-nano.

This script runs a baseline evaluation on the HumanEval task app using gpt-5-nano
and reports the pass rate.
"""

import asyncio
import os
import sys
from pathlib import Path
from typing import Optional

import httpx
from dotenv import load_dotenv

# Add repo root to path for imports
REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

load_dotenv()


async def run_baseline(
    task_app_url: str = "http://127.0.0.1:8113",
    seeds: Optional[list[int]] = None,
    model: str = "gpt-5-nano",
    num_seeds: int = 10,
) -> dict[str, float]:
    """Run baseline evaluation on HumanEval task app.
    
    Args:
        task_app_url: URL of the task app (default: local)
        seeds: List of seeds to evaluate (default: 0 to num_seeds-1)
        model: Model to use (default: gpt-5-nano)
        num_seeds: Number of seeds if seeds not provided
        
    Returns:
        Dictionary with metrics: {'pass_rate', 'mean_reward', 'num_passed', 'total'}
    """
    if seeds is None:
        seeds = list(range(num_seeds))
    
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable required")
    
    passed_count = 0
    total_reward = 0.0
    results = []
    
    async with httpx.AsyncClient(timeout=300.0) as client:
        # Check health
        try:
            health_resp = await client.get(f"{task_app_url}/health")
            if health_resp.status_code != 200:
                print(f"Warning: Task app health check failed: {health_resp.status_code}")
        except Exception as e:
            print(f"Warning: Could not reach task app at {task_app_url}: {e}")
            print("Make sure the task app is running: python -m examples.task_apps.other_langprobe_benchmarks.humaneval_task_app")
        
        print(f"Running baseline evaluation with {model} on {len(seeds)} seeds...")
        
        for seed in seeds:
            rollout_request = {
                "run_id": f"humaneval_baseline_{model}_{seed}",
                "env": {
                    "env_name": "humaneval",
                    "seed": seed,
                    "config": {
                        "split": "test",
                    },
                },
                "policy": {
                    "policy_name": "humaneval_baseline",
                    "config": {
                        "model": model,
                        "inference_url": "https://api.openai.com/v1",
                        "temperature": 1.0,
                        "max_completion_tokens": 2048,
                        "reasoning_effort": "minimal",
                    },
                },
                "ops": ["policy"],
                "mode": "eval",
            }
            
            try:
                response = await client.post(
                    f"{task_app_url}/rollout",
                    json=rollout_request,
                    headers={"X-API-Key": os.getenv("ENVIRONMENT_API_KEY", "")},
                )
                response.raise_for_status()
                data = response.json()
                
                # Extract metrics
                metrics = data.get("metrics", {})
                reward = metrics.get("mean_return", 0.0)
                total_reward += reward
                
                # Check if test passed from trajectory info
                trajectories = data.get("trajectories", [])
                if trajectories:
                    steps = trajectories[0].get("steps", [])
                    if steps:
                        info = steps[0].get("info", {})
                        test_passed = info.get("test_passed", False)
                        if test_passed:
                            passed_count += 1
                        results.append({
                            "seed": seed,
                            "passed": test_passed,
                            "reward": reward,
                            "entry_point": info.get("entry_point", ""),
                        })
                
                print(f"  Seed {seed}: {'✓' if reward > 0 else '✗'} (reward={reward:.3f})")
                
            except Exception as e:
                print(f"  Seed {seed}: ERROR - {e}")
                results.append({"seed": seed, "error": str(e)})
    
    pass_rate = passed_count / len(seeds) if seeds else 0.0
    mean_reward = total_reward / len(seeds) if seeds else 0.0
    
    print(f"\n{'='*60}")
    print(f"Baseline Results ({model})")
    print(f"{'='*60}")
    print(f"Total seeds: {len(seeds)}")
    print(f"Passed: {passed_count}")
    print(f"Pass rate: {pass_rate:.1%}")
    print(f"Mean reward: {mean_reward:.3f}")
    print(f"{'='*60}\n")
    
    return {
        "pass_rate": pass_rate,
        "mean_reward": mean_reward,
        "num_passed": passed_count,
        "total": len(seeds),
    }


async def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Run HumanEval baseline with gpt-5-nano")
    parser.add_argument(
        "--task-app-url",
        default="http://127.0.0.1:8113",
        help="URL of the HumanEval task app",
    )
    parser.add_argument(
        "--seeds",
        type=str,
        help="Comma-separated list of seeds (e.g., '0,1,2') or range (e.g., '0-9')",
    )
    parser.add_argument(
        "--num-seeds",
        type=int,
        default=10,
        help="Number of seeds to evaluate (default: 10)",
    )
    parser.add_argument(
        "--model",
        default="gpt-5-nano",
        help="Model to use (default: gpt-5-nano)",
    )
    
    args = parser.parse_args()
    
    # Parse seeds
    seeds = None
    if args.seeds:
        if "-" in args.seeds:
            # Range format: "0-9"
            start, end = map(int, args.seeds.split("-"))
            seeds = list(range(start, end + 1))
        else:
            # Comma-separated: "0,1,2"
            seeds = [int(s.strip()) for s in args.seeds.split(",")]
    
    results = await run_baseline(
        task_app_url=args.task_app_url,
        seeds=seeds,
        model=args.model,
        num_seeds=args.num_seeds,
    )
    
    # Exit with non-zero if pass rate is zero (indicates problem)
    if results["pass_rate"] == 0.0:
        print("WARNING: Pass rate is 0.0 - check task app and model configuration!")
        sys.exit(1)
    
    sys.exit(0)


if __name__ == "__main__":
    asyncio.run(main())

