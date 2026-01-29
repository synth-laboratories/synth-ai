#!/usr/bin/env python3
"""Run GEPA optimization using Harbor hosted sandboxes.

This script runs GEPA prompt optimization using Harbor for rollout execution.
It creates a GEPA job that uses Harbor deployments for sandbox execution.

Usage:
    export SYNTH_API_KEY=sk_live_...
    uv run python demos/harbor/run_gepa_harbor.py --deployment-name <name>
"""

import argparse
import asyncio
import json
import os
import sys
from typing import Any, Dict, List, Optional

import httpx


async def run_gepa_with_harbor(
    deployment_name: str,
    api_key: str,
    backend_url: str = "https://api-dev.usesynth.ai",
    seeds: List[int] = None,
    num_generations: int = 2,
    population_size: int = 4,
    max_concurrent: int = 5,
    rollout_timeout: int = 600,
):
    """Run GEPA optimization using Harbor for rollouts.

    This creates a GEPA job that uses the Harbor deployment for sandbox execution.
    """
    seeds = seeds if seeds is not None else list(range(13))

    print("=" * 70)
    print("GEPA OPTIMIZATION WITH HARBOR")
    print("=" * 70)
    print(f"Backend: {backend_url}")
    print(f"Deployment: {deployment_name}")
    print(f"Seeds: {seeds}")
    print(f"Generations: {num_generations}")
    print(f"Population Size: {population_size}")
    print(f"Max Concurrent: {max_concurrent}")
    print(f"Rollout Timeout: {rollout_timeout}s")
    print()

    # First, verify the deployment is ready
    async with httpx.AsyncClient(timeout=60.0) as client:
        response = await client.get(
            f"{backend_url}/api/harbor/deployments/{deployment_name}/status",
            headers={"Authorization": f"Bearer {api_key}"}
        )

        if response.status_code != 200:
            print(f"ERROR: Failed to get deployment status: {response.status_code}")
            print(response.text)
            return None

        status = response.json()
        if status["status"] != "ready":
            print(f"ERROR: Deployment is not ready (status: {status['status']})")
            return None

        print(f"Deployment ready: {status.get('deployment_name', deployment_name)}")
        print(f"Snapshot: {status.get('snapshot_id')}")
        print()

    # Create the GEPA job configuration
    # The backend expects config_body to contain the full prompt_learning config
    gepa_config = {
        "algorithm": "gepa",
        "auto_start": True,
        "config_body": {
            "prompt_learning": {
                # Use deployment-specific URL so GEPA calls /deployments/{id}/rollout
                "task_app_url": f"{backend_url}/api/harbor/deployments/{deployment_name}",
                "task_app_api_key": api_key,  # Use the user's API key for Harbor auth
                "_backend_managed_task_app": True,  # Allow task_app_api_key for Harbor
                "env_name": "engine_bench",
                "algorithm": "gepa",
                "harbor": {
                    "deployment_name": deployment_name,
                },
                "initial_prompt": {
                    "id": "enginebench_harbor",
                    "name": "EngineBench Harbor",
                    "messages": [
                        {"role": "user", "pattern": "{task_description}", "order": 0}
                    ],
                    "wildcards": {"task_description": "REQUIRED"}
                },
                "gepa": {
                    "env_name": "engine_bench",
                    "evaluation": {
                        "seeds": seeds,
                        "validation_seeds": seeds[:2],
                        "validation_pool": "train",
                        "validation_top_k": 2,
                    },
                    "mutation": {"rate": 0.5},
                    "population": {
                        "initial_size": population_size,
                        "num_generations": num_generations,
                        "children_per_generation": population_size,
                        "crossover_rate": 0.5,
                        "selection_pressure": 1.0,
                    },
                    # pareto_set_size must be < len(seeds) to have feedback seeds
                    "archive": {"pareto_set_size": max(1, min(5, len(seeds) - 1))},
                    "rollout": {
                        "budget": len(seeds) * population_size * num_generations,
                        "max_concurrent": max_concurrent,
                        "minibatch_size": 2,
                    },
                    "token": {
                        "counting_model": "gpt-4",
                        "enforce_pattern_limit": False,
                    },
                    "proposer": {"model": "claude-sonnet-4"},
                },
                "policy": {
                    "model": "gpt-4o-mini",
                    "provider": "openai",
                    "inference_mode": "synth_hosted",
                    "inference_url": "https://api.openai.com/v1",
                    "timeout": rollout_timeout,
                    "context_override": {
                        "system_prompt": """You are an expert Rust developer implementing Pokemon TCG cards.

CRITICAL: The stub file contains `todo!()` macros that YOU MUST REPLACE with working code.

Your task: Implement card effects by editing Rust files with stub functions marked with TODO comments.

Key patterns:
- Use `def_id_matches(&card.def_id, "DF", NUMBER)` to identify cards
- Implement attack modifiers in the `attack_override` function
- Use `game.queue_prompt()` for user choices
- Return `AttackOverrides::default()` if card doesn't apply

Output requirements:
1. EDIT files - replace TODO stubs with working code
2. Make code compile (`cargo check`)
3. Make tests pass (`cargo test`)"""
                    }
                }
            }
        }
    }

    print("Creating GEPA job...")
    print(f"Config: {json.dumps(gepa_config, indent=2)[:500]}...")
    print()

    # Submit the job
    async with httpx.AsyncClient(timeout=600.0) as client:
        response = await client.post(
            f"{backend_url}/api/prompt-learning/online/jobs",
            json=gepa_config,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            }
        )

        print(f"Status: {response.status_code}")
        result = response.json()
        print(f"Response: {json.dumps(result, indent=2)}")

        if response.status_code == 200:
            print()
            print("=" * 70)
            print("GEPA JOB STARTED")
            print("=" * 70)
            job_id = result.get("job_id")
            print(f"Job ID: {job_id}")
            print()
            print("Monitor progress:")
            print(f"  curl -H 'Authorization: Bearer $SYNTH_API_KEY' \\")
            print(f"       {backend_url}/api/prompt-learning/online/jobs/{job_id}")
            return result
        else:
            print(f"ERROR: Failed to create job: {result.get('detail', result)}")
            return None


async def main():
    parser = argparse.ArgumentParser(description="Run GEPA optimization with Harbor")
    parser.add_argument(
        "--deployment-name",
        type=str,
        required=True,
        help="Harbor deployment name to use for rollouts",
    )
    parser.add_argument(
        "--backend-url",
        type=str,
        default="https://api-dev.usesynth.ai",
        help="Backend API URL",
    )
    parser.add_argument(
        "--seeds",
        type=str,
        default=",".join(str(i) for i in range(13)),
        help="Comma-separated list of seeds",
    )
    parser.add_argument(
        "--generations",
        type=int,
        default=2,
        help="Number of GEPA generations",
    )
    parser.add_argument(
        "--population",
        type=int,
        default=4,
        help="Population size",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=5,
        help="Max concurrent rollouts",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=600,
        help="Rollout timeout in seconds",
    )
    args = parser.parse_args()

    # Get API key
    api_key = os.environ.get("SYNTH_API_KEY")
    if not api_key:
        print("ERROR: SYNTH_API_KEY not set")
        sys.exit(1)

    seeds = [int(s.strip()) for s in args.seeds.split(",")]

    result = await run_gepa_with_harbor(
        deployment_name=args.deployment_name,
        api_key=api_key,
        backend_url=args.backend_url,
        seeds=seeds,
        num_generations=args.generations,
        population_size=args.population,
        max_concurrent=args.concurrency,
        rollout_timeout=args.timeout,
    )

    if result:
        print("\nGEPA job submitted successfully!")
    else:
        print("\nFailed to submit GEPA job")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
