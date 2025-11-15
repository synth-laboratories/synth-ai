#!/usr/bin/env python3
"""Run Synth GEPA on Iris via backend HTTP API (localhost:8000).

This uses the backend's HTTP API instead of direct imports.
"""

import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Any, Optional

import httpx

# Add repo root to path
REPO_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO_ROOT))

from examples.blog_posts.langprobe.integrations.learning_curve_tracker import (
    LearningCurveTracker,
)


async def run_synth_gepa_via_api(
    backend_url: str = "http://localhost:8000",
    task_app_url: str = "http://127.0.0.1:8115",
    train_seeds: Optional[list[int]] = None,
    rollout_budget: int = 400,
    output_dir: Optional[Path] = None,
) -> dict[str, Any]:
    """Run Synth GEPA via backend HTTP API.

    Args:
        backend_url: Backend API URL (default: http://localhost:8000)
        task_app_url: Task app URL
        train_seeds: Training seeds (default: 0-99)
        rollout_budget: Rollout budget (default: 400)
        output_dir: Output directory

    Returns:
        Results dictionary
    """
    if train_seeds is None:
        train_seeds = list(range(100))  # Seeds 0-99

    if output_dir is None:
        output_dir = Path(__file__).parent.parent / "results" / "synth_gepa"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create TOML config content
    config_content = f"""
[prompt_learning]
algorithm = "gepa"
task_app_url = "{task_app_url}"
task_app_id = "iris"

# Seeds for optimization
evaluation_seeds = {train_seeds}

# Held-out seeds for final evaluation
test_pool = {list(range(100, 150))}

[prompt_learning.initial_prompt]
id = "iris_classification"
name = "Iris Classification"

[[prompt_learning.initial_prompt.messages]]
role = "system"
pattern = "You are a botany classification assistant. Based on the flower's measurements, classify the iris species. Respond with one of: setosa, versicolor, or virginica."
order = 0

[[prompt_learning.initial_prompt.messages]]
role = "user"
pattern = "Flower Measurements:\\n{{features}}\\n\\nClassify this iris flower. Respond with one of: setosa, versicolor, or virginica."
order = 1

[prompt_learning.initial_prompt.wildcards]
features = "REQUIRED"

[prompt_learning.policy]
model = "openai/gpt-oss-20b"
provider = "groq"
temperature = 1.0
max_completion_tokens = 512
policy_name = "iris-gepa"

[prompt_learning.gepa]
env_name = "iris"
initial_population_size = 20
num_generations = 15
mutation_rate = 0.3
crossover_rate = 0.5
selection_pressure = 1.0
minibatch_size = 8
pareto_set_size = 64
feedback_fraction = 0.5
children_per_generation = 12
patience_generations = 5
rollout_budget = {rollout_budget}
archive_size = 64
pareto_eps = 1e-6
max_concurrent_rollouts = 20
mutation_llm_model = "openai/gpt-oss-20b"
mutation_llm_provider = "groq"
"""

    # Write config to temp file
    config_path = output_dir / "iris_synth_gepa.toml"
    with open(config_path, "w") as f:
        f.write(config_content)

    print(f"✓ Created config: {config_path}")
    print(f"🚀 Submitting to backend API: {backend_url}")

    # Read config file content
    with open(config_path, "rb") as f:
        config_bytes = f.read()

    # Submit job via HTTP API to local runner endpoint
    async with httpx.AsyncClient(timeout=7200.0) as client:
        print(f"Submitting to: {backend_url}/prompt-learning/local/run")
        
        # Use local runner endpoint
        response = await client.post(
            f"{backend_url}/prompt-learning/local/run",
            json={
                "algorithm": "gepa",
                "config": config_content,
                "job_id": f"iris_gepa_{rollout_budget}_{os.getpid()}",
            },
            headers={"Content-Type": "application/json"},
        )
        
        if response.status_code != 200:
            raise RuntimeError(
                f"Optimization failed: {response.status_code} - {response.text[:500]}"
            )
        
        result_data = response.json()
        
        if result_data.get("status") == "failed":
            raise RuntimeError(f"Optimization failed: {result_data.get('error', 'Unknown error')}")
        
        # Extract results
        result = result_data.get("result", {})
        best_score = result.get("best_score", 0.0)
        best_template = result.get("best_template", {})
        
        # Save results
        results_file = output_dir / "iris_results.json"
        with open(results_file, "w") as f:
            json.dump(
                {
                    "job_id": result_data.get("job_id"),
                    "status": result_data.get("status"),
                    "best_score": best_score,
                    "best_template": best_template,
                    "rollout_budget": rollout_budget,
                    "train_seeds": train_seeds,
                    "full_result": result,
                },
                f,
                indent=2,
            )
        
        print(f"\n✅ Optimization complete!")
        print(f"   Job ID: {result_data.get('job_id')}")
        print(f"   Status: {result_data.get('status')}")
        print(f"   Best Score: {best_score:.4f}")
        print(f"   Results saved to: {results_file}")
        
        return {
            "job_id": result_data.get("job_id"),
            "best_score": best_score,
            "status": result_data.get("status"),
            "best_template": best_template,
        }


async def main() -> None:
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Run Synth GEPA on Iris via backend API")
    parser.add_argument(
        "--backend-url",
        default="http://localhost:8000",
        help="Backend API URL",
    )
    parser.add_argument(
        "--task-app-url",
        default="http://127.0.0.1:8115",
        help="Task app URL",
    )
    parser.add_argument(
        "--rollout-budget",
        type=int,
        default=400,
        help="Rollout budget",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory",
    )

    args = parser.parse_args()

    results = await run_synth_gepa_via_api(
        backend_url=args.backend_url,
        task_app_url=args.task_app_url,
        rollout_budget=args.rollout_budget,
        output_dir=args.output_dir,
    )

    print(f"\nFinal Results:")
    print(f"  Best Score: {results.get('best_score', 'N/A')}")


if __name__ == "__main__":
    asyncio.run(main())

