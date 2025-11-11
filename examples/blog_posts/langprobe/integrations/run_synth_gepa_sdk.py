#!/usr/bin/env python3
"""Run Synth GEPA on Iris using SDK API.

This uses the high-level SDK API which submits to backend.
For direct optimizer access, backend codebase is required.
"""

import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Any, Optional

# Add repo root to path
REPO_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO_ROOT))

try:
    from synth_ai.api.train.prompt_learning import PromptLearningJob
    SDK_AVAILABLE = True
except ImportError:
    SDK_AVAILABLE = False
    print("Warning: SDK not available, cannot run optimization via SDK API")
    print("Need backend codebase for direct optimizer access")


async def run_synth_gepa_via_sdk(
    task_app_url: str = "http://127.0.0.1:8115",
    train_seeds: Optional[list[int]] = None,
    rollout_budget: int = 400,
    output_dir: Optional[Path] = None,
) -> dict[str, Any]:
    """Run Synth GEPA via SDK API.

    Args:
        task_app_url: Task app URL
        train_seeds: Training seeds (default: 0-99)
        rollout_budget: Rollout budget (default: 400)
        output_dir: Output directory

    Returns:
        Results dictionary
    """
    if not SDK_AVAILABLE:
        raise RuntimeError("SDK not available")

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
model = "openai/gpt-oss-120b"
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

    print(f"Created config: {config_path}")
    print(f"Submitting job to backend...")

    # Create job
    backend_url = os.getenv("BACKEND_BASE_URL", "https://api.usesynth.ai")
    api_key = os.getenv("SYNTH_API_KEY")
    if not api_key:
        raise ValueError("SYNTH_API_KEY environment variable required")

    job = PromptLearningJob.from_config(
        config_path=str(config_path),
        backend_url=backend_url,
        api_key=api_key,
    )

    # Submit and poll
    job_id = job.submit()
    print(f"Job submitted: {job_id}")

    print("Polling until complete...")
    result = job.poll_until_complete(timeout=7200.0, interval=10.0)

    # Get results
    results_data = job.get_results()
    best_prompt = job.get_best_prompt_text(rank=1)

    # Save results
    results_file = output_dir / "iris_results.json"
    with open(results_file, "w") as f:
        json.dump(
            {
                "job_id": job_id,
                "best_score": results_data.get("best_score"),
                "best_prompt": best_prompt,
                "status": result.get("status"),
                "results": results_data,
            },
            f,
            indent=2,
        )

    return {
        "job_id": job_id,
        "best_score": results_data.get("best_score"),
        "best_prompt": best_prompt,
        "status": result.get("status"),
    }


async def main() -> None:
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Run Synth GEPA on Iris via SDK")
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

    if not SDK_AVAILABLE:
        print("Error: SDK not available")
        sys.exit(1)

    results = await run_synth_gepa_via_sdk(
        task_app_url=args.task_app_url,
        rollout_budget=args.rollout_budget,
        output_dir=args.output_dir,
    )

    print(f"\nResults:")
    print(f"  Job ID: {results['job_id']}")
    print(f"  Best Score: {results['best_score']}")
    print(f"  Status: {results['status']}")


if __name__ == "__main__":
    asyncio.run(main())

