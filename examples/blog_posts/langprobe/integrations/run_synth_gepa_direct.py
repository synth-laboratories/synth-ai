#!/usr/bin/env python3
"""Run Synth GEPA directly on Iris (requires backend codebase).

This script runs GEPA optimizer directly, accessing backend code from monorepo.
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

# Try to add backend to path
MONOREPO_ROOT = REPO_ROOT.parent / "monorepo"
BACKEND_ROOT = MONOREPO_ROOT / "backend"
if BACKEND_ROOT.exists():
    sys.path.insert(0, str(BACKEND_ROOT))
    print(f"✓ Added backend to path: {BACKEND_ROOT}")
else:
    print(f"⚠️  Backend not found at {BACKEND_ROOT}")
    print("   Falling back to SDK API approach")

try:
    from app.routes.prompt_learning.algorithm.gepa.optimizer import (
        GEPAOptimizer,
        GEPAConfig,
    )
    from app.routes.prompt_learning.core.runtime import LocalRuntime
    from app.routes.prompt_learning.core.patterns import PromptPattern, MessagePattern
    BACKEND_AVAILABLE = True
except ImportError as e:
    BACKEND_AVAILABLE = False
    print(f"⚠️  Backend imports failed: {e}")
    print("   Use run_synth_gepa_sdk.py instead for SDK API approach")


async def run_synth_gepa_direct(
    task_app_url: str = "http://127.0.0.1:8115",
    train_seeds: Optional[list[int]] = None,
    rollout_budget: int = 400,
    output_dir: Optional[Path] = None,
) -> dict[str, Any]:
    """Run Synth GEPA directly using backend optimizer.

    Args:
        task_app_url: Task app URL
        train_seeds: Training seeds (default: 0-99)
        rollout_budget: Rollout budget (default: 400)
        output_dir: Output directory

    Returns:
        Results dictionary
    """
    if not BACKEND_AVAILABLE:
        raise RuntimeError(
            "Backend not available. Use run_synth_gepa_sdk.py for SDK API approach."
        )

    if train_seeds is None:
        train_seeds = list(range(100))  # Seeds 0-99

    if output_dir is None:
        output_dir = Path(__file__).parent.parent / "results" / "synth_gepa"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Initial prompt messages
    initial_prompt_messages = [
        {
            "role": "system",
            "pattern": (
                "You are a botany classification assistant. Based on the flower's measurements, "
                "classify the iris species. Respond with one of: setosa, versicolor, or virginica."
            ),
        },
        {
            "role": "user",
            "pattern": (
                "Flower Measurements:\n{features}\n\n"
                "Classify this iris flower. Respond with one of: setosa, versicolor, or virginica."
            ),
        },
    ]

    # Create prompt pattern
    message_patterns = [
        MessagePattern(
            role=msg["role"],
            pattern=msg["pattern"],
            order=i,
        )
        for i, msg in enumerate(initial_prompt_messages)
    ]
    initial_pattern = PromptPattern(messages=message_patterns)

    # Create GEPA config
    config = GEPAConfig(
        task_app_url=task_app_url,
        task_app_api_key=os.getenv("ENVIRONMENT_API_KEY", ""),
        env_name="iris",
        initial_population_size=20,
        num_generations=15,
        mutation_rate=0.3,
        crossover_rate=0.5,
        pareto_set_size=64,
        minibatch_size=8,
        feedback_fraction=0.5,
        rollout_budget=rollout_budget,
        policy_config={
            "model": "openai/gpt-oss-20b",
            "provider": "groq",
            "temperature": 1.0,
            "max_completion_tokens": 512,
        },
        mutation_llm_model="openai/gpt-oss-20b",
        mutation_llm_provider="groq",
    )

    # Create runtime and optimizer
    runtime = LocalRuntime()
    optimizer = GEPAOptimizer(config=config, runtime=runtime)
    optimizer.job_id = "iris_gepa_blog_post"

    print(f"🚀 Starting Synth GEPA optimization...")
    print(f"   Task app: {task_app_url}")
    print(f"   Train seeds: {len(train_seeds)} seeds")
    print(f"   Rollout budget: {rollout_budget}")
    print()

    # Run optimization
    result = await optimizer.optimize(
        initial_pattern=initial_pattern,
        train_seeds=train_seeds,
    )

    # Extract results
    best_template = result.best_template
    best_score = result.best_score

    # Save results
    results_file = output_dir / "iris_results.json"
    with open(results_file, "w") as f:
        json.dump(
            {
                "best_score": best_score,
                "best_template": best_template.to_dict() if best_template else {},
                "rollout_budget": rollout_budget,
                "train_seeds": train_seeds,
            },
            f,
            indent=2,
        )

    print(f"\n✅ Optimization complete!")
    print(f"   Best score: {best_score:.4f}")
    print(f"   Results saved to: {results_file}")

    return {
        "best_score": best_score,
        "best_template": best_template.to_dict() if best_template else {},
        "rollout_budget": rollout_budget,
    }


async def main() -> None:
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Run Synth GEPA directly on Iris")
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

    if not BACKEND_AVAILABLE:
        print("Error: Backend not available")
        print("   Make sure monorepo/backend exists and is accessible")
        sys.exit(1)

    results = await run_synth_gepa_direct(
        task_app_url=args.task_app_url,
        rollout_budget=args.rollout_budget,
        output_dir=args.output_dir,
    )

    print(f"\nFinal Results:")
    print(f"  Best Score: {results['best_score']:.4f}")


if __name__ == "__main__":
    asyncio.run(main())

