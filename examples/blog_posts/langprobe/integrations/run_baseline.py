#!/usr/bin/env python3
"""Baseline runner for Iris with gpt-oss-120b.

Run from repo root:
    python3 examples/blog_posts/langprobe/integrations/run_baseline.py
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

from examples.blog_posts.langprobe.integrations.task_app_client import TaskAppClient


async def run_baseline(
    task_app_url: str = "http://127.0.0.1:8115",
    seeds: Optional[list[int]] = None,
    model: str = "openai/gpt-oss-20b",
    output_path: Optional[Path] = None,
) -> dict:
    """Run baseline evaluation on Iris task app.

    Args:
        task_app_url: Task app URL
        seeds: List of seeds to evaluate (default: 0-9)
        model: Model to use (default: gpt-oss-120b)
        output_path: Path to save results JSON

    Returns:
        Dictionary with metrics: {'accuracy', 'mean_reward', 'num_correct', 'total', 'results'}
    """
    if seeds is None:
        seeds = list(range(10))  # Seeds 0-9 for baseline

    # Initial prompt messages (from iris_task_app.py)
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

    async with TaskAppClient(task_app_url) as client:
        correct_count = 0
        total_reward = 0.0
        results = []

        for seed in seeds:
            response = await client.evaluate_prompt(
                prompt_messages=initial_prompt_messages,
                seed=seed,
                task_app_id="iris",
                model=model,
                provider="groq",
            )

            metrics = response.get("metrics", {})
            outcome_score = metrics.get("outcome_score", 0.0)
            is_correct = int(outcome_score > 0.5)  # 1.0 for correct, 0.0 for incorrect

            correct_count += is_correct
            total_reward += outcome_score

            results.append(
                {
                    "seed": seed,
                    "outcome_score": outcome_score,
                    "correct": is_correct,
                }
            )

        accuracy = correct_count / len(seeds) if seeds else 0.0
        mean_reward = total_reward / len(seeds) if seeds else 0.0

        baseline_results = {
            "accuracy": accuracy,
            "mean_reward": mean_reward,
            "num_correct": correct_count,
            "total": len(seeds),
            "model": model,
            "seeds": seeds,
            "results": results,
        }

        # Save results
        if output_path:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "w") as f:
                json.dump(baseline_results, f, indent=2)

        return baseline_results


async def main() -> None:
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Run Iris baseline with gpt-oss-120b")
    parser.add_argument(
        "--task-app-url",
        default="http://127.0.0.1:8115",
        help="Task app URL",
    )
    parser.add_argument(
        "--seeds",
        type=str,
        help="Comma-separated list of seeds (e.g., '0,1,2') or range (e.g., '0-9')",
    )
    parser.add_argument(
        "--model",
        default="openai/gpt-oss-20b",
        help="Model to use (default: gpt-oss-20b)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output JSON path (default: results/baseline/iris_baseline.json)",
    )

    args = parser.parse_args()

    # Parse seeds
    seeds = None
    if args.seeds:
        if "-" in args.seeds:
            start, end = map(int, args.seeds.split("-"))
            seeds = list(range(start, end + 1))
        else:
            seeds = [int(s.strip()) for s in args.seeds.split(",")]

    output_path = args.output
    if output_path is None:
        output_path = (
            Path(__file__).parent.parent / "results" / "baseline" / "iris_baseline.json"
        )

    results = await run_baseline(
        task_app_url=args.task_app_url,
        seeds=seeds,
        model=args.model,
        output_path=output_path,
    )

    print(f"\nBaseline Results:")
    print(f"  Model: {results['model']}")
    print(f"  Seeds: {results['seeds']}")
    print(f"  Accuracy: {results['accuracy']:.2%}")
    print(f"  Mean Reward: {results['mean_reward']:.4f}")
    print(f"  Correct: {results['num_correct']}/{results['total']}")
    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    asyncio.run(main())
