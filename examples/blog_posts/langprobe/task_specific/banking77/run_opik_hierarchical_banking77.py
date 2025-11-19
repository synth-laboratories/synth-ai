#!/usr/bin/env python3
"""Run Opik MetaPromptOptimizer optimization on Banking77 intent classification."""

from __future__ import annotations

import asyncio
from pathlib import Path

from opik_banking77_adapter import run_opik_hierarchical_banking77


async def main():
    """Main entry point for Banking77 Opik HierarchicalReflectiveOptimizer runner."""
    import argparse

    parser = argparse.ArgumentParser(description="Run Opik MetaPromptOptimizer on Banking77")
    parser.add_argument(
        "--task-app-url",
        default="http://127.0.0.1:8102",
        help="Task app URL (for reference)",
    )
    parser.add_argument(
        "--rollout-budget",
        type=int,
        default=200,
        help="Rollout budget (default: 200)",
    )
    parser.add_argument(
        "--max-trials",
        type=int,
        default=5,
        help="Maximum optimization trials (default: 5)",
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=100,
        help="Number of samples per trial (default: 100)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Model name (default: groq/llama-3.1-8b-instant)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory",
    )

    args = parser.parse_args()

    # Run optimization with seeds matching other adapters
    train_seeds = list(range(50))  # 0-49: 50 training examples
    val_seeds = list(range(50, 80))  # 50-79: 30 validation examples

    results = await run_opik_hierarchical_banking77(
        task_app_url=args.task_app_url,
        train_seeds=train_seeds,
        val_seeds=val_seeds,
        rollout_budget=args.rollout_budget,
        max_trials=args.max_trials,
        n_samples=args.n_samples,
        model=args.model,
        output_dir=args.output_dir,
    )

    print(f"\n✅ Opik MetaPromptOptimizer Optimization complete!")
    print(f"   Baseline score: {results['baseline_score']:.4f}")
    print(f"   Best score: {results['best_score']:.4f}")
    print(f"   Lift: {results['lift']:.4f}")
    print(f"   Total rollouts: {results['total_rollouts']}")
    print(f"   Elapsed time: {results['elapsed_time']:.1f}s")


if __name__ == "__main__":
    asyncio.run(main())

