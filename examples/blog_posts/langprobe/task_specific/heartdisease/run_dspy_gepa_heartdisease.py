#!/usr/bin/env python3
"""Run DSPy GEPA optimization on Heart Disease classification."""

from __future__ import annotations

import asyncio
from pathlib import Path

from dspy_heartdisease_adapter import run_dspy_gepa_heartdisease


async def main():
    """Main entry point for Heart Disease GEPA runner."""
    import argparse

    parser = argparse.ArgumentParser(description="Run DSPy GEPA on Heart Disease")
    parser.add_argument(
        "--task-app-url",
        default="http://127.0.0.1:8114",
        help="Task app URL (for reference)",
    )
    parser.add_argument(
        "--rollout-budget",
        type=int,
        default=300,
        help="Rollout budget (default: 300)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory",
    )

    args = parser.parse_args()

    # Run GEPA optimization with seeds matching heartdisease_gepa.toml
    train_seeds = list(range(30))  # 0-29: 30 training examples
    val_seeds = list(range(30, 80))  # 30-79: 50 validation examples

    results = await run_dspy_gepa_heartdisease(
        task_app_url=args.task_app_url,
        train_seeds=train_seeds,
        val_seeds=val_seeds,
        rollout_budget=args.rollout_budget,
        output_dir=args.output_dir,
    )

    print(f"\n✅ GEPA Optimization complete!")
    print(f"   Best score: {results['best_score']:.4f}")
    print(f"   Val score: {results.get('val_score', 'N/A')}")
    print(f"   Total rollouts: {results['total_rollouts']}")
    print(f"   Prompt file: {results.get('prompt_file', 'N/A')}")


if __name__ == "__main__":
    asyncio.run(main())
