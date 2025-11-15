#!/usr/bin/env python3
"""Run DSPy GEPA optimization on Banking77 intent classification."""

from __future__ import annotations

import asyncio
from pathlib import Path

from dspy_banking77_adapter import run_dspy_gepa_banking77


async def main():
    """Main entry point for Banking77 GEPA runner."""
    import argparse

    parser = argparse.ArgumentParser(description="Run DSPy GEPA on Banking77")
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
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory",
    )

    args = parser.parse_args()

    # Run GEPA optimization with seeds matching banking77_gepa.toml
    train_seeds = list(range(50))  # 0-49: 50 training examples
    val_seeds = list(range(50, 80))  # 50-79: 30 validation examples

    results = await run_dspy_gepa_banking77(
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
