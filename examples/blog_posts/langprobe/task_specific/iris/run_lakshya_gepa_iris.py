#!/usr/bin/env python3
"""Runner script for Lakshya's GEPA on Iris."""

import argparse
import asyncio
import sys
from pathlib import Path

# Add repo root to path
REPO_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO_ROOT))

from .lakshya_gepa_adapter import run_lakshya_gepa_iris


async def main():
    """Run Lakshya's GEPA with modest budget."""
    parser = argparse.ArgumentParser(description="Run Lakshya's GEPA on Iris")
    parser.add_argument(
        "--rollout-budget",
        type=int,
        default=100,
        help="Rollout budget (default: 100)",
    )
    parser.add_argument(
        "--task-app-url",
        default="http://127.0.0.1:8115",
        help="Task app URL (default: http://127.0.0.1:8115)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory (default: iris/results/lakshya_gepa)",
    )

    args = parser.parse_args()

    results = await run_lakshya_gepa_iris(
        task_app_url=args.task_app_url,
        rollout_budget=args.rollout_budget,
        output_dir=args.output_dir,
    )

    print(f"\n✅ Lakshya GEPA Optimization Complete!")
    print(f"   Best score: {results['best_score']:.4f}")
    print(f"   Val score: {results.get('val_score', 'N/A')}")
    print(f"   Total rollouts: {results['total_rollouts']}")
    print(f"   Prompt file: {results.get('prompt_file', 'N/A')}")


if __name__ == "__main__":
    asyncio.run(main())

