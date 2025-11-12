#!/usr/bin/env python3
"""Quick runner script for DSPy GEPA on Iris with modest budget."""

import argparse
import asyncio
import sys
from pathlib import Path

# Add repo root to path
REPO_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO_ROOT))

from .dspy_iris_adapter import run_dspy_gepa_iris


async def main():
    """Run DSPy GEPA with modest budget."""
    parser = argparse.ArgumentParser(description="Run DSPy GEPA on Iris")
    parser.add_argument(
        "--rollout-budget",
        type=int,
        default=50,
        help="Rollout budget (default: 50)",
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
        help="Output directory (default: results/dspy_gepa/)",
    )
    
    args = parser.parse_args()
    
    results = await run_dspy_gepa_iris(
        task_app_url=args.task_app_url,
        rollout_budget=args.rollout_budget,
        output_dir=args.output_dir,
    )
    
    print(f"\n✅ DSPy GEPA Optimization Complete!")
    print(f"   Best score: {results['best_score']:.4f}")
    print(f"   Total rollouts: {results['total_rollouts']}")
    print(f"   Max metric calls: {results.get('max_metric_calls', 'N/A')}")
    print(f"   Val score: {results.get('val_score')}")


if __name__ == "__main__":
    asyncio.run(main())

