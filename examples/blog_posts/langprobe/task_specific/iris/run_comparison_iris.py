#!/usr/bin/env python3
"""Run unified comparison of all frameworks on Iris."""

import argparse
import asyncio
import sys
from pathlib import Path

# Add repo root to path
REPO_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO_ROOT))

from .run_comparison import evaluate_on_test_set, run_all_frameworks


async def main():
    """Run comparison of all frameworks."""
    parser = argparse.ArgumentParser(description="Run unified comparison of all frameworks on Iris")
    parser.add_argument(
        "--rollout-budget",
        type=int,
        required=True,
        help="Rollout budget for each framework (e.g., 100 or 500)",
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
        help="Output directory (default: iris/results/comparison/budget_{rollout_budget})",
    )
    parser.add_argument(
        "--skip-test-eval",
        action="store_true",
        help="Skip test set evaluation (only run optimization)",
    )

    args = parser.parse_args()

    # Run all frameworks
    results = await run_all_frameworks(
        rollout_budget=args.rollout_budget,
        task_app_url=args.task_app_url,
        output_base_dir=args.output_dir,
    )

    # Evaluate on test set
    if not args.skip_test_eval:
        test_results = await evaluate_on_test_set(
            framework_results=results,
            task_app_url=args.task_app_url,
        )
        results["test_evaluation"] = test_results

        # Save updated results with test evaluation
        output_dir = Path(results["frameworks"]["synth_gepa"]["output_dir"]).parent
        results_file = output_dir / "comparison_results.json"
        with open(results_file, "w") as f:
            import json
            json.dump(results, f, indent=2)

        print(f"\n{'='*80}")
        print(f"Test Set Evaluation Summary")
        print(f"{'='*80}\n")
        for framework_name, test_result in test_results.items():
            if test_result.get("status") == "completed":
                print(f"  {framework_name:20s} test_accuracy={test_result.get('test_accuracy', 0.0):.4f}")
            else:
                print(f"  {framework_name:20s} {test_result.get('status', 'unknown')}")


if __name__ == "__main__":
    asyncio.run(main())

