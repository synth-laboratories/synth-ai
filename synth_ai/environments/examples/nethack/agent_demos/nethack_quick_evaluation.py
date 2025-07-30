#!/usr/bin/env python3
"""
Runner script for NetHack evaluation framework.
"""

import asyncio
import argparse
import time
import os

from eval_framework import run_nethack_eval


async def main():
    """Main evaluation runner."""
    parser = argparse.ArgumentParser(description="Run NetHack evaluation framework")
    parser.add_argument(
        "--models",
        nargs="+",
        default=["gemini-1.5-flash-latest"],
        help="Model names to evaluate",
    )
    parser.add_argument("--difficulties", nargs="+", default=["easy"], help="Difficulty levels")
    parser.add_argument(
        "--num-trajectories",
        type=int,
        default=3,
        help="Number of trajectories per condition",
    )
    parser.add_argument("--max-turns", type=int, default=50, help="Maximum turns per trajectory")

    args = parser.parse_args()

    print(f"Starting NetHack evaluation...")
    print(f"Models: {args.models}")
    print(f"Difficulties: {args.difficulties}")
    print(f"Trajectories per condition: {args.num_trajectories}")
    print(f"Max turns: {args.max_turns}")

    start_time = time.time()

    try:
        report = await run_nethack_eval(
            model_names=args.models,
            difficulties=args.difficulties,
            num_trajectories=args.num_trajectories,
            max_turns=args.max_turns,
        )

        end_time = time.time()
        duration = end_time - start_time

        print(f"\n✅ Evaluation completed in {duration:.1f} seconds")

        # Save report to file
        import json

        output_file = f"nethack_eval_results_{int(time.time())}.json"
        with open(output_file, "w") as f:
            json.dump(report, f, indent=2, default=str)
        print(f"📁 Full report saved to {output_file}")

    except Exception as e:
        print(f"❌ Evaluation failed: {e}")
        import traceback

        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit(asyncio.run(main()))
