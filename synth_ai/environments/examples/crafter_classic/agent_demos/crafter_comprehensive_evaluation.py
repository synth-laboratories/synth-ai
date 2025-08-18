#!/usr/bin/env python3
"""
Run script for Full Enchilada Crafter Evaluation
"""

import argparse
import asyncio

from src.synth_env.examples.crafter_classic.agent_demos.full_enchilada import (
    run_full_enchilada_eval,
)


async def main():
    parser = argparse.ArgumentParser(description="Run Full Enchilada Crafter Evaluation")
    parser.add_argument(
        "--models", nargs="+", default=["gpt-4o-mini"], help="Model names to evaluate"
    )
    parser.add_argument(
        "--difficulties",
        nargs="+",
        default=["easy", "hard"],
        help="Difficulty levels to test",
    )
    parser.add_argument(
        "--num-trajectories",
        type=int,
        default=3,
        help="Number of trajectories per condition",
    )
    parser.add_argument("--max-turns", type=int, default=30, help="Maximum turns per trajectory")
    parser.add_argument("--no-images", action="store_true", help="Disable image capture")
    parser.add_argument(
        "--no-viewer",
        action="store_true",
        help="Don't launch the viewer after evaluation",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory (default: src/evals/crafter/run_TIMESTAMP)",
    )

    args = parser.parse_args()

    await run_full_enchilada_eval(
        model_names=args.models,
        difficulties=args.difficulties,
        num_trajectories=args.num_trajectories,
        max_turns=args.max_turns,
        capture_images=not args.no_images,
        launch_viewer=not args.no_viewer,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    asyncio.run(main())
