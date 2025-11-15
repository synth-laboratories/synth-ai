#!/usr/bin/env python3
"""Run Synth MIPRO on HotpotQA."""

import argparse
import asyncio
import sys
from pathlib import Path

# Add repo root to path
REPO_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO_ROOT))

from .synth_hotpotqa_adapter import run_synth_mipro_hotpotqa_inprocess


async def main():
    """Run Synth MIPRO on HotpotQA."""
    parser = argparse.ArgumentParser(description="Run Synth MIPRO on HotpotQA")
    parser.add_argument(
        "--task-app-url",
        default="http://127.0.0.1:8110",
        help="Task app URL (default: http://127.0.0.1:8110)",
    )
    parser.add_argument(
        "--rollout-budget",
        type=int,
        default=400,
        help="Rollout budget (default: 400)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory (default: results/synth_mipro/)",
    )

    args = parser.parse_args()

    await run_synth_mipro_hotpotqa_inprocess(
        task_app_url=args.task_app_url,
        rollout_budget=args.rollout_budget,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    asyncio.run(main())

