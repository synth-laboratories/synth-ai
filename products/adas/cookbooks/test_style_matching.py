#!/usr/bin/env python3
"""
ADAS Style Matching Test
========================

Demonstrates ADAS workflow optimization with contrastive judging for
style matching tasks.

Usage:
    # CLI
    uvx synth-ai train --type adas --dataset products/adas/cookbooks/style_matching_dataset.json --poll

    # Or run this script directly
    uv run python products/adas/cookbooks/test_style_matching.py

Requirements:
    - SYNTH_API_KEY in environment
    - BACKEND_BASE_URL (optional, defaults to production)
"""

import os
import sys
from pathlib import Path

from synth_ai.sdk import ADASJob, load_adas_taskset


DATASET_PATH = Path(__file__).parent / "style_matching_dataset.json"


def main() -> None:
    from dotenv import load_dotenv

    load_dotenv()

    api_key = os.getenv("SYNTH_API_KEY")
    if not api_key:
        print("Error: SYNTH_API_KEY required")
        sys.exit(1)

    backend_url = os.getenv("BACKEND_BASE_URL")

    print("=" * 80)
    print("ADAS Style Matching Test")
    print("=" * 80)
    print(f"Backend URL: {backend_url or '(default)'}")
    print()

    print(f"Loading dataset: {DATASET_PATH}")
    dataset = load_adas_taskset(DATASET_PATH)

    print(f"Dataset: {dataset.metadata.name}")
    print(f"  Tasks: {len(dataset.tasks)}")
    print(f"  Gold outputs: {len(dataset.gold_outputs)}")
    print(f"  Judge mode: {dataset.judge_config.mode}")
    print()

    print("Creating ADAS job...")
    job = ADASJob.from_dataset(
        dataset=dataset,
        policy_model="gpt-4o-mini",
        rollout_budget=120,
        proposer_effort="medium",
        backend_url=backend_url,
        api_key=api_key,
        auto_start=True,
    )

    print("Submitting job...")
    try:
        result = job.submit()
    except RuntimeError as e:
        print(f"Error submitting job: {e}")
        sys.exit(1)

    print(f"  ADAS Job ID: {result.adas_job_id}")
    print(f"  Status: {result.status}")
    print()

    print("Streaming job progress...")
    try:
        final_status = job.stream_until_complete(timeout=3600.0, interval=5.0)
    except TimeoutError as e:
        print(f"Timeout: {e}")
        sys.exit(1)

    status = final_status.get("status") if isinstance(final_status, dict) else "unknown"
    print()
    print(f"Final status: {status}")

    if status in ("succeeded", "completed"):
        print()
        print("=" * 80)
        print("Optimized Prompt")
        print("=" * 80)
        try:
            prompt = job.download_prompt()
            print(prompt)
        except Exception as e:
            print(f"Could not download prompt: {e}")

        print()
        print("=" * 80)
        print("Example Inference")
        print("=" * 80)
        example_input = dataset.tasks[0].input
        try:
            out = job.run_inference(example_input)
            print(out)
        except Exception as e:
            print(f"Inference failed: {e}")

    print()
    print("=" * 80)
    print("Done!")
    print("=" * 80)


if __name__ == "__main__":
    main()

