#!/usr/bin/env python3
"""
ADAS Banking77 Intent Classification Test
==========================================

Demonstrates ADAS workflow optimization with rubric-based judge mode
for intent classification tasks.

Usage:
    # CLI
    uvx synth-ai train --type adas --dataset products/adas/cookbooks/banking77_dataset.json --poll

    # Or run this script directly
    uv run python products/adas/cookbooks/test_banking77.py

Requirements:
    - SYNTH_API_KEY in environment
    - BACKEND_BASE_URL (optional, defaults to production)
    
Note: For integration testing, ensure your backend has ADAS routes enabled.
Set BACKEND_BASE_URL to your dev or local backend to override production.
"""

import os
import sys
from pathlib import Path

from synth_ai.sdk import ADASJob, load_adas_taskset


DATASET_PATH = Path(__file__).parent / "banking77_dataset.json"


def main():
    """Run ADAS banking77 intent classification optimization."""

    # Load environment
    from dotenv import load_dotenv
    load_dotenv()

    api_key = os.getenv("SYNTH_API_KEY")
    if not api_key:
        print("Error: SYNTH_API_KEY required")
        sys.exit(1)

    # Default to production unless BACKEND_BASE_URL is set
    backend_url = os.getenv("BACKEND_BASE_URL")

    print("=" * 80)
    print("ADAS Banking77 Intent Classification Test")
    print("=" * 80)
    print(f"Backend URL: {backend_url or '(default)'}")
    print()

    # Load dataset
    print(f"Loading dataset: {DATASET_PATH}")
    dataset = load_adas_taskset(DATASET_PATH)

    print(f"Dataset: {dataset.metadata.name}")
    print(f"  Tasks: {len(dataset.tasks)}")
    print(f"  Gold outputs: {len(dataset.gold_outputs)}")
    print(f"  Judge mode: {dataset.judge_config.mode}")
    print(f"  Judge model: {dataset.judge_config.model}")
    print()

    # Show rubric criteria
    if dataset.default_rubric and dataset.default_rubric.outcome:
        print("Rubric criteria:")
        for c in dataset.default_rubric.outcome.criteria:
            print(f"  - {c.name} (weight={c.weight}): {c.description}")
        print()

    # Create ADAS job
    print("Creating ADAS job...")
    job = ADASJob.from_dataset(
        dataset=dataset,
        policy_model="gpt-4o-mini",
        rollout_budget=100,  # More budget for classification task
        proposer_effort="medium",
        backend_url=backend_url,
        api_key=api_key,
        auto_start=True,
    )

    print(f"  Policy model: {job.config.policy_model}")
    print(f"  Rollout budget: {job.config.rollout_budget}")
    print(f"  Proposer effort: {job.config.proposer_effort}")
    print()

    # Submit job
    print("Submitting job...")
    try:
        result = job.submit()
        print(f"  ADAS Job ID: {result.adas_job_id}")
        print(f"  Status: {result.status}")
    except RuntimeError as e:
        print(f"Error submitting job: {e}")
        sys.exit(1)

    print()
    print("=" * 80)
    print("Streaming job progress...")
    print("=" * 80)
    print()

    # Stream until complete
    try:
        final_status = job.stream_until_complete(
            timeout=3600.0,  # 1 hour
            interval=5.0,
        )
    except TimeoutError as e:
        print(f"Timeout: {e}")
        sys.exit(1)

    status = final_status.get("status") if isinstance(final_status, dict) else "unknown"
    print()
    print(f"Final status: {status}")

    # Download optimized prompt if successful
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

        # Test inference with multiple examples
        print()
        print("=" * 80)
        print("Test Inference")
        print("=" * 80)

        test_queries = [
            {"query": "I want to cancel my credit card", "context": "Customer wants to close account"},
            {"query": "Why was I charged twice for the same purchase?", "context": "Duplicate charge issue"},
            {"query": "Can I increase my credit limit?", "context": "Customer requesting limit increase"},
        ]

        for i, test_input in enumerate(test_queries, 1):
            print(f"\nTest {i}:")
            print(f"  Query: {test_input['query']}")
            try:
                output = job.run_inference(test_input)
                print(f"  Predicted intent: {output}")
            except Exception as e:
                print(f"  Inference failed: {e}")

    print()
    print("=" * 80)
    print("Done!")
    print("=" * 80)


if __name__ == "__main__":
    main()
