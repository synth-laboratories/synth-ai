#!/usr/bin/env python3
"""
Example usage of Crafter datasets
"""

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))

from run_dataset import CrafterDatasetRunner


def main():
    runner = CrafterDatasetRunner()
    
    print("=== Example 1: Run 5 easy instances ===")
    runner.run_batch(
        dataset_name="crafter_balanced_v1",
        num_instances=5,
        difficulties=["easy"],
        max_steps=500
    )
    
    print("\n\n=== Example 2: Run validation set instances ===")
    runner.run_batch(
        dataset_name="crafter_balanced_v1",
        num_instances=5,
        split="val",
        max_steps=500
    )
    
    print("\n\n=== Example 3: Run speedrun challenges ===")
    runner.run_batch(
        dataset_name="crafter_balanced_v1",
        num_instances=5,
        impetus_types=["speedrun"],
        max_steps=500
    )
    
    print("\n\n=== Example 4: Compare difficulties ===")
    for difficulty in ["easy", "normal", "hard"]:
        print(f"\n--- Testing {difficulty} ---")
        runner.run_batch(
            dataset_name="crafter_progression_v1",
            num_instances=3,
            difficulties=[difficulty],
            max_steps=300
        )


if __name__ == "__main__":
    main()