#!/usr/bin/env python3
"""
Run script for Crafter dataset instances
"""

import json
import argparse
import random
from pathlib import Path
from typing import List, Optional, Dict, Any
import uuid
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))

from crafter import Env


class CrafterDatasetRunner:
    """Run Crafter instances from a dataset"""

    def __init__(self, dataset_path: Path = Path("dataset")):
        self.dataset_path = dataset_path

    def load_dataset(self, dataset_name: str) -> Dict[str, Any]:
        """Load a dataset from disk"""
        dataset_dir = self.dataset_path / dataset_name

        # Load metadata
        with open(dataset_dir / "metadata.json", "r") as f:
            metadata = json.load(f)

        # Load instances
        with open(dataset_dir / "instances.json", "r") as f:
            instances = json.load(f)

        return {"metadata": metadata, "instances": instances}

    def filter_instances(
        self,
        instances: List[Dict[str, Any]],
        difficulties: Optional[List[str]] = None,
        impetus_types: Optional[List[str]] = None,
        split: Optional[str] = None,
        split_info: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """Filter instances based on criteria"""
        filtered = instances

        # Filter by difficulty
        if difficulties:
            filtered = [inst for inst in filtered if inst["metadata"]["difficulty"] in difficulties]

        # Filter by impetus type
        if impetus_types:
            filtered = [inst for inst in filtered if self._get_impetus_type(inst) in impetus_types]

        # Filter by split
        if split and split_info:
            if split == "train":
                val_ids = set(split_info["val_instance_ids"])
                test_ids = set(split_info["test_instance_ids"])
                filtered = [
                    inst
                    for inst in filtered
                    if inst["id"] not in val_ids and inst["id"] not in test_ids
                ]
            elif split == "val":
                val_ids = set(split_info["val_instance_ids"])
                filtered = [inst for inst in filtered if inst["id"] in val_ids]
            elif split == "test":
                test_ids = set(split_info["test_instance_ids"])
                filtered = [inst for inst in filtered if inst["id"] in test_ids]

        return filtered

    def _get_impetus_type(self, instance: Dict[str, Any]) -> str:
        """Determine impetus type from instructions"""
        instructions = instance["impetus"]["instructions"].lower()
        if "speedrun" in instructions:
            return "speedrun"
        elif "focus on" in instructions:
            return "focused"
        else:
            return "general"

    def run_instance(
        self, instance: Dict[str, Any], render: bool = False, max_steps: int = 1000, agent_fn=None
    ):
        """Run a single instance"""

        # Extract parameters
        difficulty = instance["metadata"]["difficulty"]
        seed = instance["metadata"]["world_seed"]

        print(f"\n{'=' * 60}")
        print(f"Running instance: {instance['id']}")
        print(f"Difficulty: {difficulty}")
        print(f"Seed: {seed}")
        print(f"Instructions: {instance['impetus']['instructions']}")
        if instance["impetus"].get("achievement_focus"):
            print(f"Focus: {', '.join(instance['impetus']['achievement_focus'])}")
        print(f"{'=' * 60}")

        # Create environment
        env = Env(seed=seed, world_config=difficulty)

        obs = env.reset()

        # Run agent or random policy
        total_reward = 0
        achievements = set()

        for step in range(max_steps):
            if agent_fn:
                action = agent_fn(obs, instance)
            else:
                action = env.action_space.sample()

            obs, reward, done, info = env.step(action)
            total_reward += reward

            # Track achievements
            if "achievements" in info:
                for ach, unlocked in info["achievements"].items():
                    if unlocked:
                        achievements.add(ach)

            if done:
                break

        # Evaluate based on intent
        success = self._evaluate_instance(instance, achievements, total_reward, step)

        print(f"\nResults:")
        print(f"Steps: {step}")
        print(f"Total reward: {total_reward}")
        print(f"Achievements: {len(achievements)} - {list(achievements)}")
        print(f"Success: {success}")

        return {
            "instance_id": instance["id"],
            "difficulty": difficulty,
            "seed": seed,
            "steps": step,
            "total_reward": total_reward,
            "achievements": list(achievements),
            "success": success,
        }

    def _evaluate_instance(
        self, instance: Dict[str, Any], achievements: set, total_reward: float, steps: int
    ) -> bool:
        """Evaluate if instance was successful based on intent"""
        intent = instance["intent"]

        # Check minimum score
        if intent.get("minimum_score"):
            if len(achievements) < intent["minimum_score"]:
                return False

        # Check target achievements
        if intent.get("target_achievements"):
            targets = set(intent["target_achievements"])
            if not achievements.intersection(targets):
                return False

        return True

    def run_batch(
        self,
        dataset_name: str,
        num_instances: int = 10,
        difficulties: Optional[List[str]] = None,
        impetus_types: Optional[List[str]] = None,
        split: Optional[str] = None,
        render: bool = False,
        max_steps: int = 1000,
        agent_fn=None,
    ):
        """Run a batch of instances"""

        # Load dataset
        dataset = self.load_dataset(dataset_name)
        instances = dataset["instances"]

        # Filter instances
        filtered = self.filter_instances(
            instances,
            difficulties=difficulties,
            impetus_types=impetus_types,
            split=split,
            split_info=dataset["metadata"].get("split_info"),
        )

        if not filtered:
            print("No instances match the filter criteria!")
            return []

        # Sample instances
        if num_instances > len(filtered):
            print(f"Only {len(filtered)} instances available, running all")
            selected = filtered
        else:
            selected = random.sample(filtered, num_instances)

        print(f"\nRunning {len(selected)} instances from {dataset_name}")
        print(f"Difficulties: {difficulties or 'all'}")
        print(f"Impetus types: {impetus_types or 'all'}")
        print(f"Split: {split or 'all'}")

        # Run instances
        results = []
        for instance in selected:
            result = self.run_instance(
                instance, render=render, max_steps=max_steps, agent_fn=agent_fn
            )
            results.append(result)

        # Summary statistics
        self._print_summary(results)

        return results

    def _print_summary(self, results: List[Dict[str, Any]]):
        """Print summary statistics"""
        print(f"\n{'=' * 60}")
        print("SUMMARY")
        print(f"{'=' * 60}")

        # Group by difficulty
        by_difficulty = {}
        for result in results:
            diff = result["difficulty"]
            if diff not in by_difficulty:
                by_difficulty[diff] = []
            by_difficulty[diff].append(result)

        print(f"\nResults by difficulty:")
        print(
            f"{'Difficulty':<15} {'Count':<8} {'Success':<10} {'Avg Steps':<12} {'Avg Achievements'}"
        )
        print("-" * 60)

        for diff in sorted(by_difficulty.keys()):
            diff_results = by_difficulty[diff]
            count = len(diff_results)
            success_rate = sum(1 for r in diff_results if r["success"]) / count
            avg_steps = sum(r["steps"] for r in diff_results) / count
            avg_achievements = sum(len(r["achievements"]) for r in diff_results) / count

            print(
                f"{diff:<15} {count:<8} {success_rate:<10.1%} {avg_steps:<12.1f} {avg_achievements:.1f}"
            )

        # Overall stats
        total_success = sum(1 for r in results if r["success"])
        print(
            f"\nOverall success rate: {total_success}/{len(results)} ({total_success / len(results):.1%})"
        )


def main():
    parser = argparse.ArgumentParser(description="Run Crafter dataset instances")
    parser.add_argument("dataset", help="Dataset name")
    parser.add_argument(
        "-n", "--num-instances", type=int, default=10, help="Number of instances to run"
    )
    parser.add_argument(
        "-d",
        "--difficulties",
        nargs="+",
        choices=["easy", "normal", "hard", "peaceful", "resource_rich"],
        help="Filter by difficulties",
    )
    parser.add_argument(
        "-t",
        "--impetus-types",
        nargs="+",
        choices=["general", "focused", "speedrun"],
        help="Filter by impetus types",
    )
    parser.add_argument(
        "-s", "--split", choices=["train", "val", "test"], help="Filter by dataset split"
    )
    parser.add_argument("--render", action="store_true", help="Render the environment")
    parser.add_argument("--max-steps", type=int, default=1000, help="Maximum steps per episode")

    args = parser.parse_args()

    runner = CrafterDatasetRunner()
    runner.run_batch(
        dataset_name=args.dataset,
        num_instances=args.num_instances,
        difficulties=args.difficulties,
        impetus_types=args.impetus_types,
        split=args.split,
        render=args.render,
        max_steps=args.max_steps,
    )


if __name__ == "__main__":
    main()
