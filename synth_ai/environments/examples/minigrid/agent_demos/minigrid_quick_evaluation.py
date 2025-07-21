#!/usr/bin/env python3
"""
Run Gemini 1.5 Flash evaluation on MiniGrid tasks.
"""

import asyncio
from eval_framework import run_minigrid_eval, get_success_rate


async def run_gemini_evaluation():
    """Run Gemini 1.5 Flash on 5 instances and display results."""

    print("🚀 Running Gemini 1.5 Flash MiniGrid Evaluation")
    print("=" * 60)
    print("Model: gemini-1.5-flash-latest")
    print("Instances: 5 trajectories per condition")
    print("Tasks: Empty-5x5-v0, DoorKey-5x5-v0")
    print("Difficulties: easy, medium")
    print("=" * 60)

    # Run the evaluation
    report = await run_minigrid_eval(
        model_names=["gemini-1.5-flash-latest"],
        difficulties=["easy", "medium"],
        task_types=["Empty-5x5-v0", "DoorKey-5x5-v0"],  # Start with simple tasks
        num_trajectories=5,  # 5 instances as requested
        max_turns=30,
    )

    print("\n" + "=" * 60)
    print("📊 QUICK SUCCESS RATE SUMMARY")
    print("=" * 60)

    # Extract success rates for quick reference
    for difficulty in ["easy", "medium"]:
        success_rate = get_success_rate(report, "gemini-1.5-flash-latest", difficulty)
        print(f"Gemini 1.5 Flash ({difficulty:6}): {success_rate:5.1f}%")

    overall_success = get_success_rate(report, "gemini-1.5-flash-latest")
    print(f"Overall Average:              {overall_success:5.1f}%")

    return report


if __name__ == "__main__":
    # Run the evaluation
    asyncio.run(run_gemini_evaluation())
