#!/usr/bin/env python3
"""Run DSPy GEPA in parallel for Heart Disease, HotPotQA, and Banking77 tasks."""

import asyncio
import json
import subprocess
import sys
import yaml
from pathlib import Path
from typing import Dict, List, Optional


# Task configurations
# Note: Script paths are relative to langprobe directory, not comparisons directory
TASKS = {
    "Heart Disease": {
        "script": "task_specific/heartdisease/run_dspy_gepa_heartdisease.py",
        "results_dir": "task_specific/heartdisease/results/dspy_gepa",
        "benchmark_name": "heartdisease",
        "config_key": "heart_disease",
    },
    "HotPotQA": {
        "script": "task_specific/hotpotqa/run_dspy_gepa_hotpotqa.py",
        "results_dir": "task_specific/hotpotqa/results/dspy_gepa",
        "benchmark_name": "hotpotqa",
        "config_key": "hotpotqa",
    },
    "Banking77": {
        "script": "task_specific/banking77/run_dspy_gepa_banking77.py",
        "results_dir": "task_specific/banking77/results/dspy_gepa",
        "benchmark_name": "banking77",
        "config_key": "banking77",
    },
}


def load_config(config_path: Optional[Path] = None) -> Dict:
    """Load configuration from YAML file."""
    if config_path is None:
        config_path = Path(__file__).parent / "dspy_gepa_config.yaml"

    if not config_path.exists():
        print(f"⚠️  Config file not found: {config_path}")
        print("Using default budgets")
        return {
            "budgets": {"heart_disease": 300, "hotpotqa": 200, "banking77": 200},
            "model": {"policy_model": "groq/openai/gpt-oss-20b"},
        }

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    print(f"✓ Loaded config from: {config_path}")
    return config


async def run_dspy_gepa_task(task_name: str, task_config: Dict, budget: Optional[int] = None) -> Dict:
    """Run a single DSPy GEPA task and return results."""
    # Script paths are relative to langprobe directory (parent of comparisons)
    script_path = Path(__file__).parent.parent / task_config["script"]
    budget = budget or task_config["default_budget"]

    print(f"\n🚀 Starting DSPy GEPA for {task_name} (budget={budget})...")

    # Run the script
    cmd = [
        sys.executable, str(script_path),
        "--rollout-budget", str(budget),
    ]

    process = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )

    stdout, stderr = await process.communicate()

    if process.returncode != 0:
        error_msg = stderr.decode()[:500] if stderr else "Unknown error"
        print(f"❌ {task_name} FAILED: {error_msg}")
        return {
            "task": task_name,
            "status": "failed",
            "error": error_msg,
        }

    # Extract results from detailed results JSON file
    # Results paths are relative to langprobe directory (parent of comparisons)
    results_dir = Path(__file__).parent.parent / task_config["results_dir"]
    detailed_results_file = results_dir / "dspy_gepa_detailed_results.json"

    print(f"[{task_name}] Looking for results at: {detailed_results_file}")

    if detailed_results_file.exists():
        try:
            with open(detailed_results_file, "r") as f:
                data = json.load(f)

            print(f"[{task_name}] Found results JSON with {len(data.get('candidates', []))} candidates")

            # Extract final performance score and other stats
            best_score = data.get("best_score", 0.0)
            baseline_score = data.get("baseline_score", 0.0)
            total_time = data.get("total_time")
            total_rollouts = data.get("total_rollouts")
            actual_rollouts = data.get("actual_rollouts")

            print(f"✅ {task_name} completed: best={best_score:.4f}, baseline={baseline_score:.4f}, time={total_time}s")

            # Get policy model from config
            config = load_config()
            policy_model = config.get("model", {}).get("policy_model", "groq/openai/gpt-oss-20b")

            return {
                "task": task_name,
                "status": "completed",
                "baseline_score": baseline_score,
                "final_score": best_score,
                "lift": best_score - baseline_score,
                "total_rollouts": total_rollouts,
                "actual_rollouts": actual_rollouts,
                "total_time": total_time,
                "policy_model": policy_model,
                "num_candidates": len(data.get("candidates", [])),
            }
        except Exception as e:
            print(f"⚠️  {task_name}: Error reading results: {str(e)}")
            import traceback
            traceback.print_exc()

    return {
        "task": task_name,
        "status": "completed",
        "error": "Results file not found or parsing failed",
    }


async def main():
    """Run all DSPy GEPA tasks in parallel and show aggregate stats."""
    import argparse

    parser = argparse.ArgumentParser(description="Run DSPy GEPA in parallel for all tasks")
    parser.add_argument(
        "--rollout-budget",
        type=int,
        default=None,
        help="Override rollout budget for all tasks (ignores config file)",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to config YAML file (default: dspy_gepa_config.yaml)",
    )
    args = parser.parse_args()

    # Load config
    config_path = Path(args.config) if args.config else None
    config = load_config(config_path)

    budgets = config.get("budgets", {})
    policy_model = config.get("model", {}).get("policy_model", "groq/openai/gpt-oss-20b")

    print("=" * 80)
    print("DSPy GEPA PARALLEL RUNNER")
    print("=" * 80)
    print(f"Policy Model: {policy_model}")
    print(f"Budgets: {budgets}")
    print()

    # Run all tasks in parallel
    tasks = []
    for task_name, task_config in TASKS.items():
        config_key = task_config["config_key"]
        budget = args.rollout_budget if args.rollout_budget else budgets.get(config_key, 200)
        tasks.append(run_dspy_gepa_task(task_name, task_config, budget))

    results = await asyncio.gather(*tasks)

    # Print aggregate stats table (analogous to run_gepa_parallel.py format)
    from datetime import datetime

    output_lines = []
    output_lines.append("\n" + "=" * 186)
    output_lines.append("AGGREGATE STATS ACROSS ALL TASKS (dspy_gepa)")
    output_lines.append("=" * 186)
    output_lines.append("")
    output_lines.append(f"{'Task':<20} {'Policy Model':<25} {'Baseline':<12} {'Best':<14} {'Lift':<12} {'Rollouts':<12} {'Time':<12} {'Candidates':<12}")
    output_lines.append("-" * 186)

    valid_results = [r for r in results if r.get("status") == "completed" and "error" not in r]

    for result in results:
        task = result.get("task", "Unknown")
        if result.get("status") != "completed" or "error" in result:
            output_lines.append(f"{task:<20} {'ERROR':<25} {'ERROR':<12} {'ERROR':<14} {'ERROR':<12} {'ERROR':<12} {'ERROR':<12} {'ERROR':<12}")
        else:
            baseline = result.get("baseline_score")
            final = result.get("final_score")
            lift = result.get("lift")
            actual_rollouts = result.get("actual_rollouts")
            total_time = result.get("total_time")
            policy_model = result.get("policy_model", "N/A")
            num_candidates = result.get("num_candidates")

            baseline_str = f"{baseline:.4f}" if baseline is not None else "N/A"
            final_str = f"{final:.4f}" if final is not None else "N/A"
            lift_str = f"{lift:+.4f}" if lift is not None else "N/A"
            rollouts_str = str(actual_rollouts) if actual_rollouts is not None else "N/A"
            candidates_str = str(num_candidates) if num_candidates is not None else "N/A"

            # Format time (convert to minutes if > 60 seconds)
            if total_time is not None:
                if total_time >= 60:
                    time_str = f"{total_time / 60:.1f}m"
                else:
                    time_str = f"{total_time:.1f}s"
            else:
                time_str = "N/A"

            output_lines.append(f"{task:<20} {policy_model:<25} {baseline_str:<12} {final_str:<14} {lift_str:<12} {rollouts_str:<12} {time_str:<12} {candidates_str:<12}")

    if valid_results:
        output_lines.append("-" * 186)
        avg_baseline = sum(r.get("baseline_score", 0) or 0 for r in valid_results) / len(valid_results)
        avg_final = sum(r.get("final_score", 0) or 0 for r in valid_results) / len(valid_results)
        avg_lift = sum(r.get("lift", 0) or 0 for r in valid_results) / len(valid_results)
        total_actual_rollouts = sum(r.get("actual_rollouts", 0) or 0 for r in valid_results)
        total_time = sum(r.get("total_time", 0) or 0 for r in valid_results)
        total_candidates = sum(r.get("num_candidates", 0) or 0 for r in valid_results)

        # Format total time
        if total_time >= 60:
            total_time_str = f"{total_time / 60:.1f}m"
        else:
            total_time_str = f"{total_time:.1f}s"

        output_lines.append(f"{'TOTAL':<20} {'':<25} {'':<12} {'':<14} {'':<12} {total_actual_rollouts:<12} {total_time_str:<12} {total_candidates:<12}")
        output_lines.append(f"{'AVERAGE':<20} {'':<25} {avg_baseline:.4f}     {avg_final:.4f}     {avg_lift:+.4f} {'':<12} {'':<12}")

    output_lines.append("=" * 186)

    # Print to console
    output_text = "\n".join(output_lines)
    print(output_text)

    # Save to file (analogous to run_gepa_parallel.py)
    comparisons_dir = Path(__file__).parent
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = comparisons_dir / f"dspy_gepa_comparison_readout_{timestamp}.txt"

    try:
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(output_text)
        print(f"\n📄 Comparison results saved to: {output_file}")
    except Exception as e:
        print(f"\n⚠️  Could not save comparison results file: {e}")


if __name__ == "__main__":
    asyncio.run(main())
