#!/usr/bin/env python3
"""Run all DSPy scaling law experiments (1, 3, 5 LLM calls) for HotpotQA and HeartDisease."""

import asyncio
import subprocess
import sys
from pathlib import Path

# Benchmarks to run
BENCHMARKS = ["hotpotqa", "heartdisease"]
NUM_CALLS = [1, 3, 5]

# Rollout budgets (matching original configs)
BUDGETS = {
    "hotpotqa": 200,
    "heartdisease": 300,
}


async def run_experiment(benchmark: str, num_calls: int):
    """Run a single experiment."""
    script_path = Path(__file__).parent / benchmark / f"dspy_{benchmark}_scaling_adapter.py"
    
    if not script_path.exists():
        print(f"❌ Script not found: {script_path}")
        return False

    print(f"\n{'='*80}")
    print(f"Running: {benchmark.upper()} with {num_calls} LLM call(s)")
    print(f"{'='*80}\n")

    cmd = [
        sys.executable,
        str(script_path),
        "--num-calls", str(num_calls),
        "--rollout-budget", str(BUDGETS[benchmark]),
    ]

    try:
        result = subprocess.run(cmd, check=True, capture_output=False)
        print(f"\n✅ Completed: {benchmark} with {num_calls} calls")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Failed: {benchmark} with {num_calls} calls")
        print(f"Error: {e}")
        return False


async def main():
    """Run all experiments."""
    print("🚀 Starting DSPy Scaling Law Experiments")
    print(f"Benchmarks: {', '.join(BENCHMARKS)}")
    print(f"LLM Call Configurations: {', '.join(map(str, NUM_CALLS))}")
    print(f"\nTotal experiments: {len(BENCHMARKS) * len(NUM_CALLS)}")

    results = {}
    for benchmark in BENCHMARKS:
        results[benchmark] = {}
        for num_calls in NUM_CALLS:
            success = await run_experiment(benchmark, num_calls)
            results[benchmark][num_calls] = success

    # Print summary
    print("\n" + "="*80)
    print("EXPERIMENT SUMMARY")
    print("="*80)
    for benchmark in BENCHMARKS:
        print(f"\n{benchmark.upper()}:")
        for num_calls in NUM_CALLS:
            status = "✅" if results[benchmark][num_calls] else "❌"
            print(f"  {num_calls} call(s): {status}")

    # Check if all succeeded
    all_success = all(
        results[b][c] for b in BENCHMARKS for c in NUM_CALLS
    )
    
    if all_success:
        print("\n✅ All experiments completed successfully!")
    else:
        print("\n⚠️  Some experiments failed. Check logs above.")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())


