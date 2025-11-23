#!/usr/bin/env python3
"""Fetch GEPA job results from the backend."""

import os
import sys
import json
import asyncio
from pathlib import Path
from dotenv import load_dotenv

# Load environment
env_path = Path(__file__).resolve().parents[3] / ".env"
load_dotenv(env_path)

# Add parent to path
parent_dir = Path(__file__).resolve().parents[3]
if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))

from synth_ai.learning.prompt_learning_client import PromptLearningClient
from synth_ai.api.train.utils import ensure_api_base

# Configuration
job_id = "pl_f606023a927b47ea"
backend_url = os.getenv("BACKEND_BASE_URL", "https://synth-backend-dev-docker.onrender.com")
api_key = os.getenv("SYNTH_API_KEY")

if not api_key:
    print("❌ Error: SYNTH_API_KEY not found in environment")
    sys.exit(1)

print(f"Fetching results for job: {job_id}")
print(f"Backend: {backend_url}\n")

# Create client
client = PromptLearningClient(
    ensure_api_base(backend_url),
    api_key,
)

async def fetch_results():
    """Fetch job results asynchronously."""
    job_results = await client.get_job(job_id)
    return job_results

# Get job results
try:
    job_results = asyncio.run(fetch_results())

    print("=" * 80)
    print("GEPA Optimization Results - Banking77 2-Step Pipeline")
    print("=" * 80)
    print(f"\nJob ID: {job_id}")
    print(f"Status: {job_results.get('status', 'unknown')}")

    # Extract results from metadata
    metadata = job_results.get('metadata', {})
    stats = metadata.get('stats', {})
    best_score = metadata.get('prompt_best_score')

    print(f"\nPerformance:")
    print(f"  Best Score:     {best_score if best_score is not None else 'N/A'}")
    if isinstance(best_score, (int, float)):
        print(f"                  {best_score * 100:.1f}%")

    print(f"\nExecution:")
    print(f"  Optimization Trials:  {stats.get('optimization_trials_evaluated', 'N/A')}")
    print(f"  Optimization Rollouts: {stats.get('optimization_rollouts_executed', 'N/A')}")
    print(f"  Validation Rollouts:   {stats.get('validation_rollouts_executed', 'N/A')}")
    print(f"  Total Trials Tried:    {stats.get('trials_tried', 'N/A')}")

    # Save full results to file
    results_dir = Path(__file__).parent / "results" / "gepa_banking77_2step"
    results_dir.mkdir(parents=True, exist_ok=True)

    results_file = results_dir / f"job_results_{job_id}.json"
    with open(results_file, 'w') as f:
        json.dump(job_results, f, indent=2, default=str)
    print(f"\n✅ Full results saved to: {results_file}")

    # Get best snapshot (optimized prompt) if available
    best_snapshot = job_results.get("best_snapshot")
    if best_snapshot:
        print(f"\n✅ Optimized prompt snapshot ID: {job_results.get('best_snapshot_id', 'N/A')}")

        # Save just the best snapshot
        snapshot_file = results_dir / f"optimized_prompt_{job_id}.json"
        with open(snapshot_file, 'w') as f:
            json.dump(best_snapshot, f, indent=2, default=str)
        print(f"✅ Optimized prompt saved to: {snapshot_file}")

        # Print the optimized instructions if available
        prompt_data = best_snapshot.get("prompt_data", {})
        prompt_metadata = prompt_data.get("prompt_metadata", {})
        pipeline_modules = prompt_metadata.get("pipeline_modules", [])

        if pipeline_modules:
            print("\n" + "=" * 80)
            print("Optimized Module Instructions")
            print("=" * 80)
            for module in pipeline_modules:
                print(f"\n[{module.get('name', 'unknown')}]")
                print(f"  {module.get('instruction_text', 'N/A')}")
    else:
        print("\n⚠️  No best snapshot found in results")

    print("\n" + "=" * 80)

    # Compare to baseline
    baseline_accuracy = 0.42  # 42% from baseline tests
    if isinstance(best_score, (int, float)):
        improvement = (best_score - baseline_accuracy) * 100
        print(f"\n📊 Comparison to Baseline:")
        print(f"   Baseline:  {baseline_accuracy * 100:.1f}%")
        print(f"   GEPA:      {best_score * 100:.1f}%")
        print(f"   Change:    {improvement:+.1f} percentage points")
        if improvement > 0:
            print(f"   Relative:  {(improvement / (baseline_accuracy * 100)) * 100:.1f}% improvement")

except Exception as e:
    print(f"❌ Error fetching job results: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
