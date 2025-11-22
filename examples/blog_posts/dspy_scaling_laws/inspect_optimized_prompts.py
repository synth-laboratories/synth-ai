#!/usr/bin/env python3
"""Inspect and analyze optimized prompts from GEPA."""

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

async def fetch_prompts():
    """Fetch and analyze optimized prompts."""
    client = PromptLearningClient(ensure_api_base(backend_url), api_key)

    print(f"Fetching prompts for job: {job_id}")
    print(f"Backend: {backend_url}\n")

    # Get job results with full details
    job_results = await client.get_job(job_id)

    # Check available fields
    print("=" * 80)
    print("Available fields in job_results:")
    print("=" * 80)
    for key in job_results.keys():
        print(f"  - {key}")
    print()

    # Try to get the best snapshot
    best_snapshot_id = job_results.get('best_snapshot_id')
    if best_snapshot_id:
        print(f"Best snapshot ID: {best_snapshot_id}")
        print("\nAttempting to fetch best snapshot details...")

        # Try different methods to get the snapshot
        try:
            # Method 1: Check if snapshot is in job_results
            best_snapshot = job_results.get('best_snapshot')
            if best_snapshot:
                print("\n✅ Found best_snapshot in job results")
                return analyze_snapshot(best_snapshot)
        except Exception as e:
            print(f"Could not get snapshot from job results: {e}")

        # Method 2: Try to fetch snapshot separately
        try:
            snapshot = await client.get_snapshot(best_snapshot_id)
            print("\n✅ Retrieved snapshot via get_snapshot()")
            return analyze_snapshot(snapshot)
        except AttributeError:
            print("get_snapshot() method not available")
        except Exception as e:
            print(f"Error fetching snapshot: {e}")

    # Method 3: Check metadata for candidate prompts
    metadata = job_results.get('metadata', {})
    attempted_candidates = metadata.get('attempted_candidates', [])

    if attempted_candidates:
        print(f"\n✅ Found {len(attempted_candidates)} attempted candidates in metadata")
        analyze_candidates(attempted_candidates)
    else:
        print("\n⚠️  No candidates found in metadata")

    # Save full job results for inspection
    results_dir = Path(__file__).parent / "results" / "gepa_banking77_2step"
    results_file = results_dir / "full_job_results.json"
    with open(results_file, 'w') as f:
        json.dump(job_results, f, indent=2, default=str)
    print(f"\n💾 Full job results saved to: {results_file}")

    return job_results

def analyze_snapshot(snapshot):
    """Analyze a prompt snapshot."""
    print("\n" + "=" * 80)
    print("OPTIMIZED PROMPT ANALYSIS")
    print("=" * 80)

    prompt_data = snapshot.get('prompt_data', {})
    prompt_metadata = prompt_data.get('prompt_metadata', {})
    pipeline_modules = prompt_metadata.get('pipeline_modules', [])

    if pipeline_modules:
        print(f"\n📋 Pipeline has {len(pipeline_modules)} modules:\n")

        for i, module in enumerate(pipeline_modules, 1):
            name = module.get('name', 'unknown')
            instruction = module.get('instruction_text', 'N/A')

            print(f"Module {i}: {name}")
            print("-" * 80)
            print(f"Instruction:\n{instruction}\n")

    return snapshot

def analyze_candidates(candidates):
    """Analyze candidate prompts to find the best one."""
    print("\n" + "=" * 80)
    print("ANALYZING PROMPT CANDIDATES")
    print("=" * 80)

    # Find the best performing candidate
    best_candidate = None
    best_score = -1

    for i, candidate in enumerate(candidates):
        candidate_type = candidate.get('type')
        obj = candidate.get('object', {})
        metadata = obj.get('metadata', {})

        # Try to extract score
        score = None
        rollout_responses = metadata.get('rollout_responses', [])
        if rollout_responses:
            # Average score across rollouts
            scores = []
            for response in rollout_responses:
                metrics = response.get('metrics', {})
                mean_return = metrics.get('mean_return')
                if mean_return is not None:
                    scores.append(mean_return)
            if scores:
                score = sum(scores) / len(scores)

        if score is not None and score > best_score:
            best_score = score
            best_candidate = candidate

        print(f"\nCandidate {i + 1} (type: {candidate_type})")
        if score is not None:
            print(f"  Score: {score:.3f}")

    if best_candidate:
        print("\n" + "=" * 80)
        print(f"BEST CANDIDATE (Score: {best_score:.3f})")
        print("=" * 80)

        obj = best_candidate.get('object', {})
        prompt_str = obj.get('prompt', '')

        if prompt_str:
            print(f"\nPrompt:\n{prompt_str}")

        # Try to extract module-level instructions
        metadata = obj.get('metadata', {})
        if 'prompt_metadata' in metadata:
            prompt_metadata = metadata['prompt_metadata']
            if 'pipeline_modules' in prompt_metadata:
                modules = prompt_metadata['pipeline_modules']
                print(f"\n📋 Pipeline Modules ({len(modules)}):\n")
                for module in modules:
                    name = module.get('name', 'unknown')
                    instruction = module.get('instruction_text', 'N/A')
                    print(f"{name}:")
                    print("-" * 80)
                    print(f"{instruction}\n")

# Run the analysis
if __name__ == "__main__":
    try:
        asyncio.run(fetch_prompts())
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
