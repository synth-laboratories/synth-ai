#!/usr/bin/env python3
"""Test script to verify Gemini 2.5 family support in Synth AI GEPA and MIPRO.

This script tests:
1. GEPA with Gemini policy model (synth_hosted)
2. GEPA with Gemini mutation/reflection model
3. MIPRO with Gemini policy model (synth_hosted)
4. MIPRO with Gemini meta model (proposal generation)

Usage:
    # Start backend locally first:
    cd /Users/joshpurtell/Documents/GitHub/monorepo && bash scripts/run_backend_local.sh
    
    # Then run this test:
    cd /Users/joshpurtell/Documents/GitHub/synth-ai && source .env && uv run python examples/blog_posts/langprobe/dspy_scaling_laws/test_gemini_synth_integration.py
"""

import argparse
import asyncio
import os
import sys
import tempfile
from pathlib import Path
from typing import Dict, Any, Optional

from dotenv import load_dotenv

# Add synth-ai to path
REPO_ROOT = Path(__file__).resolve().parents[6]
sys.path.insert(0, str(REPO_ROOT))

load_dotenv()

try:
    from synth_ai.api.train.prompt_learning import PromptLearningJob
    from synth_ai.api.train.task_app import check_task_app_health
except ImportError:
    print("ERROR: synth-ai SDK not found. Install with: pip install synth-ai")
    sys.exit(1)


def create_gepa_config_gemini(
    task_app_url: str,
    policy_model: str = "gemini-2.5-flash-lite",
    mutation_model: str = "gemini-2.5-flash",
    rollout_budget: int = 20,
) -> str:
    """Create GEPA config with Gemini models."""
    return f"""[prompt_learning]
algorithm = "gepa"
task_app_url = "{task_app_url}"
task_app_api_key = "${{ENVIRONMENT_API_KEY}}"
env_file_path = "../../../../../.env"
results_folder = "results"

[prompt_learning.initial_prompt]
id = "hotpotqa_pattern"
name = "HotpotQA Multi-Hop Question Answering Pattern"

[[prompt_learning.initial_prompt.messages]]
role = "system"
pattern = "You are a research assistant that answers multi-hop questions. Read the passages carefully and respond in the format:\\nAnswer: <short answer>\\nSupport: <brief justification citing passages>."
order = 0

[[prompt_learning.initial_prompt.messages]]
role = "user"
pattern = "Question: {{question}}\\n\\nPassages:\\n{{context}}\\n\\nProvide the final answer."
order = 1

[prompt_learning.initial_prompt.wildcards]
question = "REQUIRED"
context = "REQUIRED"

[prompt_learning.policy]
inference_mode = "synth_hosted"
model = "{policy_model}"
provider = "google"
temperature = 0.0
max_completion_tokens = 512

[prompt_learning.gepa]
env_name = "hotpotqa"

[prompt_learning.gepa.evaluation]
train_seeds = [0, 1, 2, 3, 4]
val_seeds = [50, 51, 52, 53, 54]
validation_pool = "train"
validation_top_k = 1

[prompt_learning.gepa.rollout]
budget = {rollout_budget}
max_concurrent = 5

[prompt_learning.gepa.mutation]
rate = 0.3
llm_model = "{mutation_model}"
llm_provider = "google"
temperature = 0.7
max_tokens = 512

[prompt_learning.gepa.population]
initial_size = 5
num_generations = 2
children_per_generation = 2

[prompt_learning.gepa.archive]
max_size = 10
min_score_threshold = 0.0
feedback_fraction = 0.33

[prompt_learning.gepa.token]
max_limit = 4096
counting_model = "gpt-4"
enforce_limit = false

[prompt_learning.termination_config]
max_cost_usd = 1.0
max_trials = {rollout_budget}

[display]
local_backend = true
tui = false
show_curve = true
verbose_summary = true
show_trial_results = true
show_transformations = false
show_validation = true
"""


def create_mipro_config_gemini(
    task_app_url: str,
    policy_model: str = "gemini-2.5-flash-lite",
    meta_model: str = "gemini-2.5-flash",
    rollout_budget: int = 20,
) -> str:
    """Create MIPRO config with Gemini models."""
    return f"""[prompt_learning]
algorithm = "mipro"
task_app_url = "{task_app_url}"
task_app_api_key = "${{ENVIRONMENT_API_KEY}}"
env_file_path = "../../../../../.env"
results_folder = "results"

[prompt_learning.initial_prompt]
id = "hotpotqa_pattern"
name = "HotpotQA Multi-Hop Question Answering Pattern"

[[prompt_learning.initial_prompt.messages]]
role = "system"
pattern = "You are a research assistant that answers multi-hop questions. Read the passages carefully and respond in the format:\\nAnswer: <short answer>\\nSupport: <brief justification citing passages>."
order = 0

[[prompt_learning.initial_prompt.messages]]
role = "user"
pattern = "Question: {{question}}\\n\\nPassages:\\n{{context}}\\n\\nProvide the final answer."
order = 1

[prompt_learning.initial_prompt.wildcards]
question = "REQUIRED"
context = "REQUIRED"

[prompt_learning.policy]
inference_mode = "synth_hosted"
model = "{policy_model}"
provider = "google"
temperature = 0.0
max_completion_tokens = 512

[prompt_learning.mipro]
env_name = "hotpotqa"
num_iterations = 5
num_evaluations_per_iteration = 4
batch_size = 10
max_concurrent = 5
meta_model = "{meta_model}"
meta_model_provider = "google"
few_shot_score_threshold = 0.8
max_instructions = 5
max_demo_set_size = 10
instructions_per_batch = 20
bootstrap_train_seeds = [0, 1, 2, 3, 4]
online_pool = [5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
test_pool = [50, 51, 52, 53, 54]

[prompt_learning.termination_config]
max_cost_usd = 1.0
max_trials = {rollout_budget}

[display]
local_backend = true
tui = false
show_curve = true
verbose_summary = true
show_trial_results = true
show_transformations = false
show_validation = true
"""


async def run_gemini_test(
    algorithm: str,
    task_app_url: str = "http://127.0.0.1:8110",
    backend_url: str = "http://localhost:8000",
    rollout_budget: int = 20,
    policy_model: str = "gemini-2.5-flash-lite",
    meta_model: Optional[str] = None,
) -> Dict[str, Any]:
    """Run a Gemini test for GEPA or MIPRO.
    
    Args:
        algorithm: "gepa" or "mipro"
        task_app_url: Task app URL
        backend_url: Backend URL
        rollout_budget: Rollout budget for test
        policy_model: Policy model (default: gemini-2.5-flash-lite)
        meta_model: Meta/mutation model (default: gemini-2.5-flash)
    
    Returns:
        Results dictionary
    """
    if meta_model is None:
        meta_model = "gemini-2.5-flash"
    
    print(f"\n{'='*80}")
    print(f"🧪 Testing {algorithm.upper()} with Gemini Models")
    print(f"{'='*80}")
    print(f"Policy Model: {policy_model}")
    print(f"Meta/Mutation Model: {meta_model}")
    print(f"Task App: {task_app_url}")
    print(f"Backend: {backend_url}")
    print(f"Rollout Budget: {rollout_budget}")
    print(f"{'='*80}\n")
    
    # Check API keys
    api_key = os.getenv("SYNTH_API_KEY")
    task_app_api_key = os.getenv("ENVIRONMENT_API_KEY") or os.getenv("SYNTH_API_KEY")
    gemini_api_key = os.getenv("GEMINI_API_KEY")
    
    if not api_key:
        raise ValueError("SYNTH_API_KEY must be set")
    if not task_app_api_key:
        raise ValueError("ENVIRONMENT_API_KEY or SYNTH_API_KEY must be set")
    if not gemini_api_key:
        raise ValueError("GEMINI_API_KEY must be set for Gemini models")
    
    # Check task app health
    print("Checking task app health...")
    health = check_task_app_health(task_app_url, task_app_api_key)
    if not health.ok:
        raise ValueError(f"Task app health check failed: {health.detail}")
    print(f"✅ Task app healthy\n")
    
    # Create config
    if algorithm == "gepa":
        config_content = create_gepa_config_gemini(
            task_app_url=task_app_url,
            policy_model=policy_model,
            mutation_model=meta_model,
            rollout_budget=rollout_budget,
        )
    elif algorithm == "mipro":
        config_content = create_mipro_config_gemini(
            task_app_url=task_app_url,
            policy_model=policy_model,
            meta_model=meta_model,
            rollout_budget=rollout_budget,
        )
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")
    
    # Write config to temp file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.toml', delete=False) as f:
        f.write(config_content)
        config_path = Path(f.name)
    
    try:
        print(f"Creating {algorithm.upper()} job with Gemini models...")
        print(f"Config written to: {config_path}")
        print(f"\nConfig preview:")
        print(config_content[:500] + "...\n")
        
        # Create job
        job = PromptLearningJob.from_config(
            config_path=str(config_path),
            backend_url=backend_url,
            api_key=api_key,
            task_app_api_key=task_app_api_key,
            overrides={"overrides": {"run_local": True}},  # Run locally in-process
        )
        
        print(f"✅ Job created")
        print(f"Submitting job...\n")
        
        # Submit job
        job_id = job.submit()
        print(f"✅ Job submitted: {job_id}")
        print(f"Polling until complete...\n")
        
        # Poll until complete
        results = job.poll_until_complete(timeout=600.0)  # 10 minute timeout
        
        print(f"\n{'='*80}")
        status = results.get('status', 'unknown')
        if status == 'failed':
            print(f"❌ {algorithm.upper()} Optimization Failed!")
        else:
            print(f"✅ {algorithm.upper()} Optimization Complete!")
        print(f"{'='*80}")
        print(f"Job ID: {job_id}")
        print(f"Status: {status}")
        
        # Get detailed error if failed
        if status == 'failed':
            try:
                import asyncio
                from synth_ai.learning.prompt_learning_client import PromptLearningClient
                client = PromptLearningClient(
                    backend_url.replace('http://', 'http://').replace('https://', 'https://'),
                    api_key,
                    timeout=30.0,
                )
                detailed_status = await client.get_job(job_id)
                error_msg = (
                    detailed_status.get('error_message') or
                    detailed_status.get('error') or
                    detailed_status.get('failure_reason') or
                    results.get('error') or
                    'Unknown error'
                )
                print(f"\nError Details:")
                print(f"{'-'*80}")
                print(error_msg)
                print(f"{'-'*80}")
            except Exception as e:
                print(f"Could not fetch error details: {e}")
                import traceback
                traceback.print_exc()
        
        best_score = results.get('best_score')
        if best_score is not None:
            print(f"Best Score: {best_score:.4f}")
        else:
            print(f"Best Score: N/A")
        val_score = results.get('val_score')
        if val_score is not None:
            print(f"Val Score: {val_score:.4f}")
        test_score = results.get('test_score')
        if test_score is not None:
            print(f"Test Score: {test_score:.4f}")
        print(f"{'='*80}\n")
        
        return {
            "algorithm": algorithm,
            "job_id": job_id,
            "results": results,
            "policy_model": policy_model,
            "meta_model": meta_model,
        }
        
    finally:
        # Clean up temp config file
        if config_path.exists():
            config_path.unlink()


async def main():
    """Run all Gemini integration tests."""
    parser = argparse.ArgumentParser(
        description="Test Gemini 2.5 family support in Synth AI GEPA and MIPRO"
    )
    parser.add_argument(
        "--task-app-url",
        default="http://127.0.0.1:8110",
        help="Task app URL (default: http://127.0.0.1:8110)",
    )
    parser.add_argument(
        "--backend-url",
        default="http://localhost:8000",
        help="Backend URL (default: http://localhost:8000)",
    )
    parser.add_argument(
        "--rollout-budget",
        type=int,
        default=20,
        help="Rollout budget for tests (default: 20)",
    )
    parser.add_argument(
        "--policy-model",
        default="gemini-2.5-flash-lite",
        help="Policy model (default: gemini-2.5-flash-lite)",
    )
    parser.add_argument(
        "--meta-model",
        default="gemini-2.5-flash",
        help="Meta/mutation model (default: gemini-2.5-flash)",
    )
    parser.add_argument(
        "--algorithm",
        choices=["gepa", "mipro", "both"],
        default="both",
        help="Algorithm to test (default: both)",
    )
    parser.add_argument(
        "--skip-gepa",
        action="store_true",
        help="Skip GEPA test",
    )
    parser.add_argument(
        "--skip-mipro",
        action="store_true",
        help="Skip MIPRO test",
    )
    
    args = parser.parse_args()
    
    # Determine which tests to run
    run_gepa = (args.algorithm in ["gepa", "both"]) and not args.skip_gepa
    run_mipro = (args.algorithm in ["mipro", "both"]) and not args.skip_mipro
    
    if not run_gepa and not run_mipro:
        print("ERROR: No tests to run (both skipped)")
        sys.exit(1)
    
    print("=" * 80)
    print("🧪 Gemini 2.5 Family Integration Tests")
    print("=" * 80)
    print(f"Tests to run:")
    if run_gepa:
        print(f"  ✅ GEPA with Gemini policy + mutation models")
    if run_mipro:
        print(f"  ✅ MIPRO with Gemini policy + meta models")
    print("=" * 80)
    
    results = {}
    
    # Run GEPA test
    if run_gepa:
        try:
            gepa_results = await run_gemini_test(
                algorithm="gepa",
                task_app_url=args.task_app_url,
                backend_url=args.backend_url,
                rollout_budget=args.rollout_budget,
                policy_model=args.policy_model,
                meta_model=args.meta_model,
            )
            results["gepa"] = gepa_results
            print("✅ GEPA test PASSED\n")
        except Exception as e:
            print(f"❌ GEPA test FAILED: {e}")
            import traceback
            traceback.print_exc()
            results["gepa"] = {"error": str(e)}
            print()
    
    # Run MIPRO test
    if run_mipro:
        try:
            mipro_results = await run_gemini_test(
                algorithm="mipro",
                task_app_url=args.task_app_url,
                backend_url=args.backend_url,
                rollout_budget=args.rollout_budget,
                policy_model=args.policy_model,
                meta_model=args.meta_model,
            )
            results["mipro"] = mipro_results
            print("✅ MIPRO test PASSED\n")
        except Exception as e:
            print(f"❌ MIPRO test FAILED: {e}")
            import traceback
            traceback.print_exc()
            results["mipro"] = {"error": str(e)}
            print()
    
    # Summary
    print("=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)
    for test_name, test_results in results.items():
        if "error" in test_results:
            print(f"{test_name.upper()}: ❌ FAILED - {test_results['error']}")
        else:
            score = test_results.get("results", {}).get("best_score")
            if score is not None:
                print(f"{test_name.upper()}: ✅ PASSED - Best Score: {score:.4f}")
            else:
                status = test_results.get("results", {}).get("status", "unknown")
                if status == "failed":
                    print(f"{test_name.upper()}: ❌ FAILED - Job status: {status}")
                else:
                    print(f"{test_name.upper()}: ⚠️  Status: {status} - No score available")
    print("=" * 80)
    
    # Exit with error if any test failed
    if any("error" in r for r in results.values()):
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())

