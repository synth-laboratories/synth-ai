#!/usr/bin/env python3
"""Run MIPRO locally on Iris via backend endpoint (localhost:8000).

This script uses the backend API endpoint with proper authentication, ensuring
balance checking works correctly. It emulates the real flow but bypasses Modal.

Usage:
    python run_mipro_local.py --task-app-url http://127.0.0.1:8115 --rollout-budget 20
"""

import asyncio
import sys
import os
import tempfile
import logging
from pathlib import Path
from typing import Optional

# Load environment variables
from dotenv import load_dotenv

# Add synth-ai source to path to use local changes (not installed package)
synth_ai_root = Path(__file__).parent.parent.parent.parent.parent.parent
if str(synth_ai_root) not in sys.path:
    sys.path.insert(0, str(synth_ai_root))

load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)

# Suppress verbose HTTP logs
logging.getLogger('httpx').setLevel(logging.WARNING)
logging.getLogger('httpcore').setLevel(logging.WARNING)

# Import SDK
try:
    from synth_ai.api.train.prompt_learning import PromptLearningJob
    from synth_ai.learning.prompt_learning_client import PromptLearningClient
    from synth_ai.api.train.task_app import check_task_app_health
except ImportError:
    print("ERROR: synth-ai SDK not found. Install with: pip install synth-ai")
    sys.exit(1)

def create_iris_mipro_toml(
    task_app_url: str,
    rollout_budget: int = 20,
    bootstrap_seeds: Optional[list[int]] = None,
    online_seeds: Optional[list[int]] = None,
    test_seeds: Optional[list[int]] = None,
) -> str:
    """Create TOML config for Iris MIPRO."""
    
    # Auto-scale seeds based on budget
    if bootstrap_seeds is None:
        bootstrap_seeds = list(range(0, min(5, rollout_budget // 4)))
    
    if online_seeds is None:
        max_bootstrap = max(bootstrap_seeds) if bootstrap_seeds else -1
        online_size = min(10, rollout_budget // 2)
        online_seeds = list(range(max_bootstrap + 1, max_bootstrap + 1 + online_size))
    
    if test_seeds is None:
        max_online = max(online_seeds) if online_seeds else max(bootstrap_seeds) if bootstrap_seeds else -1
        test_seeds = list(range(max_online + 1, max_online + 1 + 10))
    
    # Auto-scale iterations
    num_iterations = max(2, min(5, rollout_budget // 10))
    num_evaluations_per_iteration = max(2, min(5, rollout_budget // (num_iterations * 2)))
    
    # Get API keys
    task_app_api_key = os.getenv("ENVIRONMENT_API_KEY") or os.getenv("SYNTH_API_KEY")
    if not task_app_api_key:
        raise ValueError("ENVIRONMENT_API_KEY or SYNTH_API_KEY must be set")
    
    policy_model = os.getenv("POLICY_MODEL", "openai/gpt-oss-20b")
    meta_model = os.getenv("META_MODEL", "llama-3.3-70b-versatile")
    
    # Build TOML config
    toml_content = f"""[prompt_learning]
algorithm = "mipro"
task_app_url = "{task_app_url}"
task_app_api_key = "{task_app_api_key}"
env_name = "iris"

[prompt_learning.initial_prompt]
id = "iris_pattern"
name = "Iris Classification Pattern"

[[prompt_learning.initial_prompt.messages]]
role = "system"
pattern = "You are a botany classification assistant. Based on the flower's measurements, classify the iris species. Respond with one of: setosa, versicolor, or virginica."
order = 0

[[prompt_learning.initial_prompt.messages]]
role = "user"
pattern = "Flower Measurements:\\n{{features}}\\n\\nClassify this iris flower. Respond with one of: setosa, versicolor, or virginica."
order = 1

[prompt_learning.initial_prompt.wildcards]
features = "REQUIRED"

[prompt_learning.policy]
inference_mode = "synth_hosted"
model = "{policy_model}"
provider = "groq"
temperature = 0.0
max_completion_tokens = 128

[prompt_learning.mipro]
rollout_budget = {rollout_budget}
bootstrap_train_seeds = {bootstrap_seeds}
online_pool = {online_seeds}
test_pool = {test_seeds}
num_iterations = {num_iterations}
num_evaluations_per_iteration = {num_evaluations_per_iteration}
batch_size = {min(5, num_evaluations_per_iteration)}
few_shot_score_threshold = 0.7
meta_model = "{meta_model}"
meta_model_provider = "groq"
meta_model_inference_url = "https://api.groq.com"

[prompt_learning.mipro.seeds]
bootstrap = {bootstrap_seeds}
online = {online_seeds}
test = {test_seeds}

[prompt_learning.mipro.meta]
temperature = 0.7
max_tokens = 512

[[prompt_learning.mipro.modules]]
module_id = "iris_classification"
description = "Single-stage Iris classification module"

[[prompt_learning.mipro.modules.stages]]
stage_id = "classify"
description = "Classify iris flower based on measurements"
max_instruction_slots = 2
max_demo_slots = 3

[prompt_learning.trace_pipeline]
base_url = "http://127.0.0.1:8081"
lease_ttl = 30.0

[prompt_learning.termination_config]
max_cost_usd = {max(0.10, rollout_budget * 0.001 * 10)}
max_trials = {num_iterations * num_evaluations_per_iteration * 2}
max_category_costs_usd = {{"rollout" = {max(0.10, rollout_budget * 0.001 * 10) * 0.7}, "proposal" = {max(0.10, rollout_budget * 0.001 * 10) * 0.3}}}
"""
    
    return toml_content


async def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Run MIPRO locally on Iris via backend endpoint")
    parser.add_argument(
        "--task-app-url",
        type=str,
        default="http://127.0.0.1:8115",
        help="Task app URL (default: http://127.0.0.1:8115)",
    )
    parser.add_argument(
        "--rollout-budget",
        type=int,
        default=20,
        help="Rollout budget (default: 20)",
    )
    parser.add_argument(
        "--bootstrap-seeds",
        type=int,
        nargs="+",
        help="Bootstrap seeds (default: auto-scale)",
    )
    parser.add_argument(
        "--online-seeds",
        type=int,
        nargs="+",
        help="Online pool seeds (default: auto-scale)",
    )
    parser.add_argument(
        "--test-seeds",
        type=int,
        nargs="+",
        help="Test seeds (default: auto-scale)",
    )
    parser.add_argument(
        "--backend-url",
        type=str,
        default="http://localhost:8000",
        help="Backend URL (default: http://localhost:8000)",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="API key (defaults to SYNTH_API_KEY env var)",
    )
    
    args = parser.parse_args()
    
    # Get API key
    api_key = args.api_key or os.getenv("SYNTH_API_KEY")
    if not api_key:
        raise ValueError("API key required (provide --api-key or set SYNTH_API_KEY env var)")
    
    print("=" * 80)
    print("MIPRO Local Test: Iris (via Backend Endpoint)")
    print("=" * 80)
    print(f"Backend URL: {args.backend_url}")
    print(f"Task app URL: {args.task_app_url}")
    print(f"Rollout budget: {args.rollout_budget}")
    print("=" * 80)
    print()
    
    # Create temporary TOML config
    toml_content = create_iris_mipro_toml(
        task_app_url=args.task_app_url,
        rollout_budget=args.rollout_budget,
        bootstrap_seeds=args.bootstrap_seeds,
        online_seeds=args.online_seeds,
        test_seeds=args.test_seeds,
    )
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.toml', delete=False) as f:
        f.write(toml_content)
        config_path = Path(f.name)
    
    try:
        # Create job using SDK
        task_app_api_key = (os.getenv("ENVIRONMENT_API_KEY") or os.getenv("SYNTH_API_KEY") or "").strip()
        if not task_app_api_key:
            raise ValueError(
                "ENVIRONMENT_API_KEY (or SYNTH_API_KEY) must be set for task app authentication."
            )

        # Validate task app availability before submitting job
        health = check_task_app_health(args.task_app_url, task_app_api_key)
        if not health.ok:
            raise ValueError(f"Task app health check failed before submission: {health.detail}")
        else:
            print(f"Task app healthy (auth ok, detail: {health.detail})")

        # Ensure downstream helpers see the same credentials
        os.environ["ENVIRONMENT_API_KEY"] = task_app_api_key
        os.environ["TASK_APP_URL"] = args.task_app_url

        job = PromptLearningJob.from_config(
            config_path=str(config_path),
            backend_url=args.backend_url,
            api_key=api_key,
            task_app_api_key=task_app_api_key,  # Explicitly pass task app API key for health check
            overrides={"overrides": {"run_local": True}},  # Run locally in-process instead of Modal
        )
        
        # Validate config before submission
        print("Validating MIPRO config...")
        try:
            from synth_ai.api.train.validators import validate_prompt_learning_config_from_file
            validate_prompt_learning_config_from_file(config_path, algorithm="mipro")
            print("✓ Config validated successfully")
        except ImportError:
            # Fallback to basic validation if validator not available
            print("⚠️  Config validator not found, skipping validation")
        except Exception as e:
            print(f"\n{'=' * 80}")
            print("❌ Config Validation Failed")
            print(f"{'=' * 80}")
            print(str(e))
            print(f"{'=' * 80}\n")
            raise
        print()
        
        print("Submitting job to backend...")
        job_id = job.submit()
        print(f"✓ Job submitted: {job_id}")
        print()
        
        # Poll until complete
        print("Polling for completion...")
        final_status = job.poll_until_complete(
            timeout=3600.0,
            interval=5.0,
            on_status=lambda status: print(f"  Status: {status.get('status')} | Best score: {status.get('best_score', 'N/A')}"),
        )
        
        print()
        print("=" * 80)
        print("✅ MIPRO Optimization Complete!")
        print("=" * 80)
        print(f"Job ID: {job_id}")
        print(f"Status: {final_status.get('status')}")
        print(f"Best Score: {final_status.get('best_score', 'N/A')}")
        
        # Show failure reason if failed
        if final_status.get('status') == 'failed':
            print("\n⚠️  Job failed. Fetching detailed error information...")
            try:
                from error_display import fetch_and_display_error
                client = PromptLearningClient(args.backend_url, api_key)
                await fetch_and_display_error(client, job_id, args.backend_url)
            except ImportError:
                # Fallback to basic error display
                try:
                    client = PromptLearningClient(args.backend_url, api_key)
                    job_detail = await client.get_job(job_id)
                    error_message = job_detail.get('error_message') or job_detail.get('error') or 'Unknown error'
                    print(f"\n❌ Failure reason: {error_message}")
                except Exception as detail_err:
                    print(f"❌ Could not fetch failure details: {detail_err}")
            except Exception as detail_err:
                print(f"❌ Error displaying failure details: {detail_err}")
                import traceback
                traceback.print_exc()
        
        print()
        
        # Get detailed results (cost, balance, etc.)
        client = PromptLearningClient(args.backend_url, api_key)
        job_detail = await client.get_job(job_id)
        
        # Extract cost and balance from best_snapshot
        best_snapshot = job_detail.get("best_snapshot")
        if best_snapshot:
            print(f"Total Cost: ${best_snapshot.get('total_cost_usd', 0.0):.4f}")
            print(f"Category Costs: {best_snapshot.get('category_costs', {})}")
            if best_snapshot.get('final_balance_usd') is not None:
                print(f"Final Balance: ${best_snapshot.get('final_balance_usd'):.2f} ({best_snapshot.get('balance_type', 'N/A')})")
        
        print("=" * 80)
        
    except Exception as e:
        print()
        print("=" * 80)
        print("❌ Error during optimization")
        print("=" * 80)
        print(f"Error: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
        
    finally:
        # Cleanup temp file
        if config_path.exists():
            config_path.unlink()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n❌ Optimization interrupted by user")
        sys.exit(1)
