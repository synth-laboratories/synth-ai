#!/usr/bin/env python3
"""Run a short MIPRO optimization for Crafter task app to verify rollouts work."""

import asyncio
import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Add synth-ai to path
synth_ai_root = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(synth_ai_root))

load_dotenv()

try:
    from synth_ai.api.train.prompt_learning import PromptLearningJob
    from synth_ai.api.train.task_app import check_task_app_health
except ImportError:
    print("ERROR: synth-ai SDK not found. Install with: pip install synth-ai")
    sys.exit(1)

async def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Run short MIPRO test for Crafter")
    parser.add_argument("--task-app-url", default="http://127.0.0.1:8116")
    parser.add_argument("--backend-url", default="http://localhost:8000")
    parser.add_argument("--rollout-budget", type=int, default=5)
    args = parser.parse_args()
    
    config_path = Path(__file__).parent / "crafter_mipro.toml"
    
    print("=" * 80)
    print("🚀 Short MIPRO Test: Crafter")
    print("=" * 80)
    print(f"Config: {config_path}")
    print(f"Task app: {args.task_app_url}")
    print(f"Backend: {args.backend_url}")
    print(f"Rollout budget: {args.rollout_budget}")
    print("=" * 80)
    print()
    
    # Get API keys
    api_key = os.getenv("SYNTH_API_KEY")
    task_app_api_key = os.getenv("ENVIRONMENT_API_KEY") or os.getenv("SYNTH_API_KEY")
    
    if not api_key:
        print("ERROR: SYNTH_API_KEY must be set")
        sys.exit(1)
    if not task_app_api_key:
        print("ERROR: ENVIRONMENT_API_KEY or SYNTH_API_KEY must be set")
        sys.exit(1)
    
    # Check task app health
    print("Checking task app health...")
    health = check_task_app_health(args.task_app_url, task_app_api_key)
    if not health.ok:
        print(f"❌ Task app health check failed: {health.detail}")
        sys.exit(1)
    print(f"✅ Task app healthy")
    print()
    
    # Create job
    print("Creating MIPRO job...")
    job = PromptLearningJob.from_config(
        config_path=str(config_path),
        backend_url=args.backend_url,
        api_key=api_key,
        task_app_api_key=task_app_api_key,
        overrides={"overrides": {"run_local": True}},
    )
    
    # Submit
    print("Submitting job...")
    job_id = job.submit()
    print(f"✅ Job submitted: {job_id}")
    print()
    
    # Poll briefly
    print("Polling for completion (timeout: 60s)...")
    try:
        final_status = job.poll_until_complete(
            timeout=60.0,
            interval=2.0,
            on_status=lambda s: print(f"  Status: {s.get('status')}"),
        )
        print()
        print("=" * 80)
        print("✅ MIPRO Test Complete!")
        print("=" * 80)
        print(f"Job ID: {job_id}")
        print(f"Status: {final_status.get('status')}")
        print(f"Best Score: {final_status.get('best_score', 'N/A')}")
        print("=" * 80)
    except Exception as e:
        print(f"\n⚠️  Polling timeout or error: {e}")
        print(f"Job ID: {job_id} - Check backend for status")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n❌ Interrupted by user")
        sys.exit(1)

