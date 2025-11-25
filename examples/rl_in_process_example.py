#!/usr/bin/env python3
"""
In-Process RL/GSPO Training Example
=====================================

This script demonstrates running RL/GSPO training with:
- Task app started in-process (no external bash commands)
- Cloudflare tunnel opened automatically
- Automatic cleanup when done

NO EXTERNAL PROCESSES NEEDED - everything happens in this one Python script!

Usage:
    cd /Users/joshpurtell/Documents/GitHub/synth-ai
    source .env
    uv run python examples/rl_in_process_example.py

Requirements:
    - SYNTH_API_KEY in .env
    - ENVIRONMENT_API_KEY in .env
    - cloudflared binary (will auto-install if missing)
    - Backend running (default: https://synth-backend-dev-docker.onrender.com)
    
Configuration:
    Default: Uses dev backend
    Override: Set BACKEND_BASE_URL env var to use different backend
    
    The script automatically matches tunnel mode:
    - If BACKEND_BASE_URL is localhost → both backend and task app use localhost
    - If BACKEND_BASE_URL is a tunnel URL → both backend and task app use tunnels
"""

from __future__ import annotations

import asyncio
import os
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

# Load environment from repo root
env_path = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(env_path)

# Add parent to path for imports
parent_dir = Path(__file__).resolve().parent.parent
if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))

from synth_ai.sdk.api.train.rl import RLJob
from synth_ai.sdk.task.in_process import InProcessTaskApp


async def main():
    """Run RL/GSPO training with in-process task app."""

    print("\n" + "=" * 80)
    print("In-Process RL/GSPO Training Demo")
    print("=" * 80 + "\n")

    # Check requirements
    if not os.getenv("SYNTH_API_KEY"):
        print("❌ Error: SYNTH_API_KEY required in .env")
        sys.exit(1)

    if not os.getenv("ENVIRONMENT_API_KEY"):
        print("❌ Error: ENVIRONMENT_API_KEY required in .env")
        sys.exit(1)

    # Configuration
    config_path = (
        Path(__file__).parent.parent
        / "synth_ai"
        / "cli"
        / "demo_apps"
        / "demo_task_apps"
        / "crafter"
        / "configs"
        / "rl_from_base_qwen4b.toml"
    )

    if not config_path.exists():
        print(f"❌ Error: Config file not found: {config_path}")
        print("   Please provide a valid RL config file path")
        sys.exit(1)

    # Default to dev backend, allow override via BACKEND_BASE_URL env var
    backend_url = os.getenv(
        "BACKEND_BASE_URL", "https://synth-backend-dev-docker.onrender.com"
    )
    api_key = os.getenv("SYNTH_API_KEY", "test")
    task_app_api_key = os.getenv("ENVIRONMENT_API_KEY", "test")

    # Determine tunnel mode based on backend URL
    is_backend_localhost = backend_url.startswith("http://localhost") or backend_url.startswith(
        "http://127.0.0.1"
    )

    if is_backend_localhost:
        # Backend is localhost → use local mode for task app (no tunnel)
        os.environ["SYNTH_TUNNEL_MODE"] = "local"
        use_local_mode = True
        print("ℹ️  Configuration: local/local")
        print("   Backend: localhost:8000")
        print("   Task App: localhost (no tunnel)")
    else:
        # Backend is tunneled → use tunnel mode for task app
        os.environ["SYNTH_TUNNEL_MODE"] = "quick"
        use_local_mode = False
        print("ℹ️  Configuration: tunnel/tunnel")
        print(f"   Backend tunnel: {backend_url}")
        print(f"   Task app: will create its own tunnel")

    print("Configuration:")
    print(f"  Config: {config_path.name}")
    print(f"  Backend: {backend_url}")
    print(f"  Task App: Starting in-process...")
    print()

    # Import task app config factory
    # For this example, we'll use a simple task app path
    # In practice, you'd point to your actual task app
    task_app_path = (
        Path(__file__).resolve().parent.parent
        / "synth_ai"
        / "cli"
        / "demo_apps"
        / "demo_task_apps"
        / "crafter"
        / "crafter_task_app.py"
    )

    if not task_app_path.exists():
        print(f"⚠️  Warning: Task app not found: {task_app_path}")
        print("   Please update task_app_path to point to your task app")
        print("   Continuing with example...")
        # For demo purposes, we'll continue but the job will fail
        # In real usage, you'd have a valid task app

    # Run RL training with in-process task app
    try:
        if task_app_path.exists():
            async with InProcessTaskApp(
                task_app_path=task_app_path,
                port=8114,
                api_key=task_app_api_key,
            ) as task_app:
                print(f"✅ Task app running at: {task_app.url}")
                if use_local_mode:
                    print(f"✅ Using local mode (no tunnel)")
                else:
                    print(f"✅ Cloudflare tunnel active")
                print()

                # Create RL job
                print("=" * 80)
                print("Running RL/GSPO Training")
                print("=" * 80 + "\n")

                job = RLJob.from_config(
                    config_path=config_path,
                    backend_url=backend_url,
                    api_key=api_key,
                    task_app_url=task_app.url,
                    task_app_api_key=task_app_api_key,
                )

                print(f"Task app URL: {task_app.url}")
                print(f"Backend URL: {backend_url}\n")
                print(f"Submitting job...\n")

                try:
                    job_id = job.submit()
                    print(f"✅ Job submitted: {job_id}\n")
                except Exception as e:
                    print(f"\n❌ Error submitting job:")
                    print(f"   Type: {type(e).__name__}")
                    print(f"   Message: {str(e)}")
                    print(f"\n   Full error details:")
                    import traceback

                    traceback.print_exc()
                    raise

                # Poll for completion
                start_time = time.time()
                last_status = None

                def on_status(status):
                    nonlocal last_status
                    elapsed = time.time() - start_time
                    state = status.get("status", "unknown")

                    # Only print if status changed or every 30 seconds
                    if state != last_status or int(elapsed) % 30 == 0:
                        timestamp = time.strftime("%H:%M:%S")
                        progress = status.get("progress", {})
                        metrics = status.get("metrics", {})

                        if progress:
                            completed = progress.get("completed", 0)
                            total = progress.get("total", 0)
                            if total > 0:
                                pct = (completed / total) * 100
                                print(
                                    f"[{timestamp}] {elapsed:6.1f}s  Status: {state} ({completed}/{total} = {pct:.1f}%)"
                                )
                            else:
                                print(f"[{timestamp}] {elapsed:6.1f}s  Status: {state}")
                        elif metrics:
                            # Print key metrics if available
                            reward = metrics.get("reward", metrics.get("mean_reward"))
                            if reward is not None:
                                print(
                                    f"[{timestamp}] {elapsed:6.1f}s  Status: {state} | Reward: {reward:.3f}"
                                )
                            else:
                                print(f"[{timestamp}] {elapsed:6.1f}s  Status: {state}")
                        else:
                            print(f"[{timestamp}] {elapsed:6.1f}s  Status: {state}")
                        last_status = state

                result = await asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda: job.poll_until_complete(
                        timeout=7200.0,  # 2 hours for RL jobs
                        interval=10.0,  # Poll every 10 seconds
                        on_status=on_status,
                    ),
                )

                total_time = time.time() - start_time
                print(f"\n✅ RL training complete in {total_time:.1f}s\n")

                # Get results
                print("=" * 80)
                print("Results")
                print("=" * 80 + "\n")

                status = result.get("status", "unknown")
                print(f"Status: {status}")

                if status == "succeeded":
                    results = result.get("results", {})
                    metrics = results.get("metrics", {})
                    if metrics:
                        print("\nFinal Metrics:")
                        for key, value in metrics.items():
                            if isinstance(value, (int, float)):
                                print(f"  {key}: {value:.4f}")
                            else:
                                print(f"  {key}: {value}")
                else:
                    error = result.get("error", result.get("message", "Unknown error"))
                    print(f"\n❌ Job failed: {error}")

                print()

        else:
            print("⚠️  Skipping RL job submission (task app not found)")
            print("   Update task_app_path to point to a valid task app")

    except KeyboardInterrupt:
        print("\n⚠️  Interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)

    print("=" * 80)
    print("✅ In-process RL/GSPO training demo complete!")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    asyncio.run(main())

