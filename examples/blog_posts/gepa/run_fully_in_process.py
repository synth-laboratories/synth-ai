#!/usr/bin/env python3
"""
FULLY IN-PROCESS GEPA Example
==============================

This script demonstrates running GEPA optimization with:
- Task app started in-process (no external bash commands)
- Cloudflare tunnel opened automatically
- Automatic cleanup when done

NO EXTERNAL PROCESSES NEEDED - everything happens in this one Python script!

Usage:
    python run_fully_in_process.py

Requirements:
    - GROQ_API_KEY in .env
    - cloudflared binary (will auto-install if missing)
    - synth-ai backend running (localhost:8000)
"""

from __future__ import annotations

import asyncio
import os
from pathlib import Path

from dotenv import load_dotenv

# Load environment
load_dotenv()

# NOTE: This is a PROTOTYPE showing what the API would look like
# The InProcessTaskApp class needs to be implemented in synth_ai/task/apps.py
# See in-process-task-app.txt for full implementation plan


async def main_with_proposed_api():
    """Example using the PROPOSED InProcessTaskApp API (not yet implemented)."""

    print("\n" + "="*80)
    print("FULLY IN-PROCESS GEPA - Proposed API")
    print("="*80 + "\n")

    # This is what the API WOULD look like once implemented:
    """
    from synth_ai.task.apps import InProcessTaskApp
    from synth_ai.api.train.prompt_learning import PromptLearningJob
    from heartdisease_baseline import HeartDiseaseTaskRunner

    async with InProcessTaskApp(
        runner_class=HeartDiseaseTaskRunner,
        app_id="heartdisease",
        port=8114,
        tunnel_mode="quick",
    ) as task_app:

        print(f"✓ Task app running at: {task_app.url}")

        job = PromptLearningJob.from_config(
            config_path=Path("configs/heartdisease_gepa_local.toml"),
            backend_url="http://localhost:8000",
            api_key="test",
            task_app_url=task_app.url,
        )

        job_id = job.submit()
        results = await job.poll_until_complete()

        print(f"Best score: {results['best_score']:.2%}")

    print("✓ Everything cleaned up automatically!")
    """

    print("This is the PROPOSED API (not yet implemented).")
    print("See in-process-task-app.txt for full details.")
    print("\nTo implement, we need to create:")
    print("  - synth_ai/task/apps.py:InProcessTaskApp class")
    print("  - Helper to convert BaselineTaskRunner -> FastAPI app")
    print("  - Integration with existing cloudflare.py tunnel code")
    print("\nEstimated effort: 4-6 hours")


async def main_with_manual_approach():
    """
    Working example using MANUAL approach (what you can do TODAY).

    This manually does what InProcessTaskApp would do automatically.
    """

    print("\n" + "="*80)
    print("MANUAL IN-PROCESS APPROACH (Works Today)")
    print("="*80 + "\n")

    # Check requirements
    if not os.getenv("GROQ_API_KEY"):
        print("❌ Error: GROQ_API_KEY required in .env")
        return

    print("This approach manually combines:")
    print("  1. Starting uvicorn in background thread")
    print("  2. Opening Cloudflare tunnel")
    print("  3. Running GEPA")
    print("  4. Cleaning up")
    print()

    # Import the necessary modules
    from synth_ai.cloudflare import (
        ensure_cloudflared_installed,
        open_quick_tunnel,
        stop_tunnel,
    )
    from synth_ai.api.train.prompt_learning import PromptLearningJob
    import threading
    import uvicorn
    import sys
    import time

    # Add parent to path for imports
    parent_dir = Path(__file__).resolve().parent.parent.parent.parent
    if str(parent_dir) not in sys.path:
        sys.path.insert(0, str(parent_dir))

    # Import task app
    task_app_path = Path(__file__).parent.parent.parent / "task_apps" / "other_langprobe_benchmarks" / "heartdisease_task_app.py"

    print(f"Task app path: {task_app_path}")
    print()

    # Ensure cloudflared is installed
    print("Checking cloudflared installation...")
    ensure_cloudflared_installed()
    print("✓ cloudflared ready\n")

    # Set environment for task app
    os.environ["ENVIRONMENT_API_KEY"] = os.getenv("ENVIRONMENT_API_KEY", "test")

    # 1. Load and start task app in background thread
    print("Step 1: Starting task app in background thread...")
    from synth_ai.utils.apps import get_asgi_app, load_file_to_module
    from synth_ai.utils.paths import configure_import_paths, REPO_ROOT

    configure_import_paths(task_app_path, REPO_ROOT)
    module = load_file_to_module(task_app_path, "heartdisease_task_app_in_process")
    app = get_asgi_app(module)

    port = 8114

    def serve():
        uvicorn.run(app, host="127.0.0.1", port=port, log_level="warning")

    server_thread = threading.Thread(target=serve, daemon=False, name="task-app-server")
    server_thread.start()
    print(f"✓ Task app server started on port {port}\n")

    # Give server a moment to start
    await asyncio.sleep(2)

    # 2. Open Cloudflare tunnel
    print("Step 2: Opening Cloudflare tunnel...")
    try:
        tunnel_url, tunnel_proc = open_quick_tunnel(port, wait_s=15.0)
        print(f"✓ Tunnel opened: {tunnel_url}\n")
    except Exception as e:
        print(f"❌ Failed to open tunnel: {e}")
        return

    try:
        # 3. Run GEPA
        print("Step 3: Running GEPA optimization...")
        print("="*80)

        config_path = Path(__file__).parent / "configs" / "heartdisease_gepa_local.toml"

        job = PromptLearningJob.from_config(
            config_path=config_path,
            backend_url="http://localhost:8000",
            api_key=os.getenv("SYNTH_API_KEY", "test"),
            task_app_api_key=os.getenv("ENVIRONMENT_API_KEY", "test"),
        )

        # Override task_app_url with our tunnel URL
        import toml
        config = toml.load(config_path)
        config["prompt_learning"]["task_app_url"] = tunnel_url

        # For demo, use smaller budget
        config["prompt_learning"]["gepa"]["rollout"]["budget"] = 50

        print(f"\nSubmitting job to {tunnel_url}...")
        job_id = job.submit()
        print(f"✓ Job submitted: {job_id}\n")

        # Poll for completion
        start_time = time.time()

        def on_status(status):
            elapsed = time.time() - start_time
            state = status.get("status", "unknown")
            print(f"[{elapsed:6.1f}s] Status: {state}")

        result = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: job.poll_until_complete(
                timeout=600.0,
                interval=5.0,
                on_status=on_status,
            )
        )

        print(f"\n✓ Job complete in {time.time() - start_time:.1f}s")

        # Get results
        from synth_ai.learning.prompt_learning_client import PromptLearningClient
        from synth_ai.api.train.utils import ensure_api_base

        client = PromptLearningClient(
            ensure_api_base("http://localhost:8000"),
            os.getenv("SYNTH_API_KEY", "test"),
        )
        prompt_results = await client.get_prompts(job._job_id)

        print(f"\n📊 Results:")
        print(f"   Best score: {prompt_results.best_score:.2%}")
        print()

    finally:
        # 4. Cleanup
        print("Step 4: Cleaning up...")
        print("-"*80)

        # Stop tunnel
        if tunnel_proc:
            stop_tunnel(tunnel_proc)
            print("✓ Tunnel stopped")

        # Server thread will stop when main exits (daemon=False ensures cleanup)
        print("✓ Server will stop on exit")
        print()

        print("="*80)
        print("✓ Fully in-process execution complete!")
        print("="*80 + "\n")


async def main():
    """Main entry point - shows both proposed and manual approaches."""

    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "--manual":
        # Run manual approach (works today)
        await main_with_manual_approach()
    else:
        # Show proposed API
        await main_with_proposed_api()

        print("\n" + "="*80)
        print("To see WORKING example with manual approach, run:")
        print("  python run_fully_in_process.py --manual")
        print("="*80 + "\n")


if __name__ == "__main__":
    asyncio.run(main())
