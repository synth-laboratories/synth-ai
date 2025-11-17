#!/usr/bin/env python3
"""
In-Process GEPA Banking77 Demo
================================

This script runs GEPA optimization for Banking77 with a task app started entirely 
in-process - no separate terminals or manual process management needed!

Usage:
    cd examples/blog_posts/vendored_prompt_learning
    uv run scripts/run_gepa_banking77_in_process.py

Requirements:
    - GROQ_API_KEY in .env (for policy model)
    - OPENAI_API_KEY in .env (for mutation LLM)
    - cloudflared binary (will auto-install if missing)
    - synth-ai backend running (localhost:8000)
"""

from __future__ import annotations

import asyncio
import os
import sys
import tempfile
import time
from pathlib import Path

from dotenv import load_dotenv

# Load environment from repo root
# Script is at: examples/blog_posts/vendored_prompt_learning/scripts/
# Need to go up 5 levels to get to repo root
env_path = Path(__file__).resolve().parents[4] / ".env"
load_dotenv(env_path)
# Also try loading from current directory and parent directories
load_dotenv()  # Try current dir and parents

# Add parent to path for imports
parent_dir = Path(__file__).resolve().parents[4]  # Repo root
if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))

from synth_ai.api.train.prompt_learning import PromptLearningJob
from synth_ai.task import InProcessTaskApp


async def main():
    """Run GEPA optimization for Banking77 with in-process task app."""

    print("\n" + "=" * 80)
    print("In-Process GEPA Banking77 Demo")
    print("=" * 80 + "\n")

    # Check requirements
    if not os.getenv("GROQ_API_KEY"):
        print("❌ Error: GROQ_API_KEY required in .env (for policy model)")
        sys.exit(1)

    # Configuration
    config_path = (
        Path(__file__).parent.parent
        / "configs"
        / "banking77_gepa_local.toml"
    )

    if not config_path.exists():
        print(f"❌ Error: Config file not found: {config_path}")
        sys.exit(1)

    backend_url = os.getenv("BACKEND_BASE_URL", "http://localhost:8000")
    api_key = os.getenv("SYNTH_API_KEY", "test")
    task_app_api_key = os.getenv("ENVIRONMENT_API_KEY", "test")

    print("Configuration:")
    print(f"  Config: {config_path.name}")
    print(f"  Backend: {backend_url}")
    print(f"  Task App: Starting in-process...")
    print()

    # Import task app config factory
    task_app_path = (
        Path(__file__).resolve().parents[4]  # Repo root
        / "examples"
        / "task_apps"
        / "banking77"
        / "banking77_task_app.py"
    )

    if not task_app_path.exists():
        print(f"❌ Error: Task app not found: {task_app_path}")
        sys.exit(1)

    # Run GEPA with in-process task app
    try:
        async with InProcessTaskApp(
            task_app_path=task_app_path,
            port=8102,
            api_key=task_app_api_key,
        ) as task_app:
            print(f"✅ Task app running at: {task_app.url}")
            print(f"✅ Cloudflare tunnel active")
            print()

            # Create GEPA job
            print("=" * 80)
            print("Running GEPA Optimization")
            print("=" * 80 + "\n")

            # Load and modify config before creating job
            import toml

            config = toml.load(config_path)
            config["prompt_learning"]["task_app_url"] = task_app.url

            # Reduce budget for very fast demo (~1 min)
            if "gepa" in config["prompt_learning"] and "rollout" in config["prompt_learning"]["gepa"]:
                original_budget = config["prompt_learning"]["gepa"]["rollout"].get("budget", 100)
                config["prompt_learning"]["gepa"]["rollout"]["budget"] = 5  # Minimal: 5 rollouts
                print(f"📊 Reduced rollout budget to 5 for very fast demo")
                print(f"   (Original: {original_budget})\n")

            # Reduce population size for faster demo
            if "gepa" in config["prompt_learning"] and "population" in config["prompt_learning"]["gepa"]:
                original_size = config["prompt_learning"]["gepa"]["population"].get("initial_size", 10)
                config["prompt_learning"]["gepa"]["population"]["initial_size"] = 2  # Minimal population
                config["prompt_learning"]["gepa"]["population"]["num_generations"] = 1  # Just 1 generation
                config["prompt_learning"]["gepa"]["population"]["children_per_generation"] = 2  # Minimal children
                print(f"📊 Reduced to minimal: population=2, generations=1, children=2 (~5 rollouts)\n")

            # Write modified config to temp file
            with tempfile.NamedTemporaryFile(mode="w", suffix=".toml", delete=False) as f:
                toml.dump(config, f)
                temp_config_path = f.name

            try:
                job = PromptLearningJob.from_config(
                    config_path=temp_config_path,
                    backend_url=backend_url,
                    api_key=api_key,
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
            finally:
                # Clean up temp config file
                if os.path.exists(temp_config_path):
                    os.unlink(temp_config_path)

            # Poll for completion
            start_time = time.time()
            last_status = None

            def on_status(status):
                nonlocal last_status
                elapsed = time.time() - start_time
                state = status.get("status", "unknown")

                # Only print if status changed or every 10 seconds
                if state != last_status or int(elapsed) % 10 == 0:
                    timestamp = time.strftime("%H:%M:%S")
                    progress = status.get("progress", {})
                    best_score = status.get("best_score")

                    if progress:
                        completed = progress.get("completed", 0)
                        total = progress.get("total", 0)
                        if total > 0:
                            pct = (completed / total) * 100
                            score_str = f" | Best: {best_score:.3f}" if best_score is not None else ""
                            print(
                                f"[{timestamp}] {elapsed:6.1f}s  Status: {state} ({completed}/{total} = {pct:.1f}%){score_str}"
                            )
                        else:
                            score_str = f" | Best: {best_score:.3f}" if best_score is not None else ""
                            print(f"[{timestamp}] {elapsed:6.1f}s  Status: {state}{score_str}")
                    else:
                        score_str = f" | Best: {best_score:.3f}" if best_score is not None else ""
                        print(f"[{timestamp}] {elapsed:6.1f}s  Status: {state}{score_str}")
                    last_status = state

            result = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: job.poll_until_complete(
                    timeout=300.0,  # 5 minutes max (should finish in ~1 min)
                    interval=2.0,  # Check every 2 seconds
                    on_status=on_status,
                ),
            )

            total_time = time.time() - start_time
            print(f"\n✅ GEPA optimization complete in {total_time:.1f}s\n")

            # Get results
            from synth_ai.learning.prompt_learning_client import PromptLearningClient
            from synth_ai.api.train.utils import ensure_api_base

            client = PromptLearningClient(
                ensure_api_base(backend_url),
                api_key,
            )
            prompt_results = await client.get_prompts(job._job_id)

            print("=" * 80)
            print("Results")
            print("=" * 80 + "\n")

            if prompt_results.best_score is not None:
                print(f"Best score: {prompt_results.best_score:.2%}")
            else:
                print("Best score: N/A (job may have failed)")

            # Parse and display candidates info
            if prompt_results.attempted_candidates is not None:
                candidates = prompt_results.attempted_candidates
                if isinstance(candidates, list):
                    if len(candidates) > 0:
                        print(f"Total candidates: {len(candidates)}")
                    else:
                        print(f"Total candidates: 0 (no candidates evaluated)")
                else:
                    print(f"Total candidates: {candidates}")
            else:
                print("Total candidates: N/A")
            print()

    except KeyboardInterrupt:
        print("\n⚠️  Interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)

    print("=" * 80)
    print("✅ In-process GEPA Banking77 demo complete!")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    asyncio.run(main())

