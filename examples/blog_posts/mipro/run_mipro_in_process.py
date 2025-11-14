#!/usr/bin/env python3
"""
In-Process MIPRO Demo
=====================

This script runs MIPRO optimization with a task app started entirely 
in-process - no separate terminals or manual process management needed!

Usage:
    cd /Users/joshpurtell/Documents/GitHub/synth-ai/examples/blog_posts/mipro
    source ../../../.env
    uv run python run_mipro_in_process.py

Requirements:
    - GROQ_API_KEY in .env (for policy model)
    - OPENAI_API_KEY in .env (for meta-model)
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
env_path = Path(__file__).resolve().parent.parent.parent.parent / ".env"
load_dotenv(env_path)

# Add parent to path for imports
parent_dir = Path(__file__).resolve().parent.parent.parent.parent
if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))

from synth_ai.api.train.prompt_learning import PromptLearningJob
from synth_ai.task import InProcessTaskApp


async def main():
    """Run MIPRO optimization with in-process task app."""

    print("\n" + "=" * 80)
    print("In-Process MIPRO Demo")
    print("=" * 80 + "\n")

    # Check requirements
    if not os.getenv("GROQ_API_KEY"):
        print("❌ Error: GROQ_API_KEY required in .env (for policy model)")
        sys.exit(1)
    
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ Error: OPENAI_API_KEY required in .env (for meta-model)")
        sys.exit(1)

    # Configuration
    config_path = (
        Path(__file__).parent
        / "configs"
        / "banking77_mipro_local.toml"
    )

    if not config_path.exists():
        print(f"❌ Error: Config file not found: {config_path}")
        sys.exit(1)

    backend_url = os.getenv("BACKEND_BASE_URL", "https://backend-local.usesynth.ai")
    api_key = os.getenv("SYNTH_API_KEY", "test")
    task_app_api_key = os.getenv("ENVIRONMENT_API_KEY", "test")

    print("Configuration:")
    print(f"  Config: {config_path.name}")
    print(f"  Backend: {backend_url}")
    print(f"  Task App: Starting in-process...")
    print()

    # Import task app config factory
    task_app_path = (
        Path(__file__).resolve().parent.parent.parent.parent
        / "examples"
        / "task_apps"
        / "banking77"
        / "banking77_task_app.py"
    )

    if not task_app_path.exists():
        print(f"❌ Error: Task app not found: {task_app_path}")
        sys.exit(1)

    # Run MIPRO with in-process task app
    try:
        async with InProcessTaskApp(
            task_app_path=task_app_path,
            port=8114,
            api_key=task_app_api_key,
        ) as task_app:
            print(f"✅ Task app running at: {task_app.url}")
            print(f"✅ Cloudflare tunnel active")
            print()

            # Create MIPRO job
            print("=" * 80)
            print("Running MIPRO Optimization")
            print("=" * 80 + "\n")

            # Load and modify config before creating job
            import toml

            config = toml.load(config_path)
            config["prompt_learning"]["task_app_url"] = task_app.url
            
            # Reduce budget for faster demo (optional)
            if "mipro" in config["prompt_learning"]:
                # Reduce iterations for faster demo
                original_iterations = config["prompt_learning"]["mipro"].get("num_iterations", 5)
                config["prompt_learning"]["mipro"]["num_iterations"] = min(3, original_iterations)
                print(f"📊 Reduced iterations to {config['prompt_learning']['mipro']['num_iterations']} for faster demo")
                print(f"   (Original: {original_iterations})\n")

            # Write modified config to temp file
            with tempfile.NamedTemporaryFile(mode='w', suffix='.toml', delete=False) as f:
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
                    timeout=3600.0,
                    interval=5.0,
                    on_status=on_status,
                ),
            )

            total_time = time.time() - start_time
            print(f"\n✅ MIPRO optimization complete in {total_time:.1f}s\n")

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
                        # Extract useful stats from candidates
                        accuracies = [c.get("accuracy", 0.0) for c in candidates if isinstance(c, dict)]
                        avg_accuracy = sum(accuracies) / len(accuracies) if accuracies else 0.0
                        max_accuracy = max(accuracies) if accuracies else 0.0
                        min_accuracy = min(accuracies) if accuracies else 0.0
                        
                        print(f"Total candidates: {len(candidates)}")
                        if accuracies:
                            print(f"  Accuracy range: {min_accuracy:.2%} - {max_accuracy:.2%} (avg: {avg_accuracy:.2%})")
                    else:
                        print(f"Total candidates: 0 (no candidates evaluated)")
                else:
                    print(f"Total candidates: {candidates}")
            else:
                print("Total candidates: N/A")
            print()

            if prompt_results.best_prompt:
                print("Best prompt:")
                print("-" * 80)
                # Extract prompt text
                if "prompt_sections" in prompt_results.best_prompt:
                    sections = prompt_results.best_prompt["prompt_sections"]
                    prompt_text = "\n\n".join(
                        [s.get("content", "") for s in sections if s.get("content")]
                    )
                    print(prompt_text[:500])
                    if len(prompt_text) > 500:
                        print("\n... [truncated]")
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
    print("✅ In-process MIPRO demo complete!")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    asyncio.run(main())

