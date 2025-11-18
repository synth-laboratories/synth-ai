#!/usr/bin/env python3
"""
In-Process MIPRO Demo
=====================

This script runs MIPRO optimization with a task app started entirely 
in-process - no separate terminals or manual process management needed!

Usage:
    cd examples/blog_posts/vendored_prompt_learning
    uv run scripts/run_mipro_in_process.py

Requirements:
    - GROQ_API_KEY in .env (for policy model)
    - OPENAI_API_KEY in .env (for meta-model)
    - cloudflared binary (will auto-install if missing)
    - synth-ai backend running (localhost:8000)
"""

import asyncio
import os
import sys
import tempfile
import time
from pathlib import Path

from dotenv import load_dotenv

# Load environment from current working directory
load_dotenv()  # Load from current dir and parents


from synth_ai.api.train.prompt_learning import PromptLearningJob
from synth_ai.task import InProcessTaskApp
from synth_ai.urls import BACKEND_URL_BASE


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
    config_path = Path("banking77_mipro_local.toml")

    if not config_path.exists():
        print(f"❌ Error: Config file not found: {config_path}")
        sys.exit(1)

    api_key = os.getenv("SYNTH_API_KEY", "test")
    task_app_api_key = os.getenv("ENVIRONMENT_API_KEY", "test")

    print("Configuration:")
    print(f"  Config: {config_path.name}")
    print(f"  Backend: {BACKEND_URL_BASE}")
    print(f"  Task App: Starting in-process...")
    print()

    # Import task app config factory
    task_app_path = Path("banking77_task_app.py")

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
                # Reduce iterations and evaluations for very fast demo (~1 min)
                original_iterations = config["prompt_learning"]["mipro"].get("num_iterations", 5)
                original_evals = config["prompt_learning"]["mipro"].get("num_evaluations_per_iteration", 2)
                config["prompt_learning"]["mipro"]["num_iterations"] = 1  # Minimal: 1 iteration
                config["prompt_learning"]["mipro"]["num_evaluations_per_iteration"] = 1  # 1 eval per iteration
                config["prompt_learning"]["mipro"]["batch_size"] = 1  # Single seed per eval
                config["prompt_learning"]["mipro"]["max_concurrent"] = 1  # One at a time for reliability
                # Reduce seed pools to single seed for ultra-fast reliable testing
                config["prompt_learning"]["mipro"]["bootstrap_train_seeds"] = [0]  # Single seed for bootstrap
                config["prompt_learning"]["mipro"]["online_pool"] = [1]  # Single seed for online eval
                config["prompt_learning"]["mipro"]["val_seeds"] = [2]  # Single seed for validation (required by MIPRO)
                config["prompt_learning"]["mipro"]["reference_pool"] = [3]  # Minimal reference corpus
                config["prompt_learning"]["mipro"]["test_pool"] = [4]  # Minimal test set to satisfy schema
                # Disable meta-updates (they require reference examples from trace store)
                if "meta_update" in config["prompt_learning"]["mipro"]:
                    config["prompt_learning"]["mipro"]["meta_update"]["enabled"] = False
                # Reduce TPE startup trials to match minimal budget
                if "tpe" in config["prompt_learning"]["mipro"]:
                    config["prompt_learning"]["mipro"]["tpe"]["n_startup_trials"] = 1
                print(f"📊 Reduced to minimal budget: 1 iteration × 1 eval × 1 seed = ~3 rollouts")
                print(f"   Using: bootstrap=[0], online=[1], val=[2]")
                print(f"   Reference/test pools minimized to [3] / [4] for schema compatibility")
                print(f"   (Original: {original_iterations} iterations, {original_evals} evals per iteration)\n")

            # Write modified config to temp file
            with tempfile.NamedTemporaryFile(mode='w', suffix='.toml', delete=False) as f:
                toml.dump(config, f)
                temp_config_path = f.name

            try:
                job = PromptLearningJob.from_config(
                    config_path=temp_config_path,
                    backend_url=BACKEND_URL_BASE,
                    api_key=api_key,
                    task_app_api_key=task_app_api_key,
                )

                print(f"Task app URL: {task_app.url}")
                print(f"Backend URL: {BACKEND_URL_BASE}\n")
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
            print(f"\n✅ MIPRO optimization complete in {total_time:.1f}s\n")

            # Get results
            from synth_ai.learning.prompt_learning_client import PromptLearningClient
            from synth_ai.api.train.utils import ensure_api_base

            client = PromptLearningClient(
                ensure_api_base(BACKEND_URL_BASE),
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
