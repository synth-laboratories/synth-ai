#!/usr/bin/env python3
"""
In-Process MIPRO Demo
=====================

This script runs MIPRO optimization with a task app started entirely
in-process - no separate terminals or manual process management needed!

Usage:
    uv run python run_mipro_in_process.py

Requirements:
    - Run `synth-ai setup` first to configure credentials
    - GROQ_API_KEY or OPENAI_API_KEY in environment (for policy model)
    - cloudflared binary (will auto-install if missing)
"""

import asyncio
import os
import sys
import tempfile
import time
from pathlib import Path

import toml

from synth_ai.core.env import get_synth_and_localapi_keys
from synth_ai.sdk.api.train.prompt_learning import PromptLearningJob
from synth_ai.sdk.task import InProcessTaskApp

CURRENT_DIR = Path(__file__).parent
TASK_APP = CURRENT_DIR / "task_app.py"
TRAIN_CFG = CURRENT_DIR / "train_cfg.toml"


def _validate_vendor_keys() -> None:
    """Check that at least one LLM vendor key is available."""
    if not os.getenv("GROQ_API_KEY") and not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError(
            "Either GROQ_API_KEY or OPENAI_API_KEY required. Set one in your environment."
        )


async def main():
    """Run MIPRO optimization with in-process task app."""
    synth_key, env_key = get_synth_and_localapi_keys()
    _validate_vendor_keys()

    # Run MIPRO with in-process task app
    try:
        async with InProcessTaskApp(
            task_app_path=TASK_APP,
            port=8114,
            api_key=env_key,
        ) as task_app:
            print(f"✅ Task app running at: {task_app.url}")
            print("✅ Cloudflare tunnel active")
            print()

            print("=" * 80)
            print("Running MIPRO Optimization")
            print("=" * 80 + "\n")

            cfg = toml.load(TRAIN_CFG)
            cfg["prompt_learning"]["task_app_url"] = task_app.url

            # Reduce budget for faster demo (optional)
            if "mipro" in cfg["prompt_learning"]:
                # Reduce iterations for faster demo
                original_iterations = cfg["prompt_learning"]["mipro"].get("num_iterations", 5)
                cfg["prompt_learning"]["mipro"]["num_iterations"] = min(3, original_iterations)
                print(
                    f"📊 Reduced iterations to {cfg['prompt_learning']['mipro']['num_iterations']} for faster demo"
                )
                print(f"   (Original: {original_iterations})\n")

            # Write modified config to temp file
            with tempfile.NamedTemporaryFile(mode="w", suffix=".toml", delete=False) as f:
                toml.dump(cfg, f)
                temp_config_path = f.name

            try:
                job = PromptLearningJob.from_config(
                    config_path=temp_config_path,
                    api_key=synth_key,
                    task_app_api_key=env_key,
                )

                print(f"Task app URL: {task_app.url}")
                print("Submitting job...\n")

                try:
                    job_id = job.submit()
                    print(f"✅ Job submitted: {job_id}\n")
                except Exception as e:
                    print("\n❌ Error submitting job:")
                    print(f"   Type: {type(e).__name__}")
                    print(f"   Message: {str(e)}")
                    print("\n   Full error details:")
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
                            score_str = (
                                f" | Best: {best_score:.3f}" if best_score is not None else ""
                            )
                            print(
                                f"[{timestamp}] {elapsed:6.1f}s  Status: {state} ({completed}/{total} = {pct:.1f}%){score_str}"
                            )
                        else:
                            score_str = (
                                f" | Best: {best_score:.3f}" if best_score is not None else ""
                            )
                            print(f"[{timestamp}] {elapsed:6.1f}s  Status: {state}{score_str}")
                    else:
                        score_str = f" | Best: {best_score:.3f}" if best_score is not None else ""
                        print(f"[{timestamp}] {elapsed:6.1f}s  Status: {state}{score_str}")
                    last_status = state

            await asyncio.get_event_loop().run_in_executor(
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
            from synth_ai.sdk.learning.prompt_learning_client import PromptLearningClient

            assert synth_key is not None, "synth_key must be set"
            client = PromptLearningClient(api_key=synth_key)
            prompt_results = await client.get_prompts(str(job._job_id))

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
                        accuracies = [
                            c.get("accuracy", 0.0) for c in candidates if isinstance(c, dict)
                        ]
                        avg_accuracy = sum(accuracies) / len(accuracies) if accuracies else 0.0
                        max_accuracy = max(accuracies) if accuracies else 0.0
                        min_accuracy = min(accuracies) if accuracies else 0.0

                        print(f"Total candidates: {len(candidates)}")
                        if accuracies:
                            print(
                                f"  Accuracy range: {min_accuracy:.2%} - {max_accuracy:.2%} (avg: {avg_accuracy:.2%})"
                            )
                    else:
                        print("Total candidates: 0 (no candidates evaluated)")
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
