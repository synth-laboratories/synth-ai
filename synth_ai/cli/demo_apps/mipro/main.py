#!/usr/bin/env python3
"""
In-Process MIPRO Demo
=====================

This script runs MIPRO optimization with a task app started entirely
in-process - no separate terminals or manual process management needed!

Usage:
    uv run python run_mipro_in_process.py
    uv run python run_mipro_in_process.py --env /path/to/.env

Requirements:
    - GROQ_API_KEY in .env (for policy model)
    - OPENAI_API_KEY in .env (for meta-model)
    - cloudflared binary (will auto-install if missing)
    - Dev backend running (default: https://synth-backend-dev-docker.onrender.com)
"""

import argparse
import asyncio
import os
import sys
import tempfile
import time
from pathlib import Path

import toml
from dotenv import load_dotenv
from synth_ai.core.urls import BACKEND_URL_BASE
from synth_ai.sdk.api.train.prompt_learning import PromptLearningJob
from synth_ai.sdk.task import InProcessTaskApp

CURRENT_DIR = Path(__file__).parent
TASK_APP = CURRENT_DIR / "task_app.py"
TRAIN_CFG = CURRENT_DIR / "train_cfg.toml"


def _load_dotenv(args) -> None:
    is_env_loaded = False
    dotenv_path = None
    if args.env:
        if not Path(args.env).exists():
            print(f"❌ Error: .env file not found: {args.env}")
            sys.exit(1)
        dotenv_path = Path(args.env)
        is_env_loaded = load_dotenv(dotenv_path)
    if not is_env_loaded:
        default_path = Path.cwd() / ".env"
        if default_path.exists():
            dotenv_path = default_path
            is_env_loaded = load_dotenv(dotenv_path)
        else:
            fallback_path = Path(".env")
            if fallback_path.exists():
                dotenv_path = fallback_path
                is_env_loaded = load_dotenv(dotenv_path)
    if is_env_loaded and dotenv_path:
        print(f"Loaded .env from {dotenv_path.absolute()}")
    elif is_env_loaded:
        print("Loaded .env")


def _validate_env() -> None:
    first_party_msg = "Run `uvx synth-ai setup` to fetch from your browser, load to your process environment, and save to .env in CWD"
    third_party_msg = "Pass the path to your .env via `uv run demo_mipro/main.py --env [PATH]` or load to process envrionment"
    if not os.getenv("SYNTH_API_KEY"):
        raise RuntimeError(f"SYNTH_API_KEY required. {first_party_msg}")
    if not os.getenv("ENVIRONMENT_API_KEY"):
        raise RuntimeError(f"ENVIRONMENT_API_KEY required. {first_party_msg}")
    if not os.getenv("GROQ_API_KEY") and not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError(f"Either GROQ_API_KEY or OPENAI_API_KEY required. {third_party_msg}")


async def main():
    """Run MIPRO optimization with in-process task app."""

    synth_key = os.getenv("SYNTH_API_KEY")
    env_key = os.getenv("ENVIRONMENT_API_KEY")
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
                print(f"📊 Reduced iterations to {cfg['prompt_learning']['mipro']['num_iterations']} for faster demo")
                print(f"   (Original: {original_iterations})\n")

            # Write modified config to temp file
            with tempfile.NamedTemporaryFile(mode='w', suffix='.toml', delete=False) as f:
                toml.dump(cfg, f)
                temp_config_path = f.name

            try:
                job = PromptLearningJob.from_config(
                    config_path=temp_config_path,
                    backend_url=BACKEND_URL_BASE,
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
            from synth_ai.sdk.api.train.utils import ensure_api_base
            from synth_ai.sdk.learning.prompt_learning_client import PromptLearningClient

            assert synth_key is not None, "synth_key must be set"
            client = PromptLearningClient(
                ensure_api_base(BACKEND_URL_BASE),
                synth_key,
            )
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
                        accuracies = [c.get("accuracy", 0.0) for c in candidates if isinstance(c, dict)]
                        avg_accuracy = sum(accuracies) / len(accuracies) if accuracies else 0.0
                        max_accuracy = max(accuracies) if accuracies else 0.0
                        min_accuracy = min(accuracies) if accuracies else 0.0
                        
                        print(f"Total candidates: {len(candidates)}")
                        if accuracies:
                            print(f"  Accuracy range: {min_accuracy:.2%} - {max_accuracy:.2%} (avg: {avg_accuracy:.2%})")
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
    parser = argparse.ArgumentParser(description="Run MIPRO optimization in-process")
    parser.add_argument(
        "--env",
        type=str,
        help="Path to .env file (default: .env in current directory)",
        default=None,
    )
    args = parser.parse_args()
    _load_dotenv(args)
    try:
        _validate_env()
    except Exception:
        sys.exit(1)
    asyncio.run(main())
    sys.exit(0)
