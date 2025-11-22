#!/usr/bin/env python3
"""Run GEPA optimization on Banking77 multi-step pipelines (2-step and 3-step)."""

from __future__ import annotations

import asyncio
import os
import sys
import tempfile
import time
import argparse
from pathlib import Path

from dotenv import load_dotenv

# Load environment from repo root
env_path = Path(__file__).resolve().parents[3] / ".env"
load_dotenv(env_path)

# Add parent to path for imports
parent_dir = Path(__file__).resolve().parents[3]
if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))

from synth_ai.api.train.prompt_learning import PromptLearningJob
from synth_ai.task import InProcessTaskApp


async def run_gepa(pipeline_type: str, config_path: Path):
    """Run GEPA optimization for a specific pipeline."""

    # Pipeline configuration
    pipeline_configs = {
        "2step": {
            "task_app_path": parent_dir / "examples" / "task_apps" / "banking77_pipeline" / "banking77_pipeline_task_app.py",
            "port": 8114,
            "env_name": "banking77-pipeline"
        },
        "3step": {
            "task_app_path": parent_dir / "examples" / "task_apps" / "banking77_3step" / "banking77_3step_task_app.py",
            "port": 8115,
            "env_name": "banking77-3step"
        }
    }

    if pipeline_type not in pipeline_configs:
        raise ValueError(f"Unknown pipeline type: {pipeline_type}. Must be '2step' or '3step'")

    config = pipeline_configs[pipeline_type]

    print("\n" + "=" * 80)
    print(f"GEPA Optimization: Banking77 {pipeline_type.upper()} Pipeline")
    print("=" * 80 + "\n")

    # Check requirements
    if not os.getenv("GROQ_API_KEY"):
        print("❌ Error: GROQ_API_KEY required in .env (for policy model)")
        sys.exit(1)

    if not config_path.exists():
        print(f"❌ Error: Config file not found: {config_path}")
        sys.exit(1)

    if not config["task_app_path"].exists():
        print(f"❌ Error: Task app not found: {config['task_app_path']}")
        sys.exit(1)

    # Backend configuration
    backend_url = os.getenv("BACKEND_BASE_URL", "https://synth-backend-dev-docker.onrender.com")
    api_key = os.getenv("SYNTH_API_KEY", "test")
    task_app_api_key = os.getenv("ENVIRONMENT_API_KEY", "test")

    # Determine tunnel mode
    is_backend_localhost = (
        backend_url.startswith("http://localhost")
        or backend_url.startswith("http://127.0.0.1")
    )

    if is_backend_localhost:
        os.environ["SYNTH_TUNNEL_MODE"] = "local"
        use_local_mode = True
        print("ℹ️  Configuration: local/local")
        print("   Backend: localhost:8000")
        print("   Task App: localhost (no tunnel)")
    else:
        os.environ["SYNTH_TUNNEL_MODE"] = "quick"
        use_local_mode = False
        os.environ["EXTERNAL_BACKEND_URL"] = backend_url.rstrip("/")
        print("ℹ️  Configuration: tunnel/tunnel")
        print(f"   Backend tunnel: {backend_url}")
        print(f"   Task app: will create its own tunnel")

    print("\nConfiguration:")
    print(f"  Pipeline: Banking77 {pipeline_type}")
    print(f"  Config: {config_path.name}")
    print(f"  Backend: {backend_url}")
    print(f"  Task App: {config['task_app_path'].name}")
    print()

    # Enable direct provider URLs for Groq
    os.environ["ALLOW_DIRECT_PROVIDER_URLS"] = "1"

    # Run GEPA with in-process task app
    try:
        async with InProcessTaskApp(
            task_app_path=config["task_app_path"],
            port=config["port"],
            api_key=task_app_api_key,
        ) as task_app:
            print(f"✅ Task app running at: {task_app.url}")
            if use_local_mode:
                print("✅ Running in local mode (no tunnel)\n")
            else:
                print("✅ Cloudflare tunnel active\n")
            print("=" * 80)
            print("Running GEPA Optimization")
            print("=" * 80 + "\n")

            # Load and modify config before creating job
            import toml

            job_config = toml.load(config_path)
            job_config["prompt_learning"]["task_app_url"] = task_app.url

            # Print GEPA parameters
            if "gepa" in job_config["prompt_learning"]:
                gepa_config = job_config["prompt_learning"]["gepa"]
                if "population" in gepa_config:
                    num_generations = gepa_config["population"].get("num_generations", 1)
                    children_per_gen = gepa_config["population"].get("children_per_generation", 5)
                    print(f"📊 Running {num_generations} generations with {children_per_gen} children per generation")
                    print(f"   (Total: {num_generations * children_per_gen} prompt candidates)\n")

            # Write modified config to temp file
            with tempfile.NamedTemporaryFile(mode='w', suffix='.toml', delete=False) as f:
                toml.dump(job_config, f)
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
            print(f"\n✅ GEPA optimization complete in {total_time:.1f}s\n")

            # Get results
            import httpx
            from synth_ai.learning.prompt_learning_client import PromptLearningClient
            from synth_ai.api.train.utils import ensure_api_base

            client = PromptLearningClient(
                ensure_api_base(backend_url),
                api_key,
            )

            # Get job results
            job_results = client.get_job(job_id)

            print("\n" + "=" * 80)
            print("GEPA Optimization Results")
            print("=" * 80)
            print(f"\nJob ID: {job_id}")
            print(f"Status: {job_results.get('status', 'unknown')}")
            print(f"Best Score: {job_results.get('best_score', 'N/A')}")
            print(f"Val Score: {job_results.get('val_score', 'N/A')}")
            print(f"Total Rollouts: {job_results.get('total_rollouts', 'N/A')}")
            print(f"Total Cost: ${job_results.get('total_cost_usd', 0):.4f}")

            # Get optimized prompt if available
            if "best_prompt" in job_results:
                best_prompt = job_results["best_prompt"]
                print(f"\nOptimized Prompt ID: {best_prompt.get('id', 'N/A')}")
                print(f"Optimized Prompt Name: {best_prompt.get('name', 'N/A')}")

                # Save prompt to file
                results_dir = config_path.parent / "results" / f"gepa_banking77_{pipeline_type}"
                results_dir.mkdir(parents=True, exist_ok=True)

                prompt_file = results_dir / f"optimized_prompt_{job_id}.json"
                import json
                with open(prompt_file, 'w') as f:
                    json.dump(best_prompt, f, indent=2)
                print(f"Optimized prompt saved to: {prompt_file}")

            print("\n" + "=" * 80 + "\n")

    except Exception as e:
        print(f"\n❌ Error during GEPA optimization:")
        print(f"   Type: {type(e).__name__}")
        print(f"   Message: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


async def main():
    """Main entry point for Banking77 multi-step GEPA runner."""
    parser = argparse.ArgumentParser(description="Run GEPA on Banking77 multi-step pipelines")
    parser.add_argument(
        "--pipeline",
        choices=["2step", "3step", "both"],
        default="both",
        help="Which pipeline(s) to optimize (default: both)",
    )

    args = parser.parse_args()

    # Determine which pipelines to run
    pipelines_to_run = []
    if args.pipeline in ["2step", "both"]:
        pipelines_to_run.append(("2step", Path(__file__).parent / "banking77_2step_gepa.toml"))
    if args.pipeline in ["3step", "both"]:
        pipelines_to_run.append(("3step", Path(__file__).parent / "banking77_3step_gepa.toml"))

    # Run GEPA for each pipeline
    for pipeline_type, config_path in pipelines_to_run:
        await run_gepa(pipeline_type, config_path)
        print("\n")


if __name__ == "__main__":
    asyncio.run(main())
