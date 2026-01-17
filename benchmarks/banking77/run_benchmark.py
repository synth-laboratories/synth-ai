#!/usr/bin/env python3
"""
Banking77 GEPA Benchmark Runner - LangProBe Dec 31 2024

Runs GEPA prompt optimization on Banking77 across 3 models × 3 runs:
- gpt-4.1-nano
- gpt-5-nano
- gpt-4o-mini

Usage:
    # Run all experiments
    python run_benchmark.py

    # Run specific model
    python run_benchmark.py --model gpt-4.1-nano

    # Run specific run
    python run_benchmark.py --model gpt-4.1-nano --run 1

    # Dry run (show what would be executed)
    python run_benchmark.py --dry-run

Environment Variables:
    SYNTH_API_KEY: API key for Synth backend (will mint demo key if not set)
"""

import argparse
import asyncio
import json
import os
import time
from datetime import datetime
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import httpx

try:
    import nest_asyncio

    nest_asyncio.apply()
except ImportError:
    pass

from synth_ai.core.env import mint_demo_api_key
from synth_ai.core.urls import synth_base_url
from synth_ai.sdk.api.train.prompt_learning import PromptLearningJob
from synth_ai.sdk.learning.rl import mint_environment_api_key, setup_environment_api_key
from synth_ai.sdk.task import run_server_background
from synth_ai.sdk.tunnels import (
    TunnelBackend,
    TunneledLocalAPI,
    cleanup_all,
    kill_port,
    wait_for_health_check,
)


def _load_gepa_banking77_demo_module():
    repo_root = Path(__file__).resolve().parents[2]
    demo_path = repo_root / "demos" / "gepa_banking77" / "run_demo.py"
    if not demo_path.exists():
        raise FileNotFoundError(f"Expected demo file not found: {demo_path}")
    spec = spec_from_file_location("gepa_banking77_demo", demo_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module spec for: {demo_path}")
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_demo = _load_gepa_banking77_demo_module()
create_banking77_local_api = _demo.create_banking77_local_api
BANKING77_LABELS = _demo.BANKING77_LABELS

# Constants
LOCAL_API_PORT = 8101

CONFIGS_DIR = Path(__file__).parent / "configs"
RESULTS_DIR = Path(__file__).parent / "results"

MODELS = ["gpt-4.1-nano", "gpt-5-nano", "gpt-4o-mini"]
RUNS = [1, 2, 3]

# Baseline system prompt
BASELINE_SYSTEM_PROMPT = "You are an expert banking assistant that classifies customer queries into banking intents. Given a customer message, respond with exactly one intent label from the provided list using the `banking77_classify` tool."


def model_to_short(model: str) -> str:
    """Convert model name to short form for file names."""
    return model.replace("-", "").replace(".", "")


def get_config_path(model: str, run: int) -> Path:
    """Get config file path for a model/run combination."""
    short = model_to_short(model)
    return CONFIGS_DIR / f"banking77_{short}_run{run}.toml"


def get_result_path(model: str, run: int) -> Path:
    """Get result file path for a model/run combination."""
    short = model_to_short(model)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    return RESULTS_DIR / f"banking77_{short}_run{run}_result.json"


async def run_single_experiment(
    model: str,
    run: int,
    backend_url: str,
    api_key: str,
    env_api_key: str,
    local_api_url: str,
) -> dict:
    """Run a single GEPA experiment."""

    config_path = get_config_path(model, run)
    result_path = get_result_path(model, run)

    print(f"\n{'=' * 60}")
    print(f"Running: {model} - Run {run}")
    print(f"Config: {config_path}")
    print(f"{'=' * 60}")

    # Load config
    import tomllib

    with open(config_path, "rb") as f:
        config = tomllib.load(f)

    # Update task_app_url
    config["prompt_learning"]["task_app_url"] = local_api_url

    start_time = time.time()

    # Create and submit job
    pl_job = PromptLearningJob.from_dict(
        config_dict=config,
        synth_base_url=backend_url,
        synth_user_key=api_key,
        localapi_key=env_api_key,
        skip_health_check=True,
    )

    job_id = pl_job.submit()
    print(f"Job ID: {job_id}")

    # Poll until complete
    result = pl_job.poll_until_complete(timeout=3600.0, interval=5.0, progress=True)

    elapsed = time.time() - start_time

    # Build result record
    record = {
        "model": model,
        "run": run,
        "job_id": job_id,
        "status": result.status.value,
        "best_score": result.best_score if result.succeeded else None,
        "error": result.error if result.failed else None,
        "elapsed_seconds": elapsed,
        "timestamp": datetime.now().isoformat(),
        "backend_url": backend_url,
    }

    # Save result
    with open(result_path, "w") as f:
        json.dump(record, f, indent=2)

    print(f"\nResult: {result.status.value}")
    if result.succeeded:
        print(f"Best Score: {result.best_score:.1%}")
    elif result.failed:
        print(f"Error: {result.error}")
    print(f"Elapsed: {elapsed:.1f}s")
    print(f"Saved to: {result_path}")

    return record


async def main():
    parser = argparse.ArgumentParser(description="Run Banking77 GEPA Benchmark")
    parser.add_argument("--model", choices=MODELS, help="Run only this model")
    parser.add_argument("--run", type=int, choices=RUNS, help="Run only this run number")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be executed")
    parser.add_argument("--port", type=int, default=LOCAL_API_PORT, help="Local API port")
    args = parser.parse_args()

    # Determine which experiments to run
    models_to_run = [args.model] if args.model else MODELS
    runs_to_run = [args.run] if args.run else RUNS

    experiments = [(m, r) for m in models_to_run for r in runs_to_run]

    print("Banking77 GEPA Benchmark - LangProBe Dec 31 2024")
    print("=" * 60)
    print(f"Models: {models_to_run}")
    print(f"Runs: {runs_to_run}")
    print(f"Total experiments: {len(experiments)}")

    if args.dry_run:
        print("\nDry run - would execute:")
        for model, run in experiments:
            config_path = get_config_path(model, run)
            print(f"  - {model} run {run}: {config_path}")
        return

    # Setup
    backend_url = synth_base_url()
    print(f"\nBackend: {backend_url}")

    # Check backend health
    r = httpx.get(f"{backend_url}/health", timeout=30)
    if r.status_code != 200:
        raise RuntimeError(f"Backend not healthy: {r.status_code}")
    print("Backend health: OK")

    # Get API key
    api_key = os.environ.get("SYNTH_API_KEY", "")
    if not api_key:
        print("No SYNTH_API_KEY found, minting demo key...")
        api_key = mint_demo_api_key(synth_base_url=backend_url, ttl_hours=8)
        print(f"Demo API Key: {api_key[:25]}...")
    else:
        print(f"Using SYNTH_API_KEY: {api_key[:20]}...")

    # Mint environment key
    env_api_key = mint_environment_api_key()
    print(f"Minted env key: {env_api_key[:12]}...{env_api_key[-4:]}")

    result = setup_environment_api_key(api_key, env_api_key, synth_base_url=backend_url)
    print(f"Uploaded env key: {result}")

    # Start local API
    print(f"\nStarting Banking77 local API on port {args.port}...")
    app = create_banking77_local_api(BASELINE_SYSTEM_PROMPT, env_api_key)

    kill_port(args.port)
    run_server_background(app, args.port)

    await wait_for_health_check("localhost", args.port, env_api_key, timeout=30.0)
    print("Local API ready!")

    # Create tunnel (or use localhost directly for local backend)
    if backend_url.startswith("http://localhost"):
        # When running against local backend, no tunnel needed
        local_api_url = f"http://localhost:{args.port}"
        print(f"\nUsing local URL directly: {local_api_url}")
    else:
        print("\nProvisioning Cloudflare tunnel...")
        tunnel = await TunneledLocalAPI.create(
            local_port=args.port,
            backend=TunnelBackend.CloudflareManagedTunnel,
            synth_user_key=api_key,
            localapi_key=env_api_key,
            synth_base_url=backend_url,
            progress=True,
        )
        local_api_url = tunnel.url
        print(f"Local API URL: {local_api_url}")

    # Run experiments
    all_results = []
    try:
        for model, run in experiments:
            try:
                record = await run_single_experiment(
                    model=model,
                    run=run,
                    backend_url=backend_url,
                    api_key=api_key,
                    env_api_key=env_api_key,
                    local_api_url=local_api_url,
                )
                all_results.append(record)
            except Exception as e:
                print(f"ERROR in {model} run {run}: {e}")
                all_results.append(
                    {
                        "model": model,
                        "run": run,
                        "status": "error",
                        "error": str(e),
                        "timestamp": datetime.now().isoformat(),
                    }
                )
    finally:
        print("\nCleaning up...")
        cleanup_all()

    # Summary
    print("\n" + "=" * 60)
    print("BENCHMARK SUMMARY")
    print("=" * 60)
    print(f"{'Model':<15} {'Run':<5} {'Status':<12} {'Score':<10}")
    print("-" * 60)
    for r in all_results:
        score = f"{r.get('best_score', 0):.1%}" if r.get("best_score") else "N/A"
        print(f"{r['model']:<15} {r['run']:<5} {r['status']:<12} {score:<10}")

    # Save summary
    summary_path = RESULTS_DIR / "benchmark_summary.json"
    with open(summary_path, "w") as f:
        json.dump(
            {
                "timestamp": datetime.now().isoformat(),
                "backend_url": backend_url,
                "experiments": all_results,
            },
            f,
            indent=2,
        )
    print(f"\nSummary saved to: {summary_path}")


if __name__ == "__main__":
    asyncio.run(main())
