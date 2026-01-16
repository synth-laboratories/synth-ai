#!/usr/bin/env python3
"""
Run minimal GEPA job for EngineBench.

Uses minimum viable seeds (13) and 2 generations to verify GEPA works.

Usage:
    cd /path/to/synth-ai
    uv run python demos/engine_bench/run_gepa_minimal.py
"""

import argparse
import asyncio
import time
from pathlib import Path

import httpx
from localapi_engine_bench import INSTANCE_IDS, app
from synth_ai.core.urls import synth_health_url
from synth_ai.sdk.api.train.prompt_learning import PromptLearningJob
from synth_ai.sdk.auth import get_or_mint_synth_user_key
from synth_ai.sdk.task import run_server_background
from synth_ai.sdk.tunnels import PortConflictBehavior, acquire_port

SYNTH_USER_KEY = get_or_mint_synth_user_key()

parser = argparse.ArgumentParser(description="Run minimal GEPA for EngineBench")
parser.add_argument("--port", type=int, default=8020)
parser.add_argument(
    "--config",
    type=str,
    default="enginebench_gepa_minimal.toml",
    help="Path to GEPA config file",
)
parser.add_argument("--budget", type=int, help="Override rollout budget")
parser.add_argument("--generations", type=int, help="Override number of generations")
parser.add_argument("--timeout", type=int, default=400, help="Agent timeout per rollout")
args = parser.parse_args()


def wait_for_health(host: str, port: int, api_key: str, timeout: float = 30.0) -> None:
    """Wait for task app health check."""
    url = f"http://{host}:{port}/health"
    headers = {"X-API-Key": api_key} if api_key else {}
    start = time.time()
    while time.time() - start < timeout:
        try:
            r = httpx.get(url, headers=headers, timeout=5.0)
            if r.status_code in (200, 400):
                return
        except (httpx.RequestError, httpx.TimeoutException):
            pass
        time.sleep(0.5)
    raise RuntimeError(f"Health check failed: {url}")


async def main():
    print("=" * 60)
    print("ENGINEBENCH - MINIMAL GEPA JOB")
    print("=" * 60)
    print("Config: 13 seeds (10 pareto + 3 feedback), 2 generations")
    print(f"Instances available: {len(INSTANCE_IDS)}")

    # Check backend health
    async with httpx.AsyncClient() as client:
        try:
            r = await client.get(synth_health_url(), timeout=10)
            print(f"Synth health: {r.status_code}")
        except Exception as e:
            print(f"Synth check failed: {e}")
            return

    print(f"API Key: {SYNTH_USER_KEY[:20]}...")

    # Acquire port and prepare localapi URL
    port = acquire_port(args.port, on_conflict=PortConflictBehavior.FIND_NEW)
    localapi_url = f"http://localhost:{port}"

    # Load config
    config_path = Path(__file__).parent / args.config
    if not config_path.exists():
        print(f"Config file not found: {config_path}")
        return

    print(f"\nLoading config from: {config_path}")

    # Load TOML config
    import tomllib

    with open(config_path, "rb") as f:
        config_dict = tomllib.load(f)

    # Apply overrides
    config_dict["prompt_learning"]["localapi_url"] = localapi_url

    if args.budget:
        if "gepa" not in config_dict["prompt_learning"]:
            config_dict["prompt_learning"]["gepa"] = {}
        if "rollout" not in config_dict["prompt_learning"]["gepa"]:
            config_dict["prompt_learning"]["gepa"]["rollout"] = {}
        config_dict["prompt_learning"]["gepa"]["rollout"]["budget"] = args.budget

    if args.generations:
        if "gepa" not in config_dict["prompt_learning"]:
            config_dict["prompt_learning"]["gepa"] = {}
        if "population" not in config_dict["prompt_learning"]["gepa"]:
            config_dict["prompt_learning"]["gepa"]["population"] = {}
        config_dict["prompt_learning"]["gepa"]["population"]["num_generations"] = args.generations

    print("\nSubmitting GEPA job...")
    print(f"  Localapi URL: {localapi_url}")
    job = PromptLearningJob.from_dict(
        config_dict=config_dict,
        synth_user_key=SYNTH_USER_KEY,
    )
    print(f"  Localapi API key: {job.config.localapi_key[:20]}...")

    # Start localapi server
    run_server_background(app, port)
    wait_for_health("localhost", port, job.config.localapi_key)
    print(f"Localapi ready on port {port}")

    job_id = job.submit()
    print(f"Job ID: {job_id}")

    # Poll results
    print("\nPolling for results...")
    result = job.poll_until_complete(timeout=7200.0, interval=15.0, progress=True)

    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"Status: {result.status}")

    if result.failed:
        print(f"Job failed: {result.error}")
        if result.raw:
            print(f"\nRaw response keys: {list(result.raw.keys())}")
            # Print error and recent_events for debugging
            if result.raw.get("error"):
                print(f"Error details: {result.raw['error']}")
            if result.raw.get("recent_events"):
                print("\nRecent events:")
                for event in result.raw.get("recent_events", [])[-5:]:
                    print(f"  - {event}")
    else:
        if result.best_score is not None:
            print(f"Best score: {result.best_score:.4f}")

        if result.best_prompt:
            if isinstance(result.best_prompt, str):
                print("\nBest prompt (first 500 chars):")
                print(result.best_prompt[:500] + ("..." if len(result.best_prompt) > 500 else ""))
            elif isinstance(result.best_prompt, dict):
                print("\nBest prompt:")
                messages = result.best_prompt.get("messages", [])
                for msg in messages:
                    role = msg.get("role", "unknown")
                    pattern = msg.get("pattern", "")
                    print(f"\n[{role.upper()}]")
                    print(pattern[:500] + ("..." if len(pattern) > 500 else ""))

        if result.raw:
            best_train_score = result.raw.get("best_train_score")
            best_validation_score = result.raw.get("best_validation_score")
            if best_train_score is not None:
                print(f"\nBest train score: {best_train_score:.4f}")
            if best_validation_score is not None:
                print(f"Best validation score: {best_validation_score:.4f}")

    print("\nDone!")


if __name__ == "__main__":
    asyncio.run(main())
