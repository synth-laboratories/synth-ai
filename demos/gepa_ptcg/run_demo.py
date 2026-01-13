#!/usr/bin/env python3
"""
Run GEPA optimization for Pokemon TCG game playing.

This script:
1. Starts the local task app
2. Runs eval to test the LLM agent against AI v4
"""

import argparse
import asyncio
import os

# Parse args early
parser = argparse.ArgumentParser(description="Run GEPA for Pokemon TCG")
parser.add_argument("--local", action="store_true", help="Use localhost:8000 backend")
parser.add_argument("--local-host", type=str, default="localhost")
parser.add_argument("--port", type=int, default=8017, help="Port for task app")
parser.add_argument("--model", type=str, default="gpt-4.1-mini", help="Model to use")
parser.add_argument("--num-games", type=int, default=3, help="Number of games to run")
args = parser.parse_args()

LOCAL_MODE = args.local
LOCAL_HOST = args.local_host
PORT = args.port
MODEL = args.model
NUM_GAMES = args.num_games

import time

import httpx  # noqa: E402
from localapi_ptcg import DEFAULT_SYSTEM_PROMPT, INSTANCE_IDS, app  # noqa: E402
from synth_ai.core.env import PROD_BASE_URL, mint_demo_api_key  # noqa: E402
from synth_ai.sdk.api.eval import EvalJob, EvalJobConfig  # noqa: E402
from synth_ai.sdk.localapi.auth import ensure_localapi_auth  # noqa: E402
from synth_ai.sdk.task import run_server_background  # noqa: E402
from synth_ai.sdk.tunnels import PortConflictBehavior, acquire_port  # noqa: E402


def wait_for_health_check_sync(host: str, port: int, api_key: str, timeout: float = 30.0) -> None:
    """Wait for task app to be ready."""
    health_url = f"http://{host}:{port}/health"
    headers = {"X-API-Key": api_key} if api_key else {}
    start = time.time()
    while time.time() - start < timeout:
        try:
            response = httpx.get(health_url, headers=headers, timeout=5.0)
            if response.status_code in (200, 400):
                return
        except (httpx.RequestError, httpx.TimeoutException):
            pass
        time.sleep(0.5)
    raise RuntimeError(f"Health check failed: {health_url}")


async def main():
    """Main entry point."""
    print("=" * 60)
    print("POKEMON TCG GEPA DEMO")
    print("=" * 60)

    if LOCAL_MODE:
        synth_api_base = f"http://{LOCAL_HOST}:8000"
        print(f"\nLOCAL MODE - using {synth_api_base} backend")
    else:
        synth_api_base = PROD_BASE_URL
        print(f"\nPROD MODE - using {synth_api_base}")

    # Check backend health
    async with httpx.AsyncClient() as client:
        try:
            resp = await client.get(f"{synth_api_base}/health", timeout=10)
            print(f"Backend health: {resp.status_code}")
        except Exception as e:
            print(f"Backend health check failed: {e}")
            return

    # Get API key
    api_key = os.getenv("SYNTH_API_KEY")
    if not api_key:
        print("No SYNTH_API_KEY, minting demo key...")
        api_key = mint_demo_api_key(backend_url=synth_api_base)
        print(f"API Key: {api_key[:20]}...")

    env_key = ensure_localapi_auth(backend_base=synth_api_base, synth_api_key=api_key)
    print(f"Env key: {env_key[:15]}...")

    # Acquire port and start task app
    port = acquire_port(PORT, on_conflict=PortConflictBehavior.FIND_NEW)
    if port != PORT:
        print(f"Port {PORT} in use, using {port} instead")

    run_server_background(app, port)
    wait_for_health_check_sync(LOCAL_HOST, port, env_key, timeout=30.0)
    print(f"Task app ready on port {port}")

    task_app_url = f"http://{LOCAL_HOST}:{port}"
    print(f"Task app URL: {task_app_url}")

    print("\n" + "=" * 60)
    print(f"Model: {MODEL}")
    print(f"Number of games: {NUM_GAMES}")
    print(f"Available instances: {len(INSTANCE_IDS)}")
    print("=" * 60)

    # Generate seeds for evaluation
    seeds = list(range(NUM_GAMES))
    print(f"\nSubmitting eval job with seeds: {seeds}")
    print(f"Instance IDs: {[INSTANCE_IDS[s % len(INSTANCE_IDS)] for s in seeds]}")

    config = EvalJobConfig(
        local_api_url=task_app_url,
        backend_url=synth_api_base,
        api_key=api_key,
        env_name="ptcg",
        seeds=seeds,
        policy_config={
            "model": MODEL,
            "system_prompt": DEFAULT_SYSTEM_PROMPT,
        },
        env_config={},
        concurrency=1,  # Run one at a time for now
    )

    job = EvalJob(config)

    try:
        job_id = job.submit()
        print(f"Job submitted: {job_id}")

        # Poll for results
        result = job.poll_until_complete(
            timeout=600.0,
            interval=5.0,
            progress=True,
        )

        print("\n" + "=" * 60)
        print("EVAL RESULT")
        print("=" * 60)
        print(f"Status: {result.status}")
        if result.mean_reward is None:
            print("Mean reward (win rate): n/a")
        else:
            print(f"Mean reward (win rate): {result.mean_reward:.2%}")
        print(f"Error: {result.error}")

        if result.seed_results:
            print(f"\nGame results ({len(result.seed_results)}):")
            for sr in result.seed_results:
                metadata = sr.get("metadata", {}) or sr.get("rollout_metadata", {}) or {}
                details = sr.get("details", {}) or {}
                instance_id = metadata.get("instance_id") or details.get("instance_id") or "?"
                winner = metadata.get("winner") or details.get("winner") or "?"
                reward = sr.get("outcome_reward", 0)
                print(f"  - {instance_id}: winner={winner}, reward={reward:.2f}")

    except Exception as e:
        print(f"\nEval job failed: {e}")
        import traceback

        traceback.print_exc()

    print("\nDone!")


if __name__ == "__main__":
    asyncio.run(main())
