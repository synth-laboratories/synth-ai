#!/usr/bin/env python3
"""
Run eval for Pokemon TCG Deck Building.

Tests LLM deck building capabilities with constraint satisfaction scoring.
"""

import argparse
import asyncio
import time

parser = argparse.ArgumentParser(description="Run Pokemon TCG Deck Builder eval")
parser.add_argument("--port", type=int, default=8018)
parser.add_argument("--model", type=str, default="gpt-4.1-mini")
parser.add_argument("--num-seeds", type=int, default=5, help="Number of seeds to evaluate")
args = parser.parse_args()

import httpx  # noqa: E402
from localapi_deckbuilder import DEFAULT_SYSTEM_PROMPT, INSTANCE_IDS, app  # noqa: E402
from synth_ai.core.urls import synth_base_url, synth_health_url  # noqa: E402
from synth_ai.sdk.api.eval import EvalJob, EvalJobConfig  # noqa: E402
from synth_ai.sdk.auth import get_or_mint_synth_user_key  # noqa: E402
from synth_ai.sdk.task import run_server_background  # noqa: E402
from synth_ai.sdk.tunnels import PortConflictBehavior, acquire_port  # noqa: E402

SYNTH_USER_KEY = get_or_mint_synth_user_key()


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
    print("POKEMON TCG DECK BUILDER EVAL")
    print("=" * 60)

    # Backend setup
    print(f"Backend: {synth_base_url()}")

    # Check backend
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

    print(f"\nModel: {args.model}")
    print(f"Seeds: {args.num_seeds}")
    print(f"Challenges: {INSTANCE_IDS}")

    # Run eval
    seeds = list(range(args.num_seeds))
    print(f"\nSubmitting eval with seeds: {seeds}")

    config = EvalJobConfig(
        localapi_url=localapi_url,
        synth_user_key=SYNTH_USER_KEY,
        env_name="deckbuilder",
        seeds=seeds,
        policy_config={
            "model": args.model,
            "system_prompt": DEFAULT_SYSTEM_PROMPT,
        },
        env_config={},
        concurrency=2,
    )

    # Start localapi server (after config creation to get localapi_key)
    run_server_background(app, port)
    wait_for_health("localhost", port, config.localapi_key)
    print(f"Localapi ready on port {port}")

    job = EvalJob(config)
    job_id = job.submit()
    print(f"Job ID: {job_id}")

    # Poll results
    result = job.poll_until_complete(timeout=600.0, interval=5.0, progress=True)

    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"Status: {result.status}")
    mean_reward = result.mean_reward
    if mean_reward is None:
        print("Mean reward: n/a")
    else:
        print(f"Mean reward: {mean_reward:.2%}")

    if result.seed_results:
        print(f"\nPer-seed results ({len(result.seed_results)}):")
        for i, sr in enumerate(result.seed_results):
            reward = sr.get("outcome_reward", 0)
            details = sr.get("details", {})
            instance_id = details.get("instance_id", "?")
            deck_size = details.get("deck_size", 0)
            constraint_results = details.get("constraint_results", [])

            satisfied = sum(1 for c in constraint_results if c.get("satisfied"))
            total = len(constraint_results)

            print(f"\n  Seed {i}: {instance_id}")
            print(f"    Score: {reward:.2f}")
            print(f"    Deck size: {deck_size}")
            print(f"    Constraints: {satisfied}/{total} satisfied")

            # Show constraint details
            for cr in constraint_results:
                status = "PASS" if cr.get("satisfied") else "FAIL"
                print(f"      [{status}] {cr.get('type')}: {cr.get('explanation')}")

            # Check if deck is valid
            error = details.get("error")
            if error:
                print(f"    Error: {error}")

            # Show deck sample if valid
            deck = details.get("deck", [])
            if deck and len(deck) > 0:
                print(f"    Deck sample: {deck[:5]}...")

    print("\nDone!")


if __name__ == "__main__":
    asyncio.run(main())
