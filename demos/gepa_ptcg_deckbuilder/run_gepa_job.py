#!/usr/bin/env python3
"""
Run GEPA prompt learning job for Pokemon TCG Deck Building.

Optimizes prompts for building Pokemon TCG decks that satisfy constraints
and win battles using genetic algorithm (GEPA).
"""

import argparse
import asyncio
import time
from pathlib import Path

parser = argparse.ArgumentParser(
    description="Run GEPA prompt learning for Pokemon TCG Deck Builder"
)
parser.add_argument("--port", type=int, default=8018)
parser.add_argument(
    "--config", type=str, default="gepa_deckbuilder.toml", help="Path to GEPA config file"
)
parser.add_argument("--budget", type=int, help="Override rollout budget")
parser.add_argument("--generations", type=int, help="Override number of generations")
args = parser.parse_args()

import httpx  # noqa: E402
from localapi_deckbuilder import DEFAULT_SYSTEM_PROMPT, INSTANCE_IDS, app  # noqa: E402
from synth_ai.core.urls import synth_base_url, synth_health_url  # noqa: E402
from synth_ai.sdk.api.train.prompt_learning import PromptLearningJob  # noqa: E402
from synth_ai.sdk.auth import get_or_mint_synth_api_key  # noqa: E402
from synth_ai.sdk.task import run_server_background  # noqa: E402
from synth_ai.sdk.tunnels import PortConflictBehavior, acquire_port  # noqa: E402

SYNTH_USER_KEY = get_or_mint_synth_api_key()


def wait_for_health(host: str, port: int, api_key: str, timeout: float = 30.0) -> None:
    """Wait for localapi health check."""
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
    print("POKEMON TCG DECK BUILDER - GEPA PROMPT LEARNING")
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

    print(f"\nChallenges: {INSTANCE_IDS}")

    # Load config
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"Config file not found: {config_path}")
        print("Creating GEPA job from dict instead...")

        # Create config dict programmatically
        config_dict = {
            "prompt_learning": {
                "algorithm": "gepa",
                "localapi_url": localapi_url,
                "localapi_id": "ptcg_deckbuilder",
                "policy": {
                    "model": "gpt-4.1-mini",
                    "provider": "openai",
                    "temperature": 0.7,
                    "max_completion_tokens": 4096,
                },
                "initial_prompt": {
                    "id": "baseline_deckbuilder",
                    "name": "Baseline Deckbuilder Prompt",
                    "messages": [
                        {
                            "role": "system",
                            "pattern": DEFAULT_SYSTEM_PROMPT,
                            "order": 0,
                        },
                        {
                            "role": "user",
                            "pattern": "{user_prompt}",
                            "order": 1,
                        },
                    ],
                    "wildcards": {
                        "user_prompt": "REQUIRED",
                    },
                },
                "gepa": {
                    "env_name": "deckbuilder",
                    "proposer_type": "dspy",
                    "proposer_effort": "LOW",
                    "proposer_output_tokens": "FAST",
                    "rollout": {
                        "budget": args.budget or 50,
                        "max_concurrent": 5,
                        "minibatch_size": 10,
                    },
                    "evaluation": {
                        "seeds": [0, 1, 2, 3, 4],
                        "validation_seeds": [5, 6, 7, 8, 9],
                        "validation_top_k": 3,
                    },
                    "population": {
                        "initial_size": 10,
                        "num_generations": args.generations or 3,
                        "children_per_generation": 8,
                        "crossover_rate": 0.5,
                        "patience_generations": 2,
                    },
                    "mutation": {
                        "rate": 0.3,
                    },
                    "archive": {
                        "size": 30,
                        "pareto_set_size": 20,
                    },
                },
            }
        }

        print("\nSubmitting GEPA job (from dict)...")
        job = PromptLearningJob.from_dict(
            config_dict=config_dict,
            synth_user_key=SYNTH_USER_KEY,
        )

        # Start localapi server
        run_server_background(app, port)
        wait_for_health("localhost", port, job.localapi_key)
        print(f"Localapi ready on port {port}")
    else:
        print(f"\nLoading config from: {config_path}")

        # Apply overrides if provided
        overrides = {}
        if args.budget:
            overrides["prompt_learning"] = {"gepa": {"rollout": {"budget": args.budget}}}
        if args.generations:
            if "prompt_learning" not in overrides:
                overrides["prompt_learning"] = {}
            if "gepa" not in overrides["prompt_learning"]:
                overrides["prompt_learning"]["gepa"] = {}
            if "population" not in overrides["prompt_learning"]["gepa"]:
                overrides["prompt_learning"]["gepa"]["population"] = {}
            overrides["prompt_learning"]["gepa"]["population"]["num_generations"] = args.generations

        # Update localapi_url in overrides to use actual port
        if "prompt_learning" not in overrides:
            overrides["prompt_learning"] = {}
        overrides["prompt_learning"]["localapi_url"] = localapi_url

        print("\nSubmitting GEPA job (from config file)...")
        job = PromptLearningJob.from_config(
            config_path=config_path,
            synth_user_key=SYNTH_USER_KEY,
            overrides=overrides,
        )

        # Start localapi server
        run_server_background(app, port)
        wait_for_health("localhost", port, job.localapi_key)
        print(f"Localapi ready on port {port}")

    job_id = job.submit()
    print(f"Job ID: {job_id}")

    # Poll results
    print("\nPolling for results...")
    result = job.poll_until_complete(timeout=3600.0, interval=10.0, progress=True)

    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"Status: {result.status}")

    if result.failed:
        print(f"Job failed: {result.error}")
        if result.raw:
            print(f"\nRaw response keys: {list(result.raw.keys())}")
    else:
        if result.best_score is not None:
            print(f"Best score: {result.best_score:.4f}")

        if result.best_prompt:
            # best_prompt can be a string or dict
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

        # Check raw response for additional scores
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
