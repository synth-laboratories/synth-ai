#!/usr/bin/env python3
"""
Run GEPA optimization for Pokemon TCG game playing.

This script:
1. Starts the local task app
2. Runs eval to test the LLM agent against AI v4
"""

import argparse
import asyncio
import json
import os
from datetime import datetime
from pathlib import Path

# Parse args early
parser = argparse.ArgumentParser(description="Run GEPA for Pokemon TCG")
parser.add_argument("--port", type=int, default=8017, help="Port for task app")
parser.add_argument("--model", type=str, default="gpt-4.1-mini", help="Model to use")
parser.add_argument("--num-games", type=int, default=3, help="Number of games to run")
parser.add_argument(
    "--enable-verifier",
    action="store_true",
    help="Enable backend verifier evaluation and fuse verifier_reward with local_api_reward",
)
parser.add_argument("--react", action="store_true", help="Use the PTCG ReAct system prompt")
parser.add_argument(
    "--out-dir",
    type=str,
    default="",
    help="If set, write local artifacts here (rollouts JSONL, backend traces, job results)",
)
parser.add_argument(
    "--download-traces",
    action="store_true",
    help="Download backend traces into --out-dir/<run>/backend_traces (requires --out-dir)",
)
args = parser.parse_args()

PORT = args.port
MODEL = args.model
NUM_GAMES = args.num_games
ENABLE_VERIFIER = bool(args.enable_verifier)
USE_REACT = args.react
OUT_DIR_RAW = args.out_dir
DOWNLOAD_TRACES = args.download_traces

import time  # noqa: E402

import httpx  # noqa: E402
from localapi_ptcg import (  # noqa: E402
    DEFAULT_SYSTEM_PROMPT,
    INSTANCE_IDS,
    PTCG_REACT_SYSTEM_PROMPT,
    app,
)
from synth_ai.core.urls import synth_base_url, synth_health_url  # noqa: E402
from synth_ai.sdk.api.eval import EvalJob, EvalJobConfig  # noqa: E402
from synth_ai.sdk.auth import get_or_mint_synth_api_key  # noqa: E402
from synth_ai.sdk.task import run_server_background  # noqa: E402
from synth_ai.sdk.tunnels import PortConflictBehavior, acquire_port  # noqa: E402

SYNTH_USER_KEY = get_or_mint_synth_api_key()


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

    # Check backend health
    async with httpx.AsyncClient() as client:
        try:
            resp = await client.get(synth_health_url(), timeout=10)
            print(f"Synth health: {resp.status_code}")
        except Exception as e:
            print(f"Synth health check failed: {e}")
            return

    print(f"API Key: {SYNTH_USER_KEY[:20]}...")

    run_dir: Path | None = None
    if OUT_DIR_RAW:
        run_stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        run_dir = Path(OUT_DIR_RAW).expanduser().resolve() / f"ptcg_eval_{run_stamp}"
        run_dir.mkdir(parents=True, exist_ok=True)
        os.environ["PTCG_TRACE_DIR"] = str(run_dir)
        print(f"Writing local artifacts to: {run_dir}")

    # Acquire port and start task app
    port = acquire_port(PORT, on_conflict=PortConflictBehavior.FIND_NEW)
    if port != PORT:
        print(f"Port {PORT} in use, using {port} instead")

    localapi_url = f"http://localhost:{port}"
    seeds = list(range(NUM_GAMES))

    # Create config (auto-provisions localapi_key)
    config = EvalJobConfig(
        localapi_url=localapi_url,
        synth_user_key=SYNTH_USER_KEY,
        env_name="ptcg",
        seeds=seeds,
        policy_config={
            "model": MODEL,
            "system_prompt": PTCG_REACT_SYSTEM_PROMPT if USE_REACT else DEFAULT_SYSTEM_PROMPT,
        },
        env_config={},
        verifier_config=(
            {
                # Use backend verifier to evaluate "intangible" gameplay quality, then fuse it with the
                # task app's local_api_reward (outcome_reward).
                "enabled": True,
                "reward_source": "fused",
                "backend_base": synth_base_url(),
                "backend_provider": "openai",
                "backend_model": MODEL,
                # Zero-shot rubric verifier graph.
                "verifier_graph_id": "zero_shot_verifier_rubric_single",
                # Use both event-level and outcome-level rubric components when available.
                "backend_outcome_enabled": True,
                "backend_event_enabled": True,
                # Verifier execution controls
                "concurrency": 1,
                "timeout": 240.0,
                # Fusion weights
                "weight_env": 0.7,
                "weight_event": 0.15,
                "weight_outcome": 0.15,
            }
            if ENABLE_VERIFIER
            else None
        ),
        concurrency=1,  # Run one at a time for now
    )

    run_server_background(app, port)
    wait_for_health_check_sync("localhost", port, config.localapi_key, timeout=30.0)
    print(f"Localapi ready on port {port}")
    print(f"Localapi URL: {localapi_url}")

    if run_dir is not None:
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.get(
                    f"{localapi_url}/info",
                    headers={"X-API-Key": config.localapi_key, "Content-Type": "application/json"},
                )
                resp.raise_for_status()
                (run_dir / "localapi_info.json").write_text(
                    json.dumps(resp.json(), indent=2, default=str), encoding="utf-8"
                )
        except Exception as e:
            print(f"Warning: failed to fetch /info from localapi: {e}")

    print("\n" + "=" * 60)
    print(f"Model: {MODEL}")
    print(f"Prompt: {'ReAct' if USE_REACT else 'baseline'}")
    print(f"Number of games: {NUM_GAMES}")
    print(f"Available instances: {len(INSTANCE_IDS)}")
    print("=" * 60)

    print(f"\nSubmitting eval job with seeds: {seeds}")
    print(f"Instance IDs: {[INSTANCE_IDS[s % len(INSTANCE_IDS)] for s in seeds]}")

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

        if run_dir is not None:
            try:
                raw_results = job.get_results()
                (run_dir / "eval_job_results.json").write_text(
                    json.dumps(raw_results, indent=2, default=str), encoding="utf-8"
                )
                (run_dir / "eval_job_id.txt").write_text(str(job_id), encoding="utf-8")
            except Exception as e:
                print(f"Warning: failed to write eval job results: {e}")

            if DOWNLOAD_TRACES or OUT_DIR_RAW:
                try:
                    traces_dir = job.download_traces(run_dir / "backend_traces")
                    print(f"Downloaded backend traces to {traces_dir}")
                except Exception as e:
                    print(f"Warning: failed to download backend traces: {e}")

        print("\n" + "=" * 60)
        print("EVAL RESULT")
        print("=" * 60)
        print(f"Status: {result.status}")
        if result.mean_reward is None:
            print("Mean reward: n/a")
        else:
            if ENABLE_VERIFIER:
                print(f"Mean reward (fused): {result.mean_reward:.3f}")
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
                local_api_reward = (
                    sr.get("local_api_reward")
                    if sr.get("local_api_reward") is not None
                    else sr.get("outcome_reward", 0.0)
                )
                verifier_reward = sr.get("verifier_reward")
                fused_reward = sr.get("reward")
                if ENABLE_VERIFIER:
                    print(
                        f"  - {instance_id}: winner={winner}, local_api_reward={local_api_reward:.2f}, "
                        f"verifier_reward={verifier_reward}, fused_reward={fused_reward}"
                    )
                else:
                    print(f"  - {instance_id}: winner={winner}, reward={local_api_reward:.2f}")

    except Exception as e:
        print(f"\nEval job failed: {e}")
        import traceback

        traceback.print_exc()

    print("\nDone!")


if __name__ == "__main__":
    asyncio.run(main())
