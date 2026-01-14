#!/usr/bin/env python3
"""
EngineBench Eval Job Runner

Runs an evaluation job for Pokemon TCG card implementations using synth-ai.
This evaluates coding agents on their ability to implement game mechanics in Rust.

Usage:
    uv run python demos/engine_bench/run_eval.py --local
    uv run python demos/engine_bench/run_eval.py --local --seeds 5 --model gpt-4.1-mini
"""

import argparse
import asyncio
import os
import time

import httpx

# Import the task app
from localapi_engine_bench import INSTANCE_IDS, app
from synth_ai.core.env import mint_demo_api_key
from synth_ai.core.urls import BACKEND_URL_BASE
from synth_ai.sdk.api.eval import EvalJob, EvalJobConfig
from synth_ai.sdk.localapi.auth import ensure_localapi_auth
from synth_ai.sdk.task import run_server_background
from synth_ai.sdk.tunnels import PortConflictBehavior, acquire_port


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
    parser = argparse.ArgumentParser(description="Run EngineBench eval job")
    parser.add_argument("--local", action="store_true", help="Use localhost:8000 backend")
    parser.add_argument("--local-host", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=8017, help="Port for task app")
    parser.add_argument("--seeds", type=int, default=1, help="Number of seeds to eval")
    parser.add_argument(
        "--model", type=str, default="gpt-5.2", help="Model to use for coding agent"
    )
    parser.add_argument(
        "--agent",
        type=str,
        default="opencode",
        choices=["opencode", "codex"],
        help="Agent runner to use (opencode or codex)",
    )
    parser.add_argument("--timeout", type=int, default=300, help="Agent timeout in seconds")
    parser.add_argument(
        "--split",
        type=str,
        default="df",
        choices=["df", "hp"],
        help="Dataset split (df=Dragon Frontiers, hp=Holon Phantoms)",
    )
    parser.add_argument(
        "--verifier",
        type=str,
        default=None,
        help="Verifier graph ID (e.g., 'zero_shot_verifier_rubric_single'). If set, enables fused rewards.",
    )
    parser.add_argument(
        "--verifier-model",
        type=str,
        default="gpt-4o-mini",
        help="Model to use for verifier (default: gpt-4o-mini)",
    )
    parser.add_argument(
        "--weight-env",
        type=float,
        default=0.6,
        help="Weight for environment (unit test) reward in fused mode (default: 0.6)",
    )
    parser.add_argument(
        "--weight-outcome",
        type=float,
        default=0.4,
        help="Weight for verifier outcome reward in fused mode (default: 0.4)",
    )
    args = parser.parse_args()

    local_mode = args.local
    local_host = args.local_host
    port = args.port
    num_seeds = args.seeds
    model = args.model
    agent = args.agent
    timeout = args.timeout
    split = args.split
    verifier_graph_id = args.verifier
    verifier_model = args.verifier_model
    weight_env = args.weight_env
    weight_outcome = args.weight_outcome

    synth_api_base = "http://localhost:8000" if local_mode else BACKEND_URL_BASE
    if local_mode:
        print("=" * 60)
        print("LOCAL MODE - using localhost:8000 backend")
        print("=" * 60)
    else:
        print(f"PROD MODE - using {synth_api_base}")

    r = httpx.get(f"{synth_api_base}/health", timeout=30)
    print(f"Backend health: {r.status_code}")

    api_key = os.environ.get("SYNTH_API_KEY", "")
    if not api_key:
        print("No SYNTH_API_KEY, minting demo key...")
        api_key = mint_demo_api_key(backend_url=synth_api_base)
    os.environ["SYNTH_API_KEY"] = api_key
    print(f"API Key: {api_key[:20]}...")

    environment_api_key = ensure_localapi_auth(
        backend_base=synth_api_base,
        synth_api_key=api_key,
    )
    print(f"Env key: {environment_api_key[:12]}...")

    print("\n" + "=" * 60)
    print("STARTING ENGINEBENCH EVAL")
    print("=" * 60)
    print(f"Model: {model}")
    print(f"Agent: {agent}")
    print(f"Split: {split}")
    print(f"Seeds: {num_seeds}")
    print(f"Timeout: {timeout}s")
    print(f"Available instances: {len(INSTANCE_IDS)}")
    if verifier_graph_id:
        print(f"Verifier: {verifier_graph_id} (model={verifier_model})")
        print(f"  Reward fusion: env={weight_env}, outcome={weight_outcome}")
    else:
        print("Verifier: disabled (unit test reward only)")

    # Filter instances by split
    split_instances = [i for i in INSTANCE_IDS if i.startswith(f"{split}-")]
    print(f"Instances in {split} split: {len(split_instances)}")

    # Start task app
    port = acquire_port(port, on_conflict=PortConflictBehavior.FIND_NEW)
    if port != args.port:
        print(f"Port {args.port} in use, using {port} instead")

    run_server_background(app, port)
    wait_for_health_check_sync("localhost", port, environment_api_key, timeout=30.0)
    print(f"Task app ready on port {port}")

    task_app_url = f"http://{local_host}:{port}"
    print(f"Task app URL: {task_app_url}")

    # Create eval job
    # Use seeds that map to instances in the selected split
    # Use seed 22 (df-023-tropius) - "Simple attacks" (trivial, no Poke-Body/Poke-Power)
    # This is the easiest possible case - just basic attacks
    base_seed = None
    first_df_seed = None

    # First, try to find tropius specifically
    for i, inst_id in enumerate(INSTANCE_IDS):
        if inst_id.startswith(f"{split}-"):
            if first_df_seed is None:
                first_df_seed = i  # Remember first df- instance as fallback
            if inst_id == "df-023-tropius":
                base_seed = i
                break

    # Fall back to first df- instance if tropius not found
    if base_seed is None:
        base_seed = first_df_seed if first_df_seed is not None else 0

    seeds = list(range(base_seed, base_seed + num_seeds))
    print(f"\nSubmitting eval job with seeds: {seeds}")
    print(f"Instance IDs: {[INSTANCE_IDS[s % len(INSTANCE_IDS)] for s in seeds]}")

    # Build verifier config if enabled
    verifier_config = None
    if verifier_graph_id:
        verifier_config = {
            "enabled": True,  # REQUIRED: must be True for verifier to run
            "verifier_graph_id": verifier_graph_id,
            "reward_source": "fused",
            "backend_base": synth_api_base,  # Use same backend for verifier
            "backend_model": verifier_model,
            "backend_outcome_enabled": True,
            "backend_event_enabled": True,
            "weight_env": weight_env,
            "weight_event": 0.0,  # We're not scoring events separately
            "weight_outcome": weight_outcome,
        }

    config = EvalJobConfig(
        local_api_url=task_app_url,
        backend_url=synth_api_base,
        api_key=api_key,
        env_name="engine_bench",
        seeds=seeds,
        policy_config={
            "model": model,
            "timeout": timeout,
            "agent": agent,
        },
        env_config={
            "split": split,
        },
        verifier_config=verifier_config,
        concurrency=1,  # Run one at a time (coding agent tasks are heavy)
    )

    job = EvalJob(config)

    try:
        job_id = job.submit()
        print(f"Job submitted: {job_id}")

        # Longer timeout for coding tasks
        result = job.poll_until_complete(
            timeout=timeout * num_seeds + 120.0,  # account for overhead
            interval=5.0,
            progress=True,
        )

        print("\n" + "=" * 60)
        print("EVAL RESULT")
        print("=" * 60)
        print(f"Status: {result.status}")
        print(f"Mean reward: {result.mean_reward}")
        print(f"Error: {result.error}")

        if result.seed_results:
            print(f"\nSeed results ({len(result.seed_results)}):")
            for sr in result.seed_results:
                # Seed results from backend
                seed = sr.get("seed", "?")
                # Use 'score' field which is the fused reward (if verifier enabled)
                # Falls back to outcome_reward if score not present
                fused_reward = sr.get("score") or sr.get("outcome_reward", 0)
                outcome_reward = sr.get("outcome_reward", 0)  # env/task app reward
                verifier_reward = sr.get("verifier_score")
                error = sr.get("error")

                status = "✅" if fused_reward >= 0.8 else "⚠️" if fused_reward > 0.3 else "❌"

                # Show fused reward with breakdown
                if verifier_reward is not None:
                    # Verifier was used - show breakdown
                    print(
                        f"  {status} seed={seed}: fused={fused_reward:.2f} (env={outcome_reward:.2f}, verifier={verifier_reward:.2f})"
                    )
                else:
                    # No verifier - just show env reward
                    print(f"  {status} seed={seed}: reward={outcome_reward:.2f}")

                if error:
                    print(f"    error: {error}")

        if result.failed:
            try:
                failed_results = job.get_results()
                results_items = failed_results.get("results", [])
                if isinstance(results_items, dict):
                    results_items = results_items.get("items", [])
                if results_items:
                    print("\nFailed result details:")
                    for item in results_items:
                        seed = item.get("seed")
                        error = item.get("error")
                        details = item.get("details", {}) if isinstance(item, dict) else {}
                        agent_stderr_tail = details.get("agent_stderr_tail", "")
                        print(f"  - seed={seed} error={error}")
                        if agent_stderr_tail:
                            print(f"    stderr_tail={agent_stderr_tail[-400:]}")
            except Exception as e:
                print(f"Failed to fetch results for failed job: {e}")

    except Exception as e:
        print(f"\nEval job failed: {e}")
        import traceback

        traceback.print_exc()

    print("\nDone!")


if __name__ == "__main__":
    asyncio.run(main())
