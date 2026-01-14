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

# Parse args early
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
# Verifier configuration
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

LOCAL_MODE = args.local
LOCAL_HOST = args.local_host
PORT = args.port
NUM_SEEDS = args.seeds
MODEL = args.model
AGENT = args.agent
TIMEOUT = args.timeout
SPLIT = args.split
VERIFIER_GRAPH_ID = args.verifier
VERIFIER_MODEL = args.verifier_model
WEIGHT_ENV = args.weight_env
WEIGHT_OUTCOME = args.weight_outcome

import httpx  # noqa: E402

# Import the task app
from localapi_engine_bench import INSTANCE_IDS, app  # noqa: E402
from synth_ai.core.env import PROD_BASE_URL, mint_demo_api_key  # noqa: E402
from synth_ai.sdk.api.eval import EvalJob, EvalJobConfig  # noqa: E402
from synth_ai.sdk.localapi.auth import ensure_localapi_auth  # noqa: E402
from synth_ai.sdk.task import run_server_background  # noqa: E402
from synth_ai.sdk.tunnels import PortConflictBehavior, acquire_port  # noqa: E402

# Backend config
if LOCAL_MODE:
    SYNTH_API_BASE = "http://localhost:8000"
    print("=" * 60)
    print("LOCAL MODE - using localhost:8000 backend")
    print("=" * 60)
else:
    SYNTH_API_BASE = PROD_BASE_URL
    print(f"PROD MODE - using {SYNTH_API_BASE}")

os.environ["SYNTH_API_BASE"] = SYNTH_API_BASE

# Check backend health
r = httpx.get(f"{SYNTH_API_BASE}/health", timeout=30)
print(f"Backend health: {r.status_code}")

# API Key
API_KEY = os.environ.get("SYNTH_API_KEY", "")
if not API_KEY:
    print("No SYNTH_API_KEY, minting demo key...")
    API_KEY = mint_demo_api_key(backend_url=SYNTH_API_BASE)
os.environ["SYNTH_API_KEY"] = API_KEY
print(f"API Key: {API_KEY[:20]}...")

# Environment Key
ENVIRONMENT_API_KEY = ensure_localapi_auth(
    backend_base=SYNTH_API_BASE,
    synth_api_key=API_KEY,
)
print(f"Env key: {ENVIRONMENT_API_KEY[:12]}...")


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
    print("\n" + "=" * 60)
    print("STARTING ENGINEBENCH EVAL")
    print("=" * 60)
    print(f"Model: {MODEL}")
    print(f"Agent: {AGENT}")
    print(f"Split: {SPLIT}")
    print(f"Seeds: {NUM_SEEDS}")
    print(f"Timeout: {TIMEOUT}s")
    print(f"Available instances: {len(INSTANCE_IDS)}")
    if VERIFIER_GRAPH_ID:
        print(f"Verifier: {VERIFIER_GRAPH_ID} (model={VERIFIER_MODEL})")
        print(f"  Reward fusion: env={WEIGHT_ENV}, outcome={WEIGHT_OUTCOME}")
    else:
        print("Verifier: disabled (unit test reward only)")

    # Filter instances by split
    split_instances = [i for i in INSTANCE_IDS if i.startswith(f"{SPLIT}-")]
    print(f"Instances in {SPLIT} split: {len(split_instances)}")

    # Start task app
    port = acquire_port(PORT, on_conflict=PortConflictBehavior.FIND_NEW)
    if port != PORT:
        print(f"Port {PORT} in use, using {port} instead")

    run_server_background(app, port)
    wait_for_health_check_sync("localhost", port, ENVIRONMENT_API_KEY, timeout=30.0)
    print(f"Task app ready on port {port}")

    task_app_url = f"http://{LOCAL_HOST}:{port}"
    print(f"Task app URL: {task_app_url}")

    # Create eval job
    # Use seeds that map to instances in the selected split
    # Use seed 22 (df-023-tropius) - "Simple attacks" (trivial, no Poke-Body/Poke-Power)
    # This is the easiest possible case - just basic attacks
    base_seed = None
    first_df_seed = None
    
    # First, try to find tropius specifically
    for i, inst_id in enumerate(INSTANCE_IDS):
        if inst_id.startswith(f"{SPLIT}-"):
            if first_df_seed is None:
                first_df_seed = i  # Remember first df- instance as fallback
            if inst_id == "df-023-tropius":
                base_seed = i
                break
    
    # Fall back to first df- instance if tropius not found
    if base_seed is None:
        base_seed = first_df_seed if first_df_seed is not None else 0

    seeds = list(range(base_seed, base_seed + NUM_SEEDS))
    print(f"\nSubmitting eval job with seeds: {seeds}")
    print(f"Instance IDs: {[INSTANCE_IDS[s % len(INSTANCE_IDS)] for s in seeds]}")

    # Build verifier config if enabled
    verifier_config = None
    if VERIFIER_GRAPH_ID:
        verifier_config = {
            "enabled": True,  # REQUIRED: must be True for verifier to run
            "verifier_graph_id": VERIFIER_GRAPH_ID,
            "reward_source": "fused",
            "backend_base": SYNTH_API_BASE,  # Use same backend for verifier
            "backend_model": VERIFIER_MODEL,
            "backend_outcome_enabled": True,
            "backend_event_enabled": True,
            "weight_env": WEIGHT_ENV,
            "weight_event": 0.0,  # We're not scoring events separately
            "weight_outcome": WEIGHT_OUTCOME,
        }

    config = EvalJobConfig(
        local_api_url=task_app_url,
        backend_url=SYNTH_API_BASE,
        api_key=API_KEY,
        env_name="engine_bench",
        seeds=seeds,
        policy_config={
            "model": MODEL,
            "timeout": TIMEOUT,
            "agent": AGENT,
        },
        env_config={
            "split": SPLIT,
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
            timeout=TIMEOUT * NUM_SEEDS + 120.0,  # account for overhead
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
                    print(f"  {status} seed={seed}: fused={fused_reward:.2f} (env={outcome_reward:.2f}, verifier={verifier_reward:.2f})")
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
