#!/usr/bin/env python3
"""
EngineBench Eval Job Runner

Runs an evaluation job for Pokemon TCG card implementations using synth-ai.
This evaluates coding agents on their ability to implement game mechanics in Rust.

Usage:
    uv run python demos/engine_bench/run_eval.py
    uv run python demos/engine_bench/run_eval.py --seeds 5 --model gpt-4.1-mini

    # Daytona mode (run localapi in cloud sandbox)
    DAYTONA_SYNTH_USER_KEY=... uv run python demos/engine_bench/run_eval.py --daytona
"""

import argparse
import asyncio
import os
import sys
import time
from pathlib import Path

# Parse args early
parser = argparse.ArgumentParser(description="Run EngineBench eval job")
parser.add_argument(
    "--daytona",
    action="store_true",
    help="Run localapi in Daytona sandbox (requires DAYTONA_SYNTH_USER_KEY env var)",
)
parser.add_argument("--port", type=int, default=8017, help="Port for localapi")
parser.add_argument("--seeds", type=int, default=1, help="Number of seeds to eval")
parser.add_argument("--model", type=str, default="gpt-5.2", help="Model to use for coding agent")
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
parser.add_argument(
    "--localapi-url",
    type=str,
    default=None,
    help="Public URL for localapi (e.g., cloudflare tunnel URL). Required for prod mode.",
)
args = parser.parse_args()

USE_DAYTONA = args.daytona
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

# Validate Daytona requirements
if USE_DAYTONA and not os.environ.get("DAYTONA_SYNTH_USER_KEY"):
    print("Error: --daytona requires DAYTONA_SYNTH_USER_KEY environment variable")
    print("Set it with: export DAYTONA_SYNTH_USER_KEY=your_key")
    sys.exit(1)

import httpx  # noqa: E402

# Import the localapi
from localapi_engine_bench import INSTANCE_IDS, app  # noqa: E402
from synth_ai.core.urls import synth_base_url, synth_health_url  # noqa: E402
from synth_ai.sdk.api.eval import EvalJob, EvalJobConfig  # noqa: E402
from synth_ai.sdk.auth import get_or_mint_synth_api_key  # noqa: E402
from synth_ai.sdk.task import run_server_background  # noqa: E402
from synth_ai.sdk.tunnels import PortConflictBehavior, acquire_port  # noqa: E402

# Import Daytona helper if needed
if USE_DAYTONA:
    try:
        from daytona_helper import DaytonaLocalapiRunner  # noqa: E402
    except ImportError:
        print("Error: Daytona helper not found. Make sure daytona_helper.py exists.")
        sys.exit(1)

# Backend config
SYNTH_API_BASE = synth_base_url()
print(f"Backend: {SYNTH_API_BASE}")

# Check backend health
r = httpx.get(synth_health_url(), timeout=30)
print(f"Backend health: {r.status_code}")

# API Key
SYNTH_USER_KEY = get_or_mint_synth_api_key()
print(f"API Key: {SYNTH_USER_KEY[:20]}...")


def wait_for_health_check_sync(host: str, port: int, api_key: str, timeout: float = 30.0) -> None:
    """Wait for localapi to be ready."""
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
    print(f"Localapi mode: {'Daytona' if USE_DAYTONA else 'Local'}")
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

    # Start localapi (local or Daytona)
    daytona_runner = None
    localapi_url = None

    if USE_DAYTONA:
        print("\n" + "=" * 60)
        print("PROVISIONING DAYTONA SANDBOX")
        print("=" * 60)

        # Get localapi file path
        localapi_file = Path(__file__).parent / "localapi_engine_bench.py"
        if not localapi_file.exists():
            raise RuntimeError(f"Localapi file not found: {localapi_file}")

        # Create Daytona runner
        daytona_runner = DaytonaLocalapiRunner(
            api_key=os.environ.get("DAYTONA_SYNTH_USER_KEY"),
            api_url=os.environ.get("DAYTONA_API_URL"),
            target=os.environ.get("DAYTONA_TARGET"),
            localapi_port=8000,  # Daytona preview URLs use port 8000
        )

        # Provision sandbox
        await daytona_runner.provision()

        # Upload localapi
        await daytona_runner.upload_localapi(localapi_file)

        # Setup environment - Note: For Daytona, the task app will authenticate via SYNTH_USER_KEY
        env_vars = {
            "SYNTH_SYNTH_USER_KEY": SYNTH_USER_KEY,
            "SYNTH_BACKEND_URL": SYNTH_API_BASE,
            "OPENAI_SYNTH_USER_KEY": os.environ.get("OPENAI_SYNTH_USER_KEY", ""),
        }

        # Install dependencies
        install_commands = [
            "pip install --quiet fastapi uvicorn synth-ai",
        ]

        await daytona_runner.setup_environment(
            env_vars=env_vars,
            install_commands=install_commands,
        )

        # Start localapi
        localapi_url = await daytona_runner.start_localapi()
        print(f"Localapi running in Daytona: {localapi_url}")

    else:
        # Local localapi mode
        print("\n" + "=" * 60)
        print("STARTING LOCAL LOCALAPI")
        print("=" * 60)

        port = acquire_port(PORT, on_conflict=PortConflictBehavior.FIND_NEW)
        if port != PORT:
            print(f"Port {PORT} in use, using {port} instead")

        # Use tunnel URL if provided, otherwise use local URL
        if args.localapi_url:
            localapi_url = args.localapi_url
            print(f"Localapi URL (tunneled): {localapi_url}")
        else:
            localapi_url = f"http://localhost:{port}"
            print(f"Localapi URL: {localapi_url}")

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
        localapi_url=localapi_url,
        synth_user_key=SYNTH_USER_KEY,
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
        concurrency=10,  # Run up to 10 seeds in parallel
    )

    # Start localapi if using local mode (after config is created to get localapi_key)
    if not USE_DAYTONA:
        run_server_background(app, port)
        wait_for_health_check_sync("localhost", port, config.localapi_key, timeout=30.0)
        print(f"Localapi ready on port {port}")

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

            # Sort by latency (fastest first)
            sorted_results = sorted(
                result.seed_results, key=lambda sr: sr.get("latency_ms") or float("inf")
            )

            for sr in sorted_results:
                # Seed results from backend
                seed = sr.get("seed", "?")
                # Backend returns: reward (final), local_api_reward (localapi), verifier_reward
                # Legacy field names: outcome_reward, score
                outcome_reward = (
                    sr.get("local_api_reward") or sr.get("outcome_reward") or sr.get("reward") or 0
                )
                fused_reward = sr.get("reward") or sr.get("score") or outcome_reward
                verifier_reward = sr.get("verifier_reward") or sr.get("verifier_score")
                error = sr.get("error")
                latency_ms = sr.get("latency_ms")
                latency_sec = latency_ms / 1000.0 if latency_ms else None

                status = "✅" if fused_reward >= 0.8 else "⚠️" if fused_reward > 0.3 else "❌"

                # Show timing first, then reward
                timing_str = f"{latency_sec:.1f}s" if latency_sec else "?s"

                # Show fused reward with breakdown
                if verifier_reward is not None:
                    # Verifier was used - show breakdown
                    print(
                        f"  {status} seed={seed:2d}: {timing_str:>6s} | fused={fused_reward:.2f} (env={outcome_reward:.2f}, verifier={verifier_reward:.2f})"
                    )
                else:
                    # No verifier - just show env reward
                    print(
                        f"  {status} seed={seed:2d}: {timing_str:>6s} | reward={fused_reward:.2f}"
                    )

                if error:
                    print(f"    error: {error}")

            # Summary stats
            latencies = [sr.get("latency_ms") for sr in result.seed_results if sr.get("latency_ms")]
            if latencies:
                avg_latency = sum(latencies) / len(latencies) / 1000.0
                min_latency = min(latencies) / 1000.0
                max_latency = max(latencies) / 1000.0
                print("\nTiming summary:")
                print(f"  Fastest: {min_latency:.1f}s")
                print(f"  Slowest: {max_latency:.1f}s")
                print(f"  Average: {avg_latency:.1f}s")

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

        # Download traces and artifacts to local data directory
        print("\n" + "=" * 60)
        print("DOWNLOADING TRACES AND ARTIFACTS")
        print("=" * 60)
        try:
            # Save to a local data directory for later use
            # Structure: data/engine_bench/{agent}/{timestamp}_{job_id}/
            from datetime import datetime

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

            # Use a data directory at the synth-ai root level
            synth_ai_root = Path(__file__).parent.parent.parent
            data_dir = synth_ai_root / "data" / "engine_bench" / AGENT / f"{timestamp}_{job_id}"
            data_dir.mkdir(parents=True, exist_ok=True)

            # Download traces
            traces_dir = job.download_traces(data_dir / "traces")
            print(f"Traces downloaded to: {traces_dir}")

            # Save eval results with full details
            results_file = data_dir / "eval_results.json"
            with open(results_file, "w") as f:
                import json

                json.dump(
                    {
                        "job_id": job_id,
                        "agent": AGENT,
                        "model": MODEL,
                        "split": SPLIT,
                        "seeds": seeds,
                        "instance_ids": [INSTANCE_IDS[s % len(INSTANCE_IDS)] for s in seeds],
                        "timestamp": timestamp,
                        "status": result.status.value
                        if hasattr(result.status, "value")
                        else str(result.status),
                        "mean_reward": result.mean_reward,
                        "error": result.error,
                        "seed_results": result.seed_results,
                        "config": {
                            "timeout": TIMEOUT,
                            "verifier": VERIFIER_GRAPH_ID,
                            "verifier_model": VERIFIER_MODEL if VERIFIER_GRAPH_ID else None,
                            "weight_env": WEIGHT_ENV if VERIFIER_GRAPH_ID else None,
                            "weight_outcome": WEIGHT_OUTCOME if VERIFIER_GRAPH_ID else None,
                        },
                    },
                    f,
                    indent=2,
                )
            print(f"Eval results saved to: {results_file}")

            # Save summary for easy reference
            summary_file = data_dir / "summary.txt"
            with open(summary_file, "w") as f:
                f.write("EngineBench Eval Job Summary\n")
                f.write(f"{'=' * 60}\n\n")
                f.write(f"Job ID: {job_id}\n")
                f.write(f"Agent: {AGENT}\n")
                f.write(f"Model: {MODEL}\n")
                f.write(f"Split: {SPLIT}\n")
                f.write(f"Seeds: {seeds}\n")
                f.write(f"Status: {result.status}\n")
                f.write(f"Mean Reward: {result.mean_reward:.4f}\n")
                f.write("\nSeed Results:\n")
                for sr in result.seed_results:
                    seed = sr.get("seed", "?")
                    reward = sr.get("reward") or sr.get("score") or sr.get("local_api_reward") or 0
                    latency_ms = sr.get("latency_ms")
                    latency_sec = latency_ms / 1000.0 if latency_ms else None
                    f.write(f"  Seed {seed}: reward={reward:.4f}, latency={latency_sec:.1f}s\n")
                f.write(f"\nTraces location: {traces_dir}\n")
                f.write(f"Results location: {results_file}\n")
            print(f"Summary saved to: {summary_file}")

            print(f"\nAll data saved to: {data_dir}")

        except Exception as e:
            print(f"Warning: Failed to download traces/artifacts: {e}")
            import traceback

            traceback.print_exc()

    except Exception as e:
        print(f"\nEval job failed: {e}")
        import traceback

        traceback.print_exc()
    finally:
        # Cleanup Daytona sandbox if used
        if daytona_runner:
            print("\nCleaning up Daytona sandbox...")
            await daytona_runner.cleanup()

    print("\nDone!")


if __name__ == "__main__":
    asyncio.run(main())
