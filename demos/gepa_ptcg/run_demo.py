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
parser.add_argument("--local", action="store_true", help="Use localhost:8000 backend")
parser.add_argument("--local-host", type=str, default="localhost")
parser.add_argument("--port", type=int, default=8017, help="Port for task app")
parser.add_argument("--model", type=str, default="gpt-5.2", help="Model to use")
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

LOCAL_MODE = args.local
LOCAL_HOST = args.local_host
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
    # Print tcg_py version
    try:
        import importlib.metadata

        tcg_version = importlib.metadata.version("tcg-py")
        print(f"tcg_py version: {tcg_version}")
    except Exception as e:
        print(f"tcg_py version: unknown ({e})")

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

    run_server_background(app, port)
    wait_for_health_check_sync(LOCAL_HOST, port, env_key, timeout=30.0)
    print(f"Task app ready on port {port}")

    task_app_url = f"http://{LOCAL_HOST}:{port}"
    print(f"Task app URL: {task_app_url}")

    if run_dir is not None:
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.get(
                    f"{task_app_url}/info",
                    headers={"X-API-Key": env_key, "Content-Type": "application/json"},
                )
                resp.raise_for_status()
                (run_dir / "task_app_info.json").write_text(
                    json.dumps(resp.json(), indent=2, default=str), encoding="utf-8"
                )
        except Exception as e:
            print(f"Warning: failed to fetch /info from task app: {e}")

    print("\n" + "=" * 60)
    print(f"Model: {MODEL}")
    print(f"Prompt: {'ReAct' if USE_REACT else 'baseline'}")
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
            "system_prompt": PTCG_REACT_SYSTEM_PROMPT if USE_REACT else DEFAULT_SYSTEM_PROMPT,
        },
        env_config={},
        verifier_config=(
            {
                # Use backend verifier to evaluate "intangible" gameplay quality, then fuse it with the
                # task app's local_api_reward (outcome_reward).
                "enabled": True,
                "reward_source": "fused",
                "backend_base": synth_api_base,
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

        raw_results = None
        if run_dir is not None:
            try:
                raw_results = job.get_results()
                (run_dir / "eval_job_results.json").write_text(
                    json.dumps(raw_results, indent=2, default=str), encoding="utf-8"
                )
                (run_dir / "eval_job_id.txt").write_text(str(job_id), encoding="utf-8")
            except Exception as e:
                print(f"Warning: failed to write eval job results: {e}")

        # Try to get raw_results if not already available
        if raw_results is None:
            try:
                raw_results = job.get_results()
            except Exception:
                pass

        # Use raw_results if available, otherwise fall back to result.seed_results
        seed_results_to_use = None
        if raw_results and isinstance(raw_results, dict) and "seed_results" in raw_results:
            seed_results_to_use = raw_results["seed_results"]
        elif hasattr(result, "seed_results") and result.seed_results:
            seed_results_to_use = result.seed_results

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

        if seed_results_to_use:
            # Debug: Print first seed_result structure to understand format
            if len(seed_results_to_use) > 0:
                debug_sr = seed_results_to_use[0]
                print(
                    f"\n[DEBUG] First seed_result keys: {list(debug_sr.keys()) if isinstance(debug_sr, dict) else 'not a dict'}"
                )
                if isinstance(debug_sr, dict):
                    # Print all values to see what's actually there
                    print(
                        f"[DEBUG] Sample values: {[(k, v) for k, v in list(debug_sr.items())[:10]]}"
                    )
                    if "details" in debug_sr:
                        print(f"[DEBUG] Details keys: {list(debug_sr['details'].keys())}")
                    if "metadata" in debug_sr:
                        print(f"[DEBUG] Metadata keys: {list(debug_sr['metadata'].keys())}")
                    # Check if data might be in reward_info
                    if "reward_info" in debug_sr:
                        print(
                            f"[DEBUG] reward_info keys: {list(debug_sr['reward_info'].keys()) if isinstance(debug_sr['reward_info'], dict) else 'not a dict'}"
                        )

            # Print detailed statistics table
            print(f"\n{'=' * 130}")
            print("DETAILED GAME STATISTICS")
            print(f"{'=' * 130}")
            print(
                f"{'Seed':<6} {'Instance':<15} {'Winner':<8} {'Agent':<8} {'Opponent':<10} "
                f"{'Agent':<8} {'Opponent':<10} {'Agent':<8} {'Opponent':<10} {'Loss Reason':<50}"
            )
            print(
                f"{'':<6} {'ID':<15} {'':<8} {'Turns':<8} {'Turns':<10} "
                f"{'Prizes':<8} {'Prizes':<10} {'Damage':<8} {'Damage':<10} {'':<50}"
            )
            print("-" * 135)

            # Load rollouts JSONL file if available - it has the actual game results
            rollouts_data = {}
            if run_dir:
                rollouts_file = run_dir / "ptcg_rollouts.jsonl"
                if rollouts_file.exists():
                    try:
                        with open(rollouts_file) as f:
                            for line in f:
                                if line.strip():
                                    rollout = json.loads(line)
                                    seed_val = rollout.get("seed")
                                    if seed_val is not None:
                                        rollouts_data[seed_val] = rollout
                    except Exception as e:
                        print(f"[DEBUG] Failed to load rollouts: {e}")

            for idx, sr in enumerate(seed_results_to_use):
                seed = sr.get("seed", idx)

                # Try to get data from rollouts file first (most reliable)
                rollout = rollouts_data.get(seed, {})
                result_data = rollout.get("result", {})

                # Try multiple ways to extract data
                metadata = sr.get("metadata", {}) or sr.get("rollout_metadata", {}) or {}
                details = sr.get("details", {}) or {}
                reward_info = sr.get("reward_info", {}) or {}

                # Check if details is nested in reward_info
                if isinstance(reward_info, dict) and "details" in reward_info:
                    details = reward_info["details"] or details
                if isinstance(reward_info, dict) and "metadata" in reward_info:
                    metadata = reward_info["metadata"] or metadata

                # Merge rollout result data into details (rollout data takes precedence)
                if result_data:
                    details = {**details, **result_data}

                # Extract instance_id - try multiple locations (rollout data first)
                instance_id = (
                    rollout.get("instance_id")
                    or (metadata.get("instance_id") if metadata else None)
                    or (details.get("instance_id") if details else None)
                    or sr.get("instance_id")
                    or sr.get("trial_id", "").split("-")[-1]
                    if sr.get("trial_id")
                    else None or f"seed_{seed}"
                )

                # Extract winner - try multiple locations (rollout result first)
                winner = (
                    result_data.get("winner")
                    or (metadata.get("winner") if metadata else None)
                    or (details.get("winner") if details else None)
                    or sr.get("winner")
                    or "?"
                )

                # Extract game statistics - check details first, then try direct access
                # Also check reward_info.details if it exists
                reward_details = (
                    reward_info.get("details", {}) if isinstance(reward_info, dict) else {}
                )
                all_details = {**details, **reward_details} if details or reward_details else {}

                # Extract game statistics - rollout result data takes precedence
                agent_turns = (
                    result_data.get("decision_steps")
                    or all_details.get("decision_steps")
                    or all_details.get("turns")
                    or details.get("decision_steps")
                    or details.get("turns")
                    or sr.get("decision_steps")
                    or sr.get("turns")
                    or "?"
                )

                total_turns = (
                    result_data.get("turns")
                    or all_details.get("turns")
                    or details.get("turns")
                    or sr.get("turns")
                    or "?"
                )

                # Extract prize information (rollout result first)
                p1_prizes_remaining = (
                    result_data.get("p1_prizes")
                    or all_details.get("p1_prizes")
                    or details.get("p1_prizes")
                    or sr.get("p1_prizes")
                )
                if p1_prizes_remaining is None:
                    p1_prizes_remaining = 6

                p2_prizes_remaining = (
                    result_data.get("p2_prizes")
                    or all_details.get("p2_prizes")
                    or details.get("p2_prizes")
                    or sr.get("p2_prizes")
                )
                if p2_prizes_remaining is None:
                    p2_prizes_remaining = 6

                # Calculate prizes taken (start with 6, subtract remaining)
                agent_prizes_taken = (
                    6 - p1_prizes_remaining if isinstance(p1_prizes_remaining, int) else "?"
                )
                opponent_prizes_taken = (
                    6 - p2_prizes_remaining if isinstance(p2_prizes_remaining, int) else "?"
                )

                # Extract damage from rollout result first
                agent_damage = (
                    result_data.get("p1_damage_dealt")
                    or all_details.get("p1_damage_dealt")
                    or details.get("p1_damage_dealt")
                    or sr.get("p1_damage_dealt")
                    or 0
                )
                opponent_damage = (
                    result_data.get("p2_damage_dealt")
                    or all_details.get("p2_damage_dealt")
                    or details.get("p2_damage_dealt")
                    or sr.get("p2_damage_dealt")
                    or 0
                )

                # Determine loss reason (rollout result first)
                end_reason = (
                    result_data.get("end_reason")
                    or all_details.get("end_reason")
                    or details.get("end_reason")
                    or sr.get("end_reason")
                    or "Unknown"
                )
                errors = (
                    result_data.get("errors")
                    or all_details.get("errors")
                    or details.get("errors")
                    or sr.get("errors")
                    or 0
                )

                # Build descriptive loss reason
                if winner == "P1":
                    loss_reason = "Won"
                elif errors > 0:
                    # Get error messages from rollout result
                    error_msgs = (
                        result_data.get("error_messages")
                        or all_details.get("error_messages")
                        or details.get("error_messages")
                        or []
                    )

                    if error_msgs:
                        # Extract error type from most common error
                        from collections import Counter

                        error_counts = Counter(error_msgs)
                        most_common_error = error_counts.most_common(1)[0][0]
                        # Extract error type (e.g., "EnergyAlreadyAttached" from "Action failed: EnergyAlreadyAttached")
                        if ":" in most_common_error:
                            error_type = most_common_error.split(":")[-1].strip()
                        elif "failed" in most_common_error.lower():
                            # Extract from "Action failed: EnergyAlreadyAttached"
                            parts = most_common_error.split()
                            error_type = parts[-1] if parts else most_common_error
                        else:
                            error_type = most_common_error
                        loss_reason = f"{errors}× {error_type}"
                    else:
                        loss_reason = f"{errors} errors"
                else:
                    # Prefer engine-provided end_reason
                    if end_reason and end_reason not in ("Unknown",):
                        loss_reason = end_reason
                    else:
                        # Fallback: add minimal context
                        prize_diff = (
                            opponent_prizes_taken - agent_prizes_taken
                            if isinstance(opponent_prizes_taken, int)
                            and isinstance(agent_prizes_taken, int)
                            else None
                        )
                        damage_diff = (
                            opponent_damage - agent_damage
                            if isinstance(opponent_damage, (int, float))
                            and isinstance(agent_damage, (int, float))
                            else None
                        )
                        reason_parts = ["Lost"]
                        if prize_diff is not None:
                            if prize_diff > 0:
                                reason_parts.append(f"-{prize_diff} prizes")
                            elif prize_diff == 0:
                                reason_parts.append("even prizes")
                        if damage_diff is not None and abs(damage_diff) >= 10:
                            if damage_diff > 0:
                                reason_parts.append(f"-{damage_diff} dmg")
                            else:
                                reason_parts.append(f"+{abs(damage_diff)} dmg")
                        loss_reason = " | ".join(reason_parts)

                # Format opponent turns (approximate - total turns minus agent turns)
                opponent_turns = "?"
                if isinstance(total_turns, int) and isinstance(agent_turns, int):
                    opponent_turns = max(0, total_turns - agent_turns)

                print(
                    f"{seed:<6} {instance_id:<15} {winner:<8} {agent_turns:<8} {opponent_turns:<10} "
                    f"{agent_prizes_taken:<8} {opponent_prizes_taken:<10} {agent_damage:<8} {opponent_damage:<10} "
                    f"{loss_reason[:50]:<50}"
                )

            print(f"{'=' * 135}\n")

            # Also print summary format for compatibility
            print(f"Game results summary ({len(seed_results_to_use)}):")
            for idx, sr in enumerate(seed_results_to_use):
                seed = sr.get("seed", idx)
                rollout = rollouts_data.get(seed, {})
                result_data = rollout.get("result", {})

                metadata = sr.get("metadata", {}) or sr.get("rollout_metadata", {}) or {}
                details = sr.get("details", {}) or {}

                instance_id = (
                    rollout.get("instance_id")
                    or metadata.get("instance_id")
                    or details.get("instance_id")
                    or f"seed_{seed}"
                )
                winner = (
                    result_data.get("winner")
                    or metadata.get("winner")
                    or details.get("winner")
                    or "?"
                )
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
