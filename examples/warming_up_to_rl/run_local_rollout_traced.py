#!/usr/bin/env python3
"""Run a local Crafter rollout, capture tracing metadata, and optionally persist the trace."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Any

import httpx
from synth_ai._utils.user_config import load_user_config, update_user_config
from synth_ai.task import TaskAppClient

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.append(str(SCRIPT_DIR))

from shared import build_rollout_request, ops_from_pairs, parse_ops  # noqa: E402


def summarise_rollout(response: Any) -> dict[str, Any]:
    metrics = (
        response.metrics.model_dump()
        if hasattr(response, "metrics")
        else response.get("metrics", {})
    )
    return {
        "run_id": getattr(response, "run_id", None) or response.get("run_id"),
        "num_episodes": metrics.get("num_episodes"),
        "num_steps": metrics.get("num_steps"),
        "episode_returns": metrics.get("episode_returns"),
        "outcome_score": metrics.get("outcome_score"),
        "events_score": metrics.get("events_score"),
    }


def summarise_trace(trace: Any) -> dict[str, Any]:
    if trace is None:
        return {"trace": None}
    if not isinstance(trace, dict):
        return {"trace_type": type(trace).__name__}

    format_hint = "compact" if "events_count" in trace or "lm_calls" in trace else "full"
    events_count = trace.get("events_count")
    if (
        events_count is None
        and "event_history" in trace
        and isinstance(trace["event_history"], list)
    ):
        events_count = len(trace["event_history"])
    messages_count = trace.get("messages_count")
    if (
        messages_count is None
        and "markov_blanket_message_history" in trace
        and isinstance(trace["markov_blanket_message_history"], list)
    ):
        messages_count = len(trace["markov_blanket_message_history"])

    metadata = trace.get("metadata") if isinstance(trace.get("metadata"), dict) else {}
    lm_calls = trace.get("lm_calls") if isinstance(trace.get("lm_calls"), list) else []
    decision_rewards = (
        trace.get("decision_rewards") if isinstance(trace.get("decision_rewards"), list) else []
    )

    return {
        "session_id": trace.get("session_id"),
        "format": format_hint,
        "events_count": events_count,
        "messages_count": messages_count,
        "metadata_keys": sorted(metadata.keys()),
        "lm_calls_count": len(lm_calls),
        "decision_turns": len(decision_rewards),
    }
def dump_trace(trace: dict[str, Any], *, path: Path, pretty: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(trace, fh, indent=2 if pretty else None)
        fh.write("\n")


def extract_environment_rewards(trace_payload: dict[str, Any] | None) -> list[float]:
    if not trace_payload:
        return []

    rewards: list[float] = []

    def _collect(events: list[dict[str, Any]]) -> None:
        for event in events:
            reward = event.get("reward")
            if reward is not None:
                try:
                    rewards.append(float(reward))
                except Exception:
                    continue

    if isinstance(trace_payload.get("event_history"), list):
        _collect(trace_payload["event_history"])
    if isinstance(trace_payload.get("session_time_steps"), list):
        for step in trace_payload["session_time_steps"]:
            _collect(step.get("events", []))

    return rewards


def extract_decision_rewards(trace_payload: dict[str, Any] | None) -> list[dict[str, Any]]:
    if not trace_payload:
        return []
    rewards = trace_payload.get("decision_rewards")
    return rewards if isinstance(rewards, list) else []


def extract_trajectory_rewards(response: Any) -> list[float]:
    """Extract per-step rewards directly from the rollout trajectories."""

    rewards: list[float] = []

    if response is None:
        return rewards

    trajectories = getattr(response, "trajectories", None)
    if trajectories is None and isinstance(response, dict):
        trajectories = response.get("trajectories")

    if not trajectories:
        return rewards

    for traj in trajectories:
        steps = getattr(traj, "steps", None)
        if steps is None and isinstance(traj, dict):
            steps = traj.get("steps")
        if not steps:
            continue
        for step in steps:
            reward_val = getattr(step, "reward", None)
            if reward_val is None and isinstance(step, dict):
                reward_val = step.get("reward")
            if reward_val is None:
                continue
            try:
                rewards.append(float(reward_val))
            except Exception:
                continue

    return rewards


def print_reward_summary(
    trace_payload: dict[str, Any] | None,
    rollout_summary: dict[str, Any],
    trajectory_rewards: list[float],
) -> None:
    print("Reward summary:")

    env_rewards = extract_environment_rewards(trace_payload)
    reward_source = "trace"
    if not env_rewards and trajectory_rewards:
        env_rewards = trajectory_rewards
        reward_source = "trajectory"

    if env_rewards:
        print(f"  Environment rewards per step ({reward_source}): {env_rewards}")
        print(f"  Environment reward total: {sum(env_rewards):.3f}")
    else:
        print("  Environment rewards per step: none recorded")

    decision_rewards = extract_decision_rewards(trace_payload)
    if decision_rewards:
        print("  Decision rewards:")
        for entry in decision_rewards:
            turn = entry.get("turn")
            ach_delta = entry.get("ach_delta")
            unique_delta = entry.get("unique_delta")
            achievements = entry.get("achievements") or []
            print(
                f"    turn={turn}, ach_delta={ach_delta}, unique_delta={unique_delta}, achievements={achievements}"
            )
    else:
        print("  Decision rewards: none recorded")

    episode_returns = rollout_summary.get("episode_returns")
    if episode_returns:
        print(f"  Outcome rewards (episode returns): {episode_returns}")
        if env_rewards:
            try:
                total_env_reward = float(sum(env_rewards))
                target = float(episode_returns[0]) if episode_returns else 0.0
                if abs(total_env_reward - target) > 1e-6:
                    print(
                        "  ⚠️  Reward mismatch: sum(environment rewards)"
                        f"={total_env_reward:.3f} vs episode return={target:.3f}"
                    )
            except Exception:
                pass
    else:
        print("  Outcome rewards: none recorded")


async def main() -> None:
    user_config = load_user_config()

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://localhost:8001", help="Task app base URL")
    parser.add_argument("--api-key", help="Environment API key (will prompt if not provided)")
    parser.add_argument(
        "--inference-api-key", help="Inference provider API key (will prompt if not provided)"
    )
    parser.add_argument("--seed", type=int, default=42, help="Environment seed")
    parser.add_argument("--run-id", default="local-trace", help="Run identifier")
    parser.add_argument("--model", default="gpt-4o-mini", help="OpenAI-compatible model id")
    parser.add_argument(
        "--inference-url", default="https://api.openai.com", help="Inference base URL (OpenAI/Groq)"
    )
    parser.add_argument(
        "--ops", help="Comma-separated rollout ops (fallback: alternating agent/env)"
    )
    parser.add_argument(
        "--max-llm-calls",
        type=int,
        default=1,
        help="Number of agent/env pairs when --ops not supplied",
    )
    parser.add_argument(
        "--max-policy-tokens",
        type=int,
        default=None,
        help="Optional max token budget forwarded to policy",
    )
    parser.add_argument(
        "--trace-format",
        choices=["compact", "full"],
        default="compact",
        help="Trace payload format requested from the server",
    )
    parser.add_argument(
        "--trace-path",
        type=Path,
        help="Path to write the trace JSON (defaults to ./<run_id>_trace.json unless --no-trace-file is set)",
    )
    parser.add_argument(
        "--no-trace-file",
        action="store_true",
        help="Do not write the trace JSON to disk",
    )
    parser.add_argument(
        "--no-print-trace",
        action="store_true",
        help="Do not print the full trace payload to stdout",
    )
    parser.add_argument(
        "--no-trace",
        action="store_true",
        help="Disable return_trace (useful for comparing behaviour without tracing)",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=60.0,
        help="HTTP timeout in seconds for the client (default: 60)",
    )
    args = parser.parse_args()

    # Prompt for required parameters if not provided
    base_url = args.base_url
    if args.base_url == "http://localhost:8001":
        print("\nTask app configuration:")
        base_url_input = input("Task app base URL [http://localhost:8001]: ").strip()
        base_url = base_url_input if base_url_input else "http://localhost:8001"

    api_key = (
        args.api_key
        or os.getenv("ENVIRONMENT_API_KEY")
        or user_config.get("ENVIRONMENT_API_KEY")
        or user_config.get("DEV_ENVIRONMENT_API_KEY")
    )
    if not api_key:
        api_key = input("Environment API key (from ENVIRONMENT_API_KEY): ").strip()
        if not api_key:
            parser.error("Environment API key is required")
        update_user_config(
            {
                "ENVIRONMENT_API_KEY": api_key,
                "DEV_ENVIRONMENT_API_KEY": api_key,
            }
        )
        user_config["ENVIRONMENT_API_KEY"] = api_key
        user_config["DEV_ENVIRONMENT_API_KEY"] = api_key
    os.environ["ENVIRONMENT_API_KEY"] = api_key
    os.environ.setdefault("DEV_ENVIRONMENT_API_KEY", api_key)

    # Use Groq by default
    model = "llama-3.3-70b-versatile"
    inference_url = "https://api.groq.com/openai"

    print("\nInference configuration (Groq):")
    inference_api_key = (
        args.inference_api_key or os.getenv("GROQ_API_KEY") or user_config.get("GROQ_API_KEY")
    )
    if not inference_api_key:
        inference_api_key = input("Groq API key: ").strip()
        if not inference_api_key:
            parser.error("Groq API key is required")
        update_user_config({"GROQ_API_KEY": inference_api_key})
        user_config["GROQ_API_KEY"] = inference_api_key
        print("[INFO] Saved GROQ_API_KEY to user configuration")
    os.environ["GROQ_API_KEY"] = inference_api_key

    print("\nRollout configuration:")
    max_llm_calls = args.max_llm_calls
    if args.max_llm_calls == 1:
        max_llm_calls_input = input("Max LLM calls [10]: ").strip()
        max_llm_calls = int(max_llm_calls_input) if max_llm_calls_input else 10

    # Override args with prompted values
    args.base_url = base_url
    args.max_llm_calls = max_llm_calls

    if args.trace_path is not None:
        try:
            args.trace_path = Path(args.trace_path).expanduser().resolve()
        except Exception:
            print(f"Could not resolve path '{args.trace_path}'. Falling back to prompt.")
            args.trace_path = None

    default_trace_path = (Path.cwd() / f"{args.run_id}_trace.json").resolve()
    if not args.no_trace_file and args.trace_path is None:
        try:
            prompt = (
                "\nTo where do you want to save the trace JSON?\n"
                f"Press enter for default path: {default_trace_path}\n> "
            )
            trace_path_input = input(prompt).strip()
        except Exception:
            trace_path_input = ""
        if trace_path_input:
            try:
                args.trace_path = Path(trace_path_input).expanduser().resolve()
            except Exception:
                print(f"Could not resolve path '{trace_path_input}'. Using default {default_trace_path}.")
                args.trace_path = default_trace_path
        else:
            args.trace_path = default_trace_path

    explicit_ops = parse_ops(args.ops)
    ops = explicit_ops if explicit_ops is not None else ops_from_pairs(args.max_llm_calls)
    return_trace = not args.no_trace

    async with TaskAppClient(args.base_url, api_key=api_key, timeout=args.timeout) as client:
        try:
            print(f"Fetching task_info for seed {args.seed}…")
            task_info = await client.task_info(seeds=[args.seed])
            info_payload = task_info[0] if isinstance(task_info, list) else task_info
            try:
                print(json.dumps(info_payload.model_dump(), indent=2)[:600])
            except Exception:
                print(info_payload)

            request = build_rollout_request(
                seed=args.seed,
                run_id=args.run_id,
                model=model,
                inference_url=inference_url,
                inference_api_key=inference_api_key,
                ops=ops,
                return_trace=return_trace,
                trace_format=args.trace_format,
                max_policy_tokens=args.max_policy_tokens,
            )

            print("Requesting rollout…")
            response = await client.rollout(request)
            summary = summarise_rollout(response)
            print(json.dumps(summary, indent=2))

            trace_payload: dict[str, Any] | None = getattr(response, "trace", None)
            if return_trace:
                if trace_payload is None:
                    print(
                        "⚠️  Server did not include a trace. Ensure TASKAPP_TRACING_ENABLED=1 when starting the task app.",
                        file=sys.stderr,
                    )
                else:
                    trace_summary = summarise_trace(trace_payload)
                    print("Trace summary:")
                    print(json.dumps(trace_summary, indent=2))

                    trace_path = args.trace_path
                    if not args.no_trace_file:
                        if trace_path is None:
                            trace_path = Path(f"{args.run_id}_trace.json")
                        dump_trace(trace_payload, path=trace_path, pretty=True)
                        print(f"Trace written to {trace_path}")

                    if not args.no_print_trace:
                        print("Full trace payload:")
                        print(json.dumps(trace_payload, indent=2))

            trajectory_rewards = extract_trajectory_rewards(response)
            print_reward_summary(
                trace_payload if return_trace else None,
                summary,
                trajectory_rewards,
            )

            print(f"Ops executed: {ops}")

            print(
                "Tip: export TASKAPP_TRACING_ENABLED=1 and optionally TASKAPP_SFT_OUTPUT_DIR before running `uvx synth-ai serve …` to persist traces/SFT."
            )
            if not args.no_trace_file:
                path_hint = args.trace_path or Path(f"{args.run_id}_trace.json")
                print(f"Your traces have been saved to:\n{path_hint}")
        except httpx.ConnectError as exc:
            print(
                f"✖ Unable to connect to the task app at {args.base_url}: {exc}",
                file=sys.stderr,
            )
            print(
                "Hint: start the local task app (for example, `uvx synth-ai demo deploy --local`) "
                "and leave it running while you collect rollouts.",
                file=sys.stderr,
            )
            raise SystemExit(1) from None
        except httpx.HTTPStatusError as exc:
            detail = (
                exc.response.json()
                if exc.response.headers.get("content-type", "").startswith("application/json")
                else exc.response.text
            )
            print(f"HTTP error {exc.response.status_code}: {detail}", file=sys.stderr)
            if exc.response.status_code in (401, 503):
                print(
                    "Hint: ensure the task app process is using the same ENVIRONMENT_API_KEY passed via --api-key.",
                    file=sys.stderr,
                )
            if exc.response.status_code == 500:
                print(
                    "Hint: verify tracing is enabled server-side (TASKAPP_TRACING_ENABLED=1) and the inference credentials are configured.",
                    file=sys.stderr,
                )
            raise


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("Interrupted", file=sys.stderr)
