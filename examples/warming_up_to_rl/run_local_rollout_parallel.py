#!/usr/bin/env python3
"""Launch multiple local rollouts concurrently and summarise rewards/achievements."""

from __future__ import annotations

import argparse
import asyncio
import os
from collections import Counter
from pathlib import Path
from statistics import mean, median
from typing import Any

from dotenv import load_dotenv
from synth_ai.task import (
    RolloutEnvSpec,
    RolloutPolicySpec,
    RolloutRecordConfig,
    RolloutRequest,
    RolloutSafetyConfig,
    TaskAppClient,
)


def build_rollout_request(
    *,
    seed: int,
    run_id: str,
    model: str,
    inference_url: str,
    inference_api_key: str,
    ops: list[str],
    extra_headers: dict[str, str] | None = None,
    trace_format: str = "compact",
    return_trace: bool = False,
) -> RolloutRequest:
    policy_config = {
        "model": model,
        "inference_url": inference_url,
        "api_key": inference_api_key,
    }
    if extra_headers:
        policy_config["extra_headers"] = extra_headers
    record_cfg = RolloutRecordConfig(
        trajectories=True,
        trace_format=trace_format,
        return_trace=return_trace,
    )
    return RolloutRequest(
        run_id=run_id,
        env=RolloutEnvSpec(env_name="crafter", seed=seed, config={}),
        policy=RolloutPolicySpec(policy_name="crafter-react", config=policy_config),
        ops=ops,
        record=record_cfg,
        on_done="reset",
        safety=RolloutSafetyConfig(),
    )


def mask_value(value: str | None) -> str:
    if not value:
        return "<unset>"
    return f"{value[:6]}…{value[-4:]} (len={len(value)})"


def build_ops(max_llm_calls: int, explicit_ops: str | None) -> list[str]:
    if explicit_ops:
        ops = [op.strip() for op in explicit_ops.split(",") if op.strip()]
        if not ops:
            raise ValueError("--ops must contain at least one entry")
        return ops

    llm_calls = max(1, max_llm_calls)
    if llm_calls > 50:
        print("[WARN] --max-llm-calls capped at 50 per rollout; use --ops for manual control.")
        llm_calls = 50

    ops: list[str] = []
    for _ in range(llm_calls):
        ops.extend(["agent", "env"])
    return ops


def extract_achievements(step_info: dict[str, Any] | None) -> list[str]:
    achievements: list[str] = []
    if not isinstance(step_info, dict):
        return achievements

    added = step_info.get("achievements_added")
    if isinstance(added, list):
        achievements.extend(str(item) for item in added)

    meta = step_info.get("meta")
    if isinstance(meta, dict):
        decision = meta.get("decision_rewards")
        if isinstance(decision, dict):
            for key in ("all", "achievements"):
                maybe = decision.get(key)
                if isinstance(maybe, list):
                    achievements.extend(str(item) for item in maybe)
            for key in ("unique", "unique_achievements"):
                maybe = decision.get(key)
                if isinstance(maybe, list):
                    achievements.extend(str(item) for item in maybe)
    return achievements


def analyse_rollout_response(response: Any) -> dict[str, Any]:
    metrics = response.metrics
    trajectory = response.trajectories[0] if response.trajectories else None

    episode_return = metrics.episode_returns[0] if metrics.episode_returns else 0.0
    total_steps = metrics.num_steps

    step_achievements: list[str] = []
    if trajectory is not None:
        for step in trajectory.steps:
            step_achievements.extend(extract_achievements(step.info))

    trace_payload = response.trace or {}
    metadata = trace_payload.get("metadata") if isinstance(trace_payload, dict) else {}
    final_achievements = []
    if isinstance(metadata, dict):
        final_list = metadata.get("final_achievements")
        if isinstance(final_list, list):
            final_achievements = [str(item) for item in final_list]

    decision_rewards = (
        trace_payload.get("decision_rewards") if isinstance(trace_payload, dict) else []
    )
    trace_all: list[str] = []
    if isinstance(decision_rewards, list):
        for item in decision_rewards:
            if isinstance(item, dict):
                for key in ("achievements", "all", "unique", "unique_achievements"):
                    values = item.get(key)
                    if isinstance(values, list):
                        trace_all.extend(str(v) for v in values)

    combined = step_achievements + trace_all + final_achievements
    unique = sorted({str(item) for item in combined})

    return {
        "return": float(episode_return),
        "steps": int(total_steps),
        "achievements_all": combined,
        "achievements_unique": unique,
        "trace": trace_payload,
        "metrics": metrics,
    }


def summarise_runs(run_summaries: list[dict[str, Any]]) -> dict[str, Any]:
    if not run_summaries:
        return {}

    returns = [item["return"] for item in run_summaries]
    total_steps = sum(item["steps"] for item in run_summaries)

    achievements_all_counter = Counter()
    achievements_unique_counter = Counter()
    unique_count_hist = Counter()

    for summary in run_summaries:
        achievements_all_counter.update(summary["achievements_all"])
        unique_set = set(summary["achievements_unique"])
        achievements_unique_counter.update(unique_set)
        unique_count_hist[len(unique_set)] += 1

    stats = {
        "count": len(run_summaries),
        "returns": {
            "mean": mean(returns),
            "median": median(returns),
            "min": min(returns),
            "max": max(returns),
            "total": sum(returns),
        },
        "total_steps": total_steps,
        "achievements_all": achievements_all_counter,
        "achievements_unique": achievements_unique_counter,
        "unique_count_hist": unique_count_hist,
    }
    return stats


def print_summary(
    stats: dict[str, Any], *, run_details: list[dict[str, Any]], total_runs: int
) -> None:
    if not stats:
        print("No successful rollouts to summarise.")
        return

    returns = stats["returns"]
    print("Rollout summary:")
    print(f"  Runs succeeded: {stats['count']} / {total_runs}")
    print(f"  Total steps   : {stats['total_steps']}")
    print(
        "  Returns       : "
        f"mean={returns['mean']:.2f}, median={returns['median']:.2f}, "
        f"min={returns['min']:.2f}, max={returns['max']:.2f}, total={returns['total']:.2f}"
    )

    unique_hist = stats["unique_count_hist"]
    if unique_hist:
        print("  Unique achievement counts per run:")
        for count in sorted(unique_hist):
            runs = unique_hist[count]
            print(f"    {count:02d} unique -> {runs} run(s)")

    top_unique = stats["achievements_unique"].most_common()
    if top_unique:
        print("  Achievements unlocked (by runs):")
        for name, freq in top_unique:
            print(f"    {name}: {freq} run(s)")

    top_all = stats["achievements_all"].most_common()
    if top_all:
        print("  Achievement unlock events (total occurrences):")
        for name, freq in top_all:
            print(f"    {name}: {freq} event(s)")

    print("\nTop runs by return:")
    ranked = sorted(run_details, key=lambda item: item["summary"]["return"], reverse=True)
    for idx, item in enumerate(ranked[:10], start=1):
        summary = item["summary"]
        print(
            f"  {idx:02d}. run_id={item['run_id']} seed={item['seed']} "
            f"return={summary['return']:.2f} steps={summary['steps']} "
            f"achievements={summary['achievements_unique']}"
        )


async def execute_rollouts(args: argparse.Namespace) -> None:
    if args.env_file:
        env_path = Path(args.env_file).expanduser()
        if not env_path.exists():
            raise FileNotFoundError(f"Env file not found: {env_path}")
        load_dotenv(env_path, override=False)

    api_key = args.api_key or os.getenv("ENVIRONMENT_API_KEY")
    if not api_key:
        import sys

        print("Please enter your RL Environment API key:", file=sys.stderr, flush=True)
        api_key = input("> ").strip()
        if not api_key:
            raise RuntimeError("RL Environment API key is required")

    # Prompt for Groq API key if not set
    groq_api_key = os.getenv("GROQ_API_KEY")
    if not groq_api_key:
        import sys

        print("Please enter your Groq API key:", file=sys.stderr, flush=True)
        groq_api_key = input("> ").strip()
        if not groq_api_key:
            raise RuntimeError("Groq API key is required")

    synth_key = os.getenv("SYNTH_API_KEY")
    extra_headers: dict[str, str] | None = None
    if synth_key and "openai.com" not in args.inference_url.lower():
        extra_headers = {"Authorization": f"Bearer {synth_key}"}

    if args.verbose:
        print("Resolved configuration:")
        print(f"  Task app base URL  : {args.base_url}")
        print(f"  Inference base URL : {args.inference_url}")
        print(f"  Task app API key   : {mask_value(api_key)}")
        print(f"  Synth API key      : {mask_value(synth_key)}")
        print(f"  HTTP timeout       : {args.timeout:.1f}s")
        print(f"  Rollouts           : {args.count} (parallel={args.parallel})")

    ops = build_ops(args.max_llm_calls, args.ops)

    print(f"\n🚀 Starting {args.count} rollouts with {args.parallel} parallel workers...")
    print(f"📊 Each rollout: {len(ops)} ops ({args.max_llm_calls} LLM calls)\n")

    async with TaskAppClient(args.base_url, api_key=api_key, timeout=args.timeout) as client:

        async def run_single(index: int) -> dict[str, Any]:
            run_id = f"{args.run_id}-{index:03d}"
            seed = args.seed + index * args.seed_stride
            print(f"\n▶️  [{index + 1}/{args.count}] Starting rollout {run_id} (seed={seed})...")

            request = build_rollout_request(
                seed=seed,
                run_id=run_id,
                model=args.model,
                inference_url=args.inference_url,
                inference_api_key=groq_api_key,
                ops=ops,
                extra_headers=extra_headers,
                trace_format=args.trace_format,
                return_trace=True,
            )
            if args.max_policy_tokens is not None:
                request.policy.config.update(
                    {
                        "max_completion_tokens": args.max_policy_tokens,
                        "max_tokens": args.max_policy_tokens,
                    }
                )

            try:
                response = await client.rollout(request)
                summary = analyse_rollout_response(response)
                print(
                    f"\n✅ [{index + 1}/{args.count}] Completed {run_id} (outcome={summary.get('outcome_score', 'N/A')})"
                )
                return {
                    "ok": True,
                    "run_id": run_id,
                    "seed": seed,
                    "response": response,
                    "summary": summary,
                }
            except Exception as exc:  # pragma: no cover - surface errors
                print(f"\n❌ [{index + 1}/{args.count}] Failed {run_id}: {exc}")
                return {
                    "ok": False,
                    "run_id": run_id,
                    "seed": seed,
                    "error": exc,
                }

        semaphore = asyncio.Semaphore(max(1, args.parallel))

        async def guarded_run(index: int) -> dict[str, Any]:
            async with semaphore:
                return await run_single(index)

        tasks = [asyncio.create_task(guarded_run(i)) for i in range(args.count)]
        results = await asyncio.gather(*tasks)

    successes = [item for item in results if item.get("ok")]
    failures = [item for item in results if not item.get("ok")]

    print(f"\n{'=' * 100}\n")
    stats = summarise_runs([item["summary"] for item in successes])
    print_summary(stats, run_details=successes, total_runs=args.count)

    if failures:
        print("\nFailures:")
        for item in failures:
            err = item.get("error")
            print(f"  run_id={item['run_id']} seed={item['seed']} error={err}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://localhost:8001", help="Task app base URL")
    parser.add_argument("--api-key", help="Environment API key (or set via --env-file)")
    parser.add_argument("--env-file", help="Path to .env file providing API keys")
    parser.add_argument(
        "--model", default="gpt-4o-mini", help="Model identifier for the Crafter policy"
    )
    parser.add_argument(
        "--inference-url",
        default="https://api.openai.com",
        help="Inference base URL for the policy",
    )
    parser.add_argument("--seed", type=int, default=42, help="Base seed for the first rollout")
    parser.add_argument(
        "--seed-stride", type=int, default=1, help="Increment applied to the seed for each rollout"
    )
    parser.add_argument(
        "--count", type=int, default=20, help="Number of rollout trajectories to execute"
    )
    parser.add_argument("--parallel", type=int, default=4, help="Maximum concurrent rollouts")
    parser.add_argument("--ops", help="Comma-separated rollout ops (advanced override)")
    parser.add_argument(
        "--max-llm-calls",
        type=int,
        default=20,
        help="Number of agent/env pairs per rollout when --ops not provided",
    )
    parser.add_argument(
        "--max-policy-tokens",
        type=int,
        help="Optional per-call token limit forwarded to the policy config",
    )
    parser.add_argument(
        "--timeout", type=float, default=600.0, help="HTTP timeout (seconds) for task app requests"
    )
    parser.add_argument(
        "--trace-format",
        default="compact",
        choices=["compact", "full"],
        help="Trace format requested from the task app",
    )
    parser.add_argument("--run-id", default="batch-demo", help="Run ID prefix for rollouts")
    parser.add_argument("--verbose", action="store_true", help="Print resolved configuration")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    asyncio.run(execute_rollouts(args))


if __name__ == "__main__":
    main()
