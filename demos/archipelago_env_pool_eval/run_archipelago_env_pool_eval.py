"""Run an eval job against the Archipelago environment pool proxy task app."""

from __future__ import annotations

import argparse
import os
from typing import Any

from synth_ai.sdk.eval import EvalJob, EvalJobConfig


def _split_csv(value: str | None) -> list[str] | None:
    if not value:
        return None
    items = [item.strip() for item in value.split(",") if item.strip()]
    return items or None


def _parse_seeds(raw: str) -> list[int]:
    seeds: list[int] = []
    for token in raw.split(","):
        token = token.strip()
        if not token:
            continue
        seeds.append(int(token))
    return seeds


def _build_env_config(args: argparse.Namespace) -> dict[str, Any]:
    env_config: dict[str, Any] = {}
    if args.pool_id:
        env_config["pool_id"] = args.pool_id
    tags = _split_csv(args.pool_tags)
    if tags:
        env_config["pool_tags"] = tags
    if args.rollout_timeout:
        env_config["rollout_timeout"] = float(args.rollout_timeout)
    return env_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Run eval against Archipelago env pool")
    parser.add_argument(
        "--task-app-url",
        default=os.environ.get("TASK_APP_URL", "http://localhost:8001"),
        help="LocalAPI task app URL (default: http://localhost:8001)",
    )
    parser.add_argument(
        "--backend-url",
        default=os.environ.get("SYNTH_BACKEND_URL")
        or os.environ.get("SYNTH_BASE_URL")
        or "https://api-dev.usesynth.ai",
        help="Synth backend URL (default: https://api-dev.usesynth.ai)",
    )
    parser.add_argument(
        "--model",
        default=os.environ.get("POLICY_MODEL", "gpt-4o-mini"),
        help="Policy model to use for eval",
    )
    parser.add_argument(
        "--provider",
        default=os.environ.get("POLICY_PROVIDER", "openai"),
        help="Policy provider (openai, anthropic, etc)",
    )
    parser.add_argument("--seeds", default="0", help="Comma-separated seed list")
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--timeout", type=float, default=1200.0)
    parser.add_argument("--pool-id", default=os.environ.get("ARCHIPELAGO_POOL_ID"))
    parser.add_argument("--pool-tags", default=os.environ.get("ARCHIPELAGO_POOL_TAGS"))
    parser.add_argument("--rollout-timeout", type=float, default=None)
    args = parser.parse_args()

    api_key = os.environ.get("SYNTH_API_KEY", "").strip()
    if not api_key:
        raise SystemExit("SYNTH_API_KEY is required")

    seeds = _parse_seeds(args.seeds)

    env_config = _build_env_config(args)

    config = EvalJobConfig(
        task_app_url=args.task_app_url,
        backend_url=args.backend_url,
        api_key=api_key,
        env_name="archipelago_env_pool",
        seeds=seeds,
        policy_config={"model": args.model, "provider": args.provider},
        env_config=env_config,
        concurrency=args.concurrency,
        timeout=args.timeout,
    )

    job = EvalJob(config)
    job_id = job.submit()
    print(f"Submitted eval job: {job_id}")
    result = job.poll_until_complete(progress=True)

    print("\nRESULT")
    print(f"status: {result.status}")
    if result.mean_reward is not None:
        print(f"mean_reward: {result.mean_reward}")
    if result.total_cost_usd is not None:
        print(f"total_cost_usd: {result.total_cost_usd:.4f}")
    if result.error:
        print(f"error: {result.error}")


if __name__ == "__main__":
    main()
