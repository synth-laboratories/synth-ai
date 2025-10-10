#!/usr/bin/env python3
"""Request a rollout from a remote Crafter task app (e.g., Modal deployment)."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys

import httpx
from synth_ai.task import (
    RolloutEnvSpec,
    RolloutPolicySpec,
    RolloutRecordConfig,
    RolloutRequest,
    RolloutSafetyConfig,
    TaskAppClient,
)


def check_health(base_url: str, api_key: str) -> None:
    try:
        resp = httpx.get(
            f"{base_url.rstrip('/')}/health", headers={"X-API-Key": api_key}, timeout=10.0
        )
        data = (
            resp.json()
            if resp.headers.get("content-type", "").startswith("application/json")
            else resp.text
        )
        if resp.status_code != 200:
            print(f"warning: /health returned {resp.status_code}: {data}")
        else:
            print(f"/health ok: {data}")
    except Exception as exc:
        print(f"warning: failed to call /health: {exc}")


def build_request(
    *,
    run_id: str,
    seed: int,
    model: str,
    inference_url: str,
    llm_calls: int,
    max_policy_tokens: int | None,
) -> RolloutRequest:
    policy_config = {"model": model, "inference_url": inference_url}
    if max_policy_tokens is not None:
        policy_config.update(
            {
                "max_completion_tokens": max_policy_tokens,
                "max_tokens": max_policy_tokens,
            }
        )

    ops: list[str] = []
    for _ in range(max(llm_calls, 1)):
        ops.extend(["agent", "env"])

    return RolloutRequest(
        run_id=run_id,
        env=RolloutEnvSpec(env_name="crafter", seed=seed, config={}),
        policy=RolloutPolicySpec(policy_name="crafter-react", config=policy_config),
        ops=ops,
        record=RolloutRecordConfig(trajectories=True),
        on_done="reset",
        safety=RolloutSafetyConfig(),
    )


def summarise(response) -> dict[str, any]:
    metrics = response.metrics
    return {
        "run_id": response.run_id,
        "num_episodes": metrics.num_episodes,
        "num_steps": metrics.num_steps,
        "episode_returns": metrics.episode_returns,
        "outcome_score": metrics.outcome_score,
        "events_score": metrics.events_score,
    }


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base-url",
        default=None,
        help="Remote task app base URL (e.g., https://xyz.modal.run); defaults to TASK_APP_BASE_URL env",
    )
    parser.add_argument(
        "--api-key", required=True, help="Environment API key for the remote task app"
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--run-id", default="remote-demo")
    parser.add_argument("--model", default="gpt-4o-mini")
    parser.add_argument("--inference-url", default="https://api.openai.com")
    parser.add_argument("--max-llm-calls", type=int, default=1)
    parser.add_argument("--max-policy-tokens", type=int, default=None)
    args = parser.parse_args()

    base_url = args.base_url or os.getenv("TASK_APP_BASE_URL")
    if not base_url:
        parser.error("Missing --base-url (and TASK_APP_BASE_URL not set).")

    request = build_request(
        run_id=args.run_id,
        seed=args.seed,
        model=args.model,
        inference_url=args.inference_url,
        llm_calls=args.max_llm_calls,
        max_policy_tokens=args.max_policy_tokens,
    )

    async with TaskAppClient(base_url, api_key=args.api_key) as client:
        try:
            check_health(base_url, args.api_key)
            info = await client.task_info(seeds=[args.seed])
            payload = info[0] if isinstance(info, list) else info
            print(json.dumps(payload.model_dump(), indent=2)[:600])

            print("Requesting rollout…")
            response = await client.rollout(request)
            print(json.dumps(summarise(response), indent=2))
            print(f"Ops executed: {request.ops}")
        except httpx.HTTPStatusError as exc:
            detail = (
                exc.response.json()
                if exc.response.headers.get("content-type", "").startswith("application/json")
                else exc.response.text
            )
            print(f"HTTP error {exc.response.status_code}: {detail}", file=sys.stderr)
            if exc.response.status_code in (401, 403):
                print(
                    "Hint: check --api-key and ensure the remote deployment expects that value.",
                    file=sys.stderr,
                )
            if exc.response.status_code == 404:
                print(
                    "Hint: verify the --base-url includes the correct path (should be the root of the task app).",
                    file=sys.stderr,
                )
            if exc.response.status_code == 500:
                print(
                    "Hint: remote rollout failed server-side; inspect the deployment logs (Modal dashboard/logs).",
                    file=sys.stderr,
                )
            raise


if __name__ == "__main__":
    asyncio.run(main())
