#!/usr/bin/env python3
"""Rollout a Crafter task app using the Modal backend proxy."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
from pathlib import Path
from typing import Any

import sys

import httpx
from dotenv import load_dotenv

from synth_ai.task import (
    RolloutEnvSpec,
    RolloutPolicySpec,
    RolloutRecordConfig,
    RolloutRequest,
    RolloutSafetyConfig,
    TaskAppClient,
)


def build_rollout_request(seed: int, run_id: str, *, model: str, inference_url: str, ops: list[str], api_key: str) -> RolloutRequest:
    policy_config = {
        "model": model,
        "inference_url": inference_url,
        "extra_headers": {
            "Authorization": f"Bearer {api_key}",
        },
    }
    return RolloutRequest(
        run_id=run_id,
        env=RolloutEnvSpec(env_name="crafter", seed=seed, config={}),
        policy=RolloutPolicySpec(policy_name="crafter-react", config=policy_config),
        ops=ops,
        record=RolloutRecordConfig(trajectories=True),
        on_done="reset",
        safety=RolloutSafetyConfig(),
    )


def summarise_response(data: Any) -> dict[str, Any]:
    metrics = data.metrics.model_dump() if hasattr(data.metrics, "model_dump") else data.get("metrics", {})
    return {
        "run_id": getattr(data, "run_id", None) or data.get("run_id"),
        "num_episodes": metrics.get("num_episodes"),
        "num_steps": metrics.get("num_steps"),
        "episode_returns": metrics.get("episode_returns"),
        "outcome_score": metrics.get("outcome_score"),
        "events_score": metrics.get("events_score"),
    }


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://localhost:8010", help="Task app base URL")
    parser.add_argument("--env-file", type=str, default=None, help="Path to .env file with keys")
    parser.add_argument("--seed", type=int, default=42, help="Env seed to rollout")
    parser.add_argument("--run-id", default="modal-eval", help="Run identifier")
    parser.add_argument("--model", required=True, help="Model identifier for the Crafter policy")
    parser.add_argument("--inference-url", required=True, help="Modal backend inference base URL (e.g., http://localhost:8000/api)")
    parser.add_argument("--task-app-key", default=None, help="Environment API key for the task app (fallback ENVIRONMENT_API_KEY)")
    parser.add_argument("--modal-key", default=None, help="Synth/Modal API key for inference (fallback SYNTH_API_KEY)")
    parser.add_argument("--max-llm-calls", type=int, default=20, help="Number of policy inference calls")
    parser.add_argument("--ops", default=None, help="Comma-separated rollout ops (advanced override)")
    parser.add_argument("--max-policy-tokens", type=int, default=None, help="Optional per-call token limit forwarded to the policy config")
    parser.add_argument("--verbose", action="store_true", help="Print resolved configuration and headers")
    args = parser.parse_args()

    if args.env_file:
        env_path = Path(args.env_file).expanduser()
        if not env_path.exists():
            print(f"[WARN] Env file not found: {env_path}")
        else:
            load_dotenv(env_path, override=False)

    task_app_key = args.task_app_key or os.getenv("ENVIRONMENT_API_KEY")
    if not task_app_key:
        parser.error("Missing task app API key (set ENVIRONMENT_API_KEY or pass --task-app-key)")

    modal_key = args.modal_key or os.getenv("SYNTH_API_KEY")
    if not modal_key:
        parser.error("Missing Synth/Modal API key (set SYNTH_API_KEY or pass --modal-key)")

    if synth_key and "openai.com" not in args.inference_url.lower():
        os.environ["OPENAI_API_KEY"] = synth_key

    if args.ops:
        ops = [op.strip() for op in args.ops.split(",") if op.strip()]
        if not ops:
            raise ValueError("Ops must contain at least one entry")
    else:
        llm_calls = max(args.max_llm_calls, 1)
        if llm_calls > 20:
            llm_calls = 20
        ops = []
        for _ in range(llm_calls):
            ops.extend(["agent", "env"])

    if args.verbose:
        def _mask(val: str | None) -> str:
            if not val:
                return "<unset>"
            return f"{val[:6]}…{val[-4:]} (len={len(val)})"

        print("Resolved configuration:")
        print(f"  Task app base URL  : {args.base_url}")
        print(f"  Inference base URL : {args.inference_url}")
        print(f"  Task app API key   : {_mask(task_app_key)}")
        print(f"  Modal API key      : {_mask(modal_key)}")
        print(f"  Ops (count={len(ops)}) : {ops}")

    inf_url_norm = args.inference_url.rstrip('/')
    if '/api' not in inf_url_norm:
        print('[WARN] Inference URL is missing /api prefix; proxy endpoints usually live at /api/inference/v1/chat/completions.')
    elif not inf_url_norm.lower().endswith('/api'):
        print('[INFO] Using inference base URL; policy will append /v1/chat/completions automatically.')

    async with TaskAppClient(args.base_url, api_key=task_app_key) as client:
        try:
            print(f"Fetching task_info for seed {args.seed}…")
            task_info = await client.task_info(seeds=[args.seed])
            info_payload = task_info[0] if isinstance(task_info, list) else task_info
            print(json.dumps(info_payload.model_dump(), indent=2)[:600])

            request = build_rollout_request(
                args.seed,
                args.run_id,
                model=args.model,
                inference_url=args.inference_url,
                ops=ops,
                api_key=modal_key,
            )
            if args.verbose:
                print(f"Request headers: {request.policy.config.get('extra_headers', {})}")
            if args.max_policy_tokens is not None:
                request.policy.config.update({
                    "max_completion_tokens": args.max_policy_tokens,
                    "max_tokens": args.max_policy_tokens,
                })
            print("Requesting rollout…")
            response = await client.rollout(request)
            summary = summarise_response(response)
            print(json.dumps(summary, indent=2))
            print(f"Ops executed: {ops}")
        except httpx.HTTPStatusError as exc:
            detail = exc.response.json() if exc.response.headers.get("content-type", "").startswith("application/json") else exc.response.text
            print(f"HTTP error {exc.response.status_code}: {detail}", file=sys.stderr)
            if exc.response.status_code in (401, 503):
                print("Hint: ensure ENVIRONMENT_API_KEY and SYNTH_API_KEY are correctly set.", file=sys.stderr)
            raise


if __name__ == "__main__":
    asyncio.run(main())
