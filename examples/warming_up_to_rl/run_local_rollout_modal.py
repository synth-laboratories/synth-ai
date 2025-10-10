#!/usr/bin/env python3
"""Rollout a Crafter task app using the Modal backend proxy."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Any

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


def build_rollout_request(
    seed: int, run_id: str, *, model: str, inference_url: str, ops: list[str], api_key: str
) -> RolloutRequest:
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
    metrics = (
        data.metrics.model_dump()
        if hasattr(data.metrics, "model_dump")
        else data.get("metrics", {})
    )
    return {
        "run_id": getattr(data, "run_id", None) or data.get("run_id"),
        "num_episodes": metrics.get("num_episodes"),
        "num_steps": metrics.get("num_steps"),
        "episode_returns": metrics.get("episode_returns"),
        "outcome_score": metrics.get("outcome_score"),
        "events_score": metrics.get("events_score"),
    }


async def main() -> None:
    # Load .env file from current directory first if it exists
    default_env = Path.cwd() / ".env"
    if default_env.exists():
        load_dotenv(default_env, override=False)

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://localhost:8010", help="Task app base URL")
    parser.add_argument("--env-file", type=str, default=None, help="Path to .env file with keys")
    parser.add_argument("--seed", type=int, default=42, help="Env seed to rollout")
    parser.add_argument("--run-id", default="modal-eval", help="Run identifier")
    parser.add_argument(
        "--model",
        required=False,
        help="Model identifier for the Crafter policy (e.g., fft:Qwen/Qwen3-4B:job_xxx)",
    )
    parser.add_argument(
        "--inference-url",
        required=False,
        help="Modal backend inference base URL (e.g., http://localhost:8000/api)",
    )
    parser.add_argument(
        "--task-app-key",
        default=None,
        help="Environment API key for the task app (fallback ENVIRONMENT_API_KEY)",
    )
    parser.add_argument(
        "--modal-key",
        default=None,
        help="Synth/Modal API key for inference (fallback SYNTH_API_KEY)",
    )
    parser.add_argument(
        "--max-llm-calls", type=int, default=20, help="Number of policy inference calls"
    )
    parser.add_argument(
        "--ops", default=None, help="Comma-separated rollout ops (advanced override)"
    )
    parser.add_argument(
        "--max-policy-tokens",
        type=int,
        default=None,
        help="Optional per-call token limit forwarded to the policy config",
    )
    parser.add_argument(
        "--verbose", action="store_true", help="Print resolved configuration and headers"
    )
    args = parser.parse_args()

    # Also load from explicit --env-file if provided
    if args.env_file:
        env_path = Path(args.env_file).expanduser()
        if not env_path.exists():
            print(f"[WARN] Env file not found: {env_path}")
        else:
            load_dotenv(env_path, override=False)

    # Prompt for required parameters if not provided
    base_url = args.base_url
    if args.base_url == "http://localhost:8010":
        print("\nTask app configuration:")
        base_url_input = input("Task app base URL [http://localhost:8001]: ").strip()
        base_url = base_url_input if base_url_input else "http://localhost:8001"

    model = args.model
    if not model:
        print("\nFine-tuned model configuration:")
        print(
            "Note: This should be the model ID returned from training (e.g., fft:Qwen/Qwen3-4B:job_abc123)"
        )
        model_input = input("Fine-tuned model ID: ").strip()
        if not model_input:
            parser.error("Model identifier is required")
        model = model_input

    inference_url = args.inference_url
    if not inference_url:
        inference_url_input = input("Inference URL [http://localhost:8000/api]: ").strip()
        inference_url = inference_url_input if inference_url_input else "http://localhost:8000/api"

    # Override args
    args.base_url = base_url
    args.model = model
    args.inference_url = inference_url

    # Check environment variables first (loaded from .env)
    task_app_key = args.task_app_key or os.getenv("ENVIRONMENT_API_KEY")
    if not task_app_key:
        print("\n[INFO] ENVIRONMENT_API_KEY not found in environment or .env file")
        task_app_key = input("RL Environment API key: ").strip()
        if not task_app_key:
            parser.error("Missing task app API key")

    modal_key = args.modal_key or os.getenv("SYNTH_API_KEY")
    if not modal_key:
        print("[INFO] SYNTH_API_KEY not found in environment or .env file")
        modal_key = input("Synth API key: ").strip()
        if not modal_key:
            parser.error("Missing Synth/Modal API key")

    if modal_key and "openai.com" not in args.inference_url.lower():
        os.environ["OPENAI_API_KEY"] = modal_key

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

    inf_url_norm = args.inference_url.rstrip("/")
    if "/api" not in inf_url_norm:
        print(
            "[WARN] Inference URL is missing /api prefix; proxy endpoints usually live at /api/inference/v1/chat/completions."
        )
    elif not inf_url_norm.lower().endswith("/api"):
        print(
            "[INFO] Using inference base URL; policy will append /v1/chat/completions automatically."
        )

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
                request.policy.config.update(
                    {
                        "max_completion_tokens": args.max_policy_tokens,
                        "max_tokens": args.max_policy_tokens,
                    }
                )
            print("Requesting rollout…")
            response = await client.rollout(request)
            summary = summarise_response(response)
            print(json.dumps(summary, indent=2))
            print(f"Ops executed: {ops}")
        except httpx.HTTPStatusError as exc:
            detail = (
                exc.response.json()
                if exc.response.headers.get("content-type", "").startswith("application/json")
                else exc.response.text
            )
            print(f"HTTP error {exc.response.status_code}: {detail}", file=sys.stderr)
            if exc.response.status_code in (401, 503):
                print(
                    "Hint: ensure ENVIRONMENT_API_KEY and SYNTH_API_KEY are correctly set.",
                    file=sys.stderr,
                )
            raise


if __name__ == "__main__":
    asyncio.run(main())
