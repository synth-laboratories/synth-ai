#!/usr/bin/env python3
"""Hit a locally running Crafter task app and request a rollout."""

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
    seed: int,
    run_id: str,
    *,
    model: str,
    inference_url: str,
    ops: list[str],
    extra_headers: dict[str, str] | None = None,
    trace_format: str = "compact",
    return_trace: bool = False,
) -> RolloutRequest:
    policy_config = {"model": model, "inference_url": inference_url}
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


def summarise_response(data: Any) -> dict[str, Any]:
    metrics = (
        data.metrics.model_dump()
        if hasattr(data.metrics, "model_dump")
        else data.get("metrics", {})
    )
    error = None
    rollout_status = None
    try:
        trajectories = getattr(data, "trajectories", None) or data.get("trajectories")
        if isinstance(trajectories, list) and trajectories:
            final = getattr(trajectories[0], "final", None)
            if not final and isinstance(trajectories[0], dict):
                final = trajectories[0].get("final")
            if isinstance(final, dict):
                error = final.get("error")
                rollout_status = final.get("rollout_status")
    except Exception:
        pass
    return {
        "run_id": getattr(data, "run_id", None) or data.get("run_id"),
        "num_episodes": metrics.get("num_episodes"),
        "num_steps": metrics.get("num_steps"),
        "episode_returns": metrics.get("episode_returns"),
        "outcome_score": metrics.get("outcome_score"),
        "events_score": metrics.get("events_score"),
        "rollout_status": rollout_status,
        "error": error,
    }


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://localhost:8001", help="Task app base URL")
    parser.add_argument("--api-key", help="Environment API key (or set via --env-file)")
    parser.add_argument("--seed", type=int, default=42, help="Env seed to rollout")
    parser.add_argument("--run-id", default="local-demo", help="Run identifier")
    parser.add_argument(
        "--model",
        default="gpt-4o-mini",
        help="Model identifier for the Crafter policy (OpenAI-compatible)",
    )
    parser.add_argument(
        "--inference-url",
        default="https://api.openai.com",
        help="Inference base URL used by the policy (e.g., https://api.openai.com)",
    )
    parser.add_argument(
        "--env-file", type=str, default=None, help="Path to .env file with API keys"
    )
    parser.add_argument(
        "--ops", default=None, help="Comma-separated rollout ops (advanced override)"
    )
    parser.add_argument(
        "--max-llm-calls",
        type=int,
        default=1,
        help="Number of policy inference calls when --ops not provided",
    )
    parser.add_argument(
        "--max-policy-tokens",
        type=int,
        default=None,
        help="Optional per-call token limit forwarded to the policy config",
    )
    parser.add_argument(
        "--timeout", type=float, default=600.0, help="HTTP timeout (seconds) for task app requests"
    )
    parser.add_argument(
        "--verbose", action="store_true", help="Print resolved configuration and headers"
    )
    args = parser.parse_args()

    if args.env_file:
        env_path = Path(args.env_file).expanduser()
        if not env_path.exists():
            print(f"[WARN] Env file not found: {env_path}")
        else:
            load_dotenv(env_path, override=False)

    api_key = args.api_key or os.getenv("ENVIRONMENT_API_KEY")
    if not api_key:
        parser.error("Missing --api-key (or ENVIRONMENT_API_KEY not set)")

    extra_headers: dict[str, str] | None = None
    synth_key = os.getenv("SYNTH_API_KEY")
    if synth_key:
        extra_headers = {"Authorization": f"Bearer {synth_key}"}
        if "openai.com" not in args.inference_url.lower():
            os.environ["OPENAI_API_KEY"] = synth_key

    if args.verbose:

        def _mask(val: str | None) -> str:
            if not val:
                return "<unset>"
            return f"{val[:6]}…{val[-4:]} (len={len(val)})"

        print("Resolved configuration:")
        print(f"  Task app base URL  : {args.base_url}")
        print(f"  Inference base URL : {args.inference_url}")
        print(f"  Task app API key   : {_mask(api_key)}")
        print(f"  Synth API key      : {_mask(synth_key)}")
        print(f"  HTTP timeout       : {args.timeout:.1f}s")

    if args.ops:
        ops = [op.strip() for op in args.ops.split(",") if op.strip()]
        if not ops:
            raise ValueError("Ops must contain at least one entry")
    else:
        llm_calls = max(args.max_llm_calls, 1)
        if llm_calls > 20:
            print(
                "[WARN] --max-llm-calls capped at 20 to avoid excessive episodes; use --ops for manual control."
            )
            llm_calls = 20
        ops = []
        for _ in range(llm_calls):
            ops.extend(["agent", "env"])

    async with TaskAppClient(args.base_url, api_key=api_key, timeout=args.timeout) as client:
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
                extra_headers=extra_headers,
            )
            if args.max_policy_tokens is not None:
                request.policy.config.update(
                    {
                        "max_completion_tokens": args.max_policy_tokens,
                        "max_tokens": args.max_policy_tokens,
                    }
                )
            if args.verbose:
                print(f"Ops: {ops}")
                print(f"Request headers: {request.policy.config.get('extra_headers', {})}")
            print("Requesting rollout…")
            response = await client.rollout(request)
            summary = summarise_response(response)
            print(json.dumps(summary, indent=2))
            print(f"Ops executed: {ops}")
            print("Tip: use --max-llm-calls N for agent/env pairs or --ops for manual control.")
        except httpx.HTTPStatusError as exc:
            detail = (
                exc.response.json()
                if exc.response.headers.get("content-type", "").startswith("application/json")
                else exc.response.text
            )
            print(f"HTTP error {exc.response.status_code}: {detail}", file=sys.stderr)
            if exc.response.status_code in (401, 503):
                print(
                    "Hint: ensure the task app was started with ENVIRONMENT_API_KEY set and pass the same key via --api-key.",
                    file=sys.stderr,
                )
            if exc.response.status_code == 500 and args.model in str(detail):
                print(
                    "Hint: supply --model/--inference-url (and set OPENAI_API_KEY or GROQ_API_KEY) so the policy can route inference.",
                    file=sys.stderr,
                )
                print(
                    "Hint: the inference URL should be the base (e.g., https://api.openai.com); the task app appends /v1/chat/completions.",
                    file=sys.stderr,
                )
                if args.max_policy_tokens is not None:
                    print(
                        f"Hint: --max-policy-tokens={args.max_policy_tokens} is forwarded to the policy config as max_completion_tokens.",
                        file=sys.stderr,
                    )
            raise


if __name__ == "__main__":
    asyncio.run(main())
