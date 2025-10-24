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
from synth_ai._utils.user_config import load_user_config, update_user_config
from synth_ai.task import TaskAppClient

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.append(str(SCRIPT_DIR))

from shared import build_rollout_request, ops_from_pairs, parse_ops  # noqa: E402


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
    parser.add_argument("--api-key", help="Environment API key (or stored in user config)")
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

    user_config = load_user_config()

    api_key = (
        args.api_key
        or os.getenv("ENVIRONMENT_API_KEY")
        or user_config.get("ENVIRONMENT_API_KEY")
        or user_config.get("DEV_ENVIRONMENT_API_KEY")
    )
    if not api_key:
        parser.error("Missing --api-key (or ENVIRONMENT_API_KEY not set)")
    else:
        os.environ.setdefault("ENVIRONMENT_API_KEY", api_key)
        if (
            user_config.get("ENVIRONMENT_API_KEY") != api_key
            or user_config.get("DEV_ENVIRONMENT_API_KEY") != api_key
        ):
            update_user_config(
                {
                    "ENVIRONMENT_API_KEY": api_key,
                    "DEV_ENVIRONMENT_API_KEY": api_key,
                }
            )
            user_config["ENVIRONMENT_API_KEY"] = api_key
            user_config["DEV_ENVIRONMENT_API_KEY"] = api_key

    openai_key = (
        os.getenv("OPENAI_API_KEY")
        or user_config.get("OPENAI_API_KEY")
        or user_config.get("DEV_OPENAI_API_KEY")
        or ""
    ).strip()
    needs_openai = "openai" in (args.inference_url or "").lower()
    if needs_openai and not openai_key:
        try:
            openai_key = (
                input(
                    "OPENAI_API_KEY not found. Enter your OpenAI API key (required for local rollouts):\n> "
                )
                .strip()
            )
        except (KeyboardInterrupt, EOFError):
            openai_key = ""
        if openai_key:
            update_user_config(
                {
                    "OPENAI_API_KEY": openai_key,
                    "DEV_OPENAI_API_KEY": openai_key,
                }
            )
            user_config["OPENAI_API_KEY"] = openai_key
            user_config["DEV_OPENAI_API_KEY"] = openai_key
    if needs_openai and not openai_key:
        parser.error("OPENAI_API_KEY is required. Export it before running or provide it when prompted.")
    if openai_key:
        os.environ["OPENAI_API_KEY"] = openai_key

    extra_headers: dict[str, str] | None = None
    synth_key = (
        os.getenv("SYNTH_API_KEY")
        or user_config.get("SYNTH_API_KEY")
        or user_config.get("DEV_SYNTH_API_KEY")
    )
    if synth_key:
        os.environ.setdefault("SYNTH_API_KEY", synth_key)
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

    explicit_ops = parse_ops(args.ops)
    if explicit_ops is not None:
        ops = explicit_ops
    else:
        llm_calls = max(args.max_llm_calls, 1)
        if llm_calls > 20:
            print(
                "[WARN] --max-llm-calls capped at 20 to avoid excessive episodes; use --ops for manual control."
            )
        ops = ops_from_pairs(llm_calls, cap=20)

    async with TaskAppClient(args.base_url, api_key=api_key, timeout=args.timeout) as client:
        try:
            print(f"Fetching task_info for seed {args.seed}…")
            task_info = await client.task_info(seeds=[args.seed])
            info_payload = task_info[0] if isinstance(task_info, list) else task_info
            print(json.dumps(info_payload.model_dump(), indent=2)[:600])

            request = build_rollout_request(
                seed=args.seed,
                run_id=args.run_id,
                model=args.model,
                inference_url=args.inference_url,
                ops=ops,
                extra_headers=extra_headers,
                max_policy_tokens=args.max_policy_tokens,
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
            print("\nWhat's next:")
            print("  uvx python run_local_rollout_traced.py")
            print("  Inspect trace outputs under ./traces/")
            print("  Stop the local server with Ctrl+C in the deploy terminal")
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
