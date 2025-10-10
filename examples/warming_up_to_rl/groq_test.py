"""Quick smoke test that drives a rollout through the Groq proxy-backed Crafter Task App."""

from __future__ import annotations

import argparse
import asyncio
import os
from typing import Any

from synth_ai.task import (
    INTERACT_TOOL_SCHEMA,
    RolloutEnvSpec,
    RolloutPolicySpec,
    RolloutRequest,
    TaskAppClient,
    to_jsonable,
)


def _build_policy_payload(seed: int, model: str) -> dict[str, Any]:
    return {
        "model": model,
        "tools": INTERACT_TOOL_SCHEMA,
        "messages": [
            {
                "role": "system",
                "content": "You control the Crafter agent. Think briefly, then call the interact tool with 3-5 actions to maximize achievements.",
            },
            {
                "role": "user",
                "content": (
                    f"Environment seed {seed}. Plan initial survival/crafting steps and then call interact with concrete actions."
                ),
            },
        ],
    }


async def run(args: argparse.Namespace) -> None:
    client = TaskAppClient(args.base_url, api_key=args.api_key, timeout=args.timeout)

    health = await client.health()
    print("/health →", to_jsonable(health))

    info = await client.info()
    print("/info →", to_jsonable(info))

    inference_url = args.inference_url or f"{args.base_url.rstrip('/')}/proxy/groq"

    request = RolloutRequest(
        run_id=args.run_id,
        env=RolloutEnvSpec(env_name="crafter", seed=args.seed, config={"seed": args.seed}),
        policy=RolloutPolicySpec(
            policy_name="groq-smoke",
            config={"model": args.model, "inference_url": inference_url.rstrip("/")},
        ),
        ops=[
            {"type": "policy", "payload": _build_policy_payload(args.seed, args.model)},
            {"type": "env"},
        ],
    )

    response = await client.rollout(request)
    print("rollout.metrics →", to_jsonable(response.metrics.model_dump()))
    for idx, step in enumerate(response.trajectories[0].steps, start=1):
        print(
            f"step[{idx}] tool_calls={step.tool_calls} reward={step.reward} info={to_jsonable(step.info)}"
        )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base-url", default=os.getenv("TASK_APP_BASE_URL", "http://localhost:8000")
    )
    parser.add_argument(
        "--api-key",
        default=os.getenv("TASK_APP_API_KEY"),
        required=os.getenv("TASK_APP_API_KEY") is None,
    )
    parser.add_argument("--model", default=os.getenv("GROQ_MODEL", "groq/mixtral-8x7b"))
    parser.add_argument("--inference-url", default=os.getenv("TASK_APP_INFERENCE_URL"))
    parser.add_argument("--seed", type=int, default=int(os.getenv("CRAFTER_TEST_SEED", "42")))
    parser.add_argument("--run-id", default=os.getenv("TASK_APP_RUN_ID", "groq-test"))
    parser.add_argument("--timeout", type=float, default=float(os.getenv("TASK_APP_TIMEOUT", "60")))
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
