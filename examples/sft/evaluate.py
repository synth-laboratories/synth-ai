#!/usr/bin/env python3
"""Evaluate a base or fine-tuned model on Crafter via the Task App rollout.

This mirrors the minimal evaluation loop: call `/rollout` for a set of seeds
and report outcome/step metrics. If tracing is enabled server-side, you can
use the exported sqlite DB for further analysis.
"""

from __future__ import annotations

import argparse
import asyncio
import os
from dataclasses import dataclass
from typing import Any

from synth_ai.task import (
    RolloutEnvSpec,
    RolloutPolicySpec,
    RolloutRecordConfig,
    RolloutRequest,
    TaskAppClient,
)


@dataclass(slots=True)
class EvalArgs:
    base_url: str
    api_key: str
    model: str
    inference_url: str
    inference_api_key: str
    seeds: list[int]
    max_llm_calls: int
    timeout: float


def _ops(n: int) -> list[str]:
    n = max(1, n)
    ops: list[str] = []
    for _ in range(n):
        ops.extend(["agent", "env"])  # one LLM step followed by one env step
    return ops


def _request(seed: int, a: EvalArgs) -> RolloutRequest:
    return RolloutRequest(
        run_id=f"eval-{seed}",
        env=RolloutEnvSpec(env_name="crafter", seed=seed, config={}),
        policy=RolloutPolicySpec(
            policy_name="crafter-react",
            config={"model": a.model, "inference_url": a.inference_url, "api_key": a.inference_api_key},
        ),
        ops=_ops(a.max_llm_calls),
        record=RolloutRecordConfig(trajectories=True, return_trace=False, trace_format="compact"),
    )


async def _eval_seed(client: TaskAppClient, seed: int, a: EvalArgs) -> dict[str, Any]:
    resp = await client.rollout(_request(seed, a))
    m = resp.metrics
    return {"seed": seed, "num_steps": m.num_steps, "episode_returns": m.episode_returns, "outcome_score": m.outcome_score}


async def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base-url", default=os.getenv("TASK_APP_URL", "http://localhost:8001"))
    p.add_argument("--api-key", default=os.getenv("ENVIRONMENT_API_KEY"))
    p.add_argument("--model", required=True, help="Base or ft:<id> to evaluate")
    p.add_argument("--inference-url", default=os.getenv("INFERENCE_URL", "https://api.groq.com/openai"))
    p.add_argument("--inference-api-key", default=os.getenv("GROQ_API_KEY"))
    p.add_argument("--seeds", default="0,1,2,3,4,5,6,7,8,9")
    p.add_argument("--max-llm-calls", type=int, default=10)
    p.add_argument("--timeout", type=float, default=60.0)
    args = p.parse_args()

    seeds = [int(s) for s in str(args.seeds).split(",") if s.strip()]
    a = EvalArgs(
        base_url=str(args.base_url).strip(),
        api_key=str(args.api_key or "").strip(),
        model=str(args.model).strip(),
        inference_url=str(args.inference_url).strip(),
        inference_api_key=str(args.inference_api_key or "").strip(),
        seeds=seeds,
        max_llm_calls=int(args.max_llm_calls),
        timeout=float(args.timeout),
    )
    if not a.api_key:
        raise SystemExit("ENVIRONMENT_API_KEY is required")
    if not a.inference_api_key:
        raise SystemExit("Inference API key (e.g., GROQ_API_KEY) is required")

    results: list[dict[str, Any]] = []
    async with TaskAppClient(a.base_url, api_key=a.api_key, timeout=a.timeout) as client:
        for seed in a.seeds:
            r = await _eval_seed(client, seed, a)
            results.append(r)
            print(f"seed={seed} return={r.get('episode_returns')}")

    # Simple aggregate
    flat_returns: list[float] = []
    for r in results:
        ers = r.get("episode_returns") or []
        if isinstance(ers, list) and ers:
            try:
                flat_returns.append(float(ers[0]))
            except Exception:
                pass
    if flat_returns:
        mean_ret = sum(flat_returns) / len(flat_returns)
        print(f"mean_return={mean_ret:.3f} over {len(flat_returns)} episodes")


if __name__ == "__main__":
    asyncio.run(main())


