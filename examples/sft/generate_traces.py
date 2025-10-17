#!/usr/bin/env python3
"""Generate Crafter rollouts and server-side traces via the Task App.

This script is a slim wrapper around the Task App `/rollout` endpoint to
produce trajectories while the server (if configured with TASKAPP_TRACING_ENABLED)
persists traces to its sqlite database. Use `export_dataset.py` afterwards
to build an SFT JSONL.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
import time
from dataclasses import dataclass
from typing import Any

from synth_ai.task import (
    RolloutEnvSpec,
    RolloutPolicySpec,
    RolloutRecordConfig,
    RolloutRequest,
    TaskAppClient,
)


def _ensure_str(val: Any, name: str) -> str:
    s = str(val or "").strip()
    if not s:
        raise SystemExit(f"Missing required {name}")
    return s


def _build_ops(max_llm_calls: int) -> list[str]:
    max_llm_calls = max(1, int(max_llm_calls or 1))
    ops: list[str] = []
    for _ in range(max_llm_calls):
        ops.extend(["agent", "env"])  # LLM step then env step
    return ops


def _build_request(seed: int, run_id: str, model: str, inference_url: str, api_key: str, *, max_llm_calls: int, return_trace: bool) -> RolloutRequest:
    policy_cfg: dict[str, Any] = {
        "model": model,
        "inference_url": inference_url,
        "api_key": api_key,
    }
    record = RolloutRecordConfig(trajectories=True, return_trace=bool(return_trace), trace_format="compact")
    return RolloutRequest(
        run_id=run_id,
        env=RolloutEnvSpec(env_name="crafter", seed=seed, config={}),
        policy=RolloutPolicySpec(policy_name="crafter-react", config=policy_cfg),
        ops=_build_ops(max_llm_calls),
        record=record,
    )


@dataclass(slots=True)
class Args:
    base_url: str
    api_key: str
    inference_url: str
    inference_api_key: str
    model: str
    episodes: int
    start_seed: int
    max_llm_calls: int
    concurrency: int
    return_trace: bool
    timeout: float


async def _run_one(client: TaskAppClient, run_id: str, seed: int, a: Args) -> dict[str, Any]:
    req = _build_request(
        seed=seed,
        run_id=f"{run_id}-seed{seed}",
        model=a.model,
        inference_url=a.inference_url,
        api_key=a.inference_api_key,
        max_llm_calls=a.max_llm_calls,
        return_trace=a.return_trace,
    )
    resp = await client.rollout(req)
    metrics = resp.metrics.model_dump()
    return {
        "seed": seed,
        "num_steps": metrics.get("num_steps"),
        "episode_returns": metrics.get("episode_returns"),
        "outcome_score": metrics.get("outcome_score"),
    }


async def _bounded_gather(n: int, coros: list[asyncio.Future]):
    sem = asyncio.Semaphore(n)

    async def _wrap(coro):
        async with sem:
            return await coro

    return await asyncio.gather(*[_wrap(c) for c in coros])


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default=os.getenv("TASK_APP_URL", "http://localhost:8001"))
    parser.add_argument("--api-key", default=os.getenv("ENVIRONMENT_API_KEY"))
    parser.add_argument("--inference-url", default=os.getenv("INFERENCE_URL", "https://api.groq.com/openai"))
    parser.add_argument("--inference-api-key", default=os.getenv("GROQ_API_KEY"))
    parser.add_argument("--model", default=os.getenv("POLICY_MODEL", "llama-3.3-70b-versatile"))
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--start-seed", type=int, default=0)
    parser.add_argument("--max-llm-calls", type=int, default=10)
    parser.add_argument("--concurrency", type=int, default=5)
    parser.add_argument("--return-trace", action="store_true")
    parser.add_argument("--timeout", type=float, default=60.0)
    args_ns = parser.parse_args()

    a = Args(
        base_url=_ensure_str(args_ns.base_url, "--base-url"),
        api_key=_ensure_str(args_ns.api_key, "--api-key"),
        inference_url=_ensure_str(args_ns.inference_url, "--inference-url"),
        inference_api_key=_ensure_str(args_ns.inference_api_key, "--inference-api-key"),
        model=_ensure_str(args_ns.model, "--model"),
        episodes=int(args_ns.episodes),
        start_seed=int(args_ns.start_seed),
        max_llm_calls=int(args_ns.max_llm_calls),
        concurrency=max(1, int(args_ns.concurrency)),
        return_trace=bool(args_ns.return_trace),
        timeout=float(args_ns.timeout),
    )

    print(
        f"[INFO] base={a.base_url} episodes={a.episodes} start_seed={a.start_seed} model={a.model} tp={a.max_llm_calls}"
    )
    run_id = f"traces-{int(time.time())}"

    successes = 0
    failures = 0
    async with TaskAppClient(a.base_url, api_key=a.api_key, timeout=a.timeout) as client:
        tasks = [
            _run_one(client, run_id, seed, a) for seed in range(a.start_seed, a.start_seed + a.episodes)
        ]
        for result in await _bounded_gather(a.concurrency, tasks):
            if isinstance(result, dict):
                successes += 1
                print(f"[OK] seed={result['seed']} return={result.get('episode_returns')}")
            else:
                failures += 1
                print(f"[ERR] seed result not dict: {result}", file=sys.stderr)

    print(f"[DONE] successes={successes} failures={failures}")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("Interrupted", file=sys.stderr)


