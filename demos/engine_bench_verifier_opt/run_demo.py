#!/usr/bin/env python3
"""Optimize a verifier graph for Rust coding-agent traces from EngineBench.

Usage:
    uv run python demos/engine_bench_verifier_opt/run_demo.py --local
    uv run python demos/engine_bench_verifier_opt/run_demo.py --eval-dir /abs/path/to/eval_dir
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import httpx
from dotenv import load_dotenv
from synth_ai.core.env import PROD_BASE_URL
from synth_ai.sdk.api.train.graphgen import GraphGenJob
from synth_ai.sdk.api.train.graphgen_models import (
    GraphGenGoldOutput,
    GraphGenTask,
    GraphGenTaskSet,
    GraphGenTaskSetMetadata,
    GraphGenVerifierConfig,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Optimize a verifier graph for EngineBench Rust traces."
    )
    parser.add_argument(
        "--local",
        action="store_true",
        help="Use local backend (http://localhost:8000).",
    )
    parser.add_argument(
        "--local-host",
        type=str,
        default="localhost",
        help="Hostname for local API URLs (use 'host.docker.internal' if backend runs in Docker).",
    )
    parser.add_argument(
        "--agent",
        type=str,
        default="opencode",
        choices=["opencode", "codex"],
        help="Which EngineBench agent traces to load.",
    )
    parser.add_argument(
        "--eval-dir",
        type=str,
        default=None,
        help="Absolute path to an EngineBench eval directory with eval_results.json and traces/.",
    )
    parser.add_argument(
        "--max-traces",
        type=int,
        default=20,
        help="Maximum number of traces to use for optimization.",
    )
    parser.add_argument(
        "--generations",
        type=int,
        default=2,
        help="Number of generations for optimization.",
    )
    parser.add_argument(
        "--children",
        type=int,
        default=4,
        help="Children per generation (population size).",
    )
    parser.add_argument(
        "--rollout-budget",
        type=int,
        default=60,
        help="Rollout budget for optimization.",
    )
    parser.add_argument(
        "--policy-model",
        type=str,
        default="gpt-4o-mini",
        help="Model for verifier graph execution.",
    )
    parser.add_argument(
        "--verifier-model",
        type=str,
        default="gpt-4o-mini",
        help="Model for verifier evaluation.",
    )
    parser.add_argument(
        "--max-items",
        type=int,
        default=12,
        help="Max list items kept during trace truncation.",
    )
    parser.add_argument(
        "--max-chars",
        type=int,
        default=2400,
        help="Max characters kept per string during trace truncation.",
    )
    return parser.parse_args()


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _resolve_eval_dir(agent: str, eval_dir: Optional[str]) -> Path:
    if eval_dir:
        resolved = Path(eval_dir).expanduser().resolve()
        if not resolved.exists():
            raise FileNotFoundError(f"Eval dir not found: {resolved}")
        return resolved

    base_dir = _repo_root() / "data" / "engine_bench" / agent
    if not base_dir.exists():
        raise FileNotFoundError(f"EngineBench data not found: {base_dir}")

    candidates = [p for p in base_dir.iterdir() if p.is_dir()]
    if not candidates:
        raise FileNotFoundError(f"No eval runs found under: {base_dir}")

    return sorted(candidates, key=lambda p: p.name)[-1]


def _load_eval_results(eval_dir: Path) -> Dict[int, Dict[str, Any]]:
    eval_path = eval_dir / "eval_results.json"
    if not eval_path.exists():
        raise FileNotFoundError(f"Missing eval_results.json at {eval_path}")

    data = json.loads(eval_path.read_text(encoding="utf-8"))
    seed_results = data.get("seed_results") or []

    results: Dict[int, Dict[str, Any]] = {}
    for row in seed_results:
        if not isinstance(row, dict):
            continue
        seed = row.get("seed")
        reward = row.get("reward")
        if isinstance(seed, int) and isinstance(reward, (int, float)):
            results[seed] = row

    if not results:
        raise ValueError(f"No usable rewards found in {eval_path}")

    return results


def _truncate_for_context(value: Any, *, max_items: int, max_chars: int) -> Any:
    if isinstance(value, list):
        if len(value) > max_items:
            trimmed = value[:max_items]
            return [
                _truncate_for_context(item, max_items=max_items, max_chars=max_chars)
                for item in trimmed
            ] + [{"__truncated__": f"... {len(value) - max_items} more items ..."}]
        return [
            _truncate_for_context(item, max_items=max_items, max_chars=max_chars) for item in value
        ]
    if isinstance(value, dict):
        return {
            key: _truncate_for_context(val, max_items=max_items, max_chars=max_chars)
            for key, val in value.items()
        }
    if isinstance(value, str):
        if len(value) > max_chars:
            return value[:max_chars] + f"... [truncated {len(value) - max_chars} chars]"
        return value
    return value


def _load_traces(
    trace_dir: Path, seeds: Iterable[int], *, max_items: int, max_chars: int
) -> Dict[int, Dict[str, Any]]:
    traces: Dict[int, Dict[str, Any]] = {}
    for seed in seeds:
        trace_path = trace_dir / f"seed_{seed}.json"
        if not trace_path.exists():
            continue
        raw_trace = json.loads(trace_path.read_text(encoding="utf-8"))
        traces[seed] = _truncate_for_context(raw_trace, max_items=max_items, max_chars=max_chars)
    return traces


def _split_train_val(
    seeds: List[int], train_ratio: float = 0.8
) -> Tuple[List[int], List[int]]:
    split_idx = max(1, int(len(seeds) * train_ratio))
    return seeds[:split_idx], seeds[split_idx:]


def _build_graphgen_dataset(
    *,
    eval_dir: Path,
    seeds: List[int],
    seed_results: Dict[int, Dict[str, Any]],
    traces: Dict[int, Dict[str, Any]],
) -> GraphGenTaskSet:
    tasks: List[GraphGenTask] = []
    gold_outputs: List[GraphGenGoldOutput] = []

    for seed in seeds:
        trace = traces.get(seed)
        if trace is None:
            continue
        seed_row = seed_results.get(seed, {})
        reward = float(seed_row.get("reward", 0.0))
        trial_id = seed_row.get("trial_id", f"seed_{seed}")

        trace_id = trace.get("id") or seed_row.get("trace_id") or f"seed_{seed}"

        tasks.append(
            GraphGenTask(
                id=str(trace_id),
                input={
                    "trace": trace,
                    "trace_id": str(trace_id),
                    "seed": seed,
                    "trial_id": trial_id,
                },
            )
        )

        gold_outputs.append(
            GraphGenGoldOutput(
                task_id=str(trace_id),
                output={
                    "outcome_reward": reward,
                    "outcome_feedback": "Reward derived from deterministic EngineBench tests.",
                    "event_rewards": [],
                    "reward_details": {
                        "seed": seed,
                        "trial_id": trial_id,
                        "eval_dir": str(eval_dir),
                    },
                },
            )
        )

    input_schema = {
        "type": "object",
        "properties": {
            "trace": {
                "type": "object",
                "description": "EngineBench trace from a Rust coding agent.",
            },
            "trace_id": {"type": "string"},
            "seed": {"type": "number"},
            "trial_id": {"type": "string"},
        },
        "required": ["trace", "trace_id"],
    }

    output_schema = {
        "type": "object",
        "properties": {
            "outcome_reward": {
                "type": "number",
                "minimum": 0.0,
                "maximum": 1.0,
                "description": "Overall reward estimate for the trace.",
            },
            "outcome_feedback": {"type": "string", "description": "Summary feedback."},
            "event_rewards": {
                "type": "array",
                "items": {"type": "number"},
                "description": "Optional per-event rewards aligned with trace events.",
            },
        },
        "required": ["outcome_reward", "outcome_feedback"],
    }

    return GraphGenTaskSet(
        metadata=GraphGenTaskSetMetadata(
            name="enginebench_verifier_optimization",
            description=(
                "Train a verifier graph to estimate rewards for Rust coding agent traces."
            ),
            input_schema=input_schema,
            output_schema=output_schema,
        ),
        tasks=tasks,
        gold_outputs=gold_outputs,
        verifier_config=GraphGenVerifierConfig(
            mode="gold_examples",
            model="gpt-4o-mini",
            provider="openai",
        ),
        input_schema=input_schema,
        output_schema=output_schema,
    )


def main() -> None:
    args = _parse_args()

    repo_root = _repo_root()
    env_path = repo_root / ".env"
    if env_path.exists():
        load_dotenv(env_path, override=True)

    synth_api_base = "http://localhost:8000" if args.local else PROD_BASE_URL

    print("=" * 60)
    print("ENGINEBENCH VERIFIER OPTIMIZATION")
    print("=" * 60)
    print(f"Backend: {synth_api_base}")

    try:
        response = httpx.get(f"{synth_api_base}/health", timeout=30)
        if response.status_code == 200:
            print(f"Backend health: {response.json()}")
        else:
            print(f"WARNING: Backend returned status {response.status_code}")
    except Exception as exc:
        print(f"WARNING: Could not check backend health: {exc}")

    api_key = os.environ.get("SYNTH_API_KEY", "")
    if not api_key:
        print("ERROR: SYNTH_API_KEY not found in environment.")
        sys.exit(1)

    eval_dir = _resolve_eval_dir(args.agent, args.eval_dir)
    trace_dir = eval_dir / "traces"

    seed_results = _load_eval_results(eval_dir)
    sorted_seeds = sorted(seed_results.keys())
    if args.max_traces and len(sorted_seeds) > args.max_traces:
        sorted_seeds = sorted_seeds[: args.max_traces]

    traces = _load_traces(
        trace_dir, sorted_seeds, max_items=args.max_items, max_chars=args.max_chars
    )
    available_seeds = [seed for seed in sorted_seeds if seed in traces]
    if not available_seeds:
        raise RuntimeError("No traces found for the selected seeds.")

    train_seeds, val_seeds = _split_train_val(available_seeds)

    print(f"Eval dir: {eval_dir}")
    print(f"Trace dir: {trace_dir}")
    print(f"Train seeds: {len(train_seeds)} | Val seeds: {len(val_seeds)}")

    dataset = _build_graphgen_dataset(
        eval_dir=eval_dir,
        seeds=train_seeds,
        seed_results=seed_results,
        traces=traces,
    )

    print("\nSubmitting GraphGen job...")
    start = time.time()
    job = GraphGenJob.from_dataset(
        dataset=dataset,
        backend_url=synth_api_base,
        api_key=api_key,
        graph_type="verifier",
        policy_models=[args.policy_model],
        judge_model=args.verifier_model,
        rollout_budget=args.rollout_budget,
        population_size=max(2, args.children),
        num_generations=args.generations,
    )

    job_ids = job.submit()
    print(f"GraphGen job: {job_ids.graphgen_job_id}")
    print(f"Graph evolve job: {job_ids.graph_evolve_job_id}")

    print("\nStreaming progress...")
    job.stream_until_complete(timeout=3600.0)
    duration = time.time() - start
    print("\nOptimization finished.")
    print(f"Duration: {duration:.1f}s")

    # Optional: run inference on a holdout trace
    if val_seeds:
        seed = val_seeds[0]
        trace = traces.get(seed)
        if trace:
            print("\nHoldout inference sample:")
            inference = job.run_inference(
                input_data={
                    "trace": trace,
                    "trace_id": f"seed_{seed}",
                    "seed": seed,
                }
            )
            print(json.dumps(inference, indent=2, default=str))


if __name__ == "__main__":
    main()
