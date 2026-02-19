#!/usr/bin/env python3
"""Full HTTP E2E: GEPA optimization + Eval jobs + prompt deliverable.

This script performs an end-to-end run over HTTP using the Synth SDK:
1) Starts a local container task app over HTTP.
2) Submits a GEPA prompt-optimization job.
3) Retrieves optimized prompts.
4) Runs two Eval jobs (baseline prompt and optimized prompt).
5) Writes a deliverable artifact (JSON + prompt text files).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx
from synth_ai.data.enums import SuccessStatus
from synth_ai.sdk.container import InProcessContainer, ensure_container_auth
from synth_ai.sdk.container.contracts import (
    RolloutMetrics,
    RolloutRequest,
    RolloutResponse,
    TaskInfo,
)
from synth_ai.sdk.container.server import ContainerConfig, create_container
from synth_ai.sdk.eval import EvalJob, EvalJobConfig
from synth_ai.sdk.optimization.internal.learning.prompt_learning_client import get_prompts
from synth_ai.sdk.optimization.policy import PolicyOptimizationJob

APP_ID = "intent_router_http_demo"
APP_NAME = "Intent Router HTTP Demo"
LABELS = ("card_lost", "transfer_failed", "atm_withdrawal_issue")

BASELINE_SYSTEM_PROMPT = (
    "You are an intent classifier for a fintech support team. "
    "Return exactly one label from this set: card_lost, transfer_failed, atm_withdrawal_issue."
)
USER_PATTERN = (
    "Customer query: {query}\n"
    "Available labels: card_lost, transfer_failed, atm_withdrawal_issue\n"
    "Respond with exactly one label."
)

TRAIN_DATA = [
    {"query": "I misplaced my debit card and need to freeze it", "label": "card_lost"},
    {"query": "My wallet was stolen and my card is gone", "label": "card_lost"},
    {"query": "I cannot find my credit card after traveling", "label": "card_lost"},
    {"query": "Lost card, please block spending immediately", "label": "card_lost"},
    {"query": "Bank transfer failed and money bounced back", "label": "transfer_failed"},
    {"query": "Wire transfer is stuck and recipient got nothing", "label": "transfer_failed"},
    {"query": "I sent a transfer but it was declined", "label": "transfer_failed"},
    {"query": "Transfer did not go through to my landlord", "label": "transfer_failed"},
    {"query": "ATM charged me but no cash came out", "label": "atm_withdrawal_issue"},
    {"query": "Cash withdrawal failed at the machine", "label": "atm_withdrawal_issue"},
    {"query": "ATM gave the wrong amount of cash", "label": "atm_withdrawal_issue"},
    {"query": "The ATM kept my card during withdrawal", "label": "atm_withdrawal_issue"},
]

VAL_DATA = [
    {"query": "I lost my bank card yesterday", "label": "card_lost"},
    {"query": "Please freeze card, I cannot find it", "label": "card_lost"},
    {"query": "Transfer to my friend failed twice", "label": "transfer_failed"},
    {"query": "Payment transfer got rejected", "label": "transfer_failed"},
    {"query": "ATM swallowed my card and gave no money", "label": "atm_withdrawal_issue"},
    {"query": "Cash withdrawal at ATM failed", "label": "atm_withdrawal_issue"},
]

TEST_DATA = [
    {"query": "Need help, my card is missing", "label": "card_lost"},
    {"query": "Card stolen, block it now", "label": "card_lost"},
    {"query": "Recipient never received my transfer", "label": "transfer_failed"},
    {"query": "Transfer keeps failing every attempt", "label": "transfer_failed"},
    {"query": "ATM debited me without dispensing cash", "label": "atm_withdrawal_issue"},
    {"query": "Withdrawal issue at ATM this morning", "label": "atm_withdrawal_issue"},
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run GEPA + Eval E2E over HTTP via SDK")
    parser.add_argument(
        "--backend-base",
        default="http://localhost:8000",
        help="Synth backend base URL (HTTP).",
    )
    parser.add_argument(
        "--api-key",
        required=False,
        help="Synth API key (falls back to SYNTH_API_KEY env var).",
    )
    parser.add_argument(
        "--model",
        default="gpt-4.1-nano",
        help="Model used for rollout inference.",
    )
    parser.add_argument(
        "--rollout-budget",
        type=int,
        default=24,
        help="GEPA rollout budget.",
    )
    parser.add_argument(
        "--generations",
        type=int,
        default=2,
        help="GEPA generations.",
    )
    parser.add_argument(
        "--minibatch-size",
        type=int,
        default=3,
        help="GEPA minibatch size.",
    )
    parser.add_argument(
        "--max-concurrent",
        type=int,
        default=6,
        help="GEPA max concurrent rollouts.",
    )
    parser.add_argument(
        "--gepa-timeout",
        type=float,
        default=1800.0,
        help="Seconds before GEPA polling times out.",
    )
    parser.add_argument(
        "--eval-timeout",
        type=float,
        default=900.0,
        help="Seconds before eval polling times out.",
    )
    parser.add_argument(
        "--poll-interval",
        type=float,
        default=5.0,
        help="Polling interval for GEPA and Eval jobs.",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8114,
        help="Preferred local container port.",
    )
    parser.add_argument(
        "--output-dir",
        default="artifacts/http_gepa_eval",
        help="Directory where deliverable artifacts are written.",
    )
    parser.add_argument(
        "--compact-json",
        action="store_true",
        help="Print compact JSON summary to stdout.",
    )
    return parser.parse_args()


def _dataset_for_split(split: str) -> list[dict[str, str]]:
    if split == "train":
        return TRAIN_DATA
    if split == "val":
        return VAL_DATA
    if split == "test":
        return TEST_DATA
    raise ValueError(f"Unknown split: {split}")


def _normalize_label(raw: str) -> str:
    text = raw.strip().lower().replace("-", "_")
    text = " ".join(text.split())
    text = text.replace(" ", "_")
    return text


def _extract_label(raw: str) -> str:
    normalized = _normalize_label(raw)
    for label in LABELS:
        if label in normalized:
            return label
    return ""


async def _classify_via_backend_http(
    *,
    backend_base: str,
    api_key: str,
    model: str,
    system_prompt: str,
    query: str,
) -> str:
    prompt = (
        f"Customer query: {query}\n"
        f"Labels: {', '.join(LABELS)}\n"
        "Return exactly one label and nothing else."
    )
    endpoint = f"{backend_base.rstrip('/')}/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "X-API-Key": api_key,
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.0,
        "max_completion_tokens": 24,
    }
    async with httpx.AsyncClient(timeout=60.0) as client:
        response = await client.post(endpoint, headers=headers, json=payload)
    response.raise_for_status()
    data = response.json()
    content = (
        data.get("choices", [{}])[0]
        .get("message", {})
        .get("content", "")
    )
    if not isinstance(content, str):
        content = str(content)
    label = _extract_label(content)
    if label:
        return label
    return LABELS[0]


def _build_task_app(*, backend_base: str, api_key: str, default_model: str) -> Any:
    debug_counter = {"count": 0}

    async def rollout(request: RolloutRequest, fastapi_request: Any) -> RolloutResponse:
        del fastapi_request
        split = str((request.env.config or {}).get("split") or "train")
        dataset = _dataset_for_split(split)
        seed = int(request.env.seed or 0)
        sample = dataset[seed % len(dataset)]

        policy_config = request.policy.config or {}
        model = str(policy_config.get("model") or default_model)
        active_system_prompt = str(
            policy_config.get("system_prompt")
            or policy_config.get("instruction")
            or policy_config.get("prompt")
            or BASELINE_SYSTEM_PROMPT
        )
        inference_api_key = str(policy_config.get("api_key") or api_key)
        inference_base = str(policy_config.get("inference_url") or f"{backend_base.rstrip('/')}/v1")
        inference_base = inference_base.removesuffix("/chat/completions")
        inference_base = inference_base.removesuffix("/v1")

        predicted_label = await _classify_via_backend_http(
            backend_base=inference_base,
            api_key=inference_api_key,
            model=model,
            system_prompt=active_system_prompt,
            query=sample["query"],
        )
        expected_label = sample["label"]
        reward = 1.0 if predicted_label == expected_label else 0.0

        debug_counter["count"] += 1
        if debug_counter["count"] <= 3:
            print(
                f"[rollout debug] split={split} seed={seed} model={model} "
                f"expected={expected_label} predicted={predicted_label}"
            )

        return RolloutResponse(
            trace_correlation_id=request.trace_correlation_id,
            reward_info=RolloutMetrics(
                outcome_reward=reward,
                outcome_objectives={"reward": reward},
                details={
                    "split": split,
                    "seed": seed,
                    "expected_label": expected_label,
                    "predicted_label": predicted_label,
                },
            ),
            trace=None,
            inference_url=f"{inference_base.rstrip('/')}/v1",
            success_status=SuccessStatus.SUCCESS,
        )

    def provide_taskset_description() -> dict[str, Any]:
        return {
            "splits": ["train", "val", "test"],
            "sizes": {
                "train": len(TRAIN_DATA),
                "val": len(VAL_DATA),
                "test": len(TEST_DATA),
            },
            "labels": list(LABELS),
        }

    def provide_task_instances(seeds: list[int]) -> list[TaskInfo]:
        rows: list[TaskInfo] = []
        for seed in seeds:
            sample = TRAIN_DATA[int(seed) % len(TRAIN_DATA)]
            rows.append(
                TaskInfo(
                    task={"id": APP_ID, "name": APP_NAME},
                    dataset={"id": APP_ID, "split": "train", "index": int(seed)},
                    inference={"model": default_model},
                    limits={"max_turns": 1},
                    task_metadata={
                        "query": sample["query"],
                        "expected_label": sample["label"],
                        "labels": list(LABELS),
                    },
                )
            )
        return rows

    config = ContainerConfig(
        app_id=APP_ID,
        name=APP_NAME,
        description="HTTP-only intent classification container for GEPA + Eval E2E.",
        provide_taskset_description=provide_taskset_description,
        provide_task_instances=provide_task_instances,
        rollout=rollout,
        cors_origins=["*"],
    )
    return create_container(config)


def _extract_best_system_prompt(
    *,
    gepa_result: Any,
    prompts_payload: Any,
    fallback: str,
) -> str:
    candidate = gepa_result.get_system_prompt() if hasattr(gepa_result, "get_system_prompt") else None
    if isinstance(candidate, str) and candidate.strip():
        return candidate.strip()

    top_prompts = getattr(prompts_payload, "top_prompts", []) if prompts_payload is not None else []
    for prompt in top_prompts:
        if not isinstance(prompt, dict):
            continue
        full_text = prompt.get("full_text")
        if isinstance(full_text, str) and full_text.strip():
            return full_text.strip()
        pattern = prompt.get("pattern")
        if isinstance(pattern, dict):
            for message in pattern.get("messages", []):
                if not isinstance(message, dict):
                    continue
                if message.get("role") == "system":
                    content = message.get("content") or message.get("pattern")
                    if isinstance(content, str) and content.strip():
                        return content.strip()

    best_candidate = getattr(prompts_payload, "best_candidate", None)
    if isinstance(best_candidate, dict):
        for message in best_candidate.get("messages", []):
            if not isinstance(message, dict):
                continue
            if message.get("role") == "system":
                content = message.get("content") or message.get("pattern")
                if isinstance(content, str) and content.strip():
                    return content.strip()

    return fallback


@dataclass
class EvalRun:
    name: str
    job_id: str
    result: Any


def _run_eval_job(
    *,
    name: str,
    backend_base: str,
    synth_api_key: str,
    container_url: str,
    container_api_key: str,
    container_worker_token: str | None,
    model: str,
    system_prompt: str,
    timeout: float,
    poll_interval: float,
) -> EvalRun:
    config = EvalJobConfig(
        container_url=container_url,
        backend_url=backend_base,
        api_key=synth_api_key,
        container_api_key=container_api_key,
        container_worker_token=container_worker_token,
        app_id=APP_ID,
        env_name=APP_ID,
        seeds=list(range(len(TEST_DATA))),
        policy_config={
            "provider": "openai",
            "model": model,
            "inference_mode": "synth_hosted",
            "temperature": 0.0,
            "max_completion_tokens": 32,
            "system_prompt": system_prompt,
            "config": {
                "model": model,
                "system_prompt": system_prompt,
                "api_key": synth_api_key,
                "inference_url": f"{backend_base.rstrip('/')}/v1",
            },
        },
        env_config={"split": "test"},
        concurrency=4,
        timeout=120.0,
    )
    job = EvalJob(config)
    job_id = job.submit()
    result = job.poll_until_complete(
        timeout=timeout,
        interval=poll_interval,
        progress=True,
    )
    return EvalRun(name=name, job_id=job_id, result=result)


async def run() -> None:
    args = parse_args()
    synth_api_key = args.api_key
    if not synth_api_key:
        import os

        synth_api_key = os.environ.get("SYNTH_API_KEY", "").strip()
    if not synth_api_key:
        raise SystemExit("SYNTH_API_KEY is required (pass --api-key or set env var).")

    backend_base = args.backend_base.rstrip("/")
    started_at = time.time()
    timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    env_api_key = ensure_container_auth(
        backend_base=backend_base,
        synth_api_key=synth_api_key,
    )
    app = _build_task_app(
        backend_base=backend_base,
        api_key=synth_api_key,
        default_model=args.model,
    )

    print("Starting in-process container (HTTP local mode)...")
    async with InProcessContainer(
        app=app,
        host="127.0.0.1",
        port=args.port,
        tunnel_mode="local",
        api_key=env_api_key,
        auto_find_port=True,
        health_check_timeout=30.0,
        skip_tunnel_verification=True,
    ) as container:
        container_url = container.url or f"http://127.0.0.1:{container.port}"
        worker_token = container.container_worker_token
        print(f"Container URL: {container_url}")

        train_seed_count = max(16, len(TRAIN_DATA))
        train_seeds = list(range(train_seed_count))
        val_seeds = list(range(train_seed_count, train_seed_count + len(VAL_DATA)))
        safe_pareto_size = 10

        gepa_config = {
            "prompt_learning": {
                "algorithm": "gepa",
                "container_url": container_url,
                "container_id": APP_ID,
                "initial_prompt": {
                    "id": "baseline_intent_router",
                    "name": "Intent Router Baseline",
                    "messages": [
                        {"role": "system", "order": 0, "pattern": BASELINE_SYSTEM_PROMPT},
                        {"role": "user", "order": 1, "pattern": USER_PATTERN},
                    ],
                    "wildcards": {"query": "REQUIRED"},
                },
                "policy": {
                    "provider": "openai",
                    "model": args.model,
                    "inference_mode": "synth_hosted",
                    "temperature": 0.0,
                    "max_completion_tokens": 32,
                    "config": {
                        "model": args.model,
                        "api_key": synth_api_key,
                        "inference_url": f"{backend_base}/v1",
                    },
                },
                "env_config": {"split": "train"},
                "gepa": {
                    "env_name": APP_ID,
                    "evaluation": {
                        "seeds": train_seeds,
                        "validation_seeds": val_seeds,
                    },
                    "rollout": {
                        "budget": int(args.rollout_budget),
                        "max_concurrent": int(args.max_concurrent),
                        "minibatch_size": int(args.minibatch_size),
                    },
                    "population": {
                        "initial_size": 3,
                        "num_generations": int(args.generations),
                        "children_per_generation": 2,
                    },
                    "mutation": {"rate": 0.3},
                    "archive": {
                        "pareto_set_size": safe_pareto_size,
                        "feedback_fraction": 0.5,
                    },
                    "token": {"counting_model": "gpt-4"},
                },
            }
        }

        print("Submitting GEPA job...")
        policy_job = PolicyOptimizationJob.from_dict(
            config_dict=gepa_config,
            backend_url=backend_base,
            api_key=synth_api_key,
            container_api_key=env_api_key,
            container_worker_token=worker_token,
            algorithm="gepa",
            skip_health_check=True,
        )
        gepa_job_id = await asyncio.to_thread(policy_job.submit)
        print(f"GEPA Job ID: {gepa_job_id}")
        gepa_result = await asyncio.to_thread(
            policy_job.poll_until_complete,
            timeout=float(args.gepa_timeout),
            interval=float(args.poll_interval),
            progress=True,
        )
        if not gepa_result.succeeded:
            raise RuntimeError(
                f"GEPA job failed: status={gepa_result.status.value} error={gepa_result.error}"
            )

        print("Fetching optimized prompt artifacts...")
        gepa_raw_results = await asyncio.to_thread(policy_job.get_results)
        prompts_payload = await asyncio.to_thread(get_prompts, gepa_job_id, backend_base, synth_api_key)
        optimized_system_prompt = _extract_best_system_prompt(
            gepa_result=gepa_result,
            prompts_payload=prompts_payload,
            fallback=BASELINE_SYSTEM_PROMPT,
        )

        print("Running Eval jobs (baseline and optimized)...")
        baseline_eval = await asyncio.to_thread(
            _run_eval_job,
            name="baseline",
            backend_base=backend_base,
            synth_api_key=synth_api_key,
            container_url=container_url,
            container_api_key=env_api_key,
            container_worker_token=worker_token,
            model=args.model,
            system_prompt=BASELINE_SYSTEM_PROMPT,
            timeout=float(args.eval_timeout),
            poll_interval=float(args.poll_interval),
        )
        optimized_eval = await asyncio.to_thread(
            _run_eval_job,
            name="optimized",
            backend_base=backend_base,
            synth_api_key=synth_api_key,
            container_url=container_url,
            container_api_key=env_api_key,
            container_worker_token=worker_token,
            model=args.model,
            system_prompt=optimized_system_prompt,
            timeout=float(args.eval_timeout),
            poll_interval=float(args.poll_interval),
        )

    elapsed = time.time() - started_at
    baseline_score = baseline_eval.result.mean_reward or 0.0
    optimized_score = optimized_eval.result.mean_reward or 0.0
    improvement = optimized_score - baseline_score

    baseline_prompt_path = output_dir / f"{timestamp}_baseline_prompt.txt"
    optimized_prompt_path = output_dir / f"{timestamp}_optimized_prompt.txt"
    baseline_prompt_path.write_text(BASELINE_SYSTEM_PROMPT + "\n", encoding="utf-8")
    optimized_prompt_path.write_text(optimized_system_prompt + "\n", encoding="utf-8")

    deliverable = {
        "run_type": "http_gepa_eval_e2e",
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "elapsed_seconds": elapsed,
        "backend_base": backend_base,
        "container_app_id": APP_ID,
        "model": args.model,
        "jobs": {
            "gepa": {
                "job_id": gepa_job_id,
                "status": gepa_result.status.value,
                "best_reward": gepa_result.best_reward,
            },
            "eval_baseline": {
                "job_id": baseline_eval.job_id,
                "status": baseline_eval.result.status.value,
                "mean_reward": baseline_eval.result.mean_reward,
                "total_tokens": baseline_eval.result.total_tokens,
                "total_cost_usd": baseline_eval.result.total_cost_usd,
            },
            "eval_optimized": {
                "job_id": optimized_eval.job_id,
                "status": optimized_eval.result.status.value,
                "mean_reward": optimized_eval.result.mean_reward,
                "total_tokens": optimized_eval.result.total_tokens,
                "total_cost_usd": optimized_eval.result.total_cost_usd,
            },
        },
        "scores": {
            "baseline_eval_mean_reward": baseline_score,
            "optimized_eval_mean_reward": optimized_score,
            "improvement": improvement,
        },
        "prompts": {
            "baseline_system_prompt": BASELINE_SYSTEM_PROMPT,
            "optimized_system_prompt": optimized_system_prompt,
            "baseline_prompt_file": str(baseline_prompt_path),
            "optimized_prompt_file": str(optimized_prompt_path),
            "top_prompts_count": len(getattr(prompts_payload, "top_prompts", []) or []),
            "optimized_candidates_count": len(
                getattr(prompts_payload, "optimized_candidates", []) or []
            ),
        },
        "gepa_artifacts": {
            "raw_results": gepa_raw_results,
            "prompt_results_summary": {
                "best_reward": getattr(prompts_payload, "best_reward", None),
                "total_rollouts": getattr(prompts_payload, "total_rollouts", None),
                "total_proposal_calls": getattr(prompts_payload, "total_proposal_calls", None),
                "event_counts": getattr(prompts_payload, "event_counts", None),
            },
        },
    }

    json_path = output_dir / f"{timestamp}_deliverable.json"
    json_path.write_text(json.dumps(deliverable, indent=2, default=str), encoding="utf-8")

    if args.compact_json:
        print(json.dumps(deliverable, separators=(",", ":"), default=str))
    else:
        print(json.dumps(deliverable, indent=2, default=str))
    print(f"\nDeliverable JSON: {json_path}")
    print(f"Baseline prompt file: {baseline_prompt_path}")
    print(f"Optimized prompt file: {optimized_prompt_path}")


if __name__ == "__main__":
    asyncio.run(run())
