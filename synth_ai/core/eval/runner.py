"""Eval runner for executing rollouts against task apps."""

import asyncio
import json
import os
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx

from synth_ai.core.eval.config import EvalRunConfig
from synth_ai.sdk.task.client import TaskAppClient
from synth_ai.sdk.task.contracts import (
    RolloutEnvSpec,
    RolloutMode,
    RolloutPolicySpec,
    RolloutRecordConfig,
    RolloutRequest,
)

_POLL_INTERVAL_S = 2.0
_MAX_POLL_ATTEMPTS = 600


@dataclass(slots=True)
class EvalResult:
    seed: int
    score: float | None
    reward_mean: float | None
    outcome_score: float | None
    events_score: float | None
    latency_ms: float | None
    verifier_score: float | None
    tokens: int | None
    cost_usd: float | None
    error: str | None = None
    trace: dict[str, Any] | None = None


def _count_tokens_from_trace(trace: dict[str, Any] | None) -> int:
    """Extract total token count from trace."""
    if not trace:
        return 0

    usage = trace.get("usage")
    if isinstance(usage, dict):
        total = usage.get("total_tokens", 0)
        if total > 0:
            return total

    total = 0
    event_history = trace.get("event_history") or []
    for event in event_history:
        if not isinstance(event, dict):
            continue
        evt_usage = event.get("usage") or {}
        if isinstance(evt_usage, dict):
            total += evt_usage.get("total_tokens", 0)
        response = event.get("response") or {}
        if isinstance(response, dict):
            resp_usage = response.get("usage") or {}
            if isinstance(resp_usage, dict):
                total += resp_usage.get("total_tokens", 0)
    return total


def _count_tokens_from_trajectories(trajectories: list[Any]) -> int:
    """Extract token count from trajectory steps."""
    total = 0
    for traj in trajectories:
        if not hasattr(traj, "steps"):
            continue
        for step in traj.steps:
            if not hasattr(step, "info") or not isinstance(step.info, dict):
                continue
            tokens = step.info.get("tokens")
            if isinstance(tokens, int):
                total += tokens
            usage = step.info.get("usage") or {}
            if isinstance(usage, dict):
                total += usage.get("total_tokens", 0)
    return total


def _build_run_id(config: EvalRunConfig, seed: int) -> str:
    base = config.app_id or config.env_name or "eval"
    suffix = uuid.uuid4().hex[:8]
    return f"{base}-seed-{seed}-{suffix}"


def _build_rollout_request(config: EvalRunConfig, seed: int) -> RolloutRequest:
    env_config = dict(config.env_config or {})
    policy_config = dict(config.policy_config or {})

    output_mode = policy_config.pop("output_mode", None)
    structured_config = policy_config.pop("structured_config", None)

    policy_kwargs: dict[str, Any] = {
        "policy_name": config.policy_name,
        "config": policy_config,
    }
    if output_mode is not None:
        policy_kwargs["output_mode"] = output_mode
    if structured_config is not None:
        policy_kwargs["structured_config"] = structured_config

    trace_fmt: Any = config.trace_format
    record = RolloutRecordConfig(
        trajectories=True,
        logprobs=False,
        value=False,
        return_trace=config.return_trace,
        trace_format=trace_fmt,
    )

    synth_base = os.getenv("SYNTH_API_BASE") or os.getenv("SYNTH_BASE_URL")

    return RolloutRequest(
        run_id=_build_run_id(config, seed),
        env=RolloutEnvSpec(env_name=config.env_name, config=env_config, seed=seed),
        policy=RolloutPolicySpec(**policy_kwargs),
        record=record,
        on_done="reset",
        training_session_id=None,
        synth_base_url=synth_base,
        mode=config.mode or RolloutMode.EVAL,
    )


async def _eval_seed(
    client: TaskAppClient,
    config: EvalRunConfig,
    seed: int,
    semaphore: asyncio.Semaphore,
) -> EvalResult:
    """Execute a single rollout for one seed (used in direct mode)."""
    async with semaphore:
        start = time.perf_counter()
        try:
            request = _build_rollout_request(config, seed)
            response = await client.rollout(request)
            latency_ms = (time.perf_counter() - start) * 1000.0

            metrics = response.metrics
            reward_mean = metrics.reward_mean
            outcome_score = metrics.outcome_score
            events_score = metrics.events_score
            outcome_reward = metrics.outcome_reward
            outcome_objectives = metrics.outcome_objectives

            reward_val = None
            if isinstance(outcome_objectives, dict):
                reward_val = outcome_objectives.get("reward")
            if reward_val is None and outcome_reward is not None:
                reward_val = outcome_reward
            if reward_val is None and outcome_score is not None:
                reward_val = outcome_score
            if reward_val is None and reward_mean is not None:
                reward_val = reward_mean
            score = float(reward_val) if reward_val is not None else None
            verifier_score = None
            tokens = None
            cost_usd = None

            if isinstance(metrics.details, dict):
                verifier_score = metrics.details.get("verifier_score")
                tokens = metrics.details.get("tokens")
                cost_usd = metrics.details.get("cost_usd")

            trace = response.trace if config.return_trace else None

            if tokens is None:
                if trace:
                    tokens = _count_tokens_from_trace(trace)
                else:
                    trajectories = getattr(response, "trajectories", None)
                    if trajectories:
                        tokens = _count_tokens_from_trajectories(trajectories)
                if tokens == 0:
                    tokens = None

            return EvalResult(
                seed=seed,
                score=score,
                reward_mean=reward_mean,
                outcome_score=outcome_score,
                events_score=events_score,
                latency_ms=latency_ms,
                verifier_score=verifier_score,
                tokens=tokens,
                cost_usd=cost_usd,
                error=None,
                trace=trace,
            )
        except Exception as exc:
            latency_ms = (time.perf_counter() - start) * 1000.0
            return EvalResult(
                seed=seed,
                score=None,
                reward_mean=None,
                outcome_score=None,
                events_score=None,
                latency_ms=latency_ms,
                verifier_score=None,
                tokens=None,
                cost_usd=None,
                error=str(exc),
                trace=None,
            )


async def run_eval(config: EvalRunConfig) -> list[EvalResult]:
    """Run evaluation against a task app."""
    backend_url = config.backend_url or os.getenv("SYNTH_BASE_URL") or os.getenv("BACKEND_OVERRIDE")
    api_key = config.backend_api_key or os.getenv("SYNTH_API_KEY")

    if backend_url and api_key:
        return await run_eval_via_backend(config, backend_url, api_key)

    return await run_eval_direct(config)


async def run_eval_direct(config: EvalRunConfig) -> list[EvalResult]:
    """Direct mode: Call task apps directly without backend."""
    if not config.task_app_url:
        raise ValueError("task_app_url is required for eval runs")
    if not config.seeds:
        raise ValueError("No seeds provided for evaluation")

    api_key = config.task_app_api_key or os.getenv("ENVIRONMENT_API_KEY")
    semaphore = asyncio.Semaphore(max(1, int(config.concurrency or 1)))

    async with TaskAppClient(base_url=config.task_app_url, api_key=api_key) as client:
        tasks = [_eval_seed(client, config, seed, semaphore) for seed in config.seeds]
        results = await asyncio.gather(*tasks)

    results.sort(key=lambda item: item.seed)
    return results


async def run_eval_via_backend(
    config: EvalRunConfig,
    backend_url: str,
    api_key: str,
) -> list[EvalResult]:
    """Backend mode: Route through backend interceptor for trace/usage capture."""
    if not config.task_app_url:
        raise ValueError("task_app_url is required for eval runs")
    if not config.seeds:
        raise ValueError("No seeds provided for evaluation")

    base = backend_url.rstrip("/")
    if not base.endswith("/api"):
        base = f"{base}/api"

    headers = {"Authorization": f"Bearer {api_key}"}

    policy = dict(config.policy_config or {})
    policy["policy_name"] = config.policy_name

    job_request = {
        "task_app_url": config.task_app_url,
        "task_app_api_key": config.task_app_api_key or os.getenv("ENVIRONMENT_API_KEY"),
        "app_id": config.app_id,
        "env_name": config.env_name,
        "seeds": list(config.seeds),
        "policy": policy,
        "env_config": config.env_config,
        "mode": config.mode.value if hasattr(config.mode, "value") else str(config.mode or "eval"),
        "max_concurrent": config.concurrency,
        "timeout": config.timeout,
    }

    async with httpx.AsyncClient(timeout=httpx.Timeout(30.0)) as client:
        print(f"[eval] Creating eval job via backend: {base}/eval/jobs", flush=True)
        resp = await client.post(f"{base}/eval/jobs", json=job_request, headers=headers)

        if resp.status_code not in (200, 201):
            raise RuntimeError(f"Failed to create eval job: {resp.status_code} {resp.text}")

        job_data = resp.json()
        job_id = job_data.get("job_id")
        if not job_id:
            raise RuntimeError(f"No job_id in response: {job_data}")

        print(f"[eval] Job created: {job_id}", flush=True)

        for attempt in range(_MAX_POLL_ATTEMPTS):
            await asyncio.sleep(_POLL_INTERVAL_S)

            status_resp = await client.get(f"{base}/eval/jobs/{job_id}", headers=headers)
            if status_resp.status_code != 200:
                print(f"[eval] Warning: status check failed: {status_resp.status_code}", flush=True)
                continue

            status_data = status_resp.json()
            status = status_data.get("status", "")

            if status in ("completed", "failed"):
                break

            if attempt % 10 == 0:
                print(f"[eval] Job {job_id} status: {status} (attempt {attempt})", flush=True)
        else:
            raise RuntimeError(
                f"Eval job {job_id} timed out after {_MAX_POLL_ATTEMPTS * _POLL_INTERVAL_S}s"
            )

        if status == "failed":
            error = status_data.get("error", "Unknown error")
            raise RuntimeError(f"Eval job {job_id} failed: {error}")

        results_resp = await client.get(f"{base}/eval/jobs/{job_id}/results", headers=headers)
        if results_resp.status_code != 200:
            raise RuntimeError(
                f"Failed to get results: {results_resp.status_code} {results_resp.text}"
            )

        results_data = results_resp.json()
        result_rows = results_data.get("results", [])

        results: list[EvalResult] = []
        for row in result_rows:
            results.append(
                EvalResult(
                    seed=int(row.get("seed", 0)),
                    score=row.get("score"),
                    reward_mean=row.get("reward_mean"),
                    outcome_score=row.get("outcome_score"),
                    events_score=row.get("events_score"),
                    latency_ms=row.get("latency_ms"),
                    verifier_score=row.get("verifier_score"),
                    tokens=row.get("tokens"),
                    cost_usd=row.get("cost_usd"),
                    error=row.get("error"),
                    trace=None,
                )
            )

        results.sort(key=lambda item: item.seed)

        summary = results_data.get("summary", {})
        if summary:
            print(f"[eval] Backend summary: {summary}", flush=True)

        return results


async def fetch_traces_from_backend(
    job_id: str,
    backend_url: str,
    api_key: str,
    output_dir: str,
) -> str:
    """Download traces zip from backend and extract to output_dir."""
    import io
    import zipfile

    base = backend_url.rstrip("/")
    if not base.endswith("/api"):
        base = f"{base}/api"

    headers = {"Authorization": f"Bearer {api_key}"}

    async with httpx.AsyncClient(timeout=httpx.Timeout(60.0)) as client:
        resp = await client.get(f"{base}/eval/jobs/{job_id}/traces", headers=headers)

        if resp.status_code != 200:
            raise RuntimeError(f"Failed to download traces: {resp.status_code} {resp.text}")

        path = Path(output_dir)
        path.mkdir(parents=True, exist_ok=True)

        with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
            zf.extractall(path)

        return str(path)


def format_eval_table(results: list[EvalResult]) -> str:
    headers = [
        "seed",
        "score",
        "reward_mean",
        "outcome",
        "events",
        "latency_ms",
        "verifier",
        "tokens",
        "cost_usd",
        "error",
    ]

    def _fmt(value: Any) -> str:
        if value is None:
            return "-"
        if isinstance(value, float):
            return f"{value:.4f}".rstrip("0").rstrip(".")
        return str(value)

    rows = [
        [
            r.seed,
            _fmt(r.score),
            _fmt(r.reward_mean),
            _fmt(r.outcome_score),
            _fmt(r.events_score),
            _fmt(r.latency_ms),
            _fmt(r.verifier_score),
            _fmt(r.tokens),
            _fmt(r.cost_usd),
            r.error or "-",
        ]
        for r in results
    ]

    def _avg(values: list[float | int]) -> float | None:
        return sum(values) / len(values) if values else None

    scores = [r.score for r in results if isinstance(r.score, (int, float))]
    reward_means = [r.reward_mean for r in results if isinstance(r.reward_mean, (int, float))]
    outcomes = [r.outcome_score for r in results if isinstance(r.outcome_score, (int, float))]
    events = [r.events_score for r in results if isinstance(r.events_score, (int, float))]
    latencies = [r.latency_ms for r in results if isinstance(r.latency_ms, (int, float))]
    verifier_scores = [
        r.verifier_score for r in results if isinstance(r.verifier_score, (int, float))
    ]
    tokens = [r.tokens for r in results if isinstance(r.tokens, int)]
    costs = [r.cost_usd for r in results if isinstance(r.cost_usd, (int, float))]

    rows.append(
        [
            "avg",
            _fmt(_avg(scores)),
            _fmt(_avg(reward_means)),
            _fmt(_avg(outcomes)),
            _fmt(_avg(events)),
            _fmt(_avg(latencies)),
            _fmt(_avg(verifier_scores)),
            _fmt(int(sum(tokens) / len(tokens)) if tokens else None),
            _fmt(_avg(costs)),
            "-",
        ]
    )

    widths = [len(h) for h in headers]
    for row in rows:
        for idx, cell in enumerate(row):
            widths[idx] = max(widths[idx], len(str(cell)))

    def _render_row(row: list[Any]) -> str:
        return " | ".join(str(cell).ljust(widths[idx]) for idx, cell in enumerate(row))

    sep = "-+-".join("-" * width for width in widths)
    lines = [_render_row(headers), sep]
    lines.extend(_render_row(row) for row in rows)
    return "\n".join(lines)


def format_eval_report(config: EvalRunConfig, results: list[EvalResult]) -> str:
    payload = {
        "app_id": config.app_id,
        "task_app_url": config.task_app_url,
        "env_name": config.env_name,
        "policy_name": config.policy_name,
        "policy_config": config.policy_config,
        "seeds": config.seeds,
        "concurrency": config.concurrency,
    }
    header = json.dumps(payload, indent=2, default=str)
    table = format_eval_table(results)
    return f"Eval config\n{header}\n\nResults\n{table}\n"


def save_traces(results: list[EvalResult], traces_dir: str) -> int:
    """Save traces to individual JSON files in the given directory."""
    path = Path(traces_dir)
    path.mkdir(parents=True, exist_ok=True)

    saved = 0
    for result in results:
        if result.trace is not None:
            trace_file = path / f"seed_{result.seed}_trace.json"
            trace_file.write_text(json.dumps(result.trace, indent=2, default=str))
            saved += 1

    return saved


__all__ = [
    "run_eval",
    "run_eval_direct",
    "run_eval_via_backend",
    "fetch_traces_from_backend",
    "format_eval_table",
    "format_eval_report",
    "save_traces",
    "EvalResult",
]
