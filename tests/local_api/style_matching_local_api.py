"""Style matching local API for the GraphGen cookbook dataset."""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
import uuid
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any, Mapping, cast

import httpx
from dotenv import load_dotenv
from fastapi import HTTPException, Request
from style_matching_business_logic import (
    AVAILABLE_SPLITS,
    DATASET_NAME,
    DEFAULT_SPLIT,
    REPO_ROOT,
    TOOL_NAME,
    StyleMatchingDataset,
    format_notes,
    get_default_messages_templates,
    get_submit_tool_schema,
    parse_essay_from_tool_call,
)
from synth_ai.sdk.localapi.apps import LocalAPIEntry, register_local_api
from synth_ai.sdk.localapi.helpers import (
    add_metadata_endpoint,
    call_chat_completion_api,
    create_http_client_hooks,
    extract_api_key,
    preload_dataset_splits,
)
from synth_ai.sdk.localapi.server import (
    LocalAPIConfig,
    RubricBundle,
    create_local_api,
    run_local_api,
)
from synth_ai.sdk.task.contracts import RolloutMetrics, RolloutRequest, RolloutResponse, TaskInfo
from synth_ai.sdk.task.datasets import TaskDatasetRegistry, TaskDatasetSpec
from synth_ai.sdk.task.rubrics import Rubric, load_rubric
from synth_ai.sdk.task.trace_correlation_helpers import (
    build_trace_payload,
    extract_trace_correlation_id,
)


def _get_backend_url() -> str:
    return os.getenv("BACKEND_URL", "http://localhost:8000")


def _get_synth_api_key() -> str | None:
    return os.getenv("SYNTH_API_KEY")


def _get_verifier_model() -> str:
    return os.getenv("VERIFIER_MODEL", "gpt-4.1-nano")


def _get_http_retry_max() -> int:
    return int(os.getenv("STYLE_MATCHING_HTTP_RETRIES", "3"))


def _get_http_retry_backoff_seconds() -> float:
    return float(os.getenv("STYLE_MATCHING_HTTP_BACKOFF_SECONDS", "1.0"))


print(
    f"[style_matching_local_api] Module loaded: DATASET_PATH={REPO_ROOT / 'cookbooks' / 'products' / 'graphgen' / 'style_matching_dataset.json'}",
    flush=True,
)


STYLE_MATCHING_DATASET_SPEC = TaskDatasetSpec(
    id="style_matching",
    name="Style Matching Essay Task",
    version="1.0.0",
    splits=list(AVAILABLE_SPLITS),
    default_split=DEFAULT_SPLIT,
    description="Style matching essay task based on the GraphGen cookbook dataset.",
)


def _render_messages(placeholders: Mapping[str, str]) -> list[dict[str, str]]:
    rendered: list[dict[str, str]] = []
    for msg_template in get_default_messages_templates():
        role = msg_template.get("role", "user")
        pattern = msg_template.get("pattern", "")
        content = pattern.format(**placeholders)
        rendered.append({"role": role, "content": content})
    return rendered


def _extract_user_content(messages: Sequence[Mapping[str, str]]) -> str:
    for message in reversed(messages):
        if message.get("role") == "user":
            content = str(message.get("content", "")).strip()
            if content:
                return content
            break
    raise HTTPException(status_code=500, detail="No user message content available for verifier")


def _format_essay_text(title: str, content: str) -> str:
    title = title.strip()
    content = content.strip()
    if title and content:
        return f"{title}\n\n{content}"
    return title or content


def _build_session_trace(
    *, user_content: str, assistant_content: str, session_id: str
) -> dict[str, Any]:
    return {
        "session_id": session_id,
        "session_time_steps": [
            {
                "step_id": "1",
                "step_index": 0,
                "events": [
                    {
                        "event_type": "runtime",
                        "event_id": 1,
                        "type": "user_message",
                        "content": user_content,
                    },
                    {
                        "event_type": "runtime",
                        "event_id": 2,
                        "type": "assistant_message",
                        "content": assistant_content,
                    },
                ],
            }
        ],
    }


def _extract_verifier_score(result: Mapping[str, Any]) -> float:
    output = result.get("output", result)
    if isinstance(output, Mapping):
        outcome_review = output.get("outcome_review")
        if isinstance(outcome_review, Mapping) and isinstance(
            outcome_review.get("total"), (int, float)
        ):
            return float(outcome_review["total"])
        event_reviews = output.get("event_reviews")
        if isinstance(event_reviews, list):
            totals = [rev.get("total") for rev in event_reviews if isinstance(rev, Mapping)]
            totals = [t for t in totals if isinstance(t, (int, float))]
            if totals:
                return float(sum(totals) / len(totals))
        event_rewards = output.get("event_rewards")
        if isinstance(event_rewards, list):
            rewards = [r.get("reward_value") for r in event_rewards if isinstance(r, Mapping)]
            rewards = [r for r in rewards if isinstance(r, (int, float))]
            if rewards:
                return float(sum(rewards) / len(rewards))
        event_totals = output.get("event_totals")
        if isinstance(event_totals, list):
            totals = [t for t in event_totals if isinstance(t, (int, float))]
            if totals:
                return float(sum(totals) / len(totals))
        if isinstance(output.get("total"), (int, float)):
            return float(output["total"])
    raise ValueError("Verifier response missing score fields")


async def _post_json_with_retries(
    *,
    url: str,
    headers: dict[str, str],
    payload: dict[str, Any],
    timeout: float,
    label: str,
) -> httpx.Response:
    last_error: Exception | None = None
    retry_max = _get_http_retry_max()
    backoff = _get_http_retry_backoff_seconds()
    for attempt in range(1, retry_max + 1):
        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                response = await client.post(url, headers=headers, json=payload)
            if response.status_code >= 500:
                last_error = RuntimeError(
                    f"{label} HTTP {response.status_code}: {response.text[:200]}"
                )
            else:
                return response
        except httpx.RequestError as exc:
            last_error = exc
        if attempt < retry_max:
            await asyncio.sleep(backoff * attempt)
    raise RuntimeError(f"{label} failed after {retry_max} attempts: {last_error}")


def _build_gold_examples(dataset: StyleMatchingDataset) -> list[dict[str, Any]]:
    cached = getattr(dataset, "_verifier_gold_examples", None)
    if isinstance(cached, list):
        return cached

    gold_examples: list[dict[str, Any]] = []
    for gold in dataset.gold_outputs:
        task_id_raw = gold.get("task_id")
        if not task_id_raw:
            continue
        task_id = str(task_id_raw)
        task = dataset.task_by_id(task_id)
        if not task:
            raise HTTPException(status_code=500, detail=f"Missing task for gold output: {task_id}")
        task_input = task.get("input") or {}
        placeholders = {
            "system_prompt": dataset.initial_prompt,
            "outline": str(task_input.get("outline", "")),
            "topic": str(task_input.get("topic", "")),
            "notes": format_notes(task_input.get("notes") or []),
        }
        user_content = _extract_user_content(_render_messages(placeholders))

        output = gold.get("output") or {}
        assistant_content = _format_essay_text(
            str(output.get("title", "")),
            str(output.get("content", "")),
        )
        session_trace = _build_session_trace(
            user_content=user_content,
            assistant_content=assistant_content,
            session_id=f"gold-{task_id}",
        )
        gold_examples.append(
            {
                "summary": f"Gold example for {task_id}",
                "gold_score": 1.0,
                "gold_reasoning": "Reference style-matching essay.",
                "trace": session_trace,
            }
        )

    if not gold_examples:
        raise HTTPException(status_code=500, detail="No gold examples available for verifier")

    dataset._verifier_gold_examples = gold_examples
    return gold_examples


async def _score_with_verifier(
    *, session_trace: dict[str, Any], gold_examples: list[dict[str, Any]]
) -> float:
    synth_api_key = _get_synth_api_key()
    if not synth_api_key:
        raise HTTPException(
            status_code=503, detail="SYNTH_API_KEY is required for verifier scoring"
        )

    payload = {
        "job_id": "zero_shot_verifier_contrastive_single",
        "input": {
            "trace": session_trace,
            "gold_examples": gold_examples,
            "candidate_score": 0.5,
            "candidate_reasoning": "Auto-evaluated from style-matching local API",
            "options": {"model": _get_verifier_model()},
        },
    }

    headers = {
        "Authorization": f"Bearer {synth_api_key}",
        "Content-Type": "application/json",
    }

    backend_url = _get_backend_url().rstrip("/")
    response = await _post_json_with_retries(
        url=f"{backend_url}/api/graphs/completions",
        headers=headers,
        payload=payload,
        timeout=120.0,
        label="Verifier request",
    )
    if response.status_code != 200:
        raise HTTPException(
            status_code=502,
            detail=f"Verifier failed: HTTP {response.status_code} {response.text[:500]}",
        )
    try:
        return _extract_verifier_score(response.json())
    except ValueError as exc:
        raise HTTPException(
            status_code=502, detail=f"Verifier response missing score: {exc}"
        ) from exc


async def rollout_executor(request: RolloutRequest, fastapi_request: Request) -> RolloutResponse:
    dataset: StyleMatchingDataset = fastapi_request.app.state.style_matching_dataset

    split = str(((request.env.config or {}).get("split")) or DEFAULT_SPLIT)
    seed = request.env.seed or 0

    sample = dataset.sample(split=split, index=seed)

    placeholders = {
        "system_prompt": dataset.initial_prompt,
        "outline": sample["outline"],
        "topic": sample["topic"],
        "notes": format_notes(sample["notes"]),
    }

    rendered_messages = _render_messages(placeholders)
    user_content = _extract_user_content(rendered_messages)

    api_key = extract_api_key(fastapi_request, request.policy.config or {})
    http_client = getattr(fastapi_request.app.state, "http_client", None)

    tool_schema = get_submit_tool_schema()
    response_text, response_json, tool_calls = await call_chat_completion_api(
        policy_config=request.policy.config or {},
        messages=rendered_messages,
        tools=[tool_schema],
        tool_choice="required",
        api_key=api_key,
        http_client=http_client,
        expected_tool_name=TOOL_NAME,
        log_prefix="[STYLE_MATCHING]",
    )

    if not tool_calls:
        raise HTTPException(status_code=502, detail="Model did not call submit_essay tool")

    essay = {"title": "", "content": ""}
    for tc in tool_calls:
        if tc.get("function", {}).get("name") == TOOL_NAME:
            args_str = tc.get("function", {}).get("arguments", "{}")
            essay = parse_essay_from_tool_call(args_str)
            break

    title = essay.get("title", "")
    content = essay.get("content", "")
    assistant_content = _format_essay_text(title, content)

    session_trace = _build_session_trace(
        user_content=user_content,
        assistant_content=assistant_content,
        session_id=f"style-matching-{sample['task_id']}-{uuid.uuid4().hex[:8]}",
    )
    gold_examples = _build_gold_examples(dataset)
    score = await _score_with_verifier(session_trace=session_trace, gold_examples=gold_examples)

    try:
        raw_upstream = json.dumps(response_json, ensure_ascii=False)
    except Exception:
        raw_upstream = str(response_json)
    print(
        f"[STYLE_MATCHING] UPSTREAM_RESPONSE_JSON ({len(raw_upstream)} bytes): {raw_upstream}",
        flush=True,
    )

    inference_url = (request.policy.config or {}).get("inference_url")
    trace_correlation_id = extract_trace_correlation_id(
        policy_config=request.policy.config or {},
        inference_url=str(inference_url or ""),
    )

    trace_metadata = {
        "env": DATASET_NAME,
        "split": sample["split"],
        "index": sample["index"],
        "task_id": sample["task_id"],
        "verifier_score": score,
    }
    trace_payload = build_trace_payload(
        messages=rendered_messages,
        response=response_json if isinstance(response_json, dict) else None,
        correlation_id=trace_correlation_id,
        metadata=trace_metadata,
    )

    metrics = RolloutMetrics(
        outcome_reward=score,
        details={"verifier_score": score},
    )

    return RolloutResponse(
        run_id=request.run_id,
        metrics=metrics,
        trace_correlation_id=trace_correlation_id,
        trace=trace_payload,
        inference_url=str(inference_url or ""),
    )


def build_dataset() -> tuple[TaskDatasetRegistry, StyleMatchingDataset]:
    registry = TaskDatasetRegistry()
    dataset = StyleMatchingDataset()
    registry.register(STYLE_MATCHING_DATASET_SPEC, lambda _spec: dataset, cache=True)
    return registry, dataset


def _base_task_info(dataset: StyleMatchingDataset) -> TaskInfo:
    dataset_meta = STYLE_MATCHING_DATASET_SPEC.model_dump()
    dataset_meta["input_schema"] = dataset.input_schema
    dataset_meta["output_schema"] = dataset.output_schema
    return TaskInfo(
        task={
            "id": DATASET_NAME,
            "name": "Style Matching Essay Task",
            "version": "1.0.0",
            "action_space": {
                "type": "tool_call",
                "tool_name": TOOL_NAME,
                "description": "Submit essay title and content.",
            },
        },
        environment=DATASET_NAME,
        dataset=dataset_meta,
        rubric={"version": "1", "criteria_count": 1, "source": "inline"},
        inference={"supports_proxy": False, "tool": TOOL_NAME},
        limits={"max_turns": 1},
        task_metadata={"format": "tool_call"},
    )


def describe_taskset(dataset: StyleMatchingDataset) -> Mapping[str, Any]:
    return {
        **STYLE_MATCHING_DATASET_SPEC.model_dump(),
        "size": dataset.size(DEFAULT_SPLIT),
        "input_schema": dataset.input_schema,
        "output_schema": dataset.output_schema,
    }


def provide_task_instances(
    dataset: StyleMatchingDataset, seeds: Sequence[int]
) -> Iterable[TaskInfo]:
    base_info = _base_task_info(dataset)
    for seed in seeds:
        sample = dataset.sample(split=DEFAULT_SPLIT, index=seed)
        dataset_meta = dict(base_info.dataset)
        dataset_meta["split"] = sample["split"]
        dataset_meta["index"] = sample["index"]
        task_metadata = {
            **base_info.task_metadata,
            "task_id": sample["task_id"],
            "outline": sample["outline"],
            "topic": sample["topic"],
            "notes": sample["notes"],
        }
        yield TaskInfo(
            task=base_info.task,
            environment=base_info.environment,
            dataset=dataset_meta,
            rubric=base_info.rubric,
            inference=base_info.inference,
            limits=base_info.limits,
            task_metadata=task_metadata,
        )


OUTCOME_RUBRIC: Rubric = cast(
    Rubric,
    load_rubric(
        {
            "version": "1",
            "goal_text": "Return a valid essay with title and content.",
            "aggregation": "weighted_sum",
            "criteria": [
                {
                    "id": "format_valid",
                    "description": "Both title and content are present.",
                    "weight": 1.0,
                }
            ],
        }
    ),
)


def build_config() -> LocalAPIConfig:
    registry, dataset = build_dataset()
    base_info = _base_task_info(dataset)

    preload_dataset_splits(dataset, AVAILABLE_SPLITS, "style_matching_local_api")

    startup_http_client, shutdown_http_client = create_http_client_hooks(
        timeout=60.0,
        log_prefix="style_matching_local_api",
    )

    return LocalAPIConfig(
        app_id=DATASET_NAME,
        name="Style Matching Essay Task",
        description="Local API for the GraphGen style matching cookbook dataset.",
        base_task_info=base_info,
        describe_taskset=lambda: describe_taskset(dataset),
        provide_task_instances=lambda seeds: provide_task_instances(dataset, seeds),
        rollout=rollout_executor,
        dataset_registry=registry,
        rubrics=RubricBundle(outcome=OUTCOME_RUBRIC),
        proxy=None,
        app_state={"style_matching_dataset": dataset},
        cors_origins=["*"],
        startup_hooks=[startup_http_client],
        shutdown_hooks=[shutdown_http_client],
    )


register_local_api(
    entry=LocalAPIEntry(
        api_id=DATASET_NAME,
        description="Style matching local API for cookbook essays.",
        config_factory=build_config,
        aliases=("style-matching",),
    )
)


def fastapi_app():
    with contextlib.suppress(Exception):
        load_dotenv(str(REPO_ROOT / "synth-ai" / ".env"), override=False)
        load_dotenv(str(REPO_ROOT / ".env"), override=False)

    app = create_local_api(build_config())
    add_metadata_endpoint(app)
    return app


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run the style matching local API")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8120)
    parser.add_argument("--reload", action="store_true", help="Enable uvicorn autoreload")
    parser.add_argument(
        "--env-file",
        action="append",
        default=[],
        help="Additional .env files to load before startup",
    )
    args = parser.parse_args()

    default_env = Path(__file__).resolve().parents[2] / ".env"
    env_files = [str(default_env)] if default_env.exists() else []
    repo_env = Path(__file__).resolve().parents[3] / ".env"
    if repo_env.exists():
        env_files.append(str(repo_env))
    env_files.extend(args.env_file or [])

    run_local_api(
        build_config,
        host=args.host,
        port=args.port,
        reload=args.reload,
        env_files=env_files,
    )
