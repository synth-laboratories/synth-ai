"""Banking77 intent classification local API for Synth prompt optimization benchmarks."""

from __future__ import annotations

import contextlib
import json
import os
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any, Mapping, cast

# Business logic imports (no synth-ai dependencies)
from banking77_business_logic import (
    AVAILABLE_SPLITS,
    DATASET_NAME,
    DEFAULT_SPLIT,
    REPO_ROOT,
    TOOL_NAME,
    Banking77Dataset,
    Banking77Scorer,
    format_available_intents,
    get_classify_tool_schema,
    get_default_messages_templates,
)
from dotenv import load_dotenv
from fastapi import HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.requests import Request as StarletteRequest

# Synth-AI SDK imports
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
from synth_ai.sdk.task.contracts import (
    RolloutMetrics,
    RolloutRequest,
    RolloutResponse,
    TaskInfo,
)
from synth_ai.sdk.task.datasets import TaskDatasetRegistry, TaskDatasetSpec
from synth_ai.sdk.task.rubrics import Rubric, load_rubric
from synth_ai.sdk.task.trace_correlation_helpers import (
    build_trace_payload,
    extract_trace_correlation_id,
)

# Log environment at module load time for debugging
print(
    f"[banking77_local_api] Module loaded: DATASET_NAME={DATASET_NAME}, "
    f"HF_HOME={os.getenv('HF_HOME')}, "
    f"HF_DATASETS_CACHE={os.getenv('HF_DATASETS_CACHE')}, "
    f"HF_HUB_CACHE={os.getenv('HF_HUB_CACHE')}",
    flush=True,
)

# Dataset spec for registry
BANKING77_DATASET_SPEC = TaskDatasetSpec(
    id="banking77",
    name="Banking77 Intent Classification",
    version="1.0.0",
    splits=list(AVAILABLE_SPLITS),
    default_split=DEFAULT_SPLIT,
    description="Banking customer query intent classification with 77 intent categories.",
)


async def rollout_executor(request: RolloutRequest, fastapi_request: Request) -> RolloutResponse:
    """Execute a rollout for the banking77 classification task."""
    dataset: Banking77Dataset = fastapi_request.app.state.banking77_dataset

    with contextlib.suppress(Exception):
        cfg = request.policy.config or {}
        print(
            f"[LOCAL_API] INBOUND_ROLLOUT: run_id={request.run_id} seed={request.env.seed} env={request.env.env_name} "
            f"policy.model={cfg.get('model')} provider={cfg.get('provider')} api_base={cfg.get('inference_url') or cfg.get('api_base') or cfg.get('base_url')}",
            flush=True,
        )

    split = str(((request.env.config or {}).get("split")) or DEFAULT_SPLIT)
    seed = request.env.seed or 0

    sample = dataset.sample(split=split, index=seed)
    intents_list = format_available_intents(dataset.label_names)
    placeholders = {
        "query": sample["text"],
        "available_intents": intents_list,
    }

    default_messages = get_default_messages_templates()

    # Render baseline messages
    rendered_messages: list[dict[str, str]] = []
    for msg_template in default_messages:
        role = msg_template.get("role", "user")
        pattern = msg_template.get("pattern", "")
        content = pattern.format(**placeholders)
        rendered_messages.append({"role": role, "content": content})

    # Extract API key
    api_key = extract_api_key(fastapi_request, request.policy.config or {})

    http_client = getattr(fastapi_request.app.state, "http_client", None)

    classify_tool = get_classify_tool_schema()
    response_text, response_json, tool_calls = await call_chat_completion_api(
        policy_config=request.policy.config or {},
        messages=rendered_messages,
        tools=[classify_tool] if classify_tool else None,
        tool_choice="required" if classify_tool else None,
        api_key=api_key,
        http_client=http_client,
        enable_dns_preresolution=True,
        expected_tool_name=TOOL_NAME,
        log_prefix="[LOCAL_API]",
    )

    # Validate response
    try:
        raw_upstream = json.dumps(response_json, ensure_ascii=False)
    except Exception:
        raw_upstream = str(response_json)
    print(
        f"[LOCAL_API] UPSTREAM_RESPONSE_JSON ({len(raw_upstream)} bytes): {raw_upstream}",
        flush=True,
    )

    if not isinstance(response_json, dict) or not response_json:
        raise RuntimeError("Proxy returned missing/empty JSON")

    if tool_calls:
        for tc in tool_calls:
            args_str = tc.get("function", {}).get("arguments", "{}")
            try:
                args = json.loads(args_str)
            except Exception as exc:
                raise HTTPException(
                    status_code=502, detail="Tool call arguments not valid JSON"
                ) from exc
            if not str(args.get("intent", "")).strip():
                raise HTTPException(status_code=502, detail="Tool call missing 'intent'")

    # Extract predicted intent
    predicted_intent = ""
    if tool_calls:
        for tc in tool_calls:
            if tc.get("function", {}).get("name") == TOOL_NAME:
                args_str = tc.get("function", {}).get("arguments", "{}")
                try:
                    args = json.loads(args_str)
                    predicted_intent = args.get("intent", "")
                    print(f"[LOCAL_API] PARSED_TOOL_INTENT: {predicted_intent}", flush=True)
                except Exception:
                    print(f"[LOCAL_API] TOOL_PARSE_ERROR: {args_str}", flush=True)
    elif response_text:
        predicted_intent = response_text.strip().split()[0] if response_text.strip() else ""
        print(f"[LOCAL_API] CONTENT_FALLBACK_INTENT: {predicted_intent}", flush=True)

    if not str(predicted_intent or "").strip():
        raise RuntimeError("No prediction produced from proxy response")

    # Score using business logic
    expected_intent = sample["label"]
    is_correct, reward = Banking77Scorer.score(predicted_intent, expected_intent)

    print(
        f"[LOCAL_API] PREDICTION: expected={expected_intent} predicted={predicted_intent} correct={is_correct}",
        flush=True,
    )

    with contextlib.suppress(Exception):
        print(
            f"[BANKING77_ROLLOUT] run_id={request.run_id} split={sample['split']} "
            f"index={sample['index']} expected={expected_intent} predicted={predicted_intent} "
            f"reward={reward}",
            flush=True,
        )

    inference_url = (request.policy.config or {}).get("inference_url")

    metrics = RolloutMetrics(
        outcome_reward=reward,
        details={"predicted": predicted_intent, "expected": expected_intent},
    )

    policy_config = request.policy.config or {}
    trace_correlation_id = extract_trace_correlation_id(
        policy_config=policy_config,
        inference_url=str(inference_url or ""),
    )

    trace_metadata = {
        "env": "banking77",
        "split": sample["split"],
        "index": sample["index"],
        "correct": is_correct,
    }
    llm_usage = response_json.get("usage", {}) if isinstance(response_json, dict) else {}
    if llm_usage:
        trace_metadata["usage"] = dict(llm_usage)
    if isinstance(response_json, dict) and response_json.get("model"):
        trace_metadata["model"] = response_json.get("model")

    trace_payload = build_trace_payload(
        messages=rendered_messages,
        response=response_json if isinstance(response_json, dict) else None,
        correlation_id=trace_correlation_id,
        metadata=trace_metadata,
    )

    return RolloutResponse(
        run_id=request.run_id,
        metrics=metrics,
        trace=trace_payload,
        trace_correlation_id=trace_correlation_id,
        inference_url=str(inference_url or ""),
    )


def build_dataset() -> tuple[TaskDatasetRegistry, Banking77Dataset]:
    """Build the dataset registry and dataset instance."""
    registry = TaskDatasetRegistry()
    dataset = Banking77Dataset()
    registry.register(BANKING77_DATASET_SPEC, lambda _spec: dataset, cache=True)
    return registry, dataset


def _base_task_info() -> TaskInfo:
    """Get the base task info for Banking77."""
    return TaskInfo(
        task={
            "id": "banking77",
            "name": "Banking77 Intent Classification",
            "version": "1.0.0",
            "action_space": {
                "type": "tool_call",
                "tool_name": TOOL_NAME,
                "description": "Classify banking queries into one of 77 intent categories.",
            },
        },
        environment="banking77",
        dataset={
            **BANKING77_DATASET_SPEC.model_dump(),
            "hf_dataset": DATASET_NAME,
        },
        rubric={
            "version": "1",
            "criteria_count": 1,
            "source": "inline",
        },
        inference={
            "supports_proxy": True,
            "tool": TOOL_NAME,
        },
        limits={"max_turns": 1},
        task_metadata={"format": "tool_call"},
    )


def describe_taskset(dataset: Banking77Dataset) -> Mapping[str, Any]:
    """Describe the taskset for the API."""
    return {
        **BANKING77_DATASET_SPEC.model_dump(),
        "hf_dataset": DATASET_NAME,
        "num_labels": len(dataset.label_names),
        "sizes": {split: dataset.size(split) for split in AVAILABLE_SPLITS},
    }


def provide_task_instances(dataset: Banking77Dataset, seeds: Sequence[int]) -> Iterable[TaskInfo]:
    """Provide task instances for the given seeds."""
    base_info = _base_task_info()
    for seed in seeds:
        sample = dataset.sample(split=DEFAULT_SPLIT, index=seed)
        expected_intent = sample["label"]

        instance_rubric = {
            "outcome": {
                "name": "Intent Classification Accuracy",
                "criteria": [
                    {
                        "id": "intent_accuracy",
                        "description": f"Did it provide the correct intent: {expected_intent}?",
                        "weight": 1.0,
                        "expected_answer": expected_intent,
                    }
                ],
            },
        }

        dataset_dict = base_info.dataset
        if hasattr(dataset_dict, "model_dump"):
            dataset_dict = dataset_dict.model_dump()
        elif not isinstance(dataset_dict, dict):
            dataset_dict = dict(dataset_dict.__dict__) if hasattr(dataset_dict, "__dict__") else {}

        dataset_dict = {
            **dataset_dict,
            "split": sample["split"],
            "index": sample["index"],
        }

        task_metadata_dict = base_info.task_metadata
        if hasattr(task_metadata_dict, "model_dump"):
            task_metadata_dict = task_metadata_dict.model_dump()
        elif not isinstance(task_metadata_dict, dict):
            task_metadata_dict = (
                dict(task_metadata_dict.__dict__) if hasattr(task_metadata_dict, "__dict__") else {}
            )

        task_metadata_dict = {
            **task_metadata_dict,
            "query": sample["text"],
            "expected_intent": expected_intent,
        }

        yield TaskInfo(
            task=base_info.task,
            environment=base_info.environment,
            dataset=dataset_dict,
            rubric=instance_rubric,
            inference=base_info.inference,
            limits=base_info.limits,
            task_metadata=task_metadata_dict,
        )


OUTCOME_RUBRIC: Rubric = cast(
    Rubric,
    load_rubric(
        {
            "version": "1",
            "goal_text": "Classify banking customer queries into the correct intent category.",
            "aggregation": "weighted_sum",
            "criteria": [
                {
                    "id": "intent_accuracy",
                    "description": "Correctly classify the customer query into the appropriate banking intent.",
                    "weight": 1.0,
                }
            ],
        }
    ),
)

EVENTS_RUBRIC: Rubric = cast(
    Rubric,
    load_rubric(
        {
            "version": "1",
            "goal_text": "Use the banking77_classify tool correctly.",
            "aggregation": "weighted_sum",
            "criteria": [
                {
                    "id": "tool_usage",
                    "description": "Properly invoke the banking77_classify tool with the correct format.",
                    "weight": 1.0,
                }
            ],
        }
    ),
)


def build_config() -> LocalAPIConfig:
    """Build the LocalAPIConfig for the local API."""
    registry, dataset = build_dataset()
    base_info = _base_task_info()

    preload_dataset_splits(dataset, AVAILABLE_SPLITS, "banking77_local_api")

    startup_http_client, shutdown_http_client = create_http_client_hooks(
        timeout=30.0,
        log_prefix="banking77_local_api",
    )

    config = LocalAPIConfig(
        app_id="banking77",
        name="Banking77 Intent Classification Task",
        description="Banking77 dataset local API for classifying customer queries into banking intents.",
        base_task_info=base_info,
        provide_taskset_description=lambda: describe_taskset(dataset),
        provide_task_instances=lambda seeds: provide_task_instances(dataset, seeds),
        rollout=rollout_executor,
        dataset_registry=registry,
        rubrics=RubricBundle(outcome=OUTCOME_RUBRIC, events=EVENTS_RUBRIC),
        proxy=None,
        app_state={"banking77_dataset": dataset},
        cors_origins=["*"],
        startup_hooks=[startup_http_client],
        shutdown_hooks=[shutdown_http_client],
    )
    return config


register_local_api(
    entry=LocalAPIEntry(
        api_id="banking77",
        description="Banking77 intent classification local API using the banking77 dataset.",
        config_factory=build_config,
        aliases=("banking-intents",),
    )
)


def fastapi_app():
    """Return the FastAPI application for ASGI hosts."""
    with contextlib.suppress(Exception):
        load_dotenv(str(REPO_ROOT / ".env"), override=False)

    app = create_local_api(build_config())

    add_metadata_endpoint(app)

    @app.exception_handler(RequestValidationError)
    async def _on_validation_error(request: StarletteRequest, exc: RequestValidationError):
        try:
            hdr = request.headers
            snapshot = {
                "path": str(request.url.path),
                "have_x_api_key": bool(hdr.get("x-api-key")),
                "have_x_api_keys": bool(hdr.get("x-api-keys")),
                "have_authorization": bool(hdr.get("authorization")),
                "errors": exc.errors()[:5],
            }
            print("[422] validation", snapshot, flush=True)
        except Exception:
            pass
        return JSONResponse(
            status_code=422,
            content={"status": "invalid", "detail": exc.errors()[:5]},
        )

    return app


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run the Banking77 local API locally")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8102)
    parser.add_argument("--reload", action="store_true", help="Enable uvicorn autoreload")
    parser.add_argument(
        "--env-file",
        action="append",
        default=[],
        help="Additional .env files to load before startup",
    )
    args = parser.parse_args()

    default_env = Path(__file__).resolve().parents[3] / ".env"
    env_files = [str(default_env)] if default_env.exists() else []
    env_files.extend(args.env_file or [])

    run_local_api(
        build_config,
        host=args.host,
        port=args.port,
        reload=args.reload,
        env_files=env_files,
    )
