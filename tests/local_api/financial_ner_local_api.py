"""Financial NER local API for Synth prompt optimization benchmarks."""

from __future__ import annotations

import contextlib
import json
import os
import uuid
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any, Mapping, cast

from dotenv import load_dotenv
from fastapi import HTTPException, Request

# Synth-AI SDK imports
from synth_ai.sdk.localapi.helpers import (
    add_metadata_endpoint,
    call_chat_completion_api,
    create_http_client_hooks,
    extract_api_key,
    preload_dataset_splits,
)
from synth_ai.sdk.localapi.apps import LocalAPIEntry, ModalDeploymentConfig, register_local_api
from synth_ai.sdk.task.contracts import (
    RolloutMetrics,
    RolloutRequest,
    RolloutResponse,
    TaskInfo,
)
from synth_ai.sdk.task.datasets import TaskDatasetRegistry, TaskDatasetSpec
from synth_ai.sdk.task.rubrics import Rubric, load_rubric
from synth_ai.sdk.localapi.server import LocalAPIConfig, RubricBundle, create_local_api, run_local_api
from synth_ai.sdk.task.trace_correlation_helpers import extract_trace_correlation_id

# Business logic imports (no synth-ai dependencies)
from financial_ner_business_logic import (
    AVAILABLE_SPLITS,
    DATASET_NAME,
    DEFAULT_SPLIT,
    ENTITY_TYPES,
    REPO_ROOT,
    TOOL_NAME,
    FinancialNERDataset,
    FinancialNERScorer,
    get_default_messages_templates,
    get_extract_tool_schema,
    parse_entities_from_tool_call,
)


# Log environment at module load time for debugging
print(
    f"[financial_ner_task_app] Module loaded: DATASET_NAME={DATASET_NAME}, "
    f"HF_HOME={os.getenv('HF_HOME')}, "
    f"HF_DATASETS_CACHE={os.getenv('HF_DATASETS_CACHE')}, "
    f"HF_HUB_CACHE={os.getenv('HF_HUB_CACHE')}",
    flush=True,
)


# Dataset spec for registry
FINANCIAL_NER_DATASET_SPEC = TaskDatasetSpec(
    id="financial_ner",
    name="Financial NER Task",
    version="1.0.0",
    splits=list(AVAILABLE_SPLITS),
    default_split=DEFAULT_SPLIT,
    description="Financial named entity recognition task extracting 7 entity types from financial news.",
)


async def rollout_executor(request: RolloutRequest, fastapi_request: Request) -> RolloutResponse:
    """Execute a rollout for the Financial NER task."""
    dataset: FinancialNERDataset = fastapi_request.app.state.financial_ner_dataset

    with contextlib.suppress(Exception):
        cfg = (request.policy.config or {})
        print(
            f"[TASK_APP] INBOUND_ROLLOUT: run_id={request.run_id} seed={request.env.seed} env={request.env.env_name} "
            f"policy.model={cfg.get('model')} provider={cfg.get('provider')} api_base={cfg.get('inference_url') or cfg.get('api_base') or cfg.get('base_url')}",
            flush=True,
        )

    split = str(((request.env.config or {}).get("split")) or DEFAULT_SPLIT)
    seed = request.env.seed or 0

    sample = dataset.sample(split=split, index=seed)

    entity_types_list = ", ".join(ENTITY_TYPES)
    placeholders = {
        "text": sample["text"],
        "entity_types": entity_types_list,
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

    extract_tool = get_extract_tool_schema()
    response_text, response_json, tool_calls = await call_chat_completion_api(
        policy_config=request.policy.config or {},
        messages=rendered_messages,
        tools=[extract_tool] if extract_tool else None,
        tool_choice="required" if extract_tool else None,
        api_key=api_key,
        http_client=http_client,
        expected_tool_name=TOOL_NAME,
        log_prefix="[TASK_APP]",
    )

    # Validate response
    try:
        raw_upstream = json.dumps(response_json, ensure_ascii=False)
    except Exception:
        raw_upstream = str(response_json)
    print(f"[TASK_APP] UPSTREAM_RESPONSE_JSON ({len(raw_upstream)} bytes): {raw_upstream}", flush=True)

    if not isinstance(response_json, dict) or not response_json:
        raise RuntimeError("Proxy returned missing/empty JSON")

    if tool_calls:
        for tc in tool_calls:
            args_str = tc.get("function", {}).get("arguments", "{}")
            try:
                args = json.loads(args_str)
            except Exception as exc:
                raise HTTPException(status_code=502, detail="Tool call arguments not valid JSON") from exc
            if "entities" not in args or not isinstance(args["entities"], dict):
                raise HTTPException(status_code=502, detail="Tool call missing valid 'entities' dict")

    # Extract predicted entities using business logic helper
    predicted_entities: dict[str, list[str]] = {etype: [] for etype in ENTITY_TYPES}
    if tool_calls:
        for tc in tool_calls:
            if tc.get("function", {}).get("name") == TOOL_NAME:
                args_str = tc.get("function", {}).get("arguments", "{}")
                predicted_entities = parse_entities_from_tool_call(args_str)
                print(f"[TASK_APP] PARSED_TOOL_ENTITIES: {predicted_entities}", flush=True)
    elif response_text:
        try:
            parsed = json.loads(response_text)
            if isinstance(parsed, dict) and "entities" in parsed:
                predicted_entities = parsed["entities"]
        except Exception:
            print(f"[TASK_APP] CONTENT_FALLBACK_PARSE_ERROR: {response_text[:200]}", flush=True)

    if not any(predicted_entities.values()):
        print(f"[TASK_APP] WARNING: No entities extracted from proxy response, returning 0 score", flush=True)

    expected_entities = sample["entities"]

    # Score using business logic
    correct_types, total_types, reward = FinancialNERScorer.score_entities(predicted_entities, expected_entities)
    is_correct = reward == 1.0

    print(
        f"[TASK_APP] PREDICTION: expected_entities={expected_entities} predicted_entities={predicted_entities} "
        f"correct_types={correct_types}/{total_types} reward={reward:.2f}",
        flush=True,
    )

    with contextlib.suppress(Exception):
        print(
            f"[FINANCIAL_NER_ROLLOUT] run_id={request.run_id} split={sample['split']} "
            f"index={sample['index']} correct_types={correct_types}/{total_types} reward={reward:.2f}",
            flush=True,
        )

    inference_url = (request.policy.config or {}).get("inference_url")
    trace_correlation_id = extract_trace_correlation_id(
        policy_config=request.policy.config or {},
        inference_url=str(inference_url or ""),
    )

    # Build V3 trace for verifier evaluation
    trace_id = str(uuid.uuid4())
    v3_event_history = [
        {
            "type": "llm_request",
            "step_index": 0,
            "llm_request": {
                "messages": rendered_messages,
                "model": (request.policy.config or {}).get("model", "unknown"),
            },
            "llm_response": {
                "message": {
                    "content": response_text,
                    "tool_calls": tool_calls,
                },
                "model": (request.policy.config or {}).get("model", "unknown"),
            },
        },
        {
            "type": "environment_step",
            "step_index": 1,
            "observation": {
                "text": sample["text"],
                "predicted_entities": predicted_entities,
                "expected_entities": expected_entities,
            },
            "reward": reward,
            "terminated": True,
            "info": {
                "correct": is_correct,
                "correct_types": correct_types,
                "total_types": total_types,
            },
        },
    ]
    trajectory_trace = {
        "schema_version": "3.0",
        "event_history": v3_event_history,
        "markov_blanket_message_history": [],
        "metadata": {
            "trace_id": trace_id,
            "session_id": trace_id,
            "environment": "financial_ner",
            "split": sample["split"],
            "index": sample["index"],
            "correct": is_correct,
            "correct_types": correct_types,
            "expected_entities": expected_entities,
            "correlation_ids": {
                "run_id": request.run_id,
                "seed": seed,
            },
        },
    }
    if trace_correlation_id:
        metadata_block = trajectory_trace.get("metadata")
        if isinstance(metadata_block, dict):
            metadata_block["trace_correlation_id"] = trace_correlation_id
            corr_ids = metadata_block.get("correlation_ids")
            corr_map = dict(corr_ids) if isinstance(corr_ids, dict) else {}
            corr_map.setdefault("trace_correlation_id", trace_correlation_id)
            metadata_block["correlation_ids"] = corr_map

    metrics = RolloutMetrics(
        outcome_reward=reward,
        details={"correct": is_correct, "correct_types": correct_types},
    )

    trace_payload = trajectory_trace

    return RolloutResponse(
        run_id=request.run_id,
        metrics=metrics,
        trace_correlation_id=trace_correlation_id,
        trace=trace_payload,
        inference_url=str(inference_url or ""),
    )


def build_dataset() -> tuple[TaskDatasetRegistry, FinancialNERDataset]:
    """Build the dataset registry and dataset instance."""
    registry = TaskDatasetRegistry()
    dataset = FinancialNERDataset()
    registry.register(FINANCIAL_NER_DATASET_SPEC, lambda _spec: dataset, cache=True)
    return registry, dataset


def _base_task_info() -> TaskInfo:
    """Get the base task info for Financial NER."""
    return TaskInfo(
        task={
            "id": "financial_ner",
            "name": "Financial NER Task",
            "version": "1.0.0",
            "action_space": {
                "type": "tool_call",
                "tool_name": TOOL_NAME,
                "description": "Extract named entities from financial text.",
            },
        },
        environment="financial_ner",
        dataset={
            **FINANCIAL_NER_DATASET_SPEC.model_dump(),
        },
        rubric={
            "version": "1",
            "criteria_count": 7,
            "source": "inline",
        },
        inference={
            "supports_proxy": True,
            "tool": TOOL_NAME,
        },
        limits={"max_turns": 1},
        task_metadata={"format": "tool_call"},
    )


def describe_taskset(dataset: FinancialNERDataset) -> Mapping[str, Any]:
    """Describe the taskset for the API."""
    return {
        **FINANCIAL_NER_DATASET_SPEC.model_dump(),
        "num_entity_types": len(ENTITY_TYPES),
        "entity_types": ENTITY_TYPES,
        "sizes": {split: dataset.size(split) for split in AVAILABLE_SPLITS},
    }


def provide_task_instances(dataset: FinancialNERDataset, seeds: Sequence[int]) -> Iterable[TaskInfo]:
    """Provide task instances for the given seeds."""
    base_info = _base_task_info()
    for seed in seeds:
        sample = dataset.sample(split=DEFAULT_SPLIT, index=seed)
        expected_entities = sample["entities"]

        instance_rubric = {
            "outcome": {
                "name": "Entity Extraction Accuracy",
                "criteria": [
                    {
                        "id": f"{etype}_accuracy",
                        "description": f"Correctly extract {etype} entities. Expected: {expected_entities.get(etype, [])}",
                        "weight": 1.0 / len(ENTITY_TYPES),
                        "expected_answer": expected_entities.get(etype, []),
                    }
                    for etype in ENTITY_TYPES
                ]
            },
        }

        dataset_dict = base_info.dataset
        if hasattr(dataset_dict, "model_dump"):
            dataset_dict = dataset_dict.model_dump()
        elif not isinstance(dataset_dict, dict):
            if hasattr(dataset_dict, "__dict__"):
                dataset_dict = dict(dataset_dict.__dict__)
            else:
                dataset_dict = {}

        dataset_dict = {
            **dataset_dict,
            "split": sample["split"],
            "index": sample["index"],
        }

        task_metadata_dict = base_info.task_metadata
        if hasattr(task_metadata_dict, "model_dump"):
            task_metadata_dict = task_metadata_dict.model_dump()
        elif not isinstance(task_metadata_dict, dict):
            if hasattr(task_metadata_dict, "__dict__"):
                task_metadata_dict = dict(task_metadata_dict.__dict__)
            else:
                task_metadata_dict = {}

        task_metadata_dict = {
            **task_metadata_dict,
            "text": sample["text"],
            "expected_entities": expected_entities,
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
            "goal_text": "Extract all named entities of specified types from financial text.",
            "aggregation": "weighted_sum",
            "criteria": [
                {
                    "id": f"{etype}_extraction",
                    "description": f"Correctly extract all {etype} entities from the text.",
                    "weight": 1.0 / len(ENTITY_TYPES),
                }
                for etype in ENTITY_TYPES
            ],
        }
    ),
)

EVENTS_RUBRIC: Rubric = cast(
    Rubric,
    load_rubric(
        {
            "version": "1",
            "goal_text": "Use the extract_entities tool correctly.",
            "aggregation": "weighted_sum",
            "criteria": [
                {
                    "id": "tool_usage",
                    "description": "Properly invoke the extract_entities tool with correct JSON format.",
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

    preload_dataset_splits(dataset, AVAILABLE_SPLITS, "financial_ner_task_app")

    startup_http_client, shutdown_http_client = create_http_client_hooks(
        timeout=30.0,
        log_prefix="financial_ner_task_app",
    )

    config = LocalAPIConfig(
        app_id="financial_ner",
        name="Financial NER Task",
        description="Financial named entity recognition task for extracting entities from financial news.",
        base_task_info=base_info,
        describe_taskset=lambda: describe_taskset(dataset),
        provide_task_instances=lambda seeds: provide_task_instances(dataset, seeds),
        rollout=rollout_executor,
        dataset_registry=registry,
        rubrics=RubricBundle(outcome=OUTCOME_RUBRIC, events=EVENTS_RUBRIC),
        proxy=None,
        app_state={"financial_ner_dataset": dataset},
        cors_origins=["*"],
        startup_hooks=[startup_http_client],
        shutdown_hooks=[shutdown_http_client],
    )
    return config


register_local_api(
    entry=LocalAPIEntry(
        api_id="financial_ner",
        description="Financial NER local API for entity extraction from financial text.",
        config_factory=build_config,
        aliases=("financial-entities",),
        modal=ModalDeploymentConfig(
            app_name="synth-financial-ner",
            pip_packages=(
                "fastapi>=0.115.0",
                "pydantic>=2.0.0",
                "httpx>=0.26.0",
            ),
            extra_local_dirs=((str(REPO_ROOT / "synth_ai"), "/opt/synth_ai_repo/synth_ai"),),
        ),
    )
)

# Modal deployment
try:
    import modal

    app = modal.App("synth-financial-ner")

    _image = (
        modal.Image.debian_slim(python_version="3.11")
        .pip_install(
            "synth-ai",
            "fastapi>=0.115.0",
            "pydantic>=2.0.0",
            "httpx>=0.26.0",
            "python-dotenv>=1.0.0",
        )
        .env({"PYTHONPATH": "/opt/synth_ai_repo"})
        .add_local_dir(str(REPO_ROOT / "synth_ai"), "/opt/synth_ai_repo/synth_ai", copy=True)
    )
    _env_file = REPO_ROOT / ".env"
    if _env_file.exists():
        _image = _image.add_local_file(str(_env_file), "/opt/synth_ai_repo/.env")

    @app.function(
        image=_image,
        timeout=600,
    )
    @modal.asgi_app()
    def web():
        return fastapi_app()

except ImportError:
    pass


def fastapi_app():
    """Return the FastAPI application for Modal or other ASGI hosts."""
    with contextlib.suppress(Exception):
        load_dotenv(str(REPO_ROOT / ".env"), override=False)

    app = create_local_api(build_config())
    add_metadata_endpoint(app)
    return app


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run the Financial NER local API")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8001)
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
