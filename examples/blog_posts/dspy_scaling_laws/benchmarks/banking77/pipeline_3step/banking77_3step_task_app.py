"""Three-step Banking77 pipeline task app: Analyzer → Reasoner → Classifier."""

from __future__ import annotations

import contextlib
import json
import os
import uuid
from collections.abc import Iterable, Sequence
from typing import Any, Mapping, MutableMapping

from fastapi import HTTPException, Request

from synth_ai.task.apps import ModalDeploymentConfig, TaskAppEntry, register_task_app
from synth_ai.task.contracts import (
    RolloutMetrics,
    RolloutRequest,
    RolloutResponse,
    RolloutStep,
    RolloutTrajectory,
    TaskInfo,
)
from synth_ai.task.server import ProxyConfig, RubricBundle, TaskAppConfig
from synth_ai.task.vendors import normalize_vendor_keys

# Import from single-step banking77 task app
import sys
from pathlib import Path
task_apps_dir = Path(__file__).resolve().parents[4] / "task_apps"
if str(task_apps_dir) not in sys.path:
    sys.path.insert(0, str(task_apps_dir))

from banking77.banking77_task_app import (
    BANKING77_DATASET_SPEC,
    AVAILABLE_SPLITS,
    Banking77Dataset,
    TOOL_NAME,
    DATASET_NAME,
    build_dataset,
    call_chat_completion,
    describe_taskset,
    provide_task_instances,
    _base_task_info,
    banking77_router,
    OUTCOME_RUBRIC,
    EVENTS_RUBRIC,
)


ANALYZER_MODULE_NAME = "analyzer"
REASONER_MODULE_NAME = "reasoner"
CLASSIFIER_MODULE_NAME = "classifier"


def _baseline_system_instruction(module: str) -> str:
    if module == ANALYZER_MODULE_NAME:
        return (
            "You are an expert banking assistant. Analyze the customer query and identify "
            "key elements such as the action requested, entities mentioned, and context. "
            "Provide your analysis using the `banking77_classify` tool."
        )
    elif module == REASONER_MODULE_NAME:
        return (
            "You reason about banking queries. Given the query and analysis from the "
            "analyzer, identify patterns and narrow down the likely intent category. "
            "Provide your reasoning using the `banking77_classify` tool."
        )
    return (
        "You are the final classifier. Given the query, analysis, and reasoning, "
        "make the final intent classification. Always return the label using the "
        "`banking77_classify` tool."
    )


def _baseline_user_pattern(module: str) -> str:
    if module == ANALYZER_MODULE_NAME:
        return (
            "Customer Query: {query}\n\n"
            "Analyze this query and identify key elements. Return an initial classification "
            "using the tool call."
        )
    elif module == REASONER_MODULE_NAME:
        return (
            "Customer Query: {query}\n"
            "Analysis Result: {analyzer_intent}\n\n"
            "Based on the analysis, reason about the likely intent category. "
            "Return your refined classification via the tool call."
        )
    return (
        "Customer Query: {query}\n"
        "Analysis: {analyzer_intent}\n"
        "Reasoning: {reasoner_intent}\n\n"
        "Make the final intent classification using the tool call."
    )


def _format_few_shot_examples(examples: Sequence[Mapping[str, Any]] | None) -> str:
    if not examples:
        return ""

    lines: list[str] = []
    for example in examples:
        if not isinstance(example, Mapping):
            continue
        input_text = (
            str(example.get("input"))
            or str(example.get("input_text", ""))
            or str(example.get("input_data", {}).get("text", ""))
        ).strip()
        output_text = (
            str(example.get("output"))
            or str(example.get("output_text", ""))
            or str(example.get("output_data", {}).get("label", ""))
            or str(example.get("output_data", {}).get("intent", ""))
        ).strip()
        if not input_text and not output_text:
            continue
        if output_text:
            lines.append(f"- Query: {input_text} -> Intent: {output_text}")
        else:
            lines.append(f"- Query: {input_text}")
    if not lines:
        return ""
    return "Few-shot Examples:\n" + "\n".join(lines)


def _resolve_module_configs(policy_config: MutableMapping[str, Any]) -> list[dict[str, Any]]:
    template = policy_config.get("prompt_template") or {}
    metadata = template.get("prompt_metadata") or {}
    modules = metadata.get("pipeline_modules")
    if isinstance(modules, list):
        resolved: list[dict[str, Any]] = []
        for module in modules:
            if isinstance(module, Mapping):
                resolved.append(dict(module))
        if resolved:
            return resolved

    # Fall back to baseline three-module pipeline when metadata absent
    return [
        {
            "name": ANALYZER_MODULE_NAME,
            "instruction_text": "",
            "few_shots": [],
        },
        {
            "name": REASONER_MODULE_NAME,
            "instruction_text": "",
            "few_shots": [],
        },
        {
            "name": CLASSIFIER_MODULE_NAME,
            "instruction_text": "",
            "few_shots": [],
        },
    ]


def _build_module_messages(
    module_name: str,
    module_config: Mapping[str, Any],
    placeholders: Mapping[str, Any],
) -> tuple[list[dict[str, str]], dict[str, Any]]:
    instruction_text = str(module_config.get("instruction_text", "")).strip()
    if not instruction_text and module_config.get("instruction_lines"):
        try:
            instruction_text = "\n".join(str(line).strip() for line in module_config.get("instruction_lines", []) if line)
        except Exception:
            instruction_text = str(module_config.get("instruction_lines"))

    few_shots_block = _format_few_shot_examples(module_config.get("few_shots"))

    system_parts = [_baseline_system_instruction(module_name)]
    if instruction_text:
        system_parts.append(instruction_text)
    if few_shots_block:
        system_parts.append(few_shots_block)
    if placeholders.get("available_intents"):
        system_parts.append("Available Intents:\n" + placeholders["available_intents"])

    system_message = "\n\n".join(part for part in system_parts if part)

    user_pattern = _baseline_user_pattern(module_name)
    user_message = user_pattern.format(**placeholders)

    messages = [
        {"role": "system", "pattern": "{system_message}"},
        {"role": "user", "pattern": "{user_message}"},
    ]

    module_placeholders = dict(placeholders)
    module_placeholders.update({
        "system_message": system_message,
        "user_message": user_message,
    })
    return messages, module_placeholders


def _extract_intent_from_response(
    response_json: Mapping[str, Any] | None,
    response_text: str,
    tool_calls: Sequence[Mapping[str, Any]] | None,
) -> str:
    if tool_calls:
        for call in tool_calls:
            fn = (call or {}).get("function", {}) or {}
            if fn.get("name") != TOOL_NAME:
                continue
            args_raw = fn.get("arguments", "{}")
            try:
                args = json.loads(args_raw)
            except Exception:
                continue
            intent = str(args.get("intent", "")).strip()
            if intent:
                return intent
    if response_json:
        choices = (response_json.get("choices") or []) if isinstance(response_json, Mapping) else []
        if choices:
            message = (choices[0] or {}).get("message", {}) or {}
            tool_list = message.get("tool_calls", []) or []
            for call in tool_list:
                fn = (call or {}).get("function", {}) or {}
                if fn.get("name") != TOOL_NAME:
                    continue
                args_raw = fn.get("arguments", "{}")
                try:
                    args = json.loads(args_raw)
                except Exception:
                    continue
                intent = str(args.get("intent", "")).strip()
                if intent:
                    return intent
            content = str(message.get("content", "")).strip()
            if content:
                return content.split()[0]
    if response_text and response_text.strip():
        return response_text.strip().split()[0]
    return ""


async def rollout_executor(request: RolloutRequest, fastapi_request: Request) -> RolloutResponse:
    dataset: Banking77Dataset = fastapi_request.app.state.banking77_dataset

    with contextlib.suppress(Exception):
        cfg = (request.policy.config or {})
        print(
            f"[TASK_APP] INBOUND_3STEP_PIPELINE: run_id={request.run_id} seed={request.env.seed} "
            f"env={request.env.env_name} policy.model={cfg.get('model')}",
            flush=True,
        )

    split = str(((request.env.config or {}).get("split")) or "train")
    seed = request.env.seed or 0

    sample = dataset.sample(split=split, index=seed)
    observation = {
        "query": sample["text"],
        "index": sample["index"],
        "split": sample["split"],
        "available_intents": dataset.label_names,
    }

    available_intents = "\n".join(f"{i+1}. {label}" for i, label in enumerate(dataset.label_names))

    # Extract API key from request headers or environment
    api_key = fastapi_request.headers.get("X-API-Key") or os.getenv("GROQ_API_KEY")
    policy_config = request.policy.config or {}
    # Add API key to policy config so call_chat_completion can use it
    if api_key:
        policy_config = {**policy_config, "api_key": api_key}
    modules = _resolve_module_configs(policy_config)

    pipeline_records: list[dict[str, Any]] = []
    analyzer_intent = ""
    reasoner_intent = ""

    for module in modules:
        module_name = str(module.get("name", "")).strip() or ANALYZER_MODULE_NAME
        placeholders: dict[str, Any] = {
            "query": sample["text"],
            "available_intents": available_intents,
            "analyzer_intent": analyzer_intent,
            "reasoner_intent": reasoner_intent,
        }
        messages, formatted_placeholders = _build_module_messages(module_name, module, placeholders)

        # Extract API key from request headers
        api_key_from_x = fastapi_request.headers.get("X-API-Key") or fastapi_request.headers.get("x-api-key")
        api_key_from_auth = None
        if fastapi_request.headers.get("Authorization"):
            auth_header = fastapi_request.headers.get("Authorization", "")
            if auth_header.startswith("Bearer "):
                api_key_from_auth = auth_header.replace("Bearer ", "").strip()

        api_key = api_key_from_x or api_key_from_auth or None

        response_text, response_json, tool_calls = await call_chat_completion(
            policy_config,
            formatted_placeholders,
            messages,
            api_key=api_key,
        )

        if not isinstance(response_json, dict) or not response_json:
            raise RuntimeError(f"Module {module_name}: proxy returned missing/empty JSON")

        predicted_intent = _extract_intent_from_response(response_json, response_text, tool_calls)
        if not predicted_intent:
            raise RuntimeError(f"Module {module_name}: no intent extracted from response")

        pipeline_records.append(
            {
                "module": module_name,
                "instruction_text": module.get("instruction_text"),
                "few_shots": module.get("few_shots"),
                "messages": messages,
                "response": response_json,
                "tool_calls": tool_calls,
                "predicted_intent": predicted_intent,
            }
        )

        # Update intents for next module
        if module_name == ANALYZER_MODULE_NAME:
            analyzer_intent = predicted_intent
        elif module_name == REASONER_MODULE_NAME:
            reasoner_intent = predicted_intent

    expected_intent = sample["label"]
    final_intent = predicted_intent  # Last module's prediction
    is_correct = final_intent.lower().replace("_", " ") == expected_intent.lower().replace("_", " ")
    reward = 1.0 if is_correct else 0.0

    print(
        f"[TASK_APP] 3STEP_RESULT: expected={expected_intent} predicted={final_intent} correct={is_correct}",
        flush=True,
    )

    # Store messages from the first module for baseline extraction
    first_module_messages = pipeline_records[0]["messages"] if pipeline_records else []

    step = RolloutStep(
        obs=observation,
        tool_calls=pipeline_records[-1]["tool_calls"] if pipeline_records else [],
        reward=reward,
        done=True,
        info={
            "expected_intent": expected_intent,
            "predicted_intent": final_intent,
            "modules": pipeline_records,
            "correct": is_correct,
            "messages": first_module_messages,
        },
    )

    inference_url = (request.policy.config or {}).get("inference_url")
    trajectory = RolloutTrajectory(
        env_id=f"banking77_3step::{sample['split']}::{sample['index']}",
        policy_id=request.policy.policy_id or request.policy.policy_name or "policy",
        steps=[step],
        final={"observation": observation, "reward": reward},
        length=len(modules),
        inference_url=str(inference_url or ""),
    )

    metrics = RolloutMetrics(
        episode_returns=[reward],
        mean_return=reward,
        num_steps=len(modules),
        num_episodes=1,
        outcome_score=reward,
        events_score=reward,
        details={"correct": is_correct},
    )

    include_trace = bool(
        (request.record and getattr(request.record, "return_trace", False))
        or os.getenv("TASKAPP_TRACING_ENABLED")
    )
    trace_payload = None
    if include_trace:
        trace_payload = {
            "session_id": str(uuid.uuid4()),
            "events_count": len(modules),
            "decision_rewards": [reward],
            "metadata": {
                "env": "banking77_3step",
                "split": sample["split"],
                "index": sample["index"],
                "correct": is_correct,
            },
        }

    return RolloutResponse(
        run_id=request.run_id,
        trajectories=[trajectory],
        branches={},
        metrics=metrics,
        aborted=False,
        ops_executed=len(modules) * 2,
        trace=trace_payload,
        pipeline_metadata={
            "modules": pipeline_records,
            "final_intent": final_intent,
            "expected_intent": expected_intent,
        },
    )


def describe_pipeline_taskset(dataset: Banking77Dataset) -> Mapping[str, Any]:
    return describe_taskset(dataset)


def provide_pipeline_task_instances(dataset: Banking77Dataset, seeds: Sequence[int]) -> Iterable[TaskInfo]:
    for info in provide_task_instances(dataset, seeds):
        payload = info.model_dump()
        payload["environment"] = "banking77_3step"
        task_meta = payload.get("task_metadata") or {}
        task_meta["modules"] = [ANALYZER_MODULE_NAME, REASONER_MODULE_NAME, CLASSIFIER_MODULE_NAME]
        payload["task_metadata"] = task_meta
        payload.setdefault("limits", {})
        payload["limits"]["max_turns"] = 3
        task_payload = payload.get("task") or {}
        task_payload["id"] = "banking77_3step"
        task_payload["name"] = "Banking77 Three-Step Pipeline"
        task_payload["description"] = (
            "Banking77 intent classification with 3-step pipeline: analyzer, reasoner, classifier."
        )
        payload["task"] = task_payload
        yield TaskInfo(**payload)


def build_config() -> TaskAppConfig:
    registry, dataset = build_dataset()
    base_info = _base_task_info()

    proxy_keys = normalize_vendor_keys()
    proxy_config = ProxyConfig(
        enable_openai=proxy_keys.get("OPENAI_API_KEY") is not None,
        enable_groq=proxy_keys.get("GROQ_API_KEY") is not None,
        system_hint="Use the banking77_classify tool to classify queries.",
    )

    base_payload = base_info.model_dump()
    base_payload["environment"] = "banking77_3step"
    base_payload.setdefault("limits", {})
    base_payload["limits"]["max_turns"] = 3
    base_payload.setdefault("task_metadata", {})
    base_payload["task_metadata"]["modules"] = [ANALYZER_MODULE_NAME, REASONER_MODULE_NAME, CLASSIFIER_MODULE_NAME]
    base_task = base_payload.get("task") or {}
    base_task["id"] = "banking77_3step"
    base_task["name"] = "Banking77 Three-Step Pipeline"
    base_payload["task"] = base_task

    dataset_payload = BANKING77_DATASET_SPEC.model_dump()
    dataset_payload["hf_dataset"] = DATASET_NAME
    base_payload["dataset"] = dataset_payload

    config = TaskAppConfig(
        app_id="banking77-3step",
        name="Banking77 Three-Step Pipeline",
        description=(
            "Banking77 intent classification with 3-step pipeline: analyzer, reasoner, classifier."
        ),
        base_task_info=TaskInfo(**base_payload),
        describe_taskset=lambda: describe_pipeline_taskset(dataset),
        provide_task_instances=lambda seeds: provide_pipeline_task_instances(dataset, seeds),
        rollout=rollout_executor,
        dataset_registry=registry,
        rubrics=RubricBundle(outcome=OUTCOME_RUBRIC, events=EVENTS_RUBRIC),
        proxy=proxy_config,
        routers=(banking77_router,),
        app_state={"banking77_dataset": dataset},
        cors_origins=["*"],
    )
    return config


register_task_app(
    entry=TaskAppEntry(
        app_id="banking77-3step",
        description="Banking77 three-step pipeline (analyzer + reasoner + classifier).",
        config_factory=build_config,
        aliases=("banking77-threestep",),
        modal=ModalDeploymentConfig(
            app_name="synth-banking77-3step",
            pip_packages=(
                "datasets>=2.14.0",
                "fastapi>=0.115.0",
                "pydantic>=2.0.0",
                "python-dotenv>=1.0.0",
            ),
            timeout=600,
            memory=4096,
            cpu=2.0,
        ),
    )
)
