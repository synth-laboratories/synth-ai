"""Three-step HeartDisease pipeline: Analyzer → Reasoner → Classifier."""

from __future__ import annotations

import contextlib
import json
import os
import uuid
from collections.abc import Iterable, Sequence
from typing import Any, Mapping, MutableMapping
from pathlib import Path

from fastapi import HTTPException, Request, APIRouter

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

import sys
task_apps_dir = Path(__file__).resolve().parents[4] / "task_apps"
if str(task_apps_dir) not in sys.path:
    sys.path.insert(0, str(task_apps_dir))

from other_langprobe_benchmarks.heartdisease_task_app import (
    HEARTDISEASE_DATASET_SPEC,
    AVAILABLE_SPLITS,
    HeartDiseaseDataset,
    HEARTDISEASE_DATASET,
    DEFAULT_SPLIT,
    build_dataset,
    describe_taskset,
    provide_task_instances,
    _base_task_info,
    _normalize_classification,
    heartdisease_router,
    OUTCOME_RUBRIC,
    EVENTS_RUBRIC,
)
import gepa_benchmarks.common

ANALYZER_MODULE_NAME = "analyzer"
REASONER_MODULE_NAME = "reasoner"
CLASSIFIER_MODULE_NAME = "classifier"
TOOL_NAME = "heart_disease_classify"

def _baseline_system_instruction(module: str) -> str:
    if module == ANALYZER_MODULE_NAME:
        return "You are a medical analysis assistant. Analyze patient features to identify key risk factors and clinical indicators. Use the `heart_disease_classify` tool to provide your initial assessment."
    elif module == REASONER_MODULE_NAME:
        return "You are a medical reasoning assistant. Given the patient features and initial analysis, reason about the clinical significance and disease likelihood. Use the `heart_disease_classify` tool to provide your refined assessment."
    return "You are the final medical classifier. Based on all previous analysis and reasoning, make the final classification. Use the `heart_disease_classify` tool with '1' for heart disease or '0' for no disease."

def _baseline_user_pattern(module: str) -> str:
    if module == ANALYZER_MODULE_NAME:
        return "Patient Features:\n{features}\n\nAnalyze these features and provide an initial assessment using the tool."
    elif module == REASONER_MODULE_NAME:
        return "Patient Features:\n{features}\n\nInitial Analysis: {analyzer_result}\n\nReason about the disease likelihood based on the analysis."
    return "Patient Features:\n{features}\n\nAnalysis: {analyzer_result}\nReasoning: {reasoner_result}\n\nProvide the final classification (0 or 1)."

def _format_few_shot_examples(examples: Sequence[Mapping[str, Any]] | None) -> str:
    if not examples:
        return ""
    lines = []
    for ex in examples:
        if not isinstance(ex, Mapping):
            continue
        inp = str(ex.get("input") or ex.get("input_text", "")).strip()
        out = str(ex.get("output") or ex.get("output_text", "") or ex.get("output_data", {}).get("classification", "")).strip()
        if inp or out:
            lines.append(f"- Features: {inp[:100]}... -> Classification: {out}" if out else f"- Features: {inp[:100]}...")
    return "Few-shot Examples:\n" + "\n".join(lines) if lines else ""

def _resolve_module_configs(policy_config: MutableMapping[str, Any]) -> list[dict[str, Any]]:
    template = policy_config.get("prompt_template") or {}
    metadata = template.get("prompt_metadata") or {}
    modules = metadata.get("pipeline_modules")
    if isinstance(modules, list):
        resolved = [dict(m) for m in modules if isinstance(m, Mapping)]
        if resolved:
            return resolved
    return [
        {"name": ANALYZER_MODULE_NAME, "instruction_text": "", "few_shots": []},
        {"name": REASONER_MODULE_NAME, "instruction_text": "", "few_shots": []},
        {"name": CLASSIFIER_MODULE_NAME, "instruction_text": "", "few_shots": []},
    ]

def _build_module_messages(module_name: str, module_config: Mapping[str, Any], placeholders: Mapping[str, Any]) -> tuple[list[dict[str, str]], dict[str, Any]]:
    instruction = str(module_config.get("instruction_text", "")).strip()
    if not instruction and module_config.get("instruction_lines"):
        try:
            instruction = "\n".join(str(l).strip() for l in module_config.get("instruction_lines", []) if l)
        except:
            instruction = str(module_config.get("instruction_lines"))

    few_shots = _format_few_shot_examples(module_config.get("few_shots"))
    parts = [_baseline_system_instruction(module_name)]
    if instruction:
        parts.append(instruction)
    if few_shots:
        parts.append(few_shots)

    system_msg = "\n\n".join(p for p in parts if p)
    user_msg = _baseline_user_pattern(module_name).format(**placeholders)
    messages = [
        {"role": "system", "pattern": "{system_message}"},
        {"role": "user", "pattern": "{user_message}"},
    ]
    module_placeholders = dict(placeholders)
    module_placeholders.update({"system_message": system_msg, "user_message": user_msg})
    return messages, module_placeholders

def _extract_classification_from_response(response_json: Mapping[str, Any] | None, response_text: str, tool_calls: Sequence[Mapping[str, Any]] | None) -> str:
    if tool_calls:
        for call in tool_calls:
            args = call.get("arguments", {})
            if isinstance(args, dict):
                classification = str(args.get("classification", "")).strip()
                if classification:
                    return classification
    if response_json:
        choices = response_json.get("choices", []) if isinstance(response_json, Mapping) else []
        if choices:
            msg = (choices[0] or {}).get("message", {}) or {}
            for call in msg.get("tool_calls", []) or []:
                fn = (call or {}).get("function", {}) or {}
                if fn.get("name") == TOOL_NAME:
                    try:
                        args = json.loads(fn.get("arguments", "{}"))
                        classification = str(args.get("classification", "")).strip()
                        if classification:
                            return classification
                    except:
                        pass
            content = str(msg.get("content", "")).strip()
            if content:
                return content.split()[0]
    if response_text and response_text.strip():
        return response_text.strip().split()[0]
    return ""

async def rollout_executor(request: RolloutRequest, fastapi_request: Request) -> RolloutResponse:
    dataset: HeartDiseaseDataset = fastapi_request.app.state.heartdisease_dataset

    split = str(((request.env.config or {}).get("split")) or DEFAULT_SPLIT)
    seed = request.env.seed or 0
    sample = dataset.sample(split=split, index=seed)
    observation = {"features": sample["feature_text"], "index": sample["index"], "split": sample["split"]}

    # Extract API key from request headers or environment
    api_key = fastapi_request.headers.get("X-API-Key") or os.getenv("GROQ_API_KEY")
    policy_config = request.policy.config or {}
    # Add API key to policy config so call_chat_completion can use it
    if api_key:
        policy_config = {**policy_config, "api_key": api_key}
    modules = _resolve_module_configs(policy_config)

    tool_spec = [{
        "type": "function",
        "function": {
            "name": TOOL_NAME,
            "description": "Submit classification: '1' for heart disease, '0' for no disease",
            "parameters": {
                "type": "object",
                "properties": {"classification": {"type": "string", "enum": ["0", "1"]}},
                "required": ["classification"],
            },
        }
    }]

    pipeline_records = []
    results = {"analyzer_result": "", "reasoner_result": ""}

    for module in modules:
        module_name = str(module.get("name", "")).strip() or ANALYZER_MODULE_NAME
        placeholders = {"features": sample["feature_text"], **results}
        messages, formatted_placeholders = _build_module_messages(module_name, module, placeholders)

        # API key already in policy_config from earlier extraction
        response_text, response_json, raw_tool_calls = await gepa_benchmarks.common.call_chat_completion(
            policy_config, formatted_placeholders, messages, tool_spec=tool_spec, tool_choice="required"
        )

        if not isinstance(response_json, dict) or not response_json:
            raise RuntimeError(f"Module {module_name}: empty response")

        # Parse tool calls
        tool_calls_list = []
        if response_json:
            choices = response_json.get("choices", [])
            if choices:
                message = choices[0].get("message", {})
                raw_tool_calls = message.get("tool_calls", []) or []
                for call in raw_tool_calls:
                    fn = call.get("function", {})
                    name = fn.get("name")
                    arguments_str = fn.get("arguments", "{}")
                    try:
                        arguments = json.loads(arguments_str)
                    except:
                        arguments = {}
                    tool_calls_list.append({
                        "id": call.get("id"),
                        "name": name,
                        "arguments": arguments,
                        "output": "Prediction received.",
                    })

        predicted = _extract_classification_from_response(response_json, response_text, tool_calls_list)
        if not predicted:
            raise RuntimeError(f"Module {module_name}: no classification extracted")

        pipeline_records.append({
            "module": module_name,
            "instruction_text": module.get("instruction_text"),
            "few_shots": module.get("few_shots"),
            "messages": messages,
            "response": response_json,
            "tool_calls": tool_calls_list,
            "predicted_classification": predicted,
        })

        if module_name == ANALYZER_MODULE_NAME:
            results["analyzer_result"] = predicted
        elif module_name == REASONER_MODULE_NAME:
            results["reasoner_result"] = predicted

    expected_label = sample["target"]
    final_prediction = _normalize_classification(predicted)
    is_correct = final_prediction == expected_label
    reward = 1.0 if is_correct else 0.0

    print(f"[TASK_APP] HD_3STEP_RESULT: expected={expected_label} predicted={final_prediction} correct={is_correct}", flush=True)

    step = RolloutStep(
        obs=observation,
        tool_calls=pipeline_records[-1]["tool_calls"] if pipeline_records else [],
        reward=reward,
        done=True,
        info={
            "expected_label": expected_label,
            "predicted_label": final_prediction,
            "label_correct": is_correct,
            "modules": pipeline_records,
            "messages": pipeline_records[0]["messages"] if pipeline_records else [],
        },
    )

    trajectory = RolloutTrajectory(
        env_id=f"heartdisease_3step::{sample['split']}::{sample['index']}",
        policy_id=request.policy.policy_id or request.policy.policy_name or "policy",
        steps=[step],
        final={"observation": observation, "reward": reward},
        length=len(modules),
        inference_url=str((request.policy.config or {}).get("inference_url") or ""),
    )

    metrics = RolloutMetrics(
        episode_returns=[reward], mean_return=reward, num_steps=len(modules), num_episodes=1,
        outcome_score=reward, events_score=reward, details={"label_correct": is_correct}
    )

    include_trace = bool((request.record and getattr(request.record, "return_trace", False)) or os.getenv("TASKAPP_TRACING_ENABLED"))
    trace_payload = {
        "session_id": str(uuid.uuid4()), "events_count": len(modules), "decision_rewards": [reward],
        "metadata": {"env": "heartdisease_3step", "split": sample["split"], "index": sample["index"], "label_correct": is_correct}
    } if include_trace else None

    return RolloutResponse(
        run_id=request.run_id, trajectories=[trajectory], branches={}, metrics=metrics,
        aborted=False, ops_executed=len(modules) * 2, trace=trace_payload,
        pipeline_metadata={"modules": pipeline_records, "final_classification": final_prediction, "expected_label": expected_label}
    )

def describe_pipeline_taskset(dataset: HeartDiseaseDataset) -> Mapping[str, Any]:
    return describe_taskset(dataset)

def provide_pipeline_task_instances(dataset: HeartDiseaseDataset, seeds: Sequence[int]) -> Iterable[TaskInfo]:
    for info in provide_task_instances(dataset, seeds):
        payload = info.model_dump()
        payload["environment"] = "heartdisease_3step"
        task_meta = payload.get("task_metadata") or {}
        task_meta["modules"] = [ANALYZER_MODULE_NAME, REASONER_MODULE_NAME, CLASSIFIER_MODULE_NAME]
        payload["task_metadata"] = task_meta
        payload.setdefault("limits", {})["max_turns"] = 3
        task_payload = payload.get("task") or {}
        task_payload.update({"id": "heartdisease_3step", "name": "HeartDisease Three-Step Pipeline",
                            "description": "Heart disease classification with 3-step pipeline: analyzer, reasoner, classifier."})
        payload["task"] = task_payload
        yield TaskInfo(**payload)

def build_config() -> TaskAppConfig:
    registry, dataset = build_dataset()
    base_info = _base_task_info()
    proxy_keys = normalize_vendor_keys()
    proxy_config = ProxyConfig(
        enable_openai=proxy_keys.get("OPENAI_API_KEY") is not None,
        enable_groq=proxy_keys.get("GROQ_API_KEY") is not None,
        system_hint="Use the heart_disease_classify tool to classify patients."
    )

    base_payload = base_info.model_dump()
    base_payload.update({"environment": "heartdisease_3step", "limits": {"max_turns": 3},
                        "task_metadata": {"modules": [ANALYZER_MODULE_NAME, REASONER_MODULE_NAME, CLASSIFIER_MODULE_NAME]}})
    base_task = base_payload.get("task") or {}
    base_task.update({"id": "heartdisease_3step", "name": "HeartDisease Three-Step Pipeline"})
    base_payload["task"] = base_task
    base_payload["dataset"] = HEARTDISEASE_DATASET_SPEC.model_dump()
    base_payload["dataset"]["hf_dataset"] = HEARTDISEASE_DATASET

    return TaskAppConfig(
        app_id="heartdisease-3step", name="HeartDisease Three-Step Pipeline",
        description="Heart disease classification with 3-step pipeline: analyzer, reasoner, classifier.",
        base_task_info=TaskInfo(**base_payload), describe_taskset=lambda: describe_pipeline_taskset(dataset),
        provide_task_instances=lambda seeds: provide_pipeline_task_instances(dataset, seeds), rollout=rollout_executor,
        dataset_registry=registry, rubrics=RubricBundle(outcome=OUTCOME_RUBRIC, events=EVENTS_RUBRIC),
        proxy=proxy_config, routers=(heartdisease_router,), app_state={"heartdisease_dataset": dataset}, cors_origins=["*"]
    )

register_task_app(
    entry=TaskAppEntry(
        app_id="heartdisease-3step", description="HeartDisease three-step pipeline (analyzer + reasoner + classifier).",
        config_factory=build_config, aliases=("heartdisease-threestep",),
        modal=ModalDeploymentConfig(
            app_name="synth-heartdisease-3step",
            pip_packages=("datasets>=2.14.0", "fastapi>=0.115.0", "pydantic>=2.0.0", "httpx>=0.26.0"),
            timeout=600, memory=4096, cpu=2.0
        ),
    )
)
