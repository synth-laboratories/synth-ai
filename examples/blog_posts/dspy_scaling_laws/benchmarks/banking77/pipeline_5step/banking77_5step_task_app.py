"""Five-step Banking77 pipeline: Parser → Contextualizer → Reasoner → Synthesizer → Classifier."""

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

PARSER_MODULE_NAME = "parser"
CONTEXTUALIZER_MODULE_NAME = "contextualizer"
REASONER_MODULE_NAME = "reasoner"
SYNTHESIZER_MODULE_NAME = "synthesizer"
CLASSIFIER_MODULE_NAME = "classifier"

def _baseline_system_instruction(module: str) -> str:
    instructions = {
        PARSER_MODULE_NAME: "Parse and structure the customer query. Extract entities, action words, and key phrases. Return your parsed understanding using the `banking77_classify` tool.",
        CONTEXTUALIZER_MODULE_NAME: "Add relevant banking domain context. Given the parsed query, identify relevant banking concepts and background knowledge. Return your contextualized understanding using the `banking77_classify` tool.",
        REASONER_MODULE_NAME: "Perform multi-step reasoning. Connect the parsed elements with banking context to narrow down intent categories. Return your reasoning result using the `banking77_classify` tool.",
        SYNTHESIZER_MODULE_NAME: "Synthesize insights from previous steps. Combine parsing, context, and reasoning to form a confident prediction. Return your synthesis using the `banking77_classify` tool.",
        CLASSIFIER_MODULE_NAME: "Make the final classification. Given all previous analysis, select the most accurate intent. Return the final label using the `banking77_classify` tool.",
    }
    return instructions.get(module, "Classify the banking query using the `banking77_classify` tool.")

def _baseline_user_pattern(module: str) -> str:
    patterns = {
        PARSER_MODULE_NAME: "Customer Query: {query}\n\nParse this query and extract key elements. Return your initial classification.",
        CONTEXTUALIZER_MODULE_NAME: "Customer Query: {query}\nParsed Result: {parser_intent}\n\nAdd banking context to refine the classification.",
        REASONER_MODULE_NAME: "Customer Query: {query}\nParsed: {parser_intent}\nContextualized: {contextualizer_intent}\n\nReason about the likely intent category.",
        SYNTHESIZER_MODULE_NAME: "Customer Query: {query}\nParsed: {parser_intent}\nContextualized: {contextualizer_intent}\nReasoned: {reasoner_intent}\n\nSynthesize all insights into a confident prediction.",
        CLASSIFIER_MODULE_NAME: "Customer Query: {query}\nParsed: {parser_intent}\nContextualized: {contextualizer_intent}\nReasoned: {reasoner_intent}\nSynthesized: {synthesizer_intent}\n\nMake the final classification.",
    }
    return patterns.get(module, "Classify: {query}")

def _format_few_shot_examples(examples: Sequence[Mapping[str, Any]] | None) -> str:
    if not examples:
        return ""
    lines = []
    for ex in examples:
        if not isinstance(ex, Mapping):
            continue
        inp = str(ex.get("input") or ex.get("input_text", "")).strip()
        out = str(ex.get("output") or ex.get("output_text", "") or ex.get("output_data", {}).get("intent", "")).strip()
        if inp or out:
            lines.append(f"- Query: {inp} -> Intent: {out}" if out else f"- Query: {inp}")
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
        {"name": PARSER_MODULE_NAME, "instruction_text": "", "few_shots": []},
        {"name": CONTEXTUALIZER_MODULE_NAME, "instruction_text": "", "few_shots": []},
        {"name": REASONER_MODULE_NAME, "instruction_text": "", "few_shots": []},
        {"name": SYNTHESIZER_MODULE_NAME, "instruction_text": "", "few_shots": []},
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
    if placeholders.get("available_intents"):
        parts.append("Available Intents:\n" + placeholders["available_intents"])

    system_msg = "\n\n".join(p for p in parts if p)
    user_msg = _baseline_user_pattern(module_name).format(**placeholders)
    messages = [
        {"role": "system", "pattern": "{system_message}"},
        {"role": "user", "pattern": "{user_message}"},
    ]
    module_placeholders = dict(placeholders)
    module_placeholders.update({"system_message": system_msg, "user_message": user_msg})
    return messages, module_placeholders

def _extract_intent_from_response(response_json: Mapping[str, Any] | None, response_text: str, tool_calls: Sequence[Mapping[str, Any]] | None) -> str:
    if tool_calls:
        for call in tool_calls:
            fn = (call or {}).get("function", {}) or {}
            if fn.get("name") == TOOL_NAME:
                try:
                    args = json.loads(fn.get("arguments", "{}"))
                    intent = str(args.get("intent", "")).strip()
                    if intent:
                        return intent
                except:
                    pass
    if response_json:
        choices = response_json.get("choices", []) if isinstance(response_json, Mapping) else []
        if choices:
            msg = (choices[0] or {}).get("message", {}) or {}
            for call in msg.get("tool_calls", []) or []:
                fn = (call or {}).get("function", {}) or {}
                if fn.get("name") == TOOL_NAME:
                    try:
                        args = json.loads(fn.get("arguments", "{}"))
                        intent = str(args.get("intent", "")).strip()
                        if intent:
                            return intent
                    except:
                        pass
            content = str(msg.get("content", "")).strip()
            if content:
                return content.split()[0]
    if response_text and response_text.strip():
        return response_text.strip().split()[0]
    return ""

async def rollout_executor(request: RolloutRequest, fastapi_request: Request) -> RolloutResponse:
    dataset: Banking77Dataset = fastapi_request.app.state.banking77_dataset
    with contextlib.suppress(Exception):
        cfg = request.policy.config or {}
        print(f"[TASK_APP] INBOUND_5STEP: run_id={request.run_id} seed={request.env.seed} model={cfg.get('model')}", flush=True)

    split = str(((request.env.config or {}).get("split")) or "train")
    seed = request.env.seed or 0
    sample = dataset.sample(split=split, index=seed)
    observation = {"query": sample["text"], "index": sample["index"], "split": sample["split"], "available_intents": dataset.label_names}
    available_intents = "\n".join(f"{i+1}. {label}" for i, label in enumerate(dataset.label_names))

    # Extract API key from request headers or environment
    api_key = fastapi_request.headers.get("X-API-Key") or os.getenv("GROQ_API_KEY")
    policy_config = request.policy.config or {}
    # Add API key to policy config so call_chat_completion can use it
    if api_key:
        policy_config = {**policy_config, "api_key": api_key}
    modules = _resolve_module_configs(policy_config)
    pipeline_records = []
    intents = {"parser_intent": "", "contextualizer_intent": "", "reasoner_intent": "", "synthesizer_intent": ""}

    for module in modules:
        module_name = str(module.get("name", "")).strip() or PARSER_MODULE_NAME
        placeholders = {"query": sample["text"], "available_intents": available_intents, **intents}
        messages, formatted_placeholders = _build_module_messages(module_name, module, placeholders)

        api_key = (fastapi_request.headers.get("X-API-Key") or fastapi_request.headers.get("x-api-key") or
                   (fastapi_request.headers.get("Authorization", "").replace("Bearer ", "").strip() if "Bearer " in fastapi_request.headers.get("Authorization", "") else None))

        response_text, response_json, tool_calls = await call_chat_completion(policy_config, formatted_placeholders, messages, api_key=api_key)

        if not isinstance(response_json, dict) or not response_json:
            raise RuntimeError(f"Module {module_name}: empty response")

        predicted_intent = _extract_intent_from_response(response_json, response_text, tool_calls)
        if not predicted_intent:
            raise RuntimeError(f"Module {module_name}: no intent extracted")

        pipeline_records.append({"module": module_name, "instruction_text": module.get("instruction_text"), "few_shots": module.get("few_shots"),
                                "messages": messages, "response": response_json, "tool_calls": tool_calls, "predicted_intent": predicted_intent})

        if module_name == PARSER_MODULE_NAME:
            intents["parser_intent"] = predicted_intent
        elif module_name == CONTEXTUALIZER_MODULE_NAME:
            intents["contextualizer_intent"] = predicted_intent
        elif module_name == REASONER_MODULE_NAME:
            intents["reasoner_intent"] = predicted_intent
        elif module_name == SYNTHESIZER_MODULE_NAME:
            intents["synthesizer_intent"] = predicted_intent

    expected_intent = sample["label"]
    final_intent = predicted_intent
    is_correct = final_intent.lower().replace("_", " ") == expected_intent.lower().replace("_", " ")
    reward = 1.0 if is_correct else 0.0
    print(f"[TASK_APP] 5STEP_RESULT: expected={expected_intent} predicted={final_intent} correct={is_correct}", flush=True)

    step = RolloutStep(
        obs=observation, tool_calls=pipeline_records[-1]["tool_calls"] if pipeline_records else [], reward=reward, done=True,
        info={"expected_intent": expected_intent, "predicted_intent": final_intent, "modules": pipeline_records, "correct": is_correct, "messages": pipeline_records[0]["messages"] if pipeline_records else []})

    trajectory = RolloutTrajectory(
        env_id=f"banking77_5step::{sample['split']}::{sample['index']}", policy_id=request.policy.policy_id or request.policy.policy_name or "policy",
        steps=[step], final={"observation": observation, "reward": reward}, length=len(modules), inference_url=str((request.policy.config or {}).get("inference_url") or ""))

    metrics = RolloutMetrics(episode_returns=[reward], mean_return=reward, num_steps=len(modules), num_episodes=1, outcome_score=reward, events_score=reward, details={"correct": is_correct})

    include_trace = bool((request.record and getattr(request.record, "return_trace", False)) or os.getenv("TASKAPP_TRACING_ENABLED"))
    trace_payload = {"session_id": str(uuid.uuid4()), "events_count": len(modules), "decision_rewards": [reward],
                     "metadata": {"env": "banking77_5step", "split": sample["split"], "index": sample["index"], "correct": is_correct}} if include_trace else None

    return RolloutResponse(run_id=request.run_id, trajectories=[trajectory], branches={}, metrics=metrics, aborted=False, ops_executed=len(modules) * 2, trace=trace_payload,
                          pipeline_metadata={"modules": pipeline_records, "final_intent": final_intent, "expected_intent": expected_intent})

def describe_pipeline_taskset(dataset: Banking77Dataset) -> Mapping[str, Any]:
    return describe_taskset(dataset)

def provide_pipeline_task_instances(dataset: Banking77Dataset, seeds: Sequence[int]) -> Iterable[TaskInfo]:
    for info in provide_task_instances(dataset, seeds):
        payload = info.model_dump()
        payload["environment"] = "banking77_5step"
        task_meta = payload.get("task_metadata") or {}
        task_meta["modules"] = [PARSER_MODULE_NAME, CONTEXTUALIZER_MODULE_NAME, REASONER_MODULE_NAME, SYNTHESIZER_MODULE_NAME, CLASSIFIER_MODULE_NAME]
        payload["task_metadata"] = task_meta
        payload.setdefault("limits", {})["max_turns"] = 5
        task_payload = payload.get("task") or {}
        task_payload.update({"id": "banking77_5step", "name": "Banking77 Five-Step Pipeline",
                            "description": "Banking77 intent classification with 5-step pipeline: parser, contextualizer, reasoner, synthesizer, classifier."})
        payload["task"] = task_payload
        yield TaskInfo(**payload)

def build_config() -> TaskAppConfig:
    registry, dataset = build_dataset()
    base_info = _base_task_info()
    proxy_keys = normalize_vendor_keys()
    proxy_config = ProxyConfig(enable_openai=proxy_keys.get("OPENAI_API_KEY") is not None, enable_groq=proxy_keys.get("GROQ_API_KEY") is not None,
                               system_hint="Use the banking77_classify tool to classify queries.")

    base_payload = base_info.model_dump()
    base_payload.update({"environment": "banking77_5step", "limits": {"max_turns": 5},
                        "task_metadata": {"modules": [PARSER_MODULE_NAME, CONTEXTUALIZER_MODULE_NAME, REASONER_MODULE_NAME, SYNTHESIZER_MODULE_NAME, CLASSIFIER_MODULE_NAME]}})
    base_task = base_payload.get("task") or {}
    base_task.update({"id": "banking77_5step", "name": "Banking77 Five-Step Pipeline"})
    base_payload["task"] = base_task
    base_payload["dataset"] = BANKING77_DATASET_SPEC.model_dump()
    base_payload["dataset"]["hf_dataset"] = DATASET_NAME

    return TaskAppConfig(
        app_id="banking77-5step", name="Banking77 Five-Step Pipeline", description="Banking77 intent classification with 5-step pipeline: parser, contextualizer, reasoner, synthesizer, classifier.",
        base_task_info=TaskInfo(**base_payload), describe_taskset=lambda: describe_pipeline_taskset(dataset),
        provide_task_instances=lambda seeds: provide_pipeline_task_instances(dataset, seeds), rollout=rollout_executor, dataset_registry=registry,
        rubrics=RubricBundle(outcome=OUTCOME_RUBRIC, events=EVENTS_RUBRIC), proxy=proxy_config, routers=(banking77_router,),
        app_state={"banking77_dataset": dataset}, cors_origins=["*"])

register_task_app(entry=TaskAppEntry(app_id="banking77-5step", description="Banking77 five-step pipeline (parser + contextualizer + reasoner + synthesizer + classifier).",
                                    config_factory=build_config, aliases=("banking77-fivestep",),
                                    modal=ModalDeploymentConfig(app_name="synth-banking77-5step", pip_packages=("datasets>=2.14.0", "fastapi>=0.115.0", "pydantic>=2.0.0", "python-dotenv>=1.0.0"),
                                                              timeout=600, memory=4096, cpu=2.0)))
