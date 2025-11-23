"""Three-step HotpotQA pipeline: Analyzer → Reasoner → Answerer."""
from __future__ import annotations
import contextlib, json, os, uuid
from collections.abc import Iterable, Sequence
from typing import Any, Mapping, MutableMapping
from pathlib import Path
from fastapi import Request, APIRouter
from synth_ai.task.apps import ModalDeploymentConfig, TaskAppEntry, register_task_app
from synth_ai.task.contracts import RolloutMetrics, RolloutRequest, RolloutResponse, RolloutStep, RolloutTrajectory, TaskInfo
from synth_ai.task.server import ProxyConfig, RubricBundle, TaskAppConfig
from synth_ai.task.vendors import normalize_vendor_keys

import sys
task_apps_dir = Path(__file__).resolve().parents[1]
if str(task_apps_dir) not in sys.path: sys.path.insert(0, str(task_apps_dir))

from gepa_benchmarks.hotpotqa_task_app import (HOTPOTQA_DATASET_SPEC, AVAILABLE_SPLITS, HotpotQADataset, HOTPOTQA_DATASET, HOTPOTQA_CONFIG, DEFAULT_SPLIT,
    _parse_answer, hotpotqa_router, OUTCOME_RUBRIC, EVENTS_RUBRIC, normalise_answer)
from banking77.banking77_task_app import call_chat_completion

ANALYZER_MODULE, REASONER_MODULE, ANSWERER_MODULE = "analyzer", "reasoner", "answerer"

def _system_instruction(m): return {"analyzer": "Analyze the question and identify what information is needed from passages.",
    "reasoner": "Reason across passages to connect relevant information.", "answerer": "Provide the final answer with support."}.get(m, "Answer the question.")

def _user_pattern(m): return {"analyzer": "Question: {question}\nPassages:\n{context}\n\nAnalyze what's needed.",
    "reasoner": "Question: {question}\nPassages:\n{context}\nAnalysis: {analyzer_result}\n\nReason across passages.",
    "answerer": "Question: {question}\nAnalysis: {analyzer_result}\nReasoning: {reasoner_result}\n\nProvide final answer."}.get(m, "{question}")

def _resolve_configs(policy_config):
    template = policy_config.get("prompt_template") or {}
    metadata = template.get("prompt_metadata") or {}
    modules = metadata.get("pipeline_modules")
    if isinstance(modules, list):
        resolved = [dict(m) for m in modules if isinstance(m, Mapping)]
        if resolved: return resolved
    return [{"name": n, "instruction_text": "", "few_shots": []} for n in [ANALYZER_MODULE, REASONER_MODULE, ANSWERER_MODULE]]

def _build_messages(module_name, module_config, placeholders):
    instruction = str(module_config.get("instruction_text", "")).strip()
    parts = [_system_instruction(module_name)]
    if instruction: parts.append(instruction)
    system_msg = "\n\n".join(p for p in parts if p)
    user_msg = _user_pattern(module_name).format(**placeholders)
    messages = [{"role": "system", "pattern": "{system_message}"}, {"role": "user", "pattern": "{user_message}"}]
    return messages, {**placeholders, "system_message": system_msg, "user_message": user_msg}

def _extract_response(response_json, response_text):
    if response_json:
        choices = response_json.get("choices", [])
        if choices:
            content = choices[0].get("message", {}).get("content", "")
            if content: return content.strip()
    return response_text.strip() if response_text else ""

async def rollout_executor(request: RolloutRequest, fastapi_request: Request) -> RolloutResponse:
    dataset = fastapi_request.app.state.hotpotqa_dataset
    split = str(((request.env.config or {}).get("split")) or DEFAULT_SPLIT)
    seed = request.env.seed or 0
    sample = dataset.sample(split=split, index=seed)
    observation = {"question": sample["question"], "context": sample["context_text"], "supporting_titles": sample["supporting_titles"], "index": sample["index"], "split": sample["split"]}
    policy_config = request.policy.config or {}
    modules = _resolve_configs(policy_config)
    pipeline_records = []
    results = {"analyzer_result": "", "reasoner_result": ""}

    for module in modules:
        module_name = str(module.get("name", "")).strip() or ANALYZER_MODULE
        placeholders = {"question": sample["question"], "context": sample["context_text"], **results}
        messages, formatted_placeholders = _build_messages(module_name, module, placeholders)

        # Get app-level http client singleton (created at startup, reused across requests)
        http_client = getattr(fastapi_request.app.state, "http_client", None)

        response_text, response_json, _ = await call_chat_completion(
            policy_config,
            formatted_placeholders,
            messages,
            http_client=http_client,
        )
        if not isinstance(response_json, dict) or not response_json: raise RuntimeError(f"Module {module_name}: empty response")
        module_output = _extract_response(response_json, response_text)
        if not module_output: raise RuntimeError(f"Module {module_name}: no output")
        pipeline_records.append({"module": module_name, "instruction_text": module.get("instruction_text"), "few_shots": module.get("few_shots"),
                                "messages": messages, "response": response_json, "output": module_output})
        if module_name == ANALYZER_MODULE: results["analyzer_result"] = module_output
        elif module_name == REASONER_MODULE: results["reasoner_result"] = module_output

    answer_text, support_text = _parse_answer(module_output)
    expected_answer = sample["answer"]
    answer_correct = int(normalise_answer(answer_text) == normalise_answer(expected_answer))
    support_titles = sample["supporting_titles"]
    support_hits = sum(1 for title in support_titles if title.lower() in support_text.lower()) if support_titles and support_text else 0
    support_coverage = (support_hits / len(support_titles)) if support_titles else 0.0
    reward = 0.7 * answer_correct + 0.3 * support_coverage
    print(f"[HOTPOTQA_3STEP] expected={expected_answer} predicted={answer_text} em={answer_correct} support={support_hits}/{len(support_titles)}", flush=True)

    step = RolloutStep(obs=observation, tool_calls=[], reward=reward, done=True,
                      info={"expected_answer": expected_answer, "predicted_answer": answer_text, "answer_em": answer_correct, "support_coverage": support_coverage,
                           "modules": pipeline_records, "messages": pipeline_records[0]["messages"] if pipeline_records else []})
    trajectory = RolloutTrajectory(env_id=f"hotpotqa_3step::{sample['split']}::{sample['index']}", policy_id=request.policy.policy_id or "policy",
                                  steps=[step], final={"observation": observation, "reward": reward}, length=len(modules), inference_url=str((request.policy.config or {}).get("inference_url") or ""))
    metrics = RolloutMetrics(episode_returns=[reward], mean_return=reward, num_steps=len(modules), num_episodes=1, outcome_score=reward, events_score=reward,
                            details={"answer_correct": bool(answer_correct), "support_coverage": support_coverage})
    return RolloutResponse(run_id=request.run_id, trajectories=[trajectory], branches={}, metrics=metrics, aborted=False, ops_executed=len(modules)*2,
                          pipeline_metadata={"modules": pipeline_records, "final_answer": answer_text})

def build_dataset():
    from synth_ai.task.datasets import TaskDatasetRegistry
    registry = TaskDatasetRegistry()
    dataset = HotpotQADataset()
    dataset.ensure_ready([DEFAULT_SPLIT])
    registry.register(HOTPOTQA_DATASET_SPEC, lambda _: dataset, cache=True)
    return registry, dataset

def build_config():
    # Startup hook: Create aiohttp session singleton
    async def startup_http_client(app: Any) -> None:
        try:
            import aiohttp
            app.state.http_client = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=30.0), connector=aiohttp.TCPConnector(limit=10, limit_per_host=5))
            print("[hotpotqa_3step] Created app-level aiohttp client session singleton", flush=True)
        except ImportError:
            try:
                import httpx
                app.state.http_client = httpx.AsyncClient(timeout=30.0, limits=httpx.Limits(max_keepalive_connections=5, max_connections=10))
                print("[hotpotqa_3step] Created app-level httpx client singleton (fallback)", flush=True)
            except Exception as exc:
                print(f"[hotpotqa_3step] WARNING: Failed to create http client: {exc}", flush=True)
                app.state.http_client = None
        except Exception as exc:
            print(f"[hotpotqa_3step] WARNING: Failed to create aiohttp client: {exc}", flush=True)
            app.state.http_client = None

    # Shutdown hook: Clean up http client
    async def shutdown_http_client(app: Any) -> None:
        http_client = getattr(app.state, "http_client", None)
        if http_client is not None:
            try:
                if hasattr(http_client, 'close'):
                    await http_client.close()
                elif hasattr(http_client, 'aclose'):
                    await http_client.aclose()
                print("[hotpotqa_3step] Closed app-level http client", flush=True)
            except Exception as exc:
                print(f"[hotpotqa_3step] WARNING: Error closing http client: {exc}", flush=True)

    registry, dataset = build_dataset()
    proxy_keys = normalize_vendor_keys()
    proxy_config = ProxyConfig(enable_openai=proxy_keys.get("OPENAI_API_KEY") is not None, enable_groq=proxy_keys.get("GROQ_API_KEY") is not None)
    base_info = TaskInfo(task={"id": "hotpotqa_3step", "name": "HotpotQA Three-Step Pipeline"}, environment="hotpotqa_3step",
                        dataset={**HOTPOTQA_DATASET_SPEC.model_dump(), "hf_dataset": HOTPOTQA_DATASET}, limits={"max_turns": 3},
                        task_metadata={"modules": [ANALYZER_MODULE, REASONER_MODULE, ANSWERER_MODULE]},
                        rubric={"version": "1", "criteria_count": 2, "source": "inline"},
                        inference={"supports_proxy": True, "tool": None})
    return TaskAppConfig(app_id="hotpotqa-3step", name="HotpotQA Three-Step Pipeline", description="HotpotQA 3-step QA pipeline.", base_task_info=base_info,
                        describe_taskset=lambda: {**HOTPOTQA_DATASET_SPEC.model_dump(), "sizes": {s: dataset.size(s) for s in AVAILABLE_SPLITS}},
                        provide_task_instances=lambda seeds: iter([]), rollout=rollout_executor, dataset_registry=registry,
                        rubrics=RubricBundle(outcome=OUTCOME_RUBRIC, events=EVENTS_RUBRIC), proxy=proxy_config, routers=(hotpotqa_router,),
                        startup_hooks=[startup_http_client],
                        shutdown_hooks=[shutdown_http_client],
                        app_state={"hotpotqa_dataset": dataset}, cors_origins=["*"])

register_task_app(entry=TaskAppEntry(app_id="hotpotqa-3step", description="HotpotQA three-step pipeline.", config_factory=build_config, aliases=("hotpotqa-threestep",),
                                    modal=ModalDeploymentConfig(app_name="synth-hotpotqa-3step", pip_packages=("datasets>=2.14.0", "fastapi>=0.115.0", "pydantic>=2.0.0", "httpx>=0.26.0"),
                                                              timeout=600, memory=4096, cpu=2.0)))
