"""Five-step HotpotQA: Parser → Retriever → Reasoner → Synthesizer → Answerer."""
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
task_apps_dir = Path(__file__).resolve().parents[4] / "task_apps"
if str(task_apps_dir) not in sys.path: sys.path.insert(0, str(task_apps_dir))

from gepa_benchmarks.hotpotqa_task_app import (HOTPOTQA_DATASET_SPEC, AVAILABLE_SPLITS, HotpotQADataset, HOTPOTQA_DATASET, HOTPOTQA_CONFIG, DEFAULT_SPLIT,
    _parse_answer, hotpotqa_router, OUTCOME_RUBRIC, EVENTS_RUBRIC)
from gepa_benchmarks.common import normalise_answer
import gepa_benchmarks.common

PARSER_MODULE, RETRIEVER_MODULE, REASONER_MODULE, SYNTHESIZER_MODULE, ANSWERER_MODULE = "parser", "retriever", "reasoner", "synthesizer", "answerer"

def _system_instruction(m): return {"parser": "Parse the question to identify key entities and relations needed.",
    "retriever": "Identify which passages contain relevant information.", "reasoner": "Reason across retrieved passages to connect facts.",
    "synthesizer": "Synthesize multi-hop reasoning into coherent answer.", "answerer": "Provide final answer with supporting facts."}.get(m, "Answer.")

def _user_pattern(m): return {"parser": "Q: {question}\nParse the question.",
    "retriever": "Q: {question}\nPassages:\n{context}\nParsed: {parser_result}\nIdentify relevant passages.",
    "reasoner": "Q: {question}\nParsed: {parser_result}\nRelevant: {retriever_result}\nReason across facts.",
    "synthesizer": "Q: {question}\nParsed: {parser_result}\nRelevant: {retriever_result}\nReasoning: {reasoner_result}\nSynthesize answer.",
    "answerer": "Q: {question}\nParsed: {parser_result}\nRelevant: {retriever_result}\nReasoning: {reasoner_result}\nSynthesis: {synthesizer_result}\nFinal answer."}.get(m, "{question}")

def _resolve_configs(p):
    t = p.get("prompt_template") or {}
    m = t.get("prompt_metadata") or {}
    mods = m.get("pipeline_modules")
    if isinstance(mods, list):
        r = [dict(x) for x in mods if isinstance(x, Mapping)]
        if r: return r
    return [{"name": n, "instruction_text": "", "few_shots": []} for n in [PARSER_MODULE, RETRIEVER_MODULE, REASONER_MODULE, SYNTHESIZER_MODULE, ANSWERER_MODULE]]

def _build_messages(mn, mc, ph):
    inst = str(mc.get("instruction_text", "")).strip()
    parts = [_system_instruction(mn)]
    if inst: parts.append(inst)
    sys_msg = "\n\n".join(p for p in parts if p)
    usr_msg = _user_pattern(mn).format(**ph)
    msgs = [{"role": "system", "pattern": "{system_message}"}, {"role": "user", "pattern": "{user_message}"}]
    return msgs, {**ph, "system_message": sys_msg, "user_message": usr_msg}

def _extract_response(rj, rt):
    if rj:
        chs = rj.get("choices", [])
        if chs:
            cont = chs[0].get("message", {}).get("content", "")
            if cont: return cont.strip()
    return rt.strip() if rt else ""

async def rollout_executor(request: RolloutRequest, fastapi_request: Request) -> RolloutResponse:
    dataset = fastapi_request.app.state.hotpotqa_dataset
    split = str(((request.env.config or {}).get("split")) or DEFAULT_SPLIT)
    seed = request.env.seed or 0
    sample = dataset.sample(split=split, index=seed)
    observation = {"question": sample["question"], "context": sample["context_text"], "supporting_titles": sample["supporting_titles"], "index": sample["index"], "split": sample["split"]}

    # Extract API key from request headers or environment
    api_key = fastapi_request.headers.get("X-API-Key") or os.getenv("GROQ_API_KEY")
    policy_config = request.policy.config or {}
    # Add API key to policy config so call_chat_completion can use it
    if api_key:
        policy_config = {**policy_config, "api_key": api_key}
    modules = _resolve_configs(policy_config)
    pipeline_records = []
    results = {"parser_result": "", "retriever_result": "", "reasoner_result": "", "synthesizer_result": ""}

    for module in modules:
        module_name = str(module.get("name", "")).strip() or PARSER_MODULE
        placeholders = {"question": sample["question"], "context": sample["context_text"], **results}
        messages, formatted_placeholders = _build_messages(module_name, module, placeholders)
        # API key already in policy_config from earlier extraction
        response_text, response_json, _ = await gepa_benchmarks.common.call_chat_completion(policy_config, formatted_placeholders, messages)
        if not isinstance(response_json, dict) or not response_json: raise RuntimeError(f"Module {module_name}: empty")
        module_output = _extract_response(response_json, response_text)
        if not module_output: raise RuntimeError(f"Module {module_name}: no output")
        pipeline_records.append({"module": module_name, "instruction_text": module.get("instruction_text"), "few_shots": module.get("few_shots"),
                                "messages": messages, "response": response_json, "output": module_output})
        if module_name == PARSER_MODULE: results["parser_result"] = module_output
        elif module_name == RETRIEVER_MODULE: results["retriever_result"] = module_output
        elif module_name == REASONER_MODULE: results["reasoner_result"] = module_output
        elif module_name == SYNTHESIZER_MODULE: results["synthesizer_result"] = module_output

    answer_text, support_text = _parse_answer(module_output)
    expected_answer = sample["answer"]
    answer_correct = int(normalise_answer(answer_text) == normalise_answer(expected_answer))
    support_titles = sample["supporting_titles"]
    support_hits = sum(1 for title in support_titles if title.lower() in support_text.lower()) if support_titles and support_text else 0
    support_coverage = (support_hits / len(support_titles)) if support_titles else 0.0
    reward = 0.7 * answer_correct + 0.3 * support_coverage
    print(f"[HOTPOTQA_5STEP] exp={expected_answer} pred={answer_text} em={answer_correct} sup={support_hits}/{len(support_titles)}", flush=True)

    step = RolloutStep(obs=observation, tool_calls=[], reward=reward, done=True,
                      info={"expected_answer": expected_answer, "predicted_answer": answer_text, "answer_em": answer_correct, "support_coverage": support_coverage,
                           "modules": pipeline_records, "messages": pipeline_records[0]["messages"] if pipeline_records else []})
    trajectory = RolloutTrajectory(env_id=f"hotpotqa_5step::{sample['split']}::{sample['index']}", policy_id=request.policy.policy_id or "policy",
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
    registry, dataset = build_dataset()
    proxy_keys = normalize_vendor_keys()
    proxy_config = ProxyConfig(enable_openai=proxy_keys.get("OPENAI_API_KEY") is not None, enable_groq=proxy_keys.get("GROQ_API_KEY") is not None)
    base_info = TaskInfo(task={"id": "hotpotqa_5step", "name": "HotpotQA Five-Step Pipeline"}, environment="hotpotqa_5step",
                        dataset={**HOTPOTQA_DATASET_SPEC.model_dump(), "hf_dataset": HOTPOTQA_DATASET}, limits={"max_turns": 5})
    return TaskAppConfig(app_id="hotpotqa-5step", name="HotpotQA Five-Step Pipeline", description="HotpotQA 5-step QA.", base_task_info=base_info,
                        describe_taskset=lambda: {**HOTPOTQA_DATASET_SPEC.model_dump(), "sizes": {s: dataset.size(s) for s in AVAILABLE_SPLITS}},
                        provide_task_instances=lambda seeds: iter([]), rollout=rollout_executor, dataset_registry=registry,
                        rubrics=RubricBundle(outcome=OUTCOME_RUBRIC, events=EVENTS_RUBRIC), proxy=proxy_config, routers=(hotpotqa_router,),
                        app_state={"hotpotqa_dataset": dataset}, cors_origins=["*"])

register_task_app(entry=TaskAppEntry(app_id="hotpotqa-5step", description="HotpotQA five-step pipeline.", config_factory=build_config, aliases=("hotpotqa-fivestep",),
                                    modal=ModalDeploymentConfig(app_name="synth-hotpotqa-5step", pip_packages=("datasets>=2.14.0", "fastapi>=0.115.0", "pydantic>=2.0.0", "httpx>=0.26.0"),
                                                              timeout=600, memory=4096, cpu=2.0)))
