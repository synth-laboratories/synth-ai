#!/usr/bin/env python3
"""Online MIPRO demo for MIT RLM on OOLONG.

Usage:
    uv run python demos/rlm_mipro/run_demo.py
    uv run python demos/rlm_mipro/run_demo.py --rollouts 5 --model gpt-4.1-mini
    SYNTH_BACKEND_URL=https://api-dev.usesynth.ai uv run python demos/rlm_mipro/run_demo.py
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List

import httpx
from datasets import load_dataset
from dotenv import load_dotenv
from rlm import RLM
from rlm.core import rlm as rlm_core
from rlm.core import types as rlm_types
from rlm.utils import prompts as rlm_prompts
from rlm.utils.prompts import USER_PROMPT

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from demos.mipro.utils import (
    create_job,
    create_task_app_url,
    extract_candidate_text,
    get_job_detail,
    get_system_state,
    new_rollout_id,
    push_status,
    resolve_backend_url,
    should_upload_env_key,
    wait_for_health_check_sync,
)
from synth_ai.core.tunnels import PortConflictBehavior, acquire_port
from synth_ai.sdk.localapi import LocalAPIConfig, create_local_api
from synth_ai.sdk.localapi._impl import run_server_background
from synth_ai.sdk.localapi.auth import ensure_localapi_auth
from synth_ai.sdk.task.contracts import RolloutMetrics, RolloutRequest, RolloutResponse, TaskInfo

load_dotenv()

# Work around rlm QueryMetadata typing bug under Python 3.11
class PatchedQueryMetadata:
    def __init__(self, prompt):
        if isinstance(prompt, str):
            self.context_lengths = [len(prompt)]
            self.context_type = "str"
        elif isinstance(prompt, dict):
            self.context_lengths = [len(chunk) for chunk in prompt.values()]
            self.context_type = "dict"
        elif isinstance(prompt, list):
            self.context_type = "list"
            if prompt and isinstance(prompt[0], dict):
                if "content" in prompt[0]:
                    self.context_lengths = [len(chunk["content"]) for chunk in prompt]
                else:
                    self.context_lengths = [len(chunk) for chunk in prompt]
            else:
                self.context_lengths = [len(chunk) for chunk in prompt]
        else:
            raise ValueError(f"Invalid prompt type: {type(prompt)}")

        self.context_total_length = sum(self.context_lengths)


rlm_types.QueryMetadata = PatchedQueryMetadata
rlm_core.QueryMetadata = PatchedQueryMetadata


def patched_build_rlm_system_prompt(system_prompt, query_metadata=None, **_kwargs):
    return [
        {"role": "system", "content": system_prompt},
        {"role": "assistant", "content": "{context_metadata}"},
    ]


rlm_prompts.build_rlm_system_prompt = patched_build_rlm_system_prompt
rlm_core.build_rlm_system_prompt = patched_build_rlm_system_prompt

# Local API constants
APP_ID = "oolong_rlm_mipro"
APP_NAME = "OOLONG RLM (MIPRO Online)"

RLM_BASE_SYSTEM_PROMPT = (
    "You are a recursive language model. Use the REPL with the context variable to reason. "
    "Call llm_query or llm_query_batched as needed. When finished, answer with FINAL."
)
BASELINE_SYSTEM_PROMPT = "Answer questions using the context."
BASELINE_USER_PROMPT = (
    "Query: {query}\n\nContext:\n{context}\n\nAnswer the query using the context."
)

RLM_CONTEXT_METADATA_PATTERN = "{context_metadata}"
RLM_FIRST_USER_PROMPT = (
    "You have not interacted with the REPL environment or seen your prompt / context yet. "
    "Your next action should be to look through and figure out how to answer the prompt, "
    "so don't just provide a final answer yet.\n\n" + USER_PROMPT
)

COMPOSED_SYSTEM_PROMPT = RLM_BASE_SYSTEM_PROMPT + " " + BASELINE_SYSTEM_PROMPT


@dataclass
class OolongSample:
    index: int
    split: str
    query: str
    context: str
    answer: str


class OolongDataset:
    def __init__(self, hf_dataset: str = "oolongbench/oolong-real", hf_config: str = "dnd"):
        self.hf_dataset = hf_dataset
        self.hf_config = hf_config
        self._cache: Dict[str, Any] = {}

    def _load_split(self, split: str):
        if split not in self._cache:
            ds = load_dataset(self.hf_dataset, self.hf_config, split=split)
            self._cache[split] = ds
        return self._cache[split]

    def ensure_ready(self, splits: Iterable[str]) -> None:
        for split in splits:
            self._load_split(split)

    def size(self, split: str) -> int:
        return len(self._load_split(split))

    def sample(self, split: str, index: int) -> OolongSample:
        ds = self._load_split(split)
        idx = index % len(ds)
        row = ds[idx]
        query = row.get("query") or row.get("question") or ""
        context = row.get("context_window_text") or row.get("context") or row.get("text") or ""
        answer = row.get("answer") or ""
        return OolongSample(
            index=idx,
            split=split,
            query=str(query),
            context=str(context),
            answer=str(answer),
        )


# Prompt template helpers
def _normalize_prompt_template(policy_config: Dict[str, Any]) -> Dict[str, Any]:
    template = policy_config.get("prompt_template") or {}
    if not isinstance(template, dict):
        template = {}
    return template


def _get_prompt_sections(policy_config: Dict[str, Any]) -> List[Dict[str, Any]]:
    template = _normalize_prompt_template(policy_config)
    sections = (
        template.get("sections")
        or template.get("prompt_sections")
        or policy_config.get("prompt_sections")
        or []
    )
    if not isinstance(sections, list):
        return []
    return sorted(sections, key=lambda s: s.get("order", 0))


def render_prompt_sections(
    sections: List[Dict[str, Any]], placeholders: Dict[str, str]
) -> List[Dict[str, str]]:
    rendered: List[Dict[str, str]] = []
    for section in sections:
        role = section.get("role", "user")
        pattern = section.get("content") or section.get("pattern") or ""
        content = pattern.format(**placeholders)
        rendered.append({"role": role, "content": content})
    return rendered


def split_system_and_user(messages: List[Dict[str, str]]) -> tuple[str, str]:
    system_parts = [m["content"] for m in messages if m.get("role") == "system"]
    user_parts = [m["content"] for m in messages if m.get("role") != "system"]
    system_prompt = "\n\n".join(system_parts).strip()
    user_prompt = "\n\n".join(user_parts).strip()
    return system_prompt, user_prompt


def extract_final_answer(text: str) -> str:
    if not text:
        return ""

    import re

    text = str(text).strip()
    text = re.sub(r"```(?:repl|python|code)?\s*\n.*?```", "", text, flags=re.DOTALL | re.IGNORECASE)

    boxed_match = re.search(r"\\boxed\{([^}]+)\}", text)
    if boxed_match:
        answer = boxed_match.group(1).strip()
        num_match = re.search(r"\d+", answer)
        if num_match:
            return num_match.group(0)
        return answer

    final_match = re.search(r"FINAL\s*[:\-]?\s*(.+?)(?:\n|$)", text, re.IGNORECASE | re.DOTALL)
    if final_match:
        after_final = final_match.group(1).strip()
        boxed_in_final = re.search(r"\\boxed\{([^}]+)\}", after_final)
        if boxed_in_final:
            answer = boxed_in_final.group(1).strip()
            num_match = re.search(r"\d+", answer)
            if num_match:
                return num_match.group(0)
            return answer
        num_match = re.search(r"\d+", after_final)
        if num_match:
            return num_match.group(0)

    last_part = text[-500:] if len(text) > 500 else text
    final_patterns = [
        r"(?:the|final)\s+(?:answer|count|total|number|result)\s+(?:is|:)?\s*(\d+)",
        r"(?:answer|count|total|number|result)\s+(?:is|:)?\s*(\d+)",
        r"(?:is|equals?)\s+(\d+)\s*(?:\.|$|\n)",
        r"(\d+)\s*(?:\.|$|\n)\s*(?:This|Therefore|So|Thus|Hence)",
    ]
    for pattern in final_patterns:
        matches = list(re.finditer(pattern, last_part, re.IGNORECASE))
        if matches:
            return matches[-1].group(1).strip()

    if len(text) < 2000:
        numbers = re.findall(r"\d+", text)
        if numbers:
            return numbers[-1]

    return text


def normalize_answer(text: str) -> str:
    if text is None:
        return ""
    if not isinstance(text, str):
        text = str(text)

    extracted = extract_final_answer(text)
    if extracted:
        normalized = "".join(
            ch.lower() for ch in extracted.strip() if ch.isalnum() or ch.isspace()
        ).strip()
        if normalized:
            return normalized

    return "".join(ch.lower() for ch in text.strip() if ch.isalnum() or ch.isspace()).strip()


def create_oolong_rlm_local_api():
    oolong = OolongDataset()
    oolong.ensure_ready(["validation", "test"])

    async def run_rollout(request: RolloutRequest, fastapi_request) -> RolloutResponse:
        policy_config = request.policy.config or {}
        env_config = request.env.config or {}
        split = env_config.get("split", "validation")
        seed = request.env.seed or 0

        sample = oolong.sample(split=split, index=seed)
        placeholders = {
            "query": sample.query,
            "context": sample.context,
            "context_metadata": "{context_metadata}",
        }

        sections = _get_prompt_sections(policy_config)
        if not sections:
            sections = [
                {"role": "system", "content": COMPOSED_SYSTEM_PROMPT, "order": 0},
                {"role": "assistant", "content": RLM_CONTEXT_METADATA_PATTERN, "order": 1},
                {"role": "user", "content": RLM_FIRST_USER_PROMPT, "order": 2},
                {"role": "user", "content": BASELINE_USER_PROMPT, "order": 3},
            ]

        rendered = render_prompt_sections(sections, placeholders)
        messages_for_validation = []
        for section in sections:
            role = section.get("role", "user")
            pattern = section.get("content") or section.get("pattern") or ""
            messages_for_validation.append({"role": role, "content": pattern})

        system_prompt, root_prompt = split_system_and_user(rendered)
        custom_system_prompt = system_prompt or RLM_BASE_SYSTEM_PROMPT
        inference_url = (
            policy_config.get("inference_url")
            or policy_config.get("api_base")
            or policy_config.get("base_url")
        )
        if not inference_url:
            raise ValueError("Missing inference_url in policy config")

        api_key = policy_config.get("api_key")
        if not api_key:
            raise ValueError("Missing policy api_key for inference proxy")

        model_name = policy_config.get("model", "gpt-4o-mini")
        max_iterations = int(env_config.get("max_iterations", 2))
        max_depth = int(env_config.get("max_depth", 0))

        rlm = RLM(
            backend="openai",
            backend_kwargs={
                "model_name": model_name,
                "api_key": api_key,
                "base_url": inference_url,
            },
            environment="local",
            environment_kwargs={},
            custom_system_prompt=custom_system_prompt,
            max_iterations=max_iterations,
            max_depth=max_depth,
            verbose=False,
        )

        try:
            completion = rlm.completion(rendered or root_prompt)
            if isinstance(completion, str):
                predicted = completion
            else:
                predicted = (
                    getattr(completion, "response", None)
                    or getattr(completion, "answer", None)
                    or getattr(completion, "final_answer", None)
                    or getattr(completion, "completion", None)
                    or str(completion)
                )
                if predicted and "FINAL" in str(predicted).upper():
                    parts = str(predicted).upper().split("FINAL", 1)
                    if len(parts) > 1:
                        predicted = parts[1].strip()
                predicted = str(predicted) if predicted else ""
        except Exception as exc:
            print(f"[RLM ERROR seed={seed}] Completion failed: {exc}", flush=True)
            predicted = ""

        gold = sample.answer or ""
        extracted_predicted = extract_final_answer(predicted)
        reward = 1.0 if normalize_answer(extracted_predicted) == normalize_answer(gold) else 0.0

        return RolloutResponse(
            run_id=request.trace_correlation_id,
            reward_info=RolloutMetrics(
                outcome_reward=reward,
                details={
                    "messages": messages_for_validation,
                    "predicted": extracted_predicted,
                    "gold": gold,
                },
            ),
            trace=None,
            trace_correlation_id=request.trace_correlation_id or "",
        )

    def provide_taskset_description():
        return {
            "splits": ["validation", "test"],
            "sizes": {
                "validation": oolong.size("validation"),
                "test": oolong.size("test"),
            },
        }

    def provide_task_instances(seeds):
        for seed in seeds:
            sample = oolong.sample(split="validation", index=seed)
            yield TaskInfo(
                task={"id": APP_ID, "name": APP_NAME},
                dataset={"id": APP_ID, "split": sample.split, "index": sample.index},
                inference={"tool": "rlm_repl"},
                limits={"max_turns": 1},
                task_metadata={"query": sample.query},
            )

    return create_local_api(
        LocalAPIConfig(
            app_id=APP_ID,
            name=APP_NAME,
            description="OOLONG RLM local API for MIPRO online prompt optimization.",
            provide_taskset_description=provide_taskset_description,
            provide_task_instances=provide_task_instances,
            rollout=run_rollout,
            cors_origins=["*"],
        )
    )


def start_local_api(
    *,
    local_host: str,
    local_port: int,
    backend_url: str | None,
) -> tuple[str, str, int]:
    env_key = (os.environ.get("ENVIRONMENT_API_KEY") or "").strip()
    if not env_key:
        upload = should_upload_env_key(backend_url)
        env_key = ensure_localapi_auth(
            backend_base=backend_url if upload else None,
            upload=upload,
            persist=False,
        )
    os.environ["ENVIRONMENT_API_KEY"] = env_key

    port = acquire_port(local_port, on_conflict=PortConflictBehavior.FIND_NEW)
    if port != local_port:
        print(f"Port {local_port} in use, using port {port} instead")

    app = create_oolong_rlm_local_api()
    run_server_background(app, port)
    print(f"Waiting for local API on port {port}...")
    wait_for_health_check_sync("localhost", port, env_key, timeout=30.0)
    task_app_url = f"http://{local_host}:{port}"
    print(f"Local API URL: {task_app_url}")
    return task_app_url, env_key, port


def build_mipro_online_config(
    *,
    task_app_url: str,
    train_seeds: list[int],
    val_seeds: list[int],
    policy_model: str,
    proposer_model: str,
    proposer_provider: str,
    min_rollouts_before_proposal: int,
    split: str,
    max_iterations: int,
    max_depth: int,
) -> Dict[str, Any]:
    return {
        "prompt_learning": {
            "algorithm": "mipro",
            "task_app_id": APP_ID,
            "task_app_url": task_app_url,
            "initial_prompt": {
                "messages": [
                    {"role": "system", "order": 0, "pattern": COMPOSED_SYSTEM_PROMPT},
                    {"role": "assistant", "order": 1, "pattern": RLM_CONTEXT_METADATA_PATTERN},
                    {"role": "user", "order": 2, "pattern": RLM_FIRST_USER_PROMPT},
                    {"role": "user", "order": 3, "pattern": BASELINE_USER_PROMPT},
                ],
                "wildcards": {
                    "query": "REQUIRED",
                    "context": "REQUIRED",
                    "context_metadata": "REQUIRED",
                },
            },
            "policy": {
                "model": policy_model,
                "provider": "openai",
                "inference_mode": "synth_hosted",
                "temperature": 0.0,
                "max_completion_tokens": 256,
            },
            "mipro": {
                "mode": "online",
                "bootstrap_train_seeds": train_seeds,
                "val_seeds": val_seeds,
                "online_pool": train_seeds,
                "online_proposer_mode": "inline",
                "online_proposer_min_rollouts": min_rollouts_before_proposal,
                "online_rollouts_per_candidate": 5,
                "proposer": {
                    "mode": "instruction_only",
                    "model": proposer_model,
                    "provider": proposer_provider,
                    "temperature": 0.7,
                    "max_tokens": 512,
                    "generate_at_iterations": [0],
                    "instructions_per_batch": 1,
                },
            },
            "env_config": {
                "split": split,
                "max_iterations": max_iterations,
                "max_depth": max_depth,
            },
        }
    }


def run_rollout(
    *,
    task_app_url: str,
    env_key: str,
    seed: int,
    split: str,
    max_iterations: int,
    max_depth: int,
    inference_url: str,
    model: str,
    api_key: str,
    rollout_id: str,
    timeout_s: float,
) -> tuple[float, str | None]:
    rollout_start = time.perf_counter()
    payload = {
        "trace_correlation_id": rollout_id,
        "env": {"seed": seed, "config": {"seed": seed, "split": split}},
        "policy": {
            "config": {
                "model": model,
                "inference_url": inference_url,
                "api_key": api_key,
            }
        },
    }
    payload["env"]["config"]["max_iterations"] = max_iterations
    payload["env"]["config"]["max_depth"] = max_depth

    headers = {"X-API-Key": env_key}
    request_start = time.perf_counter()
    response = httpx.post(
        f"{task_app_url}/rollout",
        json=payload,
        headers=headers,
        timeout=timeout_s,
    )
    request_duration = (time.perf_counter() - request_start) * 1000.0
    response.raise_for_status()
    body = response.json()
    reward_info = body.get("reward_info", {}) if isinstance(body, dict) else {}
    reward = reward_info.get("outcome_reward")
    if reward is None and isinstance(body, dict):
        metrics = body.get("metrics", {}) or {}
        reward = metrics.get("outcome_reward")
        if reward is None:
            reward = (metrics.get("outcome_objectives") or {}).get("reward", 0.0)
    if reward is None:
        reward = (reward_info.get("outcome_objectives") or {}).get("reward", 0.0)

    metadata = body.get("metadata", {}) if isinstance(body, dict) else {}
    candidate_id = metadata.get("mipro_candidate_id")
    if not candidate_id:
        candidate_id = response.headers.get("x-mipro-candidate-id", "")
    candidate_id = str(candidate_id) if candidate_id else None

    rollout_duration = (time.perf_counter() - rollout_start) * 1000.0
    print(
        f"[TIMING] Rollout {rollout_id}: total={rollout_duration:.2}ms "
        f"(task_app_request={request_duration:.2}ms)"
    )
    return float(reward or 0.0), candidate_id


async def main() -> None:
    parser = argparse.ArgumentParser(description="Run online MIPRO for MIT RLM on OOLONG")
    parser.add_argument(
        "--backend-url",
        default=None,
        help="Backend base URL (defaults to SYNTH_BACKEND_URL or SDK default)",
    )
    parser.add_argument("--local-host", default="localhost")
    parser.add_argument("--local-port", type=int, default=8125)
    parser.add_argument("--rollouts", type=int, default=10)
    parser.add_argument("--split", default="validation", choices=["validation", "test"])
    parser.add_argument("--model", default=os.environ.get("RLM_MIPRO_MODEL", "gpt-4o-mini"))
    parser.add_argument(
        "--proposer-model",
        default=os.environ.get("RLM_MIPRO_PROPOSER_MODEL", "gpt-4.1-mini"),
    )
    parser.add_argument(
        "--proposer-provider",
        default=os.environ.get("RLM_MIPRO_PROPOSER_PROVIDER", "openai"),
    )
    parser.add_argument(
        "--min-rollouts-before-proposal",
        type=int,
        default=5,
    )
    parser.add_argument("--max-iterations", type=int, default=2)
    parser.add_argument("--max-depth", type=int, default=0)
    parser.add_argument("--rollout-timeout", type=float, default=180.0)
    args = parser.parse_args()

    backend_url = (args.backend_url or resolve_backend_url()).rstrip("/")
    api_key = (os.environ.get("SYNTH_API_KEY") or "").strip()
    if not api_key:
        raise RuntimeError("SYNTH_API_KEY is required")
    os.environ["SYNTH_API_KEY"] = api_key

    _local_task_url, env_key, local_port = start_local_api(
        local_host=args.local_host,
        local_port=args.local_port,
        backend_url=backend_url,
    )

    task_app_url, tunnel = await create_task_app_url(
        backend_url=backend_url,
        local_host=args.local_host,
        local_port=local_port,
        env_key=env_key,
        mode="online",
    )
    if tunnel:
        print("Waiting for tunnel propagation...")
        await asyncio.sleep(10.0)

    seeds = list(range(args.rollouts))
    config_body = build_mipro_online_config(
        task_app_url=task_app_url,
        train_seeds=seeds,
        val_seeds=[seed + len(seeds) for seed in range(min(2, len(seeds)))],
        policy_model=args.model,
        proposer_model=args.proposer_model,
        proposer_provider=args.proposer_provider,
        min_rollouts_before_proposal=args.min_rollouts_before_proposal,
        split=args.split,
        max_iterations=args.max_iterations,
        max_depth=args.max_depth,
    )

    job_id = create_job(backend_url, api_key, config_body)
    print(f"Online MIPRO job: {job_id}")

    detail = get_job_detail(backend_url, api_key, job_id, include_metadata=True)
    metadata = detail.get("metadata", {})
    system_id = metadata.get("mipro_system_id")
    proxy_url = metadata.get("mipro_proxy_url")
    if not system_id or not proxy_url:
        raise RuntimeError(f"Missing mipro_system_id/proxy_url in metadata: {metadata}")
    print(f"System ID: {system_id}")
    print(f"Proxy URL: {proxy_url}")

    try:
        for seed in seeds:
            rollout_id = new_rollout_id(seed)
            inference_url = f"{proxy_url}/{rollout_id}"
            reward, used_candidate_id = run_rollout(
                task_app_url=task_app_url,
                env_key=env_key,
                seed=seed,
                split=args.split,
                max_iterations=args.max_iterations,
                max_depth=args.max_depth,
                inference_url=inference_url,
                model=args.model,
                api_key=api_key,
                rollout_id=rollout_id,
                timeout_s=args.rollout_timeout,
            )
            candidate_label = used_candidate_id or "n/a"
            print(
                f"Rollout {seed}: reward={reward:.3f} id={rollout_id} candidate={candidate_label}"
            )
            push_status(
                backend_url=backend_url,
                api_key=api_key,
                system_id=system_id,
                rollout_id=rollout_id,
                reward=reward,
                candidate_id=used_candidate_id,
            )

        state = get_system_state(backend_url, api_key, system_id)
        best_candidate_id = state.get("best_candidate_id")
        candidate_text = extract_candidate_text(state, best_candidate_id)
        print(
            "Online state: "
            f"best_candidate_id={best_candidate_id} "
            f"version={state.get('version')} "
            f"candidates={len(state.get('candidates', {}))}"
        )
        if candidate_text:
            preview = candidate_text[:800] + ("..." if len(candidate_text) > 800 else "")
            print("\nBest candidate text:\n" + preview)
        else:
            print("\nBest candidate text: <not available>")
    finally:
        if tunnel:
            print("\nClosing tunnel...")
            tunnel.close()


if __name__ == "__main__":
    asyncio.run(main())
