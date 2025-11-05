"""IFBench instruction-following task app."""

from __future__ import annotations

import contextlib
import os
import re
import uuid
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any, Mapping, cast

from datasets import load_dataset
from fastapi import APIRouter, HTTPException, Request

from synth_ai.task.apps import ModalDeploymentConfig, TaskAppEntry, register_task_app
from synth_ai.task.contracts import (
    RolloutMetrics,
    RolloutRequest,
    RolloutResponse,
    RolloutStep,
    RolloutTrajectory,
    TaskInfo,
)
from synth_ai.task.datasets import TaskDatasetRegistry, TaskDatasetSpec
from synth_ai.task.rubrics import Rubric, load_rubric
from synth_ai.task.server import ProxyConfig, RubricBundle, TaskAppConfig
from synth_ai.task.vendors import normalize_vendor_keys

from .common import (
    call_chat_completion,
    count_emojis,
    count_numbers,
    count_pronouns,
    sentence_split,
    tokenize,
    unique_word_count,
)

REPO_ROOT = Path(__file__).resolve().parents[3]

DATASET_ID = "allenai/IFBench_test"
AVAILABLE_SPLITS: tuple[str, ...] = ("train",)
DEFAULT_SPLIT = "train"


ifbench_router = APIRouter()


IFBENCH_DATASET_SPEC = TaskDatasetSpec(
    id="ifbench",
    name="IFBench Instruction Following",
    version="1.0.0",
    splits=list(AVAILABLE_SPLITS),
    default_split=DEFAULT_SPLIT,
    description="Instruction following benchmark with programmatically-checked constraints.",
)

SUPPORTED_INSTRUCTIONS = {
    "count:keywords_multiple",
    "sentence:keyword",
    "count:numbers",
    "count:word_count_range",
    "count:unique_word_count",
    "count:pronouns",
    "format:list",
    "format:emoji",
}


class IFBenchDataset:
    """Load IFBench and filter to instructions we can automatically score."""

    def __init__(self) -> None:
        self._cache: dict[str, list[dict[str, Any]]] = {}

    def _load_split(self, split: str) -> list[dict[str, Any]]:
        if split not in AVAILABLE_SPLITS:
            raise ValueError(f"Unknown split '{split}'. Available: {AVAILABLE_SPLITS}")
        if split not in self._cache:
            try:
                raw = load_dataset(DATASET_ID, split=split)
            except Exception as exc:  # pragma: no cover
                raise RuntimeError(
                    f"Failed to download IFBench split '{split}'. Ensure network access."
                ) from exc
            filtered = [
                row
                for row in raw
                if set(row.get("instruction_id_list") or ()).issubset(SUPPORTED_INSTRUCTIONS)
            ]
            if not filtered:
                raise RuntimeError(
                    f"No IFBench samples remain after filtering for supported instructions ({SUPPORTED_INSTRUCTIONS})."
                )
            self._cache[split] = filtered
        return self._cache[split]

    def ensure_ready(self, splits: Sequence[str]) -> None:
        for split in splits:
            self._load_split(split)

    def size(self, split: str) -> int:
        return len(self._load_split(split))

    def sample(self, *, split: str, index: int) -> dict[str, Any]:
        dataset = self._load_split(split)
        size = len(dataset)
        if size == 0:
            raise RuntimeError(f"IFBench split '{split}' is empty")
        idx = int(index) % size
        row = dataset[int(idx)]

        instructions = []
        ids = row.get("instruction_id_list") or []
        kwargs_list = row.get("kwargs") or []
        for instr_id, kwargs in zip(ids, kwargs_list):
            instructions.append(
                {
                    "id": str(instr_id),
                    "kwargs": kwargs or {},
                }
            )

        return {
            "index": idx,
            "split": split,
            "prompt": str(row.get("prompt") or ""),
            "instructions": instructions,
        }


def _summarise_kwargs(kwargs: Mapping[str, Any]) -> str:
    items = []
    for key, value in kwargs.items():
        if value in (None, "", [], {}):
            continue
        items.append(f"{key}={value}")
    return ", ".join(items) if items else "default"


_KEYWORD_PATTERN = re.compile(
    r"keyword\s+([a-z0-9_-]+)\s+(once|twice|\d+\s+times?)",
    flags=re.IGNORECASE,
)


def _extract_keyword_targets(prompt: str, keywords: Sequence[str]) -> dict[str, int]:
    targets: dict[str, int] = {}
    for match in _KEYWORD_PATTERN.finditer(prompt):
        word = match.group(1)
        if word not in keywords:
            continue
        count_str = match.group(2).lower()
        if count_str == "once":
            targets[word] = 1
        elif count_str == "twice":
            targets[word] = 2
        else:
            digit_match = re.search(r"\d+", count_str)
            targets[word] = int(digit_match.group()) if digit_match else 1
    return targets


def _evaluate_instruction(
    instr_id: str,
    kwargs: Mapping[str, Any],
    prompt: str,
    response: str,
) -> tuple[bool, dict[str, Any]]:
    tokens = tokenize(response)
    details: dict[str, Any] = {}

    if instr_id == "count:keywords_multiple":
        keywords = [
            kwargs.get("keyword1"),
            kwargs.get("keyword2"),
            kwargs.get("keyword3"),
            kwargs.get("keyword4"),
            kwargs.get("keyword5"),
        ]
        keywords = [str(word) for word in keywords if word]
        targets = _extract_keyword_targets(prompt, keywords)
        passes = True
        occurrences: dict[str, int] = {}
        for word in keywords:
            expected = targets.get(word, 1)
            actual = len(re.findall(rf"\b{re.escape(word)}\b", response, flags=re.IGNORECASE))
            occurrences[word] = actual
            if actual < expected:
                passes = False
        details.update({"keywords": keywords, "counts": occurrences, "targets": targets})
        return passes, details

    if instr_id == "sentence:keyword":
        target_word = str(kwargs.get("word") or "").strip()
        expected = int(kwargs.get("N") or 1)
        sentences = sentence_split(response)
        satisfied = sum(
            1 for sentence in sentences if re.search(rf"\b{re.escape(target_word)}\b", sentence, re.IGNORECASE)
        )
        details.update({"word": target_word, "required": expected, "actual": satisfied})
        return satisfied >= expected, details

    if instr_id == "count:numbers":
        expected = int(kwargs.get("N") or 0)
        actual = count_numbers(response)
        details.update({"required": expected, "actual": actual})
        return actual >= expected, details

    if instr_id == "count:word_count_range":
        min_words = int(kwargs.get("min_words") or 0)
        max_words = int(kwargs.get("max_words") or 10_000)
        word_count = len(tokens)
        details.update({"min": min_words, "max": max_words, "actual": word_count})
        return min_words <= word_count <= max_words, details

    if instr_id == "count:unique_word_count":
        expected = int(kwargs.get("N") or 0)
        actual = unique_word_count(tokens)
        details.update({"required": expected, "actual": actual})
        return actual >= expected, details

    if instr_id == "count:pronouns":
        expected = int(kwargs.get("N") or 0)
        actual = count_pronouns(tokens)
        details.update({"required": expected, "actual": actual})
        return actual >= expected, details

    if instr_id == "format:list":
        separator = str(kwargs.get("sep") or "-").strip()
        lines = [line.strip() for line in response.splitlines() if line.strip()]
        bullet_lines = [line for line in lines if line.startswith(separator)]
        details.update({"separator": separator, "bullet_count": len(bullet_lines)})
        return len(bullet_lines) >= 2, details

    if instr_id == "format:emoji":
        expected = int(kwargs.get("N") or 1)
        emoji_count = count_emojis(response)
        details.update({"required": expected, "actual": emoji_count})
        return emoji_count >= expected, details

    return False, {"unsupported": True}


def evaluate_ifbench(prompt: str, instructions: Sequence[Mapping[str, Any]], response: str) -> tuple[float, dict[str, Any]]:
    results: dict[str, Any] = {}
    passed = 0
    total = 0
    for instruction in instructions:
        instr_id = str(instruction.get("id") or "")
        kwargs = instruction.get("kwargs") or {}
        ok, details = _evaluate_instruction(instr_id, kwargs, prompt, response)
        results[instr_id] = {"pass": ok, **details}
        if instr_id in SUPPORTED_INSTRUCTIONS:
            total += 1
            if ok:
                passed += 1
    reward = (passed / total) if total else 0.0
    return reward, {"passed": passed, "total": total, "details": results}


async def rollout_executor(request: RolloutRequest, fastapi_request: Request) -> RolloutResponse:
    dataset: IFBenchDataset = fastapi_request.app.state.ifbench_dataset

    split = str(((request.env.config or {}).get("split")) or DEFAULT_SPLIT)
    seed = request.env.seed or 0

    sample = dataset.sample(split=split, index=seed)

    instruction_lines = [
        f"- {instr['id']} ({_summarise_kwargs(instr['kwargs'])})" for instr in sample["instructions"]
    ]
    constraints_text = "\n".join(instruction_lines)

    observation = {
        "prompt": sample["prompt"],
        "instructions": sample["instructions"],
        "index": sample["index"],
        "split": sample["split"],
    }

    placeholders = {
        "prompt": sample["prompt"],
        "instructions": constraints_text,
    }

    default_messages = [
        {
            "role": "system",
            "pattern": (
                "You must follow every instruction exactly. Produce a single response that satisfies all constraints."
            ),
        },
        {
            "role": "user",
            "pattern": "Instructions:\n{instructions}\n\nTask:\n{prompt}",
        },
    ]

    response_json: dict[str, Any] | None = None
    response_text = ""
    error_info: dict[str, Any] = {}

    try:
        response_text, response_json, _ = await call_chat_completion(
            request.policy.config or {},
            placeholders,
            default_messages,
        )
    except HTTPException as http_err:  # pragma: no cover
        error_info = {"error": str(http_err.detail), "code": http_err.status_code}
    except Exception as exc:  # pragma: no cover
        error_info = {"error": str(exc)}

    reward, eval_details = evaluate_ifbench(sample["prompt"], sample["instructions"], response_text)
    eval_details["response_json"] = response_json
    eval_details.update(error_info)

    with contextlib.suppress(Exception):
        print(
            f"[IFBENCH_ROLLOUT] run_id={request.run_id} index={sample['index']} "
            f"passed={eval_details['passed']}/{eval_details['total']} reward={reward:.3f}",
            flush=True,
        )

    step = RolloutStep(
        obs=observation,
        tool_calls=[],
        reward=reward,
        done=True,
        info=eval_details,
    )

    inference_url = (request.policy.config or {}).get("inference_url")
    trajectory = RolloutTrajectory(
        env_id=f"ifbench::{sample['split']}::{sample['index']}",
        policy_id=request.policy.policy_id or request.policy.policy_name or "policy",
        steps=[step],
        final={"observation": observation, "reward": reward},
        length=1,
        inference_url=str(inference_url or ""),
    )

    metrics = RolloutMetrics(
        episode_returns=[reward],
        mean_return=reward,
        num_steps=1,
        num_episodes=1,
        outcome_score=reward,
        events_score=reward,
        details={"constraints_passed": eval_details.get("passed"), "constraints_total": eval_details.get("total")},
    )

    trace_payload = None
    include_trace = bool(
        (request.record and getattr(request.record, "return_trace", False))
        or os.getenv("TASKAPP_TRACING_ENABLED")
    )
    if include_trace:
        trace_payload = {
            "session_id": str(uuid.uuid4()),
            "events_count": 1,
            "decision_rewards": [reward],
            "metadata": {
                "env": "ifbench",
                "split": sample["split"],
                "index": sample["index"],
                "constraints_passed": eval_details.get("passed"),
                "constraints_total": eval_details.get("total"),
            },
        }

    return RolloutResponse(
        run_id=request.run_id,
        trajectories=[trajectory],
        branches={},
        metrics=metrics,
        aborted=False,
        ops_executed=2,
        trace=trace_payload,
    )


def build_dataset() -> tuple[TaskDatasetRegistry, IFBenchDataset]:
    registry = TaskDatasetRegistry()
    dataset = IFBenchDataset()
    dataset.ensure_ready([DEFAULT_SPLIT])
    registry.register(IFBENCH_DATASET_SPEC, lambda _spec: dataset, cache=True)
    return registry, dataset


def _base_task_info() -> TaskInfo:
    return TaskInfo(
        task={
            "id": "ifbench",
            "name": "IFBench Instruction Following",
            "version": "1.0.0",
            "action_space": {
                "type": "free_text",
                "description": "Generate a completion that satisfies all constraints.",
            },
        },
        environment="ifbench",
        dataset={
            **IFBENCH_DATASET_SPEC.model_dump(),
            "hf_dataset": DATASET_ID,
        },
        rubric={
            "version": "1",
            "criteria_count": 1,
            "source": "inline",
        },
        inference={
            "supports_proxy": True,
            "tool": None,
        },
        limits={"max_turns": 1},
        task_metadata={"supported_instructions": sorted(SUPPORTED_INSTRUCTIONS)},
    )


def describe_taskset(dataset: IFBenchDataset) -> Mapping[str, Any]:
    return {
        **IFBENCH_DATASET_SPEC.model_dump(),
        "hf_dataset": DATASET_ID,
        "supported_instructions": sorted(SUPPORTED_INSTRUCTIONS),
        "sizes": {split: dataset.size(split) for split in AVAILABLE_SPLITS},
    }


def provide_task_instances(dataset: IFBenchDataset, seeds: Sequence[int]) -> Iterable[TaskInfo]:
    base_info = _base_task_info()
    for seed in seeds:
        sample = dataset.sample(split=DEFAULT_SPLIT, index=seed)
        yield TaskInfo(
            task=base_info.task,
            environment=base_info.environment,
            dataset={
                **base_info.dataset,
                "split": sample["split"],
                "index": sample["index"],
            },
            rubric=base_info.rubric,
            inference=base_info.inference,
            limits=base_info.limits,
            task_metadata={**base_info.task_metadata, "prompt": sample["prompt"][:80]},
        )


OUTCOME_RUBRIC: Rubric = cast(
    Rubric,
    load_rubric(
        {
            "version": "1",
            "goal_text": "Satisfy the IFBench constraints.",
            "aggregation": "weighted_sum",
            "criteria": [
                {
                    "id": "constraint_satisfaction",
                    "description": "Meets all programmatically-checked constraints.",
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
            "goal_text": "Keep responses concise while following instructions.",
            "aggregation": "weighted_sum",
            "criteria": [
                {
                    "id": "concise_answer",
                    "description": "Avoid unnecessary content while satisfying constraints.",
                    "weight": 1.0,
                }
            ],
        }
    ),
)


def build_config() -> TaskAppConfig:
    registry, dataset = build_dataset()
    base_info = _base_task_info()

    proxy_keys = normalize_vendor_keys()
    proxy_config = ProxyConfig(
        enable_openai=proxy_keys.get("OPENAI_API_KEY") is not None,
        enable_groq=proxy_keys.get("GROQ_API_KEY") is not None,
        system_hint="Follow every instruction exactly. Violations are failures.",
    )

    config = TaskAppConfig(
        app_id="ifbench",
        name="IFBench Instruction Following Task",
        description="IFBench task app with automatic constraint checking for prompt optimisation.",
        base_task_info=base_info,
        describe_taskset=lambda: describe_taskset(dataset),
        provide_task_instances=lambda seeds: provide_task_instances(dataset, seeds),
        rollout=rollout_executor,
        dataset_registry=registry,
        rubrics=RubricBundle(outcome=OUTCOME_RUBRIC, events=EVENTS_RUBRIC),
        proxy=proxy_config,
        routers=(ifbench_router,),
        app_state={"ifbench_dataset": dataset},
        cors_origins=["*"],
    )
    return config


register_task_app(
    entry=TaskAppEntry(
        app_id="ifbench",
        description="IFBench task app using automatically scored constraint subsets.",
        config_factory=build_config,
        aliases=("ifbench-instructions",),
        modal=ModalDeploymentConfig(
            app_name="synth-ifbench",
            pip_packages=(
                "datasets>=2.14.0",
                "fastapi>=0.115.0",
                "pydantic>=2.0.0",
                "httpx>=0.26.0",
            ),
            extra_local_dirs=((str(REPO_ROOT / "synth_ai"), "/opt/synth_ai_repo/synth_ai"),),
        ),
    )
)


if __name__ == "__main__":  # pragma: no cover - manual helper
    import argparse
    from synth_ai.task.server import run_task_app

    parser = argparse.ArgumentParser(description="Run the IFBench task app locally")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8111)
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
    env_files.extend(args.env_file or [])

    run_task_app(
        build_config,
        host=args.host,
        port=args.port,
        reload=args.reload,
        env_files=env_files,
    )
