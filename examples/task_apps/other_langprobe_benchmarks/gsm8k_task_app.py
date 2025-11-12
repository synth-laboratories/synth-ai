"""GSM8K grade school math task app for LangProBe benchmarks."""

from __future__ import annotations

import contextlib
import os
import re
import uuid
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any, Mapping, cast

from datasets import load_dataset
from dotenv import load_dotenv
from fastapi import APIRouter, HTTPException, Request

load_dotenv()  # Load environment variables from .env

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

from ..gepa_benchmarks.common import call_chat_completion, normalise_answer

REPO_ROOT = Path(__file__).resolve().parents[3]

GSM8K_DATASET = "openai/gsm8k"
DEFAULT_SPLIT = "test"
AVAILABLE_SPLITS: tuple[str, ...] = ("train", "test")

gsm8k_router = APIRouter()


GSM8K_DATASET_SPEC = TaskDatasetSpec(
    id="gsm8k",
    name="GSM8K Grade School Math",
    version="1.0.0",
    splits=list(AVAILABLE_SPLITS),
    default_split=DEFAULT_SPLIT,
    description="Grade school math word problems from OpenAI GSM8K dataset.",
)


def _extract_final_number(text: str) -> str:
    """Extract the final numeric answer from text."""
    if not text:
        return ""
    
    # Look for patterns like "The answer is 42" or "Answer: 42" or just numbers at the end
    # Try to find the last number in the text
    numbers = re.findall(r'\d+\.?\d*', text)
    if numbers:
        return numbers[-1]
    
    # Fallback: normalize and try to extract
    normalized = normalise_answer(text)
    numbers = re.findall(r'\d+\.?\d*', normalized)
    if numbers:
        return numbers[-1]
    
    return ""


def _normalize_answer(text: str) -> str:
    """Normalize answer for comparison."""
    if not text:
        return ""
    
    # Extract final number
    answer = _extract_final_number(text)
    if answer:
        # Remove trailing zeros and decimal point if not needed
        try:
            num = float(answer)
            if num == int(num):
                return str(int(num))
            return str(num)
        except ValueError:
            return answer.strip()
    
    # Fallback to general normalization
    normalized = normalise_answer(text)
    # Try to extract number from normalized
    numbers = re.findall(r'\d+\.?\d*', normalized)
    if numbers:
        try:
            num = float(numbers[-1])
            if num == int(num):
                return str(int(num))
            return str(num)
        except ValueError:
            pass
    
    return normalized.strip()


class GSM8KDataset:
    """Lazy loader and sampler for the GSM8K dataset."""

    def __init__(self) -> None:
        self._splits: dict[str, Any] = {}

    def _load_split(self, split: str):
        if split not in AVAILABLE_SPLITS:
            raise ValueError(f"Unknown split '{split}'. Available: {AVAILABLE_SPLITS}")
        if split not in self._splits:
            try:
                self._splits[split] = load_dataset(GSM8K_DATASET, "main", split=split)
            except Exception as exc:  # pragma: no cover - network/dataset errors
                raise RuntimeError(
                    f"Failed to download GSM8K split '{split}'. "
                    f"Ensure network access to Hugging Face datasets."
                ) from exc
        return self._splits[split]

    def ensure_ready(self, required_splits: Sequence[str]) -> None:
        for split in required_splits:
            self._load_split(split)

    def size(self, split: str) -> int:
        dataset = self._load_split(split)
        return len(dataset)

    def sample(self, *, split: str, index: int) -> dict[str, Any]:
        dataset = self._load_split(split)
        size = len(dataset)
        if size == 0:
            raise RuntimeError(f"GSM8K split '{split}' is empty")
        idx = int(index) % size
        row = dataset[int(idx)]

        question = str(row.get("question") or "")
        answer = str(row.get("answer") or "")
        
        # GSM8K answer format: "### Answer: 42" or just "42"
        # Extract the numeric answer
        normalized_answer = _normalize_answer(answer)
        if not normalized_answer:
            # Try extracting from question or use raw answer
            normalized_answer = _extract_final_number(answer) or answer.strip()

        return {
            "index": idx,
            "split": split,
            "question": question,
            "answer": normalized_answer,
            "raw_answer": answer,
        }


async def rollout_executor(request: RolloutRequest, fastapi_request: Request) -> RolloutResponse:
    dataset: GSM8KDataset = fastapi_request.app.state.gsm8k_dataset

    split = str(((request.env.config or {}).get("split")) or DEFAULT_SPLIT)
    seed = request.env.seed or 0

    sample = dataset.sample(split=split, index=seed)
    observation = {
        "question": sample["question"],
        "index": sample["index"],
        "split": sample["split"],
    }

    placeholders = {
        "question": sample["question"],
    }

    default_messages = [
        {
            "role": "system",
            "pattern": (
                "You are a math problem solver. Solve the given grade school math word problem step by step. "
                "Show your reasoning and provide the final answer as a number."
            ),
        },
        {
            "role": "user",
            "pattern": "Question: {question}\n\nSolve this problem step by step and provide the final answer.",
        },
    ]

    tool_calls: list[dict[str, Any]] = []
    response_json: dict[str, Any] | None = None
    response_text = ""
    error_info: dict[str, Any] = {}

    try:
        response_text, response_json, _ = await call_chat_completion(
            request.policy.config or {},
            placeholders,
            default_messages,
        )
    except HTTPException as http_err:  # pragma: no cover - passthrough to metrics
        error_info = {"error": str(http_err.detail), "code": http_err.status_code}
    except Exception as exc:  # pragma: no cover - defensive logging
        error_info = {"error": str(exc)}

    # Extract answer from response
    predicted_answer = _normalize_answer(response_text)
    expected_answer = sample["answer"]
    answer_correct = int(predicted_answer == expected_answer)

    reward = float(answer_correct)

    info_payload = {
        "expected_answer": expected_answer,
        "predicted_answer": predicted_answer,
        "answer_correct": bool(answer_correct),
        "response_text": response_text[:500],  # Truncate for logging
        "response_json": response_json,
        **error_info,
    }

    with contextlib.suppress(Exception):
        print(
            f"[GSM8K_ROLLOUT] run_id={request.run_id} split={sample['split']} "
            f"index={sample['index']} answer_correct={answer_correct} "
            f"reward={reward:.3f}",
            flush=True,
        )

    step = RolloutStep(
        obs=observation,
        tool_calls=tool_calls,
        reward=reward,
        done=True,
        info=info_payload,
    )

    inference_url = (request.policy.config or {}).get("inference_url")

    trajectory = RolloutTrajectory(
        env_id=f"gsm8k::{sample['split']}::{sample['index']}",
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
        details={
            "answer_correct": bool(answer_correct),
        },
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
                "env": "gsm8k",
                "split": sample["split"],
                "index": sample["index"],
                "answer_correct": answer_correct,
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


def build_dataset() -> tuple[TaskDatasetRegistry, GSM8KDataset]:
    registry = TaskDatasetRegistry()
    dataset = GSM8KDataset()
    dataset.ensure_ready([DEFAULT_SPLIT])
    registry.register(GSM8K_DATASET_SPEC, lambda _spec: dataset, cache=True)
    return registry, dataset


def _base_task_info() -> TaskInfo:
    return TaskInfo(
        task={
            "id": "gsm8k",
            "name": "GSM8K Grade School Math",
            "version": "1.0.0",
            "action_space": {
                "type": "free_text",
                "description": "Solve grade school math word problems and provide final answer.",
            },
        },
        environment="gsm8k",
        dataset={
            **GSM8K_DATASET_SPEC.model_dump(),
            "hf_dataset": GSM8K_DATASET,
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
        task_metadata={
            "format": "Free-form math word problem solving",
        },
    )


def describe_taskset(dataset: GSM8KDataset) -> Mapping[str, Any]:
    return {
        **GSM8K_DATASET_SPEC.model_dump(),
        "hf_dataset": GSM8K_DATASET,
        "sizes": {split: dataset.size(split) for split in AVAILABLE_SPLITS},
    }


def provide_task_instances(dataset: GSM8KDataset, seeds: Sequence[int]) -> Iterable[TaskInfo]:
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
            task_metadata={
                **base_info.task_metadata,
                "question": sample["question"][:200],  # Preview
            },
        )


OUTCOME_RUBRIC: Rubric = cast(
    Rubric,
    load_rubric(
        {
            "version": "1",
            "goal_text": "Solve GSM8K grade school math problems correctly.",
            "aggregation": "weighted_sum",
            "criteria": [
                {
                    "id": "answer_accuracy",
                    "description": "Final answer matches the correct solution.",
                    "weight": 1.0,
                },
            ],
        }
    ),
)

EVENTS_RUBRIC: Rubric = cast(
    Rubric,
    load_rubric(
        {
            "version": "1",
            "goal_text": "Encourage clear problem-solving responses.",
            "aggregation": "weighted_sum",
            "criteria": [
                {
                    "id": "response_quality",
                    "description": "Provide a clear, well-reasoned solution.",
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
        system_hint="Solve grade school math problems step by step and provide the final answer.",
    )

    config = TaskAppConfig(
        app_id="gsm8k",
        name="GSM8K Grade School Math Task",
        description="GSM8K environment for evaluating prompt optimisers on grade school math problems.",
        base_task_info=base_info,
        describe_taskset=lambda: describe_taskset(dataset),
        provide_task_instances=lambda seeds: provide_task_instances(dataset, seeds),
        rollout=rollout_executor,
        dataset_registry=registry,
        rubrics=RubricBundle(outcome=OUTCOME_RUBRIC, events=EVENTS_RUBRIC),
        proxy=proxy_config,
        routers=(gsm8k_router,),
        app_state={"gsm8k_dataset": dataset},
        cors_origins=["*"],
    )
    return config


register_task_app(
    entry=TaskAppEntry(
        app_id="gsm8k",
        description="GSM8K grade school math task app using openai/gsm8k.",
        config_factory=build_config,
        aliases=("gsm8k-math",),
        modal=ModalDeploymentConfig(
            app_name="synth-gsm8k",
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


if __name__ == "__main__":  # pragma: no cover - manual local run helper
    import argparse
    from synth_ai.task.server import run_task_app

    parser = argparse.ArgumentParser(description="Run the GSM8K task app locally")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8112)
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

