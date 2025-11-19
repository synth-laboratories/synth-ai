"""MATH competition math task app for LangProBe benchmarks."""

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

MATH_DATASET = "hendrycks/competition_math"
DEFAULT_SPLIT = "test"
AVAILABLE_SPLITS: tuple[str, ...] = ("train", "test")

math_router = APIRouter()


MATH_DATASET_SPEC = TaskDatasetSpec(
    id="math",
    name="MATH Competition Math",
    version="1.0.0",
    splits=list(AVAILABLE_SPLITS),
    default_split=DEFAULT_SPLIT,
    description="High school and competition math problems from Hendrycks MATH dataset.",
)


_BOXED_MARKERS: tuple[str, ...] = ("\\boxed", "boxed")


def _extract_boxed(text: str) -> str | None:
    """Extract content from \\boxed{...} LaTeX expression."""
    if not text:
        return None
    for marker in _BOXED_MARKERS:
        start = text.find(marker)
        if start == -1:
            continue
        brace_start = text.find("{", start)
        if brace_start == -1:
            continue
        depth = 1
        idx = brace_start + 1
        while idx < len(text) and depth > 0:
            ch = text[idx]
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
            idx += 1
        if depth == 0:
            return text[brace_start + 1 : idx - 1].strip()
    return None


_FRAC_PATTERN = re.compile(r"\\?frac\{([^{}]+)\}\{([^{}]+)\}")
_SQRT_PATTERN = re.compile(r"\\?sqrt\{([^{}]+)\}")


def _normalize_final_answer(text: str) -> str:
    """Normalize LaTeX math answer to comparable form."""
    raw = str(text or "").strip()
    if not raw:
        return ""
    boxed = _extract_boxed(raw)
    if boxed:
        raw = boxed
    raw = raw.strip().strip("$")
    raw = raw.replace("\\left", "").replace("\\right", "")
    raw = raw.replace("\\!", "").replace("\\,", " ").replace("\\;", " ")
    raw = raw.replace("left", "").replace("right", "")
    raw = raw.replace("\\times", "*").replace("\\cdot", "*")
    raw = raw.replace("\\pi", "pi").replace("\\theta", "theta").replace("\\phi", "phi")
    raw = raw.replace("\\pm", "+/-").replace("\\mp", "-/+")
    raw = raw.replace("^{\\circ}", "deg").replace("^\\circ", "deg").replace("\\circ", "deg")

    def _frac_sub(match: re.Match[str]) -> str:
        num = match.group(1).strip()
        den = match.group(2).strip()
        return f"({num})/({den})"

    def _sqrt_sub(match: re.Match[str]) -> str:
        inner = match.group(1).strip()
        return f"sqrt({inner})"

    raw = _FRAC_PATTERN.sub(_frac_sub, raw)
    raw = _SQRT_PATTERN.sub(_sqrt_sub, raw)
    raw = raw.replace("\\", "")
    raw = raw.replace("{", "").replace("}", "")
    raw = raw.replace(" ", "")
    raw = raw.rstrip(".")
    return raw


class MathDataset:
    """Lazy loader and sampler for the MATH competition math dataset."""

    def __init__(self) -> None:
        self._splits: dict[str, Any] = {}

    def _load_split(self, split: str):
        if split not in AVAILABLE_SPLITS:
            raise ValueError(f"Unknown split '{split}'. Available: {AVAILABLE_SPLITS}")
        if split not in self._splits:
            try:
                self._splits[split] = load_dataset(MATH_DATASET, split=split)
            except Exception as exc:  # pragma: no cover - network/dataset errors
                raise RuntimeError(
                    f"Failed to download MATH split '{split}'. "
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
            raise RuntimeError(f"MATH split '{split}' is empty")
        idx = int(index) % size
        row = dataset[int(idx)]

        problem = str(row.get("problem") or "")
        solution = str(row.get("solution") or "")

        # Extract final answer from solution (look for boxed answer)
        normalized_answer = _normalize_final_answer(solution)
        if not normalized_answer:
            # Fallback: try to extract from problem or use solution as-is
            normalized_answer = _normalize_final_answer(problem) or _normalize_final_answer(solution)

        return {
            "index": idx,
            "split": split,
            "problem": problem,
            "solution": solution,
            "answer": normalized_answer,
        }


async def rollout_executor(request: RolloutRequest, fastapi_request: Request) -> RolloutResponse:
    dataset: MathDataset = fastapi_request.app.state.math_dataset

    split = str(((request.env.config or {}).get("split")) or DEFAULT_SPLIT)
    seed = request.env.seed or 0

    sample = dataset.sample(split=split, index=seed)
    observation = {
        "problem": sample["problem"],
        "index": sample["index"],
        "split": sample["split"],
    }

    placeholders = {
        "problem": sample["problem"],
    }

    default_messages = [
        {
            "role": "system",
            "pattern": (
                "You are a math problem solver. Solve the given math problem step by step. "
                "Provide your final answer clearly."
            ),
        },
        {
            "role": "user",
            "pattern": "Problem:\n{problem}\n\nSolve this problem and provide the final answer.",
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

    # Extract answer from response (look for boxed or final number)
    predicted_answer = _normalize_final_answer(response_text)
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
            f"[MATH_ROLLOUT] run_id={request.run_id} split={sample['split']} "
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
        env_id=f"math::{sample['split']}::{sample['index']}",
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
                "env": "math",
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


def build_dataset() -> tuple[TaskDatasetRegistry, MathDataset]:
    registry = TaskDatasetRegistry()
    dataset = MathDataset()
    dataset.ensure_ready([DEFAULT_SPLIT])
    registry.register(MATH_DATASET_SPEC, lambda _spec: dataset, cache=True)
    return registry, dataset


def _base_task_info() -> TaskInfo:
    return TaskInfo(
        task={
            "id": "math",
            "name": "MATH Competition Math",
            "version": "1.0.0",
            "action_space": {
                "type": "free_text",
                "description": "Solve math problems and provide final answer.",
            },
        },
        environment="math",
        dataset={
            **MATH_DATASET_SPEC.model_dump(),
            "hf_dataset": MATH_DATASET,
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
            "format": "Free-form math problem solving",
        },
    )


def describe_taskset(dataset: MathDataset) -> Mapping[str, Any]:
    return {
        **MATH_DATASET_SPEC.model_dump(),
        "hf_dataset": MATH_DATASET,
        "sizes": {split: dataset.size(split) for split in AVAILABLE_SPLITS},
    }


def provide_task_instances(dataset: MathDataset, seeds: Sequence[int]) -> Iterable[TaskInfo]:
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
                "problem": sample["problem"][:200],  # Preview
            },
        )


OUTCOME_RUBRIC: Rubric = cast(
    Rubric,
    load_rubric(
        {
            "version": "1",
            "goal_text": "Solve MATH competition math problems correctly.",
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
        system_hint="Solve math problems step by step and provide the final answer.",
    )

    config = TaskAppConfig(
        app_id="math",
        name="MATH Competition Math Task",
        description="MATH environment for evaluating prompt optimisers on competition math problems.",
        base_task_info=base_info,
        describe_taskset=lambda: describe_taskset(dataset),
        provide_task_instances=lambda seeds: provide_task_instances(dataset, seeds),
        rollout=rollout_executor,
        dataset_registry=registry,
        rubrics=RubricBundle(outcome=OUTCOME_RUBRIC, events=EVENTS_RUBRIC),
        proxy=proxy_config,
        routers=(math_router,),
        app_state={"math_dataset": dataset},
        cors_origins=["*"],
    )
    return config


register_task_app(
    entry=TaskAppEntry(
        app_id="math",
        description="MATH competition math task app using hendrycks/competition_math.",
        config_factory=build_config,
        aliases=("math-competition",),
        modal=ModalDeploymentConfig(
            app_name="synth-math",
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

    parser = argparse.ArgumentParser(description="Run the MATH task app locally")
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

