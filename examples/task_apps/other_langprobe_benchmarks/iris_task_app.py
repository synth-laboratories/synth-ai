"""Iris classification task app for LangProBe benchmarks."""

from __future__ import annotations

import contextlib
import os
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

IRIS_DATASET = "scikit-learn/iris"
DEFAULT_SPLIT = "train"
AVAILABLE_SPLITS: tuple[str, ...] = ("train", "test")

iris_router = APIRouter()


IRIS_DATASET_SPEC = TaskDatasetSpec(
    id="iris",
    name="Iris Classification",
    version="1.0.0",
    splits=list(AVAILABLE_SPLITS),
    default_split=DEFAULT_SPLIT,
    description="Classic ML classification dataset with 3 iris species.",
)


def _normalize_classification(text: str) -> str:
    """Normalize classification label to one of: setosa, versicolor, virginica."""
    if not text:
        return ""
    normalized = normalise_answer(text)
    normalized_lower = normalized.lower()
    
    # Map common variations to standard labels
    if "setosa" in normalized_lower or normalized_lower == "0":
        return "setosa"
    if "versicolor" in normalized_lower or normalized_lower == "1":
        return "versicolor"
    if "virginica" in normalized_lower or normalized_lower == "2":
        return "virginica"
    
    return normalized


class IrisDataset:
    """Lazy loader and sampler for the Iris dataset."""

    def __init__(self) -> None:
        self._splits: dict[str, Any] = {}
        self._label_names: list[str] | None = None

    def _load_split(self, split: str):
        if split not in AVAILABLE_SPLITS:
            raise ValueError(f"Unknown split '{split}'. Available: {AVAILABLE_SPLITS}")
        if split not in self._splits:
            try:
                dataset = load_dataset(IRIS_DATASET, split=split)
                self._splits[split] = dataset
                # Extract label names
                if self._label_names is None and len(dataset) > 0:
                    if "label" in dataset.features:
                        self._label_names = dataset.features["label"].names
                    elif "target" in dataset.features:
                        self._label_names = dataset.features["target"].names
                    else:
                        # Fallback: use standard iris labels
                        self._label_names = ["setosa", "versicolor", "virginica"]
            except Exception as exc:  # pragma: no cover - network/dataset errors
                raise RuntimeError(
                    f"Failed to download Iris split '{split}'. "
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
            raise RuntimeError(f"Iris split '{split}' is empty")
        idx = int(index) % size
        row = dataset[int(idx)]

        # Extract features and label
        features = {}
        label_idx = None
        label_name = None
        
        for key, value in row.items():
            if key in ("label", "target"):
                label_idx = int(value) if isinstance(value, (int, str)) else 0
                if self._label_names and 0 <= label_idx < len(self._label_names):
                    label_name = self._label_names[label_idx]
                else:
                    label_name = str(value)
            elif key not in ("species", "class"):  # Skip redundant columns
                features[key] = value

        # Format features as text
        feature_text = "\n".join([f"{key}: {value}" for key, value in features.items()])

        return {
            "index": idx,
            "split": split,
            "features": features,
            "feature_text": feature_text,
            "label": label_name or "setosa",
            "label_idx": label_idx or 0,
        }


async def rollout_executor(request: RolloutRequest, fastapi_request: Request) -> RolloutResponse:
    dataset: IrisDataset = fastapi_request.app.state.iris_dataset

    split = str(((request.env.config or {}).get("split")) or DEFAULT_SPLIT)
    seed = request.env.seed or 0

    sample = dataset.sample(split=split, index=seed)
    observation = {
        "features": sample["feature_text"],
        "index": sample["index"],
        "split": sample["split"],
    }

    placeholders = {
        "features": sample["feature_text"],
    }

    default_messages = [
        {
            "role": "system",
            "pattern": (
                "You are a botany classification assistant. Based on the flower's measurements, "
                "classify the iris species. Respond with one of: setosa, versicolor, or virginica."
            ),
        },
        {
            "role": "user",
            "pattern": (
                "Flower Measurements:\n{features}\n\n"
                "Classify this iris flower. Respond with one of: setosa, versicolor, or virginica."
            ),
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

    # Normalize and compare
    predicted_label = _normalize_classification(response_text)
    expected_label = sample["label"]
    label_correct = int(predicted_label.lower() == expected_label.lower())

    reward = float(label_correct)

    info_payload = {
        "expected_label": expected_label,
        "predicted_label": predicted_label,
        "label_correct": bool(label_correct),
        "response_text": response_text[:500],
        "response_json": response_json,
        **error_info,
    }

    with contextlib.suppress(Exception):
        print(
            f"[IRIS_ROLLOUT] run_id={request.run_id} split={sample['split']} "
            f"index={sample['index']} label_correct={label_correct} "
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
        env_id=f"iris::{sample['split']}::{sample['index']}",
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
            "label_correct": bool(label_correct),
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
                "env": "iris",
                "split": sample["split"],
                "index": sample["index"],
                "label_correct": label_correct,
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


def build_dataset() -> tuple[TaskDatasetRegistry, IrisDataset]:
    registry = TaskDatasetRegistry()
    dataset = IrisDataset()
    dataset.ensure_ready([DEFAULT_SPLIT])
    registry.register(IRIS_DATASET_SPEC, lambda _spec: dataset, cache=True)
    return registry, dataset


def _base_task_info() -> TaskInfo:
    return TaskInfo(
        task={
            "id": "iris",
            "name": "Iris Classification",
            "version": "1.0.0",
            "action_space": {
                "type": "free_text",
                "description": "Classify iris flower species: setosa, versicolor, or virginica.",
            },
        },
        environment="iris",
        dataset={
            **IRIS_DATASET_SPEC.model_dump(),
            "hf_dataset": IRIS_DATASET,
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
            "format": "Multi-class classification (setosa, versicolor, virginica)",
        },
    )


def describe_taskset(dataset: IrisDataset) -> Mapping[str, Any]:
    return {
        **IRIS_DATASET_SPEC.model_dump(),
        "hf_dataset": IRIS_DATASET,
        "sizes": {split: dataset.size(split) for split in AVAILABLE_SPLITS},
    }


def provide_task_instances(dataset: IrisDataset, seeds: Sequence[int]) -> Iterable[TaskInfo]:
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
                "features": sample["feature_text"][:200],  # Preview
            },
        )


OUTCOME_RUBRIC: Rubric = cast(
    Rubric,
    load_rubric(
        {
            "version": "1",
            "goal_text": "Correctly classify iris flowers into the correct species.",
            "aggregation": "weighted_sum",
            "criteria": [
                {
                    "id": "classification_accuracy",
                    "description": "Classification label matches the correct species.",
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
            "goal_text": "Encourage clear classification responses.",
            "aggregation": "weighted_sum",
            "criteria": [
                {
                    "id": "response_quality",
                    "description": "Provide a clear species classification.",
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
        system_hint="Classify iris flowers as setosa, versicolor, or virginica.",
    )

    config = TaskAppConfig(
        app_id="iris",
        name="Iris Classification Task",
        description="Iris environment for evaluating prompt optimisers on multi-class classification.",
        base_task_info=base_info,
        describe_taskset=lambda: describe_taskset(dataset),
        provide_task_instances=lambda seeds: provide_task_instances(dataset, seeds),
        rollout=rollout_executor,
        dataset_registry=registry,
        rubrics=RubricBundle(outcome=OUTCOME_RUBRIC, events=EVENTS_RUBRIC),
        proxy=proxy_config,
        routers=(iris_router,),
        app_state={"iris_dataset": dataset},
        cors_origins=["*"],
    )
    return config


register_task_app(
    entry=TaskAppEntry(
        app_id="iris",
        description="Iris classification task app using scikit-learn/iris.",
        config_factory=build_config,
        aliases=("iris-classification",),
        modal=ModalDeploymentConfig(
            app_name="synth-iris",
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

    parser = argparse.ArgumentParser(description="Run the Iris task app locally")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8115)
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

