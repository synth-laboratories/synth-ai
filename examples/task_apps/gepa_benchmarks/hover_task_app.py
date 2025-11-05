"""HoVer claim verification task app for Synth prompt optimisation benchmarks."""

from __future__ import annotations

import contextlib
import os
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

from .common import call_chat_completion, normalise_answer

REPO_ROOT = Path(__file__).resolve().parents[3]

DATASET_ID = "Dzeniks/hover"
DEFAULT_SPLIT = "test"
AVAILABLE_SPLITS: tuple[str, ...] = ("train", "test")


hover_router = APIRouter()


HOVER_DATASET_SPEC = TaskDatasetSpec(
    id="hover",
    name="HoVer Claim Verification",
    version="1.0.0",
    splits=list(AVAILABLE_SPLITS),
    default_split=DEFAULT_SPLIT,
    description="Claim verification with supporting evidence passages.",
)

LABEL_MAP = {
    0: "SUPPORTED",
    1: "REFUTED",
}


class HoVerDataset:
    """Thin wrapper around the HoVer dataset for sampling."""

    def __init__(self) -> None:
        self._cache: dict[str, Any] = {}

    def _load_split(self, split: str):
        if split not in AVAILABLE_SPLITS:
            raise ValueError(f"Unknown split '{split}'. Available: {AVAILABLE_SPLITS}")
        if split not in self._cache:
            try:
                self._cache[split] = load_dataset(DATASET_ID, split=split)
            except Exception as exc:  # pragma: no cover
                raise RuntimeError(
                    f"Failed to download HoVer split '{split}'. "
                    "Ensure network access to Hugging Face."
                ) from exc
        return self._cache[split]

    def ensure_ready(self, splits: Sequence[str]) -> None:
        for split in splits:
            self._load_split(split)

    def size(self, split: str) -> int:
        dataset = self._load_split(split)
        return len(dataset)

    def sample(self, *, split: str, index: int) -> dict[str, Any]:
        dataset = self._load_split(split)
        size = len(dataset)
        if size == 0:
            raise RuntimeError(f"HoVer split '{split}' is empty")
        idx = int(index) % size
        row = dataset[int(idx)]

        label_idx = int(row.get("label") or 0)
        label_text = LABEL_MAP.get(label_idx, "SUPPORTED")
        evidence = str(row.get("evidence") or "").strip()

        return {
            "index": idx,
            "split": split,
            "claim": str(row.get("claim") or ""),
            "evidence": evidence,
            "label": label_text,
        }


def _parse_label(response_text: str) -> tuple[str, str]:
    if not response_text:
        return "", ""
    lower = response_text.lower()
    label = ""
    rationale = ""
    if "label:" in lower:
        fragment = lower.split("label:", 1)[1]
        label_line = fragment.splitlines()[0]
        label = label_line.strip().upper()
    else:
        # fallback to first word
        label = response_text.strip().split()[0].upper()
    if "rationale:" in lower:
        rationale_fragment = lower.split("rationale:", 1)[1]
        rationale = rationale_fragment.strip()
    return label, rationale


async def rollout_executor(request: RolloutRequest, fastapi_request: Request) -> RolloutResponse:
    dataset: HoVerDataset = fastapi_request.app.state.hover_dataset

    split = str(((request.env.config or {}).get("split")) or DEFAULT_SPLIT)
    seed = request.env.seed or 0

    sample = dataset.sample(split=split, index=seed)
    observation = {
        "claim": sample["claim"],
        "evidence": sample["evidence"],
        "index": sample["index"],
        "split": sample["split"],
    }

    placeholders = {
        "claim": sample["claim"],
        "evidence": sample["evidence"],
    }

    default_messages = [
        {
            "role": "system",
            "pattern": (
                "You verify Wikipedia claims. Decide whether each claim is SUPPORTED or REFUTED "
                "by the evidence provided. Respond with the format:\n"
                "Label: <SUPPORTED|REFUTED>\nRationale: <short explanation>."
            ),
        },
        {
            "role": "user",
            "pattern": "Claim: {claim}\n\nEvidence:\n{evidence}\n\nReturn the label and rationale.",
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

    predicted_label, rationale = _parse_label(response_text)
    expected_label = sample["label"]

    # Normalise label (strip punctuation, match synonyms)
    normalised_prediction = normalise_answer(predicted_label)
    normalised_expected = normalise_answer(expected_label)
    is_correct = normalised_prediction.startswith(normalised_expected[:5])
    reward = 1.0 if is_correct else 0.0

    info_payload = {
        "expected_label": expected_label,
        "predicted_label": predicted_label,
        "rationale": rationale,
        "response_json": response_json,
        "correct": is_correct,
        **error_info,
    }

    with contextlib.suppress(Exception):
        print(
            f"[HOVER_ROLLOUT] run_id={request.run_id} split={sample['split']} "
            f"index={sample['index']} expected={expected_label} predicted={predicted_label} "
            f"reward={reward}",
            flush=True,
        )

    step = RolloutStep(
        obs=observation,
        tool_calls=[],
        reward=reward,
        done=True,
        info=info_payload,
    )

    inference_url = (request.policy.config or {}).get("inference_url")
    trajectory = RolloutTrajectory(
        env_id=f"hover::{sample['split']}::{sample['index']}",
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
        details={"correct": is_correct},
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
                "env": "hover",
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
        ops_executed=2,
        trace=trace_payload,
    )


def build_dataset() -> tuple[TaskDatasetRegistry, HoVerDataset]:
    registry = TaskDatasetRegistry()
    dataset = HoVerDataset()
    dataset.ensure_ready([DEFAULT_SPLIT])
    registry.register(HOVER_DATASET_SPEC, lambda _spec: dataset, cache=True)
    return registry, dataset


def _base_task_info() -> TaskInfo:
    return TaskInfo(
        task={
            "id": "hover",
            "name": "HoVer Claim Verification",
            "version": "1.0.0",
            "action_space": {
                "type": "free_text",
                "description": "Return a label (SUPPORTED/REFUTED) and short rationale.",
            },
        },
        environment="hover",
        dataset={
            **HOVER_DATASET_SPEC.model_dump(),
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
        task_metadata={"format": "Label: ... / Rationale: ..."},
    )


def describe_taskset(dataset: HoVerDataset) -> Mapping[str, Any]:
    return {
        **HOVER_DATASET_SPEC.model_dump(),
        "hf_dataset": DATASET_ID,
        "label_map": LABEL_MAP,
        "sizes": {split: dataset.size(split) for split in AVAILABLE_SPLITS},
    }


def provide_task_instances(dataset: HoVerDataset, seeds: Sequence[int]) -> Iterable[TaskInfo]:
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
                "claim": sample["claim"],
            },
        )


OUTCOME_RUBRIC: Rubric = cast(
    Rubric,
    load_rubric(
        {
            "version": "1",
            "goal_text": "Assign the correct label (SUPPORTED or REFUTED) to each claim.",
            "aggregation": "weighted_sum",
            "criteria": [
                {
                    "id": "label_accuracy",
                    "description": "Correctly classify the claim.",
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
            "goal_text": "Include a concise rationale referencing the evidence.",
            "aggregation": "weighted_sum",
            "criteria": [
                {
                    "id": "rationale_quality",
                    "description": "Provide a short rationale referencing the provided evidence.",
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
        system_hint="Return 'Label: ...' followed by 'Rationale: ...'.",
    )

    config = TaskAppConfig(
        app_id="hover",
        name="HoVer Claim Verification Task",
        description="HoVer dataset task app for verifying claims with supporting passages.",
        base_task_info=base_info,
        describe_taskset=lambda: describe_taskset(dataset),
        provide_task_instances=lambda seeds: provide_task_instances(dataset, seeds),
        rollout=rollout_executor,
        dataset_registry=registry,
        rubrics=RubricBundle(outcome=OUTCOME_RUBRIC, events=EVENTS_RUBRIC),
        proxy=proxy_config,
        routers=(hover_router,),
        app_state={"hover_dataset": dataset},
        cors_origins=["*"],
    )
    return config


register_task_app(
    entry=TaskAppEntry(
        app_id="hover",
        description="HoVer claim verification task app using the Dzeniks/hover dataset.",
        config_factory=build_config,
        aliases=("hover-claims",),
        modal=ModalDeploymentConfig(
            app_name="synth-hover",
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

    parser = argparse.ArgumentParser(description="Run the HoVer task app locally")
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
