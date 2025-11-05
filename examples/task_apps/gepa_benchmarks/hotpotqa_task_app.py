"""HotpotQA multi-hop QA task app for Synth prompt learning benchmarks."""

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

HOTPOTQA_DATASET = "hotpot_qa"
HOTPOTQA_CONFIG = "distractor"
DEFAULT_SPLIT = "validation"
AVAILABLE_SPLITS: tuple[str, ...] = ("train", "validation")


hotpotqa_router = APIRouter()


HOTPOTQA_DATASET_SPEC = TaskDatasetSpec(
    id="hotpotqa",
    name="HotpotQA Multi-Hop Question Answering",
    version="1.0.0",
    splits=list(AVAILABLE_SPLITS),
    default_split=DEFAULT_SPLIT,
    description="HotpotQA question answering with multi-hop supporting facts.",
)


class HotpotQADataset:
    """Lazy loader and sampler for the HotpotQA dataset."""

    def __init__(self) -> None:
        self._splits: dict[str, Any] = {}

    def _load_split(self, split: str):
        if split not in AVAILABLE_SPLITS:
            raise ValueError(f"Unknown split '{split}'. Available: {AVAILABLE_SPLITS}")
        if split not in self._splits:
            try:
                self._splits[split] = load_dataset(HOTPOTQA_DATASET, HOTPOTQA_CONFIG, split=split)
            except Exception as exc:  # pragma: no cover - network/dataset errors
                raise RuntimeError(
                    f"Failed to download HotpotQA split '{split}'. "
                    f"Ensure network access to Hugging Face datasets."
                ) from exc
        return self._splits[split]

    def ensure_ready(self, required_splits: Sequence[str]) -> None:
        for split in required_splits:
            self._load_split(split)

    def size(self, split: str) -> int:
        dataset = self._load_split(split)
        return len(dataset)

    @staticmethod
    def _format_context(context: Any) -> tuple[str, list[str]]:
        """Convert HotpotQA context paragraphs into display text and titles."""

        lines: list[str] = []
        titles: list[str] = []

        if isinstance(context, Mapping):
            title_list = context.get("title") or []
            sentences_list = context.get("sentences") or []
            for title, sentences in zip(title_list, sentences_list):
                title_str = str(title)
                titles.append(title_str)
                lines.append(f"### {title_str}")
                for sentence in sentences or []:
                    lines.append(str(sentence))
                lines.append("")
        elif isinstance(context, Sequence):
            for entry in context:
                if not isinstance(entry, Sequence) or len(entry) != 2:
                    continue
                title_str = str(entry[0])
                sentences = entry[1] if isinstance(entry[1], Sequence) else []
                titles.append(title_str)
                lines.append(f"### {title_str}")
                for sentence in sentences:
                    lines.append(str(sentence))
                lines.append("")

        return "\n".join(lines).strip(), titles

    def sample(self, *, split: str, index: int) -> dict[str, Any]:
        dataset = self._load_split(split)
        size = len(dataset)
        if size == 0:
            raise RuntimeError(f"HotpotQA split '{split}' is empty")
        idx = int(index) % size
        row = dataset[int(idx)]

        context_text, context_titles = self._format_context(row.get("context") or [])
        supporting = row.get("supporting_facts") or []
        supporting_titles: list[str] = []
        if isinstance(supporting, Mapping):
            supporting_titles = [str(title) for title in (supporting.get("title") or [])]
        elif isinstance(supporting, Sequence):
            supporting_titles = [
                str(entry[0]) for entry in supporting if isinstance(entry, Sequence) and entry
            ]
        supporting_titles = sorted(set(supporting_titles))

        return {
            "index": idx,
            "split": split,
            "question": str(row.get("question") or ""),
            "answer": str(row.get("answer") or ""),
            "context_text": context_text,
            "context_titles": context_titles,
            "supporting_titles": supporting_titles,
        }


def _parse_answer(response_text: str) -> tuple[str, str]:
    """Parse response text into (answer, support) segments."""

    answer = ""
    support = ""
    if not response_text:
        return answer, support

    lower = response_text.lower()
    if "answer:" in lower:
        parts = lower.split("answer:", 1)[1]
        answer_section = parts.split("support:", 1)[0] if "support:" in parts else parts
        answer = answer_section.strip()
    else:
        answer = response_text.strip()

    if "support:" in lower:
        support_section = lower.split("support:", 1)[1]
        support = support_section.strip()

    # Use original casing if possible.
    if answer:
        match_index = response_text.lower().find(answer)
        if match_index >= 0:
            answer = response_text[match_index : match_index + len(answer)].strip()
    if support:
        match_index = response_text.lower().find(support)
        if match_index >= 0:
            support = response_text[match_index : match_index + len(support)].strip()
    return answer.strip(), support.strip()


async def rollout_executor(request: RolloutRequest, fastapi_request: Request) -> RolloutResponse:
    dataset: HotpotQADataset = fastapi_request.app.state.hotpotqa_dataset

    split = str(((request.env.config or {}).get("split")) or DEFAULT_SPLIT)
    seed = request.env.seed or 0

    sample = dataset.sample(split=split, index=seed)
    observation = {
        "question": sample["question"],
        "context": sample["context_text"],
        "supporting_titles": sample["supporting_titles"],
        "index": sample["index"],
        "split": sample["split"],
    }

    placeholders = {
        "question": sample["question"],
        "context": sample["context_text"],
    }

    default_messages = [
        {
            "role": "system",
            "pattern": (
                "You are a research assistant that answers multi-hop questions. "
                "Read the passages carefully and respond in the format:\n"
                "Answer: <short answer>\nSupport: <brief justification citing passages>."
            ),
        },
        {
            "role": "user",
            "pattern": "Question: {question}\n\nPassages:\n{context}\n\nProvide the final answer.",
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

    answer_text, support_text = _parse_answer(response_text)

    expected_answer = sample["answer"]
    answer_correct = int(normalise_answer(answer_text) == normalise_answer(expected_answer))

    support_titles = sample["supporting_titles"]
    support_hits = 0
    if support_titles and support_text:
        lower_support = support_text.lower()
        support_hits = sum(1 for title in support_titles if title.lower() in lower_support)
    support_coverage = (support_hits / len(support_titles)) if support_titles else 0.0

    reward = 0.7 * answer_correct + 0.3 * support_coverage

    info_payload = {
        "expected_answer": expected_answer,
        "predicted_answer": answer_text,
        "support_text": support_text,
        "answer_em": answer_correct,
        "support_coverage": support_coverage,
        "response_json": response_json,
        **error_info,
    }

    with contextlib.suppress(Exception):
        print(
            f"[HOTPOTQA_ROLLOUT] run_id={request.run_id} split={sample['split']} "
            f"index={sample['index']} answer_em={answer_correct} "
            f"support={support_hits}/{len(support_titles) or 1} reward={reward:.3f}",
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
        env_id=f"hotpotqa::{sample['split']}::{sample['index']}",
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
            "support_coverage": support_coverage,
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
                "env": "hotpotqa",
                "split": sample["split"],
                "index": sample["index"],
                "answer_em": answer_correct,
                "support_coverage": support_coverage,
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


def build_dataset() -> tuple[TaskDatasetRegistry, HotpotQADataset]:
    registry = TaskDatasetRegistry()
    dataset = HotpotQADataset()
    dataset.ensure_ready([DEFAULT_SPLIT])
    registry.register(HOTPOTQA_DATASET_SPEC, lambda _spec: dataset, cache=True)
    return registry, dataset


def _base_task_info() -> TaskInfo:
    return TaskInfo(
        task={
            "id": "hotpotqa",
            "name": "HotpotQA Multi-Hop QA",
            "version": "1.0.0",
            "action_space": {
                "type": "free_text",
                "description": "Respond with an answer and supporting justification.",
            },
        },
        environment="hotpotqa",
        dataset={
            **HOTPOTQA_DATASET_SPEC.model_dump(),
            "hf_dataset": HOTPOTQA_DATASET,
            "hf_config": HOTPOTQA_CONFIG,
        },
        rubric={
            "version": "1",
            "criteria_count": 2,
            "source": "inline",
        },
        inference={
            "supports_proxy": True,
            "tool": None,
        },
        limits={"max_turns": 1},
        task_metadata={
            "format": "Answer: ... / Support: ...",
            "support_titles": True,
        },
    )


def describe_taskset(dataset: HotpotQADataset) -> Mapping[str, Any]:
    return {
        **HOTPOTQA_DATASET_SPEC.model_dump(),
        "hf_dataset": HOTPOTQA_DATASET,
        "hf_config": HOTPOTQA_CONFIG,
        "sizes": {split: dataset.size(split) for split in AVAILABLE_SPLITS},
    }


def provide_task_instances(dataset: HotpotQADataset, seeds: Sequence[int]) -> Iterable[TaskInfo]:
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
                "question": sample["question"],
            },
        )


OUTCOME_RUBRIC: Rubric = cast(
    Rubric,
    load_rubric(
        {
            "version": "1",
            "goal_text": "Answer HotpotQA questions accurately with supporting justification.",
            "aggregation": "weighted_sum",
            "criteria": [
                {
                    "id": "answer_accuracy",
                    "description": "Final answer matches the gold answer.",
                    "weight": 0.7,
                },
                {
                    "id": "supporting_evidence",
                    "description": "Support references the correct passages.",
                    "weight": 0.3,
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
            "goal_text": "Encourage concise responses with the requested format.",
            "aggregation": "weighted_sum",
            "criteria": [
                {
                    "id": "format_compliance",
                    "description": "Respond using 'Answer:' and 'Support:' sections.",
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
        system_hint="Provide an answer followed by supporting justification.",
    )

    config = TaskAppConfig(
        app_id="hotpotqa",
        name="HotpotQA Multi-Hop QA Task",
        description="HotpotQA environment for evaluating prompt optimisers.",
        base_task_info=base_info,
        describe_taskset=lambda: describe_taskset(dataset),
        provide_task_instances=lambda seeds: provide_task_instances(dataset, seeds),
        rollout=rollout_executor,
        dataset_registry=registry,
        rubrics=RubricBundle(outcome=OUTCOME_RUBRIC, events=EVENTS_RUBRIC),
        proxy=proxy_config,
        routers=(hotpotqa_router,),
        app_state={"hotpotqa_dataset": dataset},
        cors_origins=["*"],
    )
    return config


register_task_app(
    entry=TaskAppEntry(
        app_id="hotpotqa",
        description="HotpotQA multi-hop QA task app using the distractor split.",
        config_factory=build_config,
        aliases=("hotpotqa-multihop",),
        modal=ModalDeploymentConfig(
            app_name="synth-hotpotqa",
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

    parser = argparse.ArgumentParser(description="Run the HotpotQA task app locally")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8110)
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
