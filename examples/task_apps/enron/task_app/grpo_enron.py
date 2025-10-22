"""Task App configuration for the GRPO Enron email QA example."""

from __future__ import annotations

import contextlib
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence
from uuid import UUID, uuid4

from datasets import load_dataset

from synth_ai.environments.examples.enron.environment import EnronEnvironment
from synth_ai.environments.examples.enron.taskset import (
    EnronTaskInstance,
    EnronTaskInstanceMetadata,
)
from synth_ai.environments.tasks.core import (
    Impetus,
    Intent,
    SplitInfo,
    TaskInstanceSet,
)
from synth_ai.task.apps import ModalDeploymentConfig, TaskAppEntry, register_task_app
from synth_ai.task.contracts import (
    RolloutMetrics,
    RolloutRequest,
    RolloutResponse,
    RolloutTrajectory,
    TaskInfo,
)
from synth_ai.task.datasets import TaskDatasetRegistry, TaskDatasetSpec
from synth_ai.task.rubrics import load_rubric
from synth_ai.task.server import ProxyConfig, RubricBundle, TaskAppConfig
from synth_ai.task.tracing_utils import (
    build_tracer_factory,
    resolve_sft_output_dir,
    resolve_tracing_db_url,
    tracing_env_enabled,
)
from synth_ai.tracing_v3.session_tracer import SessionTracer

logger = logging.getLogger(__name__)

_HERE = Path(__file__).resolve()
REPO_ROOT = _HERE.parents[4]

DATASET_SPEC = TaskDatasetSpec(
    id="enron_email_qa",
    name="Enron Email QA",
    version="1.0.0",
    splits=["train", "test"],
    default_split="train",
    description="Question answering over a sample of Enron emails.",
)

HF_DATASET_ID = "corbt/enron_emails_sample_questions"
HF_CACHE_DIR = os.path.join(
    os.getenv("ENRON_DATASET_CACHE_DIR", str(REPO_ROOT / ".cache" / "hf-datasets"))
)

TOOLS = ["search_emails", "read_email", "answer_question", "terminate"]


def _load_taskset_blocking() -> TaskInstanceSet:
    """Build the Enron taskset synchronously."""

    cache_dir = Path(HF_CACHE_DIR)
    cache_dir.mkdir(parents=True, exist_ok=True)

    ds_train = load_dataset(HF_DATASET_ID, split="train", cache_dir=cache_dir)
    ds_test = load_dataset(HF_DATASET_ID, split="test", cache_dir=cache_dir)

    def _metadata_from_row(row: dict[str, Any], split: str) -> EnronTaskInstance:
        question = str(row.get("question") or "").strip()
        answer = str(row.get("answer") or "").strip()
        message_ids = row.get("message_ids") or []
        if not isinstance(message_ids, list):
            message_ids = list(message_ids)
        impetus = Impetus(instructions=question)
        intent = Intent(
            rubric={"goal": "Answer the question using the Enron emails."},
            gold_trajectories=None,
            gold_state_diff={"answer": answer},
        )
        metadata = EnronTaskInstanceMetadata(
            split=split,
            email_count=len(message_ids),
            message_ids=message_ids,
        )
        return EnronTaskInstance(
            id=uuid4(),
            impetus=impetus,
            intent=intent,
            metadata=metadata,
            is_reproducible=True,
            initial_engine_snapshot=row,
        )

    train_instances = [_metadata_from_row(r, "train") for r in ds_train]
    test_instances = [_metadata_from_row(r, "test") for r in ds_test]

    split_info = SplitInfo(
        val_instance_ids=set(),
        test_instance_ids={inst.id for inst in test_instances},
        _is_split_defined=True,
    )

    return TaskInstanceSet(
        name="Enron-QA",
        description="QA over Enron email dataset sample.",
        instances=train_instances + test_instances,
        split_info=split_info,
    )


def _safe_uuid(value: Any) -> UUID:
    if isinstance(value, UUID):
        return value
    try:
        return UUID(str(value))
    except Exception:
        return UUID(int=0)


@dataclass
class EnronDataset:
    spec: TaskDatasetSpec

    def __post_init__(self) -> None:
        self._taskset = _load_taskset_blocking()
        self.instances: list[EnronTaskInstance] = list(self._taskset.instances)
        self.instance_ids = [str(_safe_uuid(inst.id)) for inst in self.instances]
        self.default_seed = 0
        self.seed_min = 0
        self.seed_max = max(len(self.instances) - 1, 0)

    def describe(self) -> dict[str, Any]:
        return {
            **self.spec.model_dump(),
            "instance_count": len(self.instances),
            "instance_ids": self.instance_ids[:50],
        }

    def instance_by_seed(self, seed: int | None) -> EnronTaskInstance:
        if not self.instances:
            raise ValueError("Enron dataset is empty.")
        if seed is None:
            index = 0
        else:
            index = int(seed) % len(self.instances)
        return self.instances[index]


def build_dataset() -> tuple[TaskDatasetRegistry, EnronDataset]:
    registry = TaskDatasetRegistry()
    dataset = EnronDataset(DATASET_SPEC)
    registry.register(DATASET_SPEC, lambda _spec: dataset, cache=True)
    return registry, dataset


def _base_task_info(dataset: EnronDataset) -> TaskInfo:
    return TaskInfo(
        task={"id": "enron_email_qa", "name": "Enron Email QA", "version": "1.0.0"},
        environments=["enron"],
        action_space={
            "type": "tool_calls",
            "tools": TOOLS,
            "description": "Tool-assisted QA workflow over an email corpus.",
        },
        observation={
            "summary": "Text observations describing the question, tool status, and last reward.",
            "format": "text",
        },
        dataset={**dataset.describe(), "default_seed": dataset.default_seed},
        rubric={
            "version": "1",
            "criteria_count": 1,
            "source": "inline",
            "aggregation": "weighted_sum",
        },
        inference={
            "supports_proxy": False,
            "endpoints": {},
            "tool": {"name": "enron_tools", "parallel_tool_calls": False},
        },
        capabilities={
            "supports_rollout": True,
            "supports_env_lifecycle": True,
            "requires_api_key_header": True,
        },
        limits={"max_ops": 0, "max_time_s": 900},
    )


OUTCOME_RUBRIC = load_rubric(
    {
        "version": "1",
        "goal_text": "Provide the correct answer to the question using the Enron emails.",
        "aggregation": "weighted_sum",
        "criteria": [
            {
                "id": "accuracy",
                "description": "Final answer matches the gold answer.",
                "weight": 1.0,
            }
        ],
    }
)

EVENTS_RUBRIC = load_rubric(
    {
        "version": "1",
        "goal_text": "Encourage efficient use of tools when exploring the corpus.",
        "aggregation": "weighted_sum",
        "criteria": [
            {
                "id": "tool_use",
                "description": "Use search, read, and answer tools deliberately.",
                "weight": 1.0,
            }
        ],
    }
)


def describe_taskset(dataset: EnronDataset) -> dict[str, Any]:
    return dataset.describe()


def provide_task_instances(
    dataset: EnronDataset, base_info: TaskInfo, seeds: Sequence[int]
) -> Iterable[TaskInfo]:
    infos: list[TaskInfo] = []
    for seed in seeds:
        instance = dataset.instance_by_seed(seed)
        metadata = instance.metadata
        meta_dict = {
            "split": getattr(metadata, "split", None),
            "email_count": getattr(metadata, "email_count", None),
            "message_ids": getattr(metadata, "message_ids", None),
        }
        infos.append(
            TaskInfo(
                task=base_info.task,
                environments=base_info.environments,
                action_space=base_info.action_space,
                observation={**base_info.observation, "question": instance.impetus.instructions},
                dataset={
                    **base_info.dataset,
                    "instance_id": str(_safe_uuid(instance.id)),
                    "metadata": meta_dict,
                },
                rubric=base_info.rubric,
                inference=base_info.inference,
                capabilities=base_info.capabilities,
                limits=base_info.limits,
            )
        )
    return infos


def _ensure_dataset_from_state(fastapi_request, fallback: EnronDataset) -> EnronDataset:
    if fastapi_request is None:
        return fallback
    dataset = getattr(getattr(fastapi_request, "app", None), "state", None)
    candidate = getattr(dataset, "dataset", None)
    return candidate or fallback


def _normalise_observation(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if hasattr(value, "observation"):
        obs = getattr(value, "observation")
        if isinstance(obs, dict):
            return obs
        return {"text": str(obs)}
    return {"text": str(value)}


async def rollout_executor(
    request: RolloutRequest, fastapi_request
) -> RolloutResponse:
    dataset = _ensure_dataset_from_state(fastapi_request, RUNTIME_DATASET)
    env_seed = getattr(request.env, "seed", None) if request and request.env else None
    instance = dataset.instance_by_seed(env_seed)
    env = EnronEnvironment(task_instance=instance)
    initial_observation: Any
    try:
        initial_observation = await env.initialize()
    finally:
        with contextlib.suppress(Exception):
            await env.terminate()

    obs_dict = _normalise_observation(initial_observation)

    trajectory = RolloutTrajectory(
        env_id=request.env.env_id or "enron",
        policy_id=request.policy.policy_id or request.policy.policy_name or "noop-policy",
        steps=[],
        final={"observation": obs_dict},
        length=0,
        decision_samples=None,
    )

    metrics = RolloutMetrics(
        episode_returns=[0.0],
        mean_return=0.0,
        num_steps=0,
        num_episodes=1,
        outcome_score=None,
        events_score=None,
        details={"note": "Rollout captures only the initial observation."},
    )

    return RolloutResponse(
        run_id=request.run_id,
        trajectories=[trajectory],
        branches={},
        metrics=metrics,
        aborted=False,
        ops_executed=0,
        trace=None,
    )


RUNTIME_DATASET: EnronDataset
registry, RUNTIME_DATASET = build_dataset()
BASE_INFO = _base_task_info(RUNTIME_DATASET)


def build_config() -> TaskAppConfig:
    tracing_enabled = tracing_env_enabled()
    tracing_db_url = resolve_tracing_db_url()
    tracer_factory = build_tracer_factory(
        SessionTracer, enabled=tracing_enabled, db_url=tracing_db_url
    )
    sft_output_dir = resolve_sft_output_dir()

    app_state: dict[str, Any] = {
        "dataset": RUNTIME_DATASET,
        "allowed_environments": ["enron"],
        "tracing_enabled": tracing_enabled,
    }
    if tracer_factory is not None:
        app_state["session_tracer_factory"] = tracer_factory
    if sft_output_dir:
        app_state["sft_output_dir"] = sft_output_dir

    if tracing_enabled:
        logger.info("[enron:tracing] enabled (db=%s)", tracing_db_url or "default")
    else:
        logger.info("[enron:tracing] disabled")
    if sft_output_dir:
        logger.info("[enron:sft] writing JSONL to %s", sft_output_dir)

    config = TaskAppConfig(
        app_id="grpo-enron",
        name="GRPO Enron Email QA Task App",
        description="Tool-assisted QA environment over Enron emails with GRPO-compatible endpoints.",
        base_task_info=BASE_INFO,
        describe_taskset=lambda: describe_taskset(RUNTIME_DATASET),
        provide_task_instances=lambda seeds: provide_task_instances(RUNTIME_DATASET, BASE_INFO, seeds),
        rollout=rollout_executor,
        dataset_registry=registry,
        rubrics=RubricBundle(outcome=OUTCOME_RUBRIC, events=EVENTS_RUBRIC),
        proxy=ProxyConfig(enable_openai=False, enable_groq=False),
        routers=(),
        app_state=app_state,
        cors_origins=["*"],
    )
    return config


register_task_app(
    entry=TaskAppEntry(
        app_id="grpo-enron",
        description="Enron email QA task app with rollout metadata endpoints.",
        config_factory=build_config,
        aliases=("enron", "enron-task"),
        env_files=(str(REPO_ROOT / "backend" / ".env.dev"),),
        modal=ModalDeploymentConfig(
            app_name="grpo-enron-task-app",
            python_version="3.11",
            pip_packages=(
                "fastapi>=0.100.0",
                "uvicorn>=0.23.0",
                "pydantic>=2.0.0",
                "httpx>=0.24.0",
                "python-dotenv>=1.0.1",
                "datasets>=2.10.0",
            ),
            extra_local_dirs=(
                (str(REPO_ROOT), "/opt/synth_ai_repo"),
                (str(REPO_ROOT / "synth_ai"), "/opt/synth_ai_repo/synth_ai"),
                (str(_HERE.parent), "/opt/synth_ai_repo/examples/task_apps/enron/task_app"),
            ),
            secret_names=("groq-api-key", "openai-api-key"),
            memory=8192,
            cpu=2.0,
            max_containers=4,
        ),
    )
)
