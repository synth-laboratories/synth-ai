"""Task App configuration for the GRPO Verilog spec-to-RTL example."""

from __future__ import annotations

import asyncio
import contextlib
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

from synth_ai.environments.examples.verilog.environment import VerilogEnvironment
from synth_ai.environments.examples.verilog.taskset import (
    VerilogTaskInstance,
    VerilogTaskInstanceMetadata,
    create_verilog_taskset,
)
from synth_ai.environments.tasks.core import TaskInstanceSet
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
    id="verilog_eval_v2",
    name="VerilogEval Spec-to-RTL",
    version="1.0.0",
    splits=["train", "val", "test"],
    default_split="val",
    description="Spec-to-RTL problems sourced from the VerilogEval v2 benchmark.",
)

MAX_INSTANCES = int(os.getenv("VERILOG_MAX_INSTANCES", "10"))
TOOLS = ["write_file", "compile", "simulate", "submit"]


def _load_taskset_blocking(max_instances: int) -> TaskInstanceSet:
    try:
        return asyncio.run(create_verilog_taskset(max_instances=max_instances))
    except RuntimeError:
        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(create_verilog_taskset(max_instances=max_instances))
        finally:
            loop.close()


@dataclass
class VerilogDataset:
    spec: TaskDatasetSpec
    max_instances: int

    def __post_init__(self) -> None:
        self._taskset = _load_taskset_blocking(self.max_instances)
        self.instances: list[VerilogTaskInstance] = list(self._taskset.instances)
        self.instance_ids = [str(inst.id) for inst in self.instances]
        self.default_seed = 0
        self.seed_min = 0
        self.seed_max = max(len(self.instances) - 1, 0)

    def describe(self) -> dict[str, Any]:
        return {
            **self.spec.model_dump(),
            "instance_count": len(self.instances),
            "instance_ids": self.instance_ids[:50],
        }

    def instance_by_seed(self, seed: int | None) -> VerilogTaskInstance:
        if not self.instances:
            raise ValueError("Verilog dataset is empty.")
        if seed is None:
            index = 0
        else:
            index = int(seed) % len(self.instances)
        return self.instances[index]


def build_dataset() -> tuple[TaskDatasetRegistry, VerilogDataset]:
    registry = TaskDatasetRegistry()
    dataset = VerilogDataset(DATASET_SPEC, MAX_INSTANCES)
    registry.register(DATASET_SPEC, lambda _spec: dataset, cache=True)
    return registry, dataset


def _base_task_info(dataset: VerilogDataset) -> TaskInfo:
    return TaskInfo(
        task={"id": "verilog_eval_v2", "name": "VerilogEval Spec-to-RTL", "version": "1.0.0"},
        environments=["verilog"],
        action_space={
            "type": "tool_calls",
            "tools": TOOLS,
            "description": "Filesystem editing, compilation, simulation, and submission tools.",
        },
        observation={
            "summary": "Dictionary observations describing files, compilation status, simulation results, and rewards.",
            "format": "dict",
            "keys": ["files", "compile_status", "simulate_status", "reward_last"],
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
            "tool": {"name": "verilog_tools", "parallel_tool_calls": False},
        },
        capabilities={
            "supports_rollout": True,
            "supports_env_lifecycle": True,
            "requires_api_key_header": True,
        },
        limits={"max_ops": 0, "max_time_s": 3600},
    )


OUTCOME_RUBRIC = load_rubric(
    {
        "version": "1",
        "goal_text": "Produce a Verilog implementation that passes the provided testbench.",
        "aggregation": "weighted_sum",
        "criteria": [
            {
                "id": "tests_pass",
                "description": "Submission passes all compile and simulation checks.",
                "weight": 1.0,
            }
        ],
    }
)

EVENTS_RUBRIC = load_rubric(
    {
        "version": "1",
        "goal_text": "Encourage deliberate hardware design iterations.",
        "aggregation": "weighted_sum",
        "criteria": [
            {
                "id": "efficient_iterations",
                "description": "Use write/compile/simulate tools strategically before submitting.",
                "weight": 1.0,
            }
        ],
    }
)


def describe_taskset(dataset: VerilogDataset) -> dict[str, Any]:
    return dataset.describe()


def provide_task_instances(
    dataset: VerilogDataset, base_info: TaskInfo, seeds: Sequence[int]
) -> Iterable[TaskInfo]:
    infos: list[TaskInfo] = []
    for seed in seeds:
        instance = dataset.instance_by_seed(seed)
        metadata: VerilogTaskInstanceMetadata = instance.metadata  # type: ignore[assignment]
        meta_dict = {
            "problem_name": getattr(metadata, "problem_name", None),
            "difficulty": getattr(metadata, "difficulty", None),
            "description": getattr(metadata, "description", None),
            "files_provided": getattr(metadata, "files_provided", None),
        }
        infos.append(
            TaskInfo(
                task=base_info.task,
                environments=base_info.environments,
                action_space=base_info.action_space,
                observation={
                    **base_info.observation,
                    "problem_name": meta_dict["problem_name"],
                    "difficulty": meta_dict["difficulty"],
                },
                dataset={
                    **base_info.dataset,
                    "instance_id": str(instance.id),
                    "metadata": meta_dict,
                },
                rubric=base_info.rubric,
                inference=base_info.inference,
                capabilities=base_info.capabilities,
                limits=base_info.limits,
            )
        )
    return infos


def _ensure_dataset_from_state(fastapi_request, fallback: VerilogDataset) -> VerilogDataset:
    if fastapi_request is None:
        return fallback
    state = getattr(getattr(fastapi_request, "app", None), "state", None)
    candidate = getattr(state, "dataset", None)
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
    env = VerilogEnvironment(task_instance=instance)
    initial_observation: Any
    try:
        initial_observation = await env.initialize()
    finally:
        with contextlib.suppress(Exception):
            await env.terminate()

    obs_dict = _normalise_observation(initial_observation)

    trajectory = RolloutTrajectory(
        env_id=request.env.env_id or "verilog",
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


RUNTIME_DATASET: VerilogDataset
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
        "allowed_environments": ["verilog"],
        "tracing_enabled": tracing_enabled,
    }
    if tracer_factory is not None:
        app_state["session_tracer_factory"] = tracer_factory
    if sft_output_dir:
        app_state["sft_output_dir"] = sft_output_dir

    if tracing_enabled:
        logger.info("[verilog:tracing] enabled (db=%s)", tracing_db_url or "default")
    else:
        logger.info("[verilog:tracing] disabled")
    if sft_output_dir:
        logger.info("[verilog:sft] writing JSONL to %s", sft_output_dir)

    config = TaskAppConfig(
        app_id="grpo-verilog",
        name="GRPO Verilog Task App",
        description="Spec-to-RTL Verilog environment with GRPO-compatible metadata endpoints.",
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
        app_id="grpo-verilog",
        description="Verilog spec-to-RTL task app with rollout metadata endpoints.",
        config_factory=build_config,
        aliases=("verilog", "verilog-task"),
        env_files=(str(REPO_ROOT / "backend" / ".env.dev"),),
        modal=ModalDeploymentConfig(
            app_name="grpo-verilog-task-app",
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
                (str(_HERE.parent), "/opt/synth_ai_repo/examples/task_apps/verilog/task_app"),
            ),
            secret_names=("groq-api-key", "openai-api-key"),
            memory=8192,
            cpu=2.0,
            max_containers=4,
        ),
    )
)
