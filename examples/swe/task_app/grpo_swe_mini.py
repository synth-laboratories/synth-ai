"""Task app configuration for the mini-SWE agent integration."""

from __future__ import annotations

import json
import logging
import os
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from synth_ai.task.apps import ModalDeploymentConfig, TaskAppEntry, register_task_app
from synth_ai.task.contracts import RolloutMetrics, RolloutRequest, RolloutResponse, TaskInfo
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


try:
    from examples.swe.task_app.hosted.branching import (  # type: ignore
        router as branching_router,
    )
    from examples.swe.task_app.hosted.environment_routes import (  # type: ignore # noqa: E501
        router as environment_router,
    )
    from examples.swe.task_app.hosted.policy_routes import (  # type: ignore
        router as policy_router,
    )
    from examples.swe.task_app.hosted.rollout import (  # type: ignore
        RolloutEnvSpec as LegacyRolloutEnvSpec,
    )
    from examples.swe.task_app.hosted.rollout import (
        RolloutPolicySpec as LegacyRolloutPolicySpec,
    )
    from examples.swe.task_app.hosted.rollout import (
        RolloutRecordConfig as LegacyRolloutRecordConfig,
    )
    from examples.swe.task_app.hosted.rollout import (
        RolloutRequest as LegacyRolloutRequest,
    )
    from examples.swe.task_app.hosted.rollout import (
        RolloutResponse as LegacyRolloutResponse,
    )
    from examples.swe.task_app.hosted.rollout import (
        RolloutSafetyConfig as LegacyRolloutSafetyConfig,
    )
    from examples.swe.task_app.hosted.rollout import (
        execute_rollout as legacy_execute_rollout,
    )
    HAS_HOSTED = True
except Exception:
    try:  # pragma: no cover - optional dependency path
        from examples.warming_up_to_rl.task_app.synth_envs_hosted.branching import (  # type: ignore
            router as branching_router,
        )
        from examples.warming_up_to_rl.task_app.synth_envs_hosted.environment_routes import (  # type: ignore # noqa: E501
            router as environment_router,
        )
        from examples.warming_up_to_rl.task_app.synth_envs_hosted.policy_routes import (  # type: ignore
            router as policy_router,
        )
        from examples.warming_up_to_rl.task_app.synth_envs_hosted.rollout import (  # type: ignore
            RolloutEnvSpec as LegacyRolloutEnvSpec,
        )
        from examples.warming_up_to_rl.task_app.synth_envs_hosted.rollout import (
            RolloutPolicySpec as LegacyRolloutPolicySpec,
        )
        from examples.warming_up_to_rl.task_app.synth_envs_hosted.rollout import (
            RolloutRecordConfig as LegacyRolloutRecordConfig,
        )
        from examples.warming_up_to_rl.task_app.synth_envs_hosted.rollout import (
            RolloutRequest as LegacyRolloutRequest,
        )
        from examples.warming_up_to_rl.task_app.synth_envs_hosted.rollout import (
            RolloutResponse as LegacyRolloutResponse,
        )
        from examples.warming_up_to_rl.task_app.synth_envs_hosted.rollout import (
            RolloutSafetyConfig as LegacyRolloutSafetyConfig,
        )
        from examples.warming_up_to_rl.task_app.synth_envs_hosted.rollout import (
            execute_rollout as legacy_execute_rollout,
        )
        HAS_HOSTED = True
    except Exception as exc:  # pragma: no cover - optional dependency path
        logger.warning(
            "mini-swe task app running without hosted environment support: %s", exc
        )
        HAS_HOSTED = False


_HERE = Path(__file__).resolve()


def _resolve_repo_root() -> Path:
    candidates = [
        Path(os.getenv("SYNTH_AI_REPO_ROOT", "")).expanduser(),
        _HERE.parents[3],
        Path(__file__).resolve().parents[2],
        Path.cwd(),
    ]
    for candidate in candidates:
        if not candidate:
            continue
        try:
            resolved = candidate.resolve()
        except Exception:
            continue
        if (resolved / "pyproject.toml").exists():
            return resolved
        if (resolved / "synth_ai").is_dir():
            return resolved
    return Path(__file__).resolve().parents[3]


REPO_ROOT = _resolve_repo_root()
def _load_hf_dataset(source: str) -> list[dict[str, Any]]:
    try:
        from datasets import load_dataset  # type: ignore
    except Exception as exc:  # pragma: no cover - optional dependency
        raise RuntimeError(
            "datasets package is required for hf:// dataset sources. "
            "pip install datasets or set SWE_MINI_DATASET=file://<path>."
        ) from exc

    # source looks like hf://namespace/dataset:split
    cleaned = source[len("hf://") :]
    if ":" in cleaned:
        dataset_id, split = cleaned.split(":", 1)
    else:
        dataset_id, split = cleaned, "train"
    logger.info("Loading HuggingFace dataset %s split=%s", dataset_id, split)
    ds = load_dataset(dataset_id, split=split)
    return [dict(record) for record in ds]


def _parse_records(text: str) -> list[dict[str, Any]]:
    text = text.strip()
    if not text:
        return []
    if text.startswith("["):
        parsed = json.loads(text)
        if isinstance(parsed, list):
            return [dict(item) for item in parsed]
        raise ValueError("Expected list JSON for dataset file")
    records = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        records.append(json.loads(line))
    return records


def _normalize_instance(raw: dict[str, Any]) -> dict[str, Any]:
    instance_id = raw.get("instance_id") or raw.get("id")
    if not instance_id:
        raise ValueError(f"Dataset entry missing instance_id: {raw}")
    problem_statement = raw.get("problem_statement") or raw.get("prompt") or ""
    instructions = raw.get("instructions") or raw.get("extra") or ""
    metadata = raw.get("metadata") or {}
    if not isinstance(metadata, dict):
        metadata = {"raw_metadata": metadata}
    for key, value in raw.items():
        if key in {"metadata", "instructions", "problem_statement"}:
            continue
        metadata.setdefault(key, value)
    metadata.setdefault("raw_instance", raw)
    metadata.setdefault("instance_id", instance_id)
    if "image_name" not in metadata:
        iid = str(instance_id).replace("__", "_1776_")
        metadata["image_name"] = f"docker.io/swebench/sweb.eval.x86_64.{iid}:latest".lower()
    return {
        "instance_id": instance_id,
        "problem_statement": problem_statement,
        "instructions": instructions,
        "metadata": metadata,
    }


def _load_instances() -> list[dict[str, Any]]:
    default_dataset = "hf://princeton-nlp/SWE-Bench_Verified:test"
    source = os.getenv("SWE_MINI_DATASET", default_dataset).strip()
    if not source:
        source = default_dataset

    if source.startswith("file://"):
        path = Path(source[len("file://") :]).expanduser()
        if not path.exists():
            raise FileNotFoundError(f"SWE_MINI_DATASET file not found: {path}")
        data = path.read_text(encoding="utf-8")
        records = _parse_records(data)
    elif source.startswith("hf://"):
        records = _load_hf_dataset(source)
    else:
        path = Path(source).expanduser()
        if path.exists():
            data = path.read_text(encoding="utf-8")
            records = _parse_records(data)
        else:
            raise ValueError(
                f"Unsupported SWE_MINI_DATASET value '{source}'. "
                "Use file://..., or hf://dataset:split."
            )

    normalised = []
    for record in records:
        try:
            normalised.append(_normalize_instance(record))
        except Exception as exc:
            logger.warning("Skipping invalid dataset entry: %s", exc)
    if not normalised:
        raise RuntimeError("No valid mini-swe dataset entries found.")
    return normalised


@dataclass
class MiniSweDataset:
    instances: list[dict[str, Any]]

    def __post_init__(self) -> None:
        self.by_id = {item["instance_id"]: item for item in self.instances}

    def ids(self) -> list[str]:
        return [item["instance_id"] for item in self.instances]

    def get(self, instance_id: str) -> dict[str, Any]:
        if instance_id not in self.by_id:
            raise KeyError(f"Unknown mini-swe instance_id: {instance_id}")
        return self.by_id[instance_id]

    def sample_by_index(self, index: int) -> dict[str, Any]:
        if not self.instances:
            raise RuntimeError("Mini-swe dataset is empty")
        return self.instances[index % len(self.instances)]


DATASET_SPEC = TaskDatasetSpec(
    id="mini_swe_sample",
    name="mini-SWE Tasks",
    version="0.1.0",
    splits=["train"],
    default_split="train",
    description="Interactive SWE tasks executed via mini-swe-agent environments.",
)


def build_dataset() -> tuple[TaskDatasetRegistry, MiniSweDataset]:
    registry = TaskDatasetRegistry()
    dataset = MiniSweDataset(_load_instances())
    registry.register(DATASET_SPEC, lambda _spec: dataset, cache=True)
    return registry, dataset


def _base_task_info(dataset: MiniSweDataset) -> TaskInfo:
    return TaskInfo(
        task={"id": "swe_mini", "name": "mini-SWE Tasks", "version": "0.1.0"},
        environments=["swe-mini"],
        action_space={
            "type": "tool",
            "tools": ["run_command", "submit_patch"],
            "description": "Issue bash commands or submit the final patch.",
        },
        observation={
            "summary": "Step-wise command output and submission status.",
            "keys": ["task", "history", "last", "submitted"],
        },
        dataset={
            **DATASET_SPEC.model_dump(),
            "instances": dataset.ids()[:50],
        },
        rubric={
            "version": "1",
            "criteria_count": 2,
            "source": "inline",
            "aggregation": "weighted_sum",
        },
        inference={
            "supports_proxy": True,
            "endpoints": {
                "openai": "/proxy/v1/chat/completions",
                "groq": "/proxy/groq/v1/chat/completions",
            },
            "tool": {"name": "run_command", "parallel_tool_calls": False},
        },
        capabilities={
            "supports_rollout": True,
            "supports_env_lifecycle": True,
            "requires_api_key_header": True,
        },
        limits={"max_ops": 2000, "max_time_s": 7200},
    )


OUTCOME_RUBRIC = load_rubric(
    {
        "version": "1",
        "goal_text": "Complete the software engineering task and ensure tests pass.",
        "aggregation": "weighted_sum",
        "criteria": [
            {
                "id": "functional",
                "description": "All acceptance tests and lint checks succeed.",
                "weight": 1.0,
            },
            {
                "id": "quality",
                "description": "Code changes follow project conventions and include required updates.",
                "weight": 1.0,
            },
        ],
    }
)

EVENTS_RUBRIC = load_rubric(
    {
        "version": "1",
        "goal_text": "Encourage deliberate, well-scoped shell interactions.",
        "aggregation": "weighted_sum",
        "criteria": [
            {
                "id": "productive_steps",
                "description": "Commands meaningfully progress the task (editing files, running tests, inspecting context).",
                "weight": 1.0,
            }
        ],
    }
)


def describe_taskset(dataset: MiniSweDataset) -> dict[str, Any]:
    return {
        **DATASET_SPEC.model_dump(),
        "instance_ids": dataset.ids(),
    }


def provide_task_instances(
    dataset: MiniSweDataset, base_info: TaskInfo, seeds: Sequence[int]
) -> Iterable[TaskInfo]:
    infos: list[TaskInfo] = []
    for seed in seeds:
        instance = dataset.sample_by_index(int(seed))
        infos.append(
            TaskInfo(
                task=base_info.task,
                environments=base_info.environments,
                action_space=base_info.action_space,
                observation={**base_info.observation, "instance_id": instance["instance_id"]},
                dataset={**base_info.dataset, "instance_id": instance["instance_id"]},
                rubric=base_info.rubric,
                inference=base_info.inference,
                capabilities=base_info.capabilities,
                limits=base_info.limits,
            )
        )
    return infos


def _ensure_env_has_task(
    dataset: MiniSweDataset, env_spec: LegacyRolloutEnvSpec
) -> LegacyRolloutEnvSpec:
    config = dict(env_spec.config or {})
    if "task" not in config:
        instance_id = config.get("instance_id")
        if not instance_id:
            raise ValueError("mini-swe rollout request requires env.config.instance_id")
        config["task"] = dataset.get(instance_id)
    return env_spec.model_copy(update={"config": config})


def build_config() -> TaskAppConfig:
    registry, dataset = build_dataset()
    base_info = _base_task_info(dataset)

    tracing_enabled = tracing_env_enabled()
    tracing_db_url = resolve_tracing_db_url()
    tracer_factory = build_tracer_factory(SessionTracer, enabled=tracing_enabled, db_url=tracing_db_url)
    sft_output_dir = resolve_sft_output_dir()

    hosted_task_app = None
    if HAS_HOSTED:
        HostedTaskAppCls = None
        try:
            from examples.swe.task_app.hosted.hosted_app import (  # type: ignore
                TaskApp as HostedTaskApp,
            )
            HostedTaskAppCls = HostedTaskApp
        except Exception:
            try:
                from examples.warming_up_to_rl.task_app.synth_envs_hosted.hosted_app import (  # type: ignore
                    TaskApp as HostedTaskApp,
                )
                HostedTaskAppCls = HostedTaskApp
            except Exception as exc:  # pragma: no cover - optional dependency path
                logger.warning("Unable to import HostedTaskApp for swe-mini: %s", exc)
        if HostedTaskAppCls is not None:
            hosted_task_app = HostedTaskAppCls(
                service_base_url=os.getenv("SWE_MINI_SERVICE_BASE_URL"),
                vllm_base_url=os.getenv(
                    "SWE_MINI_VLLM_BASE_URL",
                    "http://localhost:8020/proxy",
                ),
                default_model=os.getenv("SWE_MINI_DEFAULT_MODEL"),
            )

    app_state: dict[str, Any] = {
        "dataset": dataset,
        "allowed_environments": ["swe-mini"],
        "tracing_enabled": tracing_enabled,
    }
    if tracer_factory is not None:
        app_state["session_tracer_factory"] = tracer_factory
    if sft_output_dir:
        app_state["sft_output_dir"] = sft_output_dir
    if hosted_task_app is not None:
        app_state["task_app"] = hosted_task_app

    if tracing_enabled:
        logger.info("[swe-mini:tracing] enabled (db=%s)", tracing_db_url or "default")
    else:
        logger.info("[swe-mini:tracing] disabled")
    if sft_output_dir:
        logger.info("[swe-mini:sft] writing JSONL to %s", sft_output_dir)

    routers: tuple = (environment_router, policy_router, branching_router) if HAS_HOSTED else ()

    async def rollout_executor(request: RolloutRequest, fastapi_request) -> RolloutResponse:
        if not HAS_HOSTED:
            return RolloutResponse(
                run_id=request.run_id,
                trajectories=[],
                branches={},
                metrics=RolloutMetrics(
                    episode_returns=[],
                    mean_return=0.0,
                    num_steps=0,
                    num_episodes=0,
                ),
                aborted=True,
                ops_executed=0,
                trace=None,
            )

        env_spec = _ensure_env_has_task(dataset, request.env)

        legacy_request = LegacyRolloutRequest(
            run_id=request.run_id,
            env=LegacyRolloutEnvSpec(
                env_id=request.env.env_id,
                env_name=env_spec.env_name or "swe-mini",
                config=env_spec.config,
                seed=request.env.seed,
            ),
            policy=LegacyRolloutPolicySpec(
                policy_id=request.policy.policy_id,
                policy_name=request.policy.policy_name or "swe-mini-react",
                config=request.policy.config,
            ),
            ops=request.ops,
            record=LegacyRolloutRecordConfig(**request.record.model_dump()),
            on_done=request.on_done,
            branch=getattr(request, "branch", None),
            safety=LegacyRolloutSafetyConfig(**request.safety.model_dump()),
            training_session_id=request.training_session_id,
            synth_base_url=request.synth_base_url,
        )

        legacy_response: LegacyRolloutResponse = await legacy_execute_rollout(
            legacy_request, fastapi_request
        )
        data = legacy_response.model_dump()
        metrics = data.get("metrics", {}) or {}
        metrics.setdefault("details", {})
        data["metrics"] = metrics
        return RolloutResponse.model_validate(data)

    async def _noop_rollout(request: RolloutRequest, fastapi_request) -> RolloutResponse:
        return RolloutResponse(
            run_id=request.run_id,
            trajectories=[],
            branches={},
            metrics=RolloutMetrics(
                episode_returns=[],
                mean_return=0.0,
                num_steps=0,
                num_episodes=0,
            ),
            aborted=True,
            ops_executed=0,
            trace=None,
        )

    rollout_callable = rollout_executor if HAS_HOSTED else _noop_rollout

    config = TaskAppConfig(
        app_id="swe-mini",
        name="mini-SWE Task App",
        description="Interactive SWE tasks executed via mini-swe-agent environments.",
        base_task_info=base_info,
        describe_taskset=lambda: describe_taskset(dataset),
        provide_task_instances=lambda seeds: provide_task_instances(dataset, base_info, seeds),
        rollout=rollout_callable,
        dataset_registry=registry,
        rubrics=RubricBundle(outcome=OUTCOME_RUBRIC, events=EVENTS_RUBRIC),
        proxy=ProxyConfig(enable_openai=True, enable_groq=True),
        routers=routers,
        app_state=app_state,
        cors_origins=["*"],
    )
    return config


register_task_app(
    entry=TaskAppEntry(
        app_id="swe-mini",
        description="mini-swe-agent task app with rollout + proxy endpoints",
        config_factory=build_config,
        aliases=("mini-swe", "swe-mini-task"),
        env_files=(str(REPO_ROOT / "backend" / ".env.dev"),),
        modal=ModalDeploymentConfig(
            app_name="swe-mini-task-app",
            python_version="3.11",
            pip_packages=(
                "fastapi>=0.109.0",
                "uvicorn>=0.23.0",
                "pydantic>=2.7.0",
                "numpy>=1.24.0",
                "aiohttp>=3.8.0",
                "httpx>=0.24.0",
                "python-dotenv>=1.0.1",
                "sqlalchemy>=2.0.42",
                "aiosqlite>=0.21.0",
                "greenlet>=3.0.3",
                "modal>=0.63.0",
                "tenacity>=8.2.3",
                "swebench[modal]>=1.1.0",
                "swe-rex[modal]>=1.4.0",
                "mini-swe-agent>=1.14.2",
                "datasets>=2.18.0",
                "litellm>=1.75.5",
                "rich>=13.7.0",
                "jinja2>=3.1.3",
            ),
            extra_local_dirs=(
                (str(REPO_ROOT / "synth_ai"), "/opt/synth_ai_repo/synth_ai"),
                (
                    str(REPO_ROOT / "examples" / "swe" / "task_app" / "hosted"),
                    "/opt/synth_ai_repo/examples/swe/task_app/hosted",
                ),
                (
                    str(_HERE.parent),
                    "/opt/synth_ai_repo/examples/swe/task_app",
                ),
            ),
            secret_names=("swe-mini-environment", "groq-api-key", "openai-api-key"),
            memory=32768,
            cpu=6.0,
            max_containers=10,
        ),
    )
)
