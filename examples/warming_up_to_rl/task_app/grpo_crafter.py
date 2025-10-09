from __future__ import annotations

"""Task App configuration for the GRPO Crafter example."""

import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

from synth_ai.task.contracts import RolloutRequest, RolloutResponse, TaskInfo, RolloutMetrics
from synth_ai.task.datasets import TaskDatasetRegistry, TaskDatasetSpec
from synth_ai.task.rubrics import load_rubric
from synth_ai.task.server import ProxyConfig, RubricBundle, TaskAppConfig
from synth_ai.task.json import to_jsonable  # noqa: F401  (imported for side-effect compatibility)
from synth_ai.task.apps import ModalDeploymentConfig, TaskAppEntry, register_task_app
from synth_ai.task.tracing_utils import (
    build_tracer_factory,
    resolve_sft_output_dir,
    resolve_tracing_db_url,
    tracing_env_enabled,
)

from synth_ai.tracing_v3.session_tracer import SessionTracer


REPO_ROOT = Path(__file__).resolve().parents[3]
TASK_APP_ROOT = REPO_ROOT / "examples" / "warming_up_to_rl" / "task_app"
SYNTH_ENVS_HOSTED_ROOT = TASK_APP_ROOT / "synth_envs_hosted"

for path in [REPO_ROOT, TASK_APP_ROOT, SYNTH_ENVS_HOSTED_ROOT]:
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

HAS_HOSTED = True
try:
    import crafter  # type: ignore
    import crafter.constants as C  # type: ignore
    from synth_ai.environments.examples.crafter_classic.taskset import TRAIT_BOUNDS
    from synth_envs_hosted.branching import router as branching_router  # type: ignore
    from synth_envs_hosted.environment_routes import router as environment_router  # type: ignore
    from synth_envs_hosted.hosted_app import TaskApp as HostedTaskApp  # type: ignore
    from synth_envs_hosted.policy_routes import router as policy_router  # type: ignore
    from synth_envs_hosted.rollout import (  # type: ignore
        RolloutEnvSpec as LegacyRolloutEnvSpec,
        RolloutPolicySpec as LegacyRolloutPolicySpec,
        RolloutRecordConfig as LegacyRolloutRecordConfig,
        RolloutRequest as LegacyRolloutRequest,
        RolloutResponse as LegacyRolloutResponse,
        RolloutSafetyConfig as LegacyRolloutSafetyConfig,
        execute_rollout as legacy_execute_rollout,
    )
except Exception as exc:  # pragma: no cover - import-time validation
    # Provide a more actionable error with the missing module and fix hints
    missing_mod = None
    if isinstance(exc, ModuleNotFoundError):
        missing_mod = (
            getattr(exc, "name", None) or str(exc).split("'")[1] if "'" in str(exc) else None
        )
    fix_hint = None
    if missing_mod:
        mapping = {
            "dotenv": "python-dotenv",
            "crafter": "crafter",
            "httpx": "httpx",
            "aiohttp": "aiohttp",
            "fastapi": "fastapi",
            "uvicorn": "uvicorn",
            "sqlalchemy": "sqlalchemy",
            "aiosqlite": "aiosqlite",
            "greenlet": "greenlet",
        }
        pkg = mapping.get(missing_mod, missing_mod)
        fix_hint = (
            f"Missing Python module '{missing_mod}'. Install the package '{pkg}'.\n"
            f"For Modal: add '{pkg}' to ModalDeploymentConfig.pip_packages in synth_ai/task/apps/grpo_crafter.py.\n"
            f"Locally: pip install {pkg}"
        )
    # Allow running without synth_envs_hosted; gate hosted features off
    if missing_mod == "synth_envs_hosted":
        HAS_HOSTED = False
    else:
        detailed = (
            "grpo_crafter task app requires example dependencies and runtime libs.\n"
            + (fix_hint + "\n" if fix_hint else "")
            + f"Original error: {exc}"
        )
        raise RuntimeError(detailed) from exc


CRAFTING_RULES_SYSTEM_HINT = (
    "Crafter crafting rules (from the paper):\n"
    "- Make Wood Pickaxe: Nearby a table; have wood in inventory.\n"
    "- Make Stone Pickaxe: Nearby a table; have wood and stone in inventory.\n"
    "- Make Iron Pickaxe: Nearby a table; furnace exists; have wood, coal, and iron in inventory.\n"
    "- Make Wood Sword: Nearby a table; have wood in inventory.\n"
    "- Make Stone Sword: Nearby a table; have wood and stone in inventory.\n"
    "- Make Iron Sword: Nearby a table; furnace exists; have wood, coal, and iron in inventory."
)


DATASET_SPEC = TaskDatasetSpec(
    id="crafter_classic_procedural",
    name="Crafter Classic Procedural Seeds",
    version="1.0.0",
    splits=["train"],
    default_split="train",
    description="Procedural Crafter Classic seeds with reproducible world traits.",
)


@dataclass
class CrafterDataset:
    spec: TaskDatasetSpec

    def __post_init__(self) -> None:
        self.default_seed = int(env_value("CRAFTER_DEFAULT_SEED", 42))
        self.seed_min = 0
        self.seed_max = int(env_value("CRAFTER_MAX_SEED", 2**31 - 1))
        area_env = env_value("CRAFTER_AREA", "64,64")
        self.area = tuple(int(x) for x in str(area_env).split(","))
        self.length = int(env_value("CRAFTER_EPISODE_LENGTH", 10000))
        self._cache: Dict[int, Dict[str, Any]] = {}

    def config_for_seed(self, seed: int) -> Dict[str, Any]:
        return {
            "seed": int(seed),
            "area": list(self.area),
            "length": self.length,
        }

    def describe_seed(self, seed: int) -> Dict[str, Any]:
        seed = int(seed)
        if seed in self._cache:
            return self._cache[seed]
        env = crafter.Env(area=self.area, length=self.length, seed=seed)
        try:
            env.reset()
            traits = _compute_world_traits(env)
            player = getattr(env, "_player", None)
            inventory = dict(getattr(player, "inventory", {})) if player else {}
            position = getattr(player, "pos", None)
        finally:
            close_fn = getattr(env, "close", None)
            if callable(close_fn):
                close_fn()
        summary = {
            "seed": seed,
            "difficulty": self._difficulty(traits),
            "traits": traits,
            "inventory": inventory,
            "player_position": list(position) if position is not None else None,
            "config": self.config_for_seed(seed),
        }
        self._cache[seed] = summary
        return summary

    def _difficulty(self, traits: Dict[str, int]) -> str:
        for difficulty, bounds in TRAIT_BOUNDS.items():
            if traits.get("trees", 0) >= bounds.get("min_trees", 0) and traits.get(
                "hostiles", 0
            ) <= bounds.get("max_hostiles", 0):
                return difficulty
        return "custom"

    @property
    def seed_range(self) -> List[int]:
        return [self.seed_min, self.seed_max]


def _compute_world_traits(env: "crafter.Env", radius: int = 10) -> Dict[str, int]:
    # Local copy to avoid import-time issues; mirrors synth_ai.environments.examples.crafter_classic.taskset.world_traits
    from crafter import objects as _objects  # type: ignore
    import numpy as _np  # type: ignore

    player = getattr(env, "_player", None)
    if player is None:
        return {"trees": 0, "cows": 0, "hostiles": 0}
    pos = _np.array(getattr(player, "pos", [0, 0]))
    counts = {"trees": 0, "cows": 0, "hostiles": 0}
    world = getattr(env, "_world", None)
    objects = getattr(world, "_objects", []) if world is not None else []
    for obj in objects:
        if obj is None or obj is player:
            continue
        try:
            if _np.abs(getattr(obj, "pos") - pos).sum() > radius:
                continue
        except Exception:
            continue
        if isinstance(obj, _objects.Plant) and getattr(obj, "kind", "") == "tree":
            counts["trees"] += 1
        elif isinstance(obj, _objects.Cow):
            counts["cows"] += 1
        elif isinstance(obj, (_objects.Zombie, _objects.Skeleton)):
            counts["hostiles"] += 1
    return counts


def env_value(key: str, default: Any) -> Any:
    import os

    return os.getenv(key, default)


def build_dataset() -> tuple[TaskDatasetRegistry, CrafterDataset]:
    registry = TaskDatasetRegistry()
    dataset = CrafterDataset(DATASET_SPEC)
    registry.register(DATASET_SPEC, lambda _spec: dataset, cache=True)
    return registry, dataset


def _base_task_info(dataset: CrafterDataset) -> TaskInfo:
    return TaskInfo(
        task={"id": "crafter_classic", "name": "Crafter Classic", "version": "1.0.0"},
        environments=["crafter"],
        action_space={
            "type": "discrete",
            "size": len(C.actions),
            "actions": list(C.actions),
        },
        observation={
            "summary": "RGB frame plus inventory, achievements, and semantic map patches.",
            "keys": ["image", "inventory", "achievements", "semantic_map_patch7"],
            "image_shape": [64, 64, 3],
        },
        dataset={
            **DATASET_SPEC.model_dump(),
            "seed_range": dataset.seed_range,
            "default_seed": dataset.default_seed,
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
            "tool": {"name": "interact", "parallel_tool_calls": False},
        },
        capabilities={
            "supports_rollout": True,
            "supports_env_lifecycle": True,
            "requires_api_key_header": True,
        },
        limits={"max_ops": 100000, "max_time_s": 3600},
    )


OUTCOME_RUBRIC = load_rubric(
    {
        "version": "1",
        "goal_text": "Reward unlocking Crafter achievements and survival.",
        "aggregation": "weighted_sum",
        "criteria": [
            {
                "id": "achievements",
                "description": "Unlock achievements or crafting milestones.",
                "weight": 1.0,
            },
            {
                "id": "survival",
                "description": "Maintain health, food, and drink levels.",
                "weight": 1.0,
            },
        ],
    }
)

EVENTS_RUBRIC = load_rubric(
    {
        "version": "1",
        "goal_text": "Encourage purposeful step-wise exploration and crafting.",
        "aggregation": "weighted_sum",
        "criteria": [
            {
                "id": "progress_steps",
                "description": "Actions progress quests, crafting, or exploration.",
                "weight": 1.0,
            }
        ],
    }
)


def describe_taskset(dataset: CrafterDataset) -> Dict[str, Any]:
    return {
        **DATASET_SPEC.model_dump(),
        "seed_range": dataset.seed_range,
        "default_seed": dataset.default_seed,
        "config": {
            "area": list(dataset.area),
            "length": dataset.length,
        },
    }


def provide_task_instances(
    dataset: CrafterDataset, base_info: TaskInfo, seeds: Sequence[int]
) -> Iterable[TaskInfo]:
    infos: list[TaskInfo] = []
    for seed_value in seeds:
        summary = dataset.describe_seed(seed_value)
        infos.append(
            TaskInfo(
                task=base_info.task,
                environments=base_info.environments,
                action_space=base_info.action_space,
                observation={
                    **base_info.observation,
                    "seed": seed_value,
                    "traits": summary["traits"],
                    "inventory": summary["inventory"],
                    "player_position": summary["player_position"],
                },
                dataset={
                    **base_info.dataset,
                    "seed": seed_value,
                    "difficulty": summary["difficulty"],
                    "config": summary["config"],
                },
                rubric=base_info.rubric,
                inference=base_info.inference,
                capabilities=base_info.capabilities,
                limits=base_info.limits,
            )
        )
    return infos


def _normalise_op(op_value: Any, index: int) -> str:
    if isinstance(op_value, str):
        candidate = op_value
    elif isinstance(op_value, dict):
        candidate = op_value.get("type") or op_value.get("op")
    else:
        candidate = None
    if not candidate:
        raise ValueError(f"Missing op type at index {index}")
    lowered = str(candidate).strip().lower()
    if lowered in {"policy", "agent", "model"}:
        return "agent"
    if lowered in {"env", "environment", "step"}:
        return "env"
    raise ValueError(f"Unsupported op type '{candidate}' at index {index}")


async def rollout_executor(request: RolloutRequest, fastapi_request) -> RolloutResponse:
    # If hosted env service code is not bundled, return a no-op rollout response compatible with contracts
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
                details={},
            ),
            aborted=False,
            ops_executed=0,
            trace=None,
        )

    converted_ops: List[str] = [_normalise_op(op, idx) for idx, op in enumerate(request.ops)]
    legacy_request = LegacyRolloutRequest(
        run_id=request.run_id,
        env=LegacyRolloutEnvSpec(
            env_id=request.env.env_id,
            env_name=request.env.env_name,
            config=request.env.config or {},
            seed=request.env.seed,
        ),
        policy=LegacyRolloutPolicySpec(
            policy_id=request.policy.policy_id,
            policy_name=request.policy.policy_name,
            config=request.policy.config or {},
        ),
        ops=converted_ops,
        record=LegacyRolloutRecordConfig(**request.record.model_dump()),
        on_done=request.on_done,
        branch=None,
        safety=LegacyRolloutSafetyConfig(**request.safety.model_dump()),
        training_session_id=request.training_session_id,
        synth_base_url=request.synth_base_url,
    )

    legacy_response: LegacyRolloutResponse = await legacy_execute_rollout(
        legacy_request, fastapi_request
    )
    data = legacy_response.model_dump()
    metrics = data.get("metrics", {}) or {}
    metrics.setdefault("outcome_score", None)
    metrics.setdefault("events_score", None)
    metrics.setdefault("details", {})
    data["metrics"] = metrics
    return RolloutResponse.model_validate(data)


def build_config() -> TaskAppConfig:
    registry, dataset = build_dataset()
    base_info = _base_task_info(dataset)

    hosted_task_app = HostedTaskApp() if HAS_HOSTED else None

    tracing_enabled = tracing_env_enabled()
    tracing_db_url = resolve_tracing_db_url()
    tracer_factory = build_tracer_factory(
        SessionTracer, enabled=tracing_enabled, db_url=tracing_db_url
    )
    sft_output_dir = resolve_sft_output_dir()

    app_state: Dict[str, Any] = {
        "task_app": hosted_task_app,
        "allowed_environments": ["crafter"],
        "tracing_enabled": tracing_enabled,
    }
    if tracer_factory is not None:
        app_state["session_tracer_factory"] = tracer_factory
    if sft_output_dir:
        app_state["sft_output_dir"] = sft_output_dir

    if tracing_enabled:
        status_msg = f"[task:tracing] enabled (db={tracing_db_url or 'default'})"
    else:
        status_msg = "[task:tracing] disabled"
    print(status_msg, flush=True)
    if sft_output_dir:
        print(f"[task:sft] writing JSONL to {sft_output_dir}", flush=True)

    def _describe_taskset() -> Dict[str, Any]:
        return describe_taskset(dataset)

    def _provide_instances(seeds: Sequence[int]):
        return provide_task_instances(dataset, base_info, seeds)

    routers: tuple = (environment_router, policy_router, branching_router) if HAS_HOSTED else ()

    config = TaskAppConfig(
        app_id="grpo-crafter",
        name="GRPO Crafter Task App",
        description="Crafter Classic environment with GRPO task endpoints and LLM proxies.",
        base_task_info=base_info,
        describe_taskset=_describe_taskset,
        provide_task_instances=_provide_instances,
        rollout=rollout_executor,
        dataset_registry=registry,
        rubrics=RubricBundle(outcome=OUTCOME_RUBRIC, events=EVENTS_RUBRIC),
        proxy=ProxyConfig(
            enable_openai=True, enable_groq=True, system_hint=CRAFTING_RULES_SYSTEM_HINT
        ),
        routers=routers,
        app_state=app_state,
        cors_origins=["*"],
    )
    return config


register_task_app(
    entry=TaskAppEntry(
        app_id="grpo-crafter",
        description="Crafter Classic task app with rollout + proxy endpoints",
        config_factory=build_config,
        aliases=("crafter", "crafter-task"),
        env_files=(str(REPO_ROOT / "backend" / ".env.dev"),),
        modal=ModalDeploymentConfig(
            app_name="grpo-crafter-task-app",
            python_version="3.11",
            pip_packages=(
                "fastapi>=0.100.0",
                "uvicorn>=0.23.0",
                "pydantic>=2.0.0",
                "numpy>=1.24.0",
                "aiohttp>=3.8.0",
                "httpx>=0.24.0",
                "python-dotenv>=1.0.1",
                # Tracing/DB runtime deps
                "sqlalchemy>=2.0.42",
                "aiosqlite>=0.21.0",
                "greenlet>=3.2.3",
                "crafter",
            ),
            extra_local_dirs=(
                (str(REPO_ROOT / "synth_ai"), "/opt/synth_ai_repo/synth_ai"),
                (str(TASK_APP_ROOT), "/opt/synth_ai_repo/examples/warming_up_to_rl/task_app"),
            ),
            secret_names=("crafter-environment-sdk", "groq-api-key", "openai-api-key"),
            memory=16384,
            cpu=4.0,
            max_containers=10,
        ),
    )
)
