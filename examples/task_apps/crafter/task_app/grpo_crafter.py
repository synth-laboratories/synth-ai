"""Task App configuration for the GRPO Crafter example."""

from __future__ import annotations

import logging
import os
import sys
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from synth_ai.task.apps import ModalDeploymentConfig, TaskAppEntry, register_task_app
from synth_ai.task.contracts import RolloutMetrics, RolloutRequest, RolloutResponse, TaskInfo
from synth_ai.task.datasets import TaskDatasetRegistry, TaskDatasetSpec
from synth_ai.task.json import to_jsonable  # noqa: F401  (imported for side-effect compatibility)
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

DEFAULT_ALIAS_OPS: list[str] = ["agent", "env"] * 10
DEFAULT_ALIAS_STEP_REWARDS: dict[str, Any] = {
    "enabled": True,
    "mode": "decision_stepwise",
    "indicator_lambda": 1.0,
    "step_beta": 0.0,
}

_HERE = Path(__file__).resolve()


def _resolve_repo_root() -> Path:
    """Best-effort detection of the Synth AI repo root across local and Modal mounts."""

    candidates: list[Path] = []
    env_root = os.getenv("SYNTH_AI_REPO_ROOT")
    if env_root:
        candidates.append(Path(env_root).expanduser())
    candidates.append(Path("/opt/synth_ai_repo"))
    candidates.extend(parent for parent in [_HERE.parent, *_HERE.parents])

    for candidate in candidates:
        try:
            resolved = candidate.resolve()
        except Exception:
            continue
        if not resolved.exists():
            continue
        if (resolved / "pyproject.toml").exists() or (resolved / "uv.lock").exists():
            return resolved
        if (resolved / "synth_ai").is_dir():
            return resolved

    try:
        return _HERE.parents[3]
    except IndexError:
        return _HERE.parent


def _resolve_task_app_root(repo_root: Path) -> Path:
    """Locate the task_app directory even when the module is copied to a temp mount."""

    preferred = (repo_root / "examples" / "warming_up_to_rl" / "task_app").resolve()
    if preferred.is_dir():
        return preferred

    local_parent = _HERE.parent.resolve()
    if (local_parent / "synth_envs_hosted").is_dir():
        return local_parent

    for parent in _HERE.parents:
        candidate = parent.resolve()
        if (candidate / "synth_envs_hosted").is_dir():
            return candidate

    fallback = Path("/opt/synth_ai_repo/examples/warming_up_to_rl/task_app")
    if fallback.is_dir():
        return fallback.resolve()

    return local_parent


REPO_ROOT = _resolve_repo_root()
TASK_APP_ROOT = _resolve_task_app_root(REPO_ROOT)
SYNTH_ENVS_HOSTED_ROOT = (TASK_APP_ROOT / "synth_envs_hosted").resolve()

EXAMPLES_ROOT = (REPO_ROOT / "examples").resolve()
RUBRICS_ROOT = (EXAMPLES_ROOT / "multi_step" / "rubrics").resolve()

for path in (REPO_ROOT, TASK_APP_ROOT, SYNTH_ENVS_HOSTED_ROOT, EXAMPLES_ROOT):
    try:
        resolved = path.resolve()
    except Exception:
        resolved = path
    if resolved.exists():
        path_str = str(resolved)
        if path_str not in sys.path:
            sys.path.insert(0, path_str)

# Fallback: explicitly add Modal mount path for 'examples' if REPO_ROOT detection fails
try:
    _hard_examples = Path("/opt/synth_ai_repo/examples")
    if _hard_examples.exists():
        _hard_examples_str = str(_hard_examples.resolve())
        if _hard_examples_str not in sys.path:
            sys.path.insert(0, _hard_examples_str)
except Exception:
    pass

HAS_HOSTED = True
try:
    import crafter  # type: ignore
    import crafter.constants as crafter_constants  # type: ignore
    from synth_ai.environments.examples.crafter_classic.taskset import TRAIT_BOUNDS
    from synth_envs_hosted.branching import router as branching_router  # type: ignore
    from synth_envs_hosted.environment_routes import router as environment_router  # type: ignore
    from synth_envs_hosted.hosted_app import TaskApp as HostedTaskApp  # type: ignore
    from synth_envs_hosted.policy_routes import router as policy_router  # type: ignore
    from synth_envs_hosted.rollout import (  # type: ignore
        RolloutEnvSpec as LegacyRolloutEnvSpec,
    )
    from synth_envs_hosted.rollout import (
        RolloutPolicySpec as LegacyRolloutPolicySpec,
    )
    from synth_envs_hosted.rollout import (
        RolloutRecordConfig as LegacyRolloutRecordConfig,
    )
    from synth_envs_hosted.rollout import (
        RolloutRequest as LegacyRolloutRequest,
    )
    from synth_envs_hosted.rollout import (
        RolloutResponse as LegacyRolloutResponse,
    )
    from synth_envs_hosted.rollout import (
        RolloutSafetyConfig as LegacyRolloutSafetyConfig,
    )
    from synth_envs_hosted.rollout import (
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
        self._cache: dict[int, dict[str, Any]] = {}

    def config_for_seed(self, seed: int) -> dict[str, Any]:
        return {
            "seed": int(seed),
            "area": list(self.area),
            "length": self.length,
        }

    def describe_seed(self, seed: int) -> dict[str, Any]:
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

    def _difficulty(self, traits: dict[str, int]) -> str:
        for difficulty, bounds in TRAIT_BOUNDS.items():
            if traits.get("trees", 0) >= bounds.get("min_trees", 0) and traits.get(
                "hostiles", 0
            ) <= bounds.get("max_hostiles", 0):
                return difficulty
        return "custom"

    @property
    def seed_range(self) -> list[int]:
        return [self.seed_min, self.seed_max]


def _compute_world_traits(env: crafter.Env, radius: int = 10) -> dict[str, int]:
    # Local copy to avoid import-time issues; mirrors synth_ai.environments.examples.crafter_classic.taskset.world_traits
    import numpy as _np  # type: ignore
    from crafter import objects as _objects  # type: ignore

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
            if _np.abs(obj.pos - pos).sum() > radius:
                continue
        except Exception:
            continue
        if isinstance(obj, _objects.Plant) and getattr(obj, "kind", "") == "tree":
            counts["trees"] += 1
        elif isinstance(obj, _objects.Cow):
            counts["cows"] += 1
        elif isinstance(obj, _objects.Zombie | _objects.Skeleton):
            counts["hostiles"] += 1
    return counts


def env_value(key: str, default: Any) -> Any:
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
            "size": len(crafter_constants.actions),
            "actions": list(crafter_constants.actions),
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


OUTCOME_RUBRIC = load_rubric(str(RUBRICS_ROOT / "crafter_outcome_rubric.json"))

EVENTS_RUBRIC = load_rubric(str(RUBRICS_ROOT / "crafter_events_rubric.json"))


def describe_taskset(dataset: CrafterDataset) -> dict[str, Any]:
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


def _coerce_math_to_crafter(request: RolloutRequest) -> RolloutRequest:
    """Map legacy math env/policy names to crafter and enrich rollout defaults."""

    def _needs_crafter(name: str | None) -> bool:
        if not name:
            return False
        lowered = str(name).strip().lower()
        return lowered.startswith("math")

    env_updates: dict[str, Any] = {}
    policy_updates: dict[str, Any] = {}
    alias_applied = False

    if _needs_crafter(request.env.env_name):
        env_updates["env_name"] = "crafter"
        alias_applied = True
    if request.env.env_id and _needs_crafter(request.env.env_id):
        env_updates["env_id"] = None
        alias_applied = True
    if _needs_crafter(request.policy.policy_name):
        policy_updates["policy_name"] = "crafter-react"
        alias_applied = True
    if request.policy.policy_id and _needs_crafter(request.policy.policy_id):
        policy_updates["policy_id"] = None
        alias_applied = True

    if not alias_applied:
        return request

    updated_env = request.env.model_copy(update=env_updates) if env_updates else request.env
    updated_policy = (
        request.policy.model_copy(update=policy_updates) if policy_updates else request.policy
    )

    env_cfg = dict(updated_env.config or {})
    env_cfg.setdefault("difficulty", "normal")
    env_cfg.setdefault("step_rewards", dict(DEFAULT_ALIAS_STEP_REWARDS))
    env_cfg.setdefault("env_params", {"max_steps_per_episode": 200})
    updated_env = updated_env.model_copy(update={"config": env_cfg})

    policy_cfg = dict(updated_policy.config or {})
    policy_cfg.setdefault("max_llm_calls", 10)
    policy_cfg.setdefault("max_completion_tokens", 1024)
    policy_cfg.setdefault("temperature", 0.2)
    policy_cfg.setdefault("step_rewards", dict(DEFAULT_ALIAS_STEP_REWARDS))
    updated_policy = updated_policy.model_copy(update={"config": policy_cfg})

    ops_override = request.ops
    if not ops_override or len(ops_override) < len(DEFAULT_ALIAS_OPS):
        ops_override = list(DEFAULT_ALIAS_OPS)

    coerced = request.model_copy(update={"env": updated_env, "policy": updated_policy, "ops": ops_override})

    try:
        print(
            "[rollout] remapped math request -> crafter "
            f"(env={request.env.env_name!r}→{coerced.env.env_name!r}, "
            f"policy={request.policy.policy_name!r}→{coerced.policy.policy_name!r})",
            flush=True,
        )
    except Exception:
        pass
    try:
        logger.info(
            "ROLLOUT_ALIAS: remapped math env/policy to crafter (env=%s→%s, policy=%s→%s)",
            request.env.env_name,
            coerced.env.env_name,
            request.policy.policy_name,
            coerced.policy.policy_name,
        )
    except Exception:
        pass

    return coerced


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

    request = _coerce_math_to_crafter(request)

    policy_cfg = dict(request.policy.config or {})
    try:
        max_llm_calls = int(policy_cfg.get("max_llm_calls") or 10)
    except Exception:
        max_llm_calls = 10
    policy_cfg.setdefault("max_llm_calls", max_llm_calls)
    policy_cfg.setdefault("max_tokens", 512)
    policy_cfg.setdefault("max_completion_tokens", 512)
    policy_cfg.setdefault("temperature", 0.2)
    policy_cfg.setdefault("top_p", 0.95)

    env_cfg = dict(request.env.config or {})
    env_params = dict(env_cfg.get("env_params") or {})
    try:
        max_steps_episode = int(env_params.get("max_steps_per_episode") or max_llm_calls)
    except Exception:
        max_steps_episode = max_llm_calls
    desired_steps = max(max_llm_calls, max_steps_episode)
    env_params["max_steps_per_episode"] = int(desired_steps)
    env_cfg["env_params"] = env_params

    updated_policy = request.policy.model_copy(update={"config": policy_cfg})
    updated_env = request.env.model_copy(update={"config": env_cfg})
    request = request.model_copy(update={"policy": updated_policy, "env": updated_env})

    converted_ops: list[str] = [_normalise_op(op, idx) for idx, op in enumerate(request.ops)]
    max_ops_allowed = max_llm_calls * 2 if max_llm_calls > 0 else len(converted_ops)
    if max_ops_allowed and len(converted_ops) > max_ops_allowed:
        converted_ops = converted_ops[:max_ops_allowed]
    legacy_request = LegacyRolloutRequest(
        run_id=request.run_id,
        env=LegacyRolloutEnvSpec(
            env_id=request.env.env_id,
            env_name=request.env.env_name,
            config=env_cfg,
            seed=request.env.seed,
        ),
        policy=LegacyRolloutPolicySpec(
            policy_id=request.policy.policy_id,
            policy_name=request.policy.policy_name,
            config=policy_cfg,
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

    app_state: dict[str, Any] = {
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

    def _describe_taskset() -> dict[str, Any]:
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
                # Mount repo root so local modules resolve when deployed on Modal
                (str(REPO_ROOT), "/opt/synth_ai_repo"),
                (str(REPO_ROOT / "synth_ai"), "/opt/synth_ai_repo/synth_ai"),
                (str(TASK_APP_ROOT), "/opt/synth_ai_repo/examples/warming_up_to_rl/task_app"),
            ),
            secret_names=("groq-api-key", "openai-api-key"),
            memory=16384,
            cpu=4.0,
            max_containers=10,
        ),
    )
)
