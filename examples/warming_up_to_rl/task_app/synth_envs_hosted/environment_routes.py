from __future__ import annotations

import contextlib
import json
import logging
from typing import Any
from uuid import uuid4

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

# Import the actual classes from synth-ai
from synth_ai.environments.examples.crafter_classic.environment import (
    CrafterClassicEnvironment,
)
from synth_ai.environments.examples.crafter_classic.taskset import (
    CrafterTaskInstance,
    CrafterTaskInstanceMetadata,
)
from synth_ai.environments.tasks.core import Impetus, Intent

from .envs.crafter.environment import CrafterEnvironmentWrapper
from .registry import registry
from .storage.volume import storage

logger = logging.getLogger(__name__)

router = APIRouter()


async def validate_environment_observation(observation: Any, context: str) -> None:
    """
    Validate that an environment observation has the correct structure.
    This ensures the environment wrapper is producing valid observations.

    Args:
        observation: The observation to validate
        context: Context string for logging (e.g., "initialize", "step")
    """
    if observation is None:
        raise ValueError(f"Environment observation cannot be None in {context}")

    if not isinstance(observation, dict):
        raise ValueError(
            f"Environment observation must be dict in {context}, got {type(observation)}"
        )

    # For Wordle environments, validate specific structure
    # Check if this looks like a Wordle observation by looking for Wordle-specific keys
    wordle_keys = {
        "text",
        "status",
        "remaining_guesses",
        "guesses",
        "feedback",
        "reward_last",
        "total_reward",
        "terminated",
    }
    if wordle_keys.issubset(set(observation.keys())):
        logger.info(f"🔍 ENV_ROUTES: Validating Wordle observation structure in {context}")
        logger.info(f"🔍 ENV_ROUTES: Observation keys: {list(observation.keys())}")

        missing_keys = wordle_keys - set(observation.keys())
        if missing_keys:
            logger.error(
                f"❌ ENV_ROUTES: Wordle observation missing required keys in {context}: {missing_keys}"
            )
            logger.error(f"❌ ENV_ROUTES: Full observation: {observation}")
            raise ValueError(
                f"Wordle observation missing required keys in {context}: {missing_keys}"
            )

        # Validate data types
        if not isinstance(observation.get("text"), str):
            raise ValueError(
                f"Wordle observation 'text' must be string in {context}, got {type(observation.get('text'))}"
            )

        if not isinstance(observation.get("guesses"), list):
            raise ValueError(
                f"Wordle observation 'guesses' must be list in {context}, got {type(observation.get('guesses'))}"
            )

        if not isinstance(observation.get("feedback"), list):
            raise ValueError(
                f"Wordle observation 'feedback' must be list in {context}, got {type(observation.get('feedback'))}"
            )

        logger.info(
            f"✅ ENV_ROUTES: Wordle observation structure validated successfully in {context}"
        )
    else:
        logger.debug(
            f"🔍 ENV_ROUTES: Observation doesn't appear to be Wordle in {context}, skipping validation"
        )


class EnvCreateRequest(BaseModel):
    env_name: str
    config: dict[str, Any] = {}
    seed: int | None = None
    parent_env_id: str | None = None
    rl_run_id: str


class EnvCreateResponse(BaseModel):
    env_id: str
    observation: dict[str, Any]
    info: dict[str, Any] | None = None
    step_idx: int


class EnvResetRequest(BaseModel):
    env_id: str
    seed: int | None = None


class EnvResetResponse(BaseModel):
    observation: dict[str, Any]
    info: dict[str, Any] | None = None
    step_idx: int


class EnvStepRequest(BaseModel):
    env_id: str
    tool_calls: list[dict[str, Any]]


class EnvStepResponse(BaseModel):
    observation: dict[str, Any]
    done: bool
    info: dict[str, Any] | None = None
    reward: float | None = None
    truncated: bool | None = None
    step_idx: int


class EnvSnapshotRequest(BaseModel):
    env_id: str


class EnvSnapshotResponse(BaseModel):
    snapshot_id: str
    path: str
    rl_run_id: str
    size: int


class EnvRestoreRequest(BaseModel):
    snapshot_id: str


class EnvRestoreResponse(BaseModel):
    env_id: str
    observation: dict[str, Any]
    info: dict[str, Any] | None = None
    step_idx: int


class EnvTerminateRequest(BaseModel):
    env_id: str


class EnvTerminateResponse(BaseModel):
    ok: bool


@router.post("/create", response_model=EnvCreateResponse)
async def create_environment(request: EnvCreateRequest) -> EnvCreateResponse:
    """Create a new environment instance."""
    try:
        # Create the underlying synth-ai environment
        env_name_lower = request.env_name.lower()
        if env_name_lower == "crafter":
            # Build a minimal Crafter task instance
            difficulty = (request.config or {}).get("difficulty", "normal")
            seed_value = request.seed if request.seed is not None else 0
            # Task object is part of the ecosystem; not required for instantiation here
            impetus = Impetus(instructions="Survive and unlock achievements.")
            intent = Intent(
                rubric={"goal": "Unlock achievements"},
                gold_trajectories=None,
                gold_state_diff={},
            )
            metadata = CrafterTaskInstanceMetadata(
                difficulty=difficulty,
                seed=seed_value,
                num_trees_radius=0,
                num_cows_radius=0,
                num_hostiles_radius=0,
            )
            instance = CrafterTaskInstance(
                id=uuid4(),
                impetus=impetus,
                intent=intent,
                metadata=metadata,
                is_reproducible=True,
                initial_engine_snapshot=None,
            )
            # Create CrafterClassicEnvironment from task instance
            base_env = CrafterClassicEnvironment(task_instance=instance)

            # Wrap it for our API
            wrapper = CrafterEnvironmentWrapper(
                env=base_env,
                seed=request.seed,
            )

            # Initialize the environment
            result = await wrapper.initialize()

            # Log a world signature for sanity: seed + starting public state hash
            try:
                pub_state = base_env.engine._get_public_state_from_env()  # type: ignore[attr-defined]
                import hashlib
                import json as _json

                sig_src = {
                    "player_position": list(pub_state.player_position),
                    "player_direction": pub_state.player_direction,
                    "semantic_map": pub_state.semantic_map,
                    "inventory": {k: v for k, v in pub_state.inventory.items() if v},
                }
                sig_str = _json.dumps(sig_src, sort_keys=True)
                sig = hashlib.md5(sig_str.encode("utf-8")).hexdigest()[:12]
                logger.info(
                    "Crafter init signature: seed=%s sig=%s pos=%s inv=%s",
                    str(seed_value),
                    sig,
                    list(pub_state.player_position),
                    {k: v for k, v in pub_state.inventory.items() if v},
                )
            except Exception as _e:
                pass

            # Handle the observation structure consistently
            # For Crafter, the result might still have the old nested structure, so we need to handle both
            if isinstance(result, dict) and "observation" in result:
                # Old nested structure - extract the inner observation
                observation_for_registry = result["observation"].copy()
            else:
                # New flat structure - remove non-observation fields
                observation_for_registry = result.copy()
                for key in ["step_idx", "info"]:
                    if key in observation_for_registry:
                        del observation_for_registry[key]

            # Register in memory
            env_id = registry.register_env(
                env=wrapper,
                seed=request.seed,
                rl_run_id=request.rl_run_id,
                last_observation=observation_for_registry,
                last_info=result.get("info"),
            )

            # Update step index in registry
            handle = registry.get_env(env_id)
            if handle:
                handle.step_idx = result["step_idx"]

            return EnvCreateResponse(
                env_id=env_id,
                observation=observation_for_registry,
                info=result.get("info"),
                step_idx=result["step_idx"],
            )
        elif env_name_lower == "wordle":
            # Defer imports to avoid hard dependency when not used
            try:
                from synth_ai.environments.examples.wordle.environment import (
                    WordleEnvironment,
                )
                from synth_ai.environments.examples.wordle.taskset import (
                    WordleTaskInstance,
                    WordleTaskInstanceMetadata,
                )
            except Exception as e:
                raise HTTPException(
                    status_code=500, detail=f"Wordle modules unavailable: {e}"
                ) from e

            # Lazy import of wrapper within branch
            try:
                from .envs.wordle.environment import WordleEnvironmentWrapper
            except Exception as e:
                raise HTTPException(
                    status_code=500, detail=f"Wordle wrapper unavailable: {e}"
                ) from e
            else:
                wordle_wrapper_cls = WordleEnvironmentWrapper

            cfg = request.config or {}
            word_length = int(cfg.get("word_length", 5))
            max_guesses = int(cfg.get("max_guesses", 6))

            # Build a single Wordle task instance with proper seed usage
            md = WordleTaskInstanceMetadata(
                word_length=word_length,
                max_guesses=max_guesses,
                target_word=None,  # Let seed determine the word
                enforce_wordlist=True,
                seed=request.seed,
                consume_invalid_attempts=True,
            )
            instance = WordleTaskInstance(
                id=uuid4(),
                impetus=Impetus(instructions="Play Wordle. Submit one 5-letter word per turn."),
                intent=Intent(rubric="guess the word", gold_trajectories=None, gold_state_diff={}),
                metadata=md,
                is_reproducible=True,
                initial_engine_snapshot=None,
            )
            base_env = WordleEnvironment(task_instance=instance)

            # Try to preserve the exact puzzle snapshot for reproducibility
            init_snap = getattr(instance, "initial_engine_snapshot", None)

            wrapper = wordle_wrapper_cls(
                env=base_env,
                seed=request.seed,
                word_length=word_length,
                max_guesses=max_guesses,
                initial_engine_snapshot=init_snap,
            )

            result = await wrapper.initialize()

            # Validate Wordle observation structure
            # After our fix, the result is now flat, so we need to extract the observation fields
            # that should be passed to the registry and response
            if isinstance(result, dict) and "observation" in result:
                # Old nested structure - extract the inner observation
                observation_for_registry = result["observation"].copy()
            else:
                # New flat structure - remove non-observation fields
                observation_for_registry = result.copy()
                for key in ["step_idx", "info"]:
                    if key in observation_for_registry:
                        del observation_for_registry[key]

            await validate_environment_observation(observation_for_registry, "initialize")

            env_id = registry.register_env(
                env=wrapper,
                seed=request.seed,
                rl_run_id=request.rl_run_id,
                last_observation=observation_for_registry,
                last_info=result.get("info"),
            )
            handle = registry.get_env(env_id)
            if handle:
                handle.step_idx = result["step_idx"]
            return EnvCreateResponse(
                env_id=env_id,
                observation=observation_for_registry,
                info=result.get("info"),
                step_idx=result["step_idx"],
            )

        elif env_name_lower == "sokoban":
            try:
                from synth_ai.environments.examples.sokoban.environment import (
                    SokobanEnvironment,
                )
                from synth_ai.environments.examples.sokoban.taskset import (
                    SokobanTaskInstance,
                    SokobanTaskInstanceMetadata,
                )
            except Exception as e:
                raise HTTPException(
                    status_code=500, detail=f"Sokoban modules unavailable: {e}"
                ) from e

            # Lazy import of wrapper within branch
            try:
                from .envs.sokoban.environment import SokobanEnvironmentWrapper
            except Exception as e:
                raise HTTPException(
                    status_code=500, detail=f"Sokoban wrapper unavailable: {e}"
                ) from e

            cfg = request.config or {}
            difficulty = cfg.get("difficulty", "easy")
            initial_state = cfg.get("initial_state")  # Optional engine snapshot

            metadata = SokobanTaskInstanceMetadata(
                difficulty=difficulty,
            )
            instance = SokobanTaskInstance(
                id=uuid4(),
                impetus=Impetus(instructions="Push boxes to targets."),
                intent=Intent(
                    rubric={"goal": "Solve the Sokoban puzzle"},
                    gold_trajectories=None,
                    gold_state_diff={},
                ),
                metadata=metadata,
                is_reproducible=True,
                initial_engine_snapshot=initial_state,
            )
            base_env = SokobanEnvironment(task_instance=instance)

            wrapper = SokobanEnvironmentWrapper(env=base_env, seed=request.seed, config=cfg)
            result = await wrapper.initialize()

            # Handle the observation structure consistently for Sokoban
            if isinstance(result, dict) and "observation" in result:
                # Old nested structure - extract the inner observation
                observation_for_registry = result["observation"].copy()
            else:
                # New flat structure - remove non-observation fields
                observation_for_registry = result.copy()
                for key in ["step_idx", "info"]:
                    if key in observation_for_registry:
                        del observation_for_registry[key]

            env_id = registry.register_env(
                env=wrapper,
                seed=request.seed,
                rl_run_id=request.rl_run_id,
                last_observation=observation_for_registry,
                last_info=result.get("info"),
            )
            handle = registry.get_env(env_id)
            if handle:
                handle.step_idx = result["step_idx"]
            return EnvCreateResponse(
                env_id=env_id,
                observation=observation_for_registry,
                info=result.get("info"),
                step_idx=result["step_idx"],
            )

        elif env_name_lower == "math":
            # Single-step math env (GSM8K-style)
            cfg = request.config or {}
            # Lazy import of wrapper within branch
            try:
                from .envs.math.environment import MathEnvironmentWrapper
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Math wrapper unavailable: {e}") from e

            wrapper = MathEnvironmentWrapper(
                seed=request.seed,
                problem_id=cfg.get("problem_id"),
                problem_text=cfg.get("problem_text"),
            )
            result = await wrapper.initialize()

            observation_for_registry = (
                result["observation"].copy()
                if isinstance(result, dict) and "observation" in result
                else result.copy()
            )
            for key in ["step_idx", "info"]:
                if key in observation_for_registry:
                    del observation_for_registry[key]

            env_id = registry.register_env(
                env=wrapper,
                seed=request.seed,
                rl_run_id=request.rl_run_id,
                last_observation=observation_for_registry,
                last_info=result.get("info"),
            )
            handle = registry.get_env(env_id)
            if handle:
                handle.step_idx = result["step_idx"]
            return EnvCreateResponse(
                env_id=env_id,
                observation=observation_for_registry,
                info=result.get("info"),
                step_idx=result["step_idx"],
            )
        else:
            raise HTTPException(
                status_code=422,
                detail=f"Unknown environment name: {request.env_name}",
            )

    except Exception as e:
        logger.error(f"Failed to create environment: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


# --- Compatibility routes for existing eval scripts that expect CrafterClassic paths ---
@router.post("/CrafterClassic/initialize", response_model=EnvCreateResponse)
async def compat_initialize(payload: dict) -> EnvCreateResponse:
    seed = payload.get("seed")
    wc = payload.get("world_config")
    cfg = payload.get("config")
    difficulty: str = "normal"
    if isinstance(wc, str) and wc:
        difficulty = wc
    elif isinstance(wc, dict) and wc.get("difficulty"):
        difficulty = str(wc.get("difficulty"))
    elif isinstance(cfg, dict) and cfg.get("difficulty"):
        difficulty = str(cfg.get("difficulty"))
    req = EnvCreateRequest(
        env_name="crafter", config={"difficulty": difficulty}, seed=seed, rl_run_id="eval"
    )
    return await create_environment(req)


@router.post("/CrafterClassic/step", response_model=EnvStepResponse)
async def compat_step(payload: dict) -> EnvStepResponse:
    env_id = payload.get("env_id")
    # eval script wraps action as {"tool_calls":[{"tool":"interact","args":{"action": <id>}}]}
    action = payload.get("action") or {}
    tool_calls = action.get("tool_calls") if isinstance(action, dict) else None
    if not isinstance(tool_calls, list):
        tool_calls = []
        # Fallback: support {action: {actions: [..]}} by expanding into tool_calls
        actions_list = action.get("actions") if isinstance(action, dict) else None
        if isinstance(actions_list, list) and actions_list:
            for a in actions_list:
                tool_calls.append(
                    {
                        "tool": "interact",
                        "args": {"action": a},
                    }
                )
    req = EnvStepRequest(env_id=env_id, tool_calls=tool_calls)
    return await step_environment(req)


@router.post("/CrafterClassic/terminate", response_model=EnvTerminateResponse)
async def compat_terminate(payload: dict) -> EnvTerminateResponse:
    env_id = payload.get("env_id")
    req = EnvTerminateRequest(env_id=env_id)
    return await terminate_environment(req)


@router.post("/reset", response_model=EnvResetResponse)
async def reset_environment(request: EnvResetRequest) -> EnvResetResponse:
    """Reset an environment to its initial state."""
    handle = registry.get_env(request.env_id)
    if not handle:
        raise HTTPException(status_code=404, detail=f"Environment {request.env_id} not found")

    try:
        # Determine wrapper type and rebuild base env if a new seed is provided
        wrapper = handle.env
        if isinstance(wrapper, CrafterEnvironmentWrapper):
            if request.seed is not None:
                try:
                    difficulty = "normal"
                    seed_value = int(request.seed)
                    metadata = CrafterTaskInstanceMetadata(
                        difficulty=difficulty,
                        seed=seed_value,
                        num_trees_radius=0,
                        num_cows_radius=0,
                        num_hostiles_radius=0,
                    )
                    instance = CrafterTaskInstance(
                        id=uuid4(),
                        impetus=Impetus(instructions="Reset"),
                        intent=Intent(
                            rubric={"goal": "Reset"},
                            gold_trajectories=None,
                            gold_state_diff={},
                        ),
                        metadata=metadata,
                        is_reproducible=True,
                        initial_engine_snapshot=None,
                    )
                    new_base_env = CrafterClassicEnvironment(task_instance=instance)
                    wrapper.env = new_base_env
                    wrapper.seed = seed_value
                    handle.seed = seed_value
                except Exception:
                    wrapper.seed = request.seed
                    handle.seed = request.seed

        elif True:
            # Try to dynamically import Wordle wrapper and check instance safely
            wordle_wrapper_cls = None
            with contextlib.suppress(Exception):
                from .envs.wordle.environment import WordleEnvironmentWrapper

                wordle_wrapper_cls = WordleEnvironmentWrapper  # type: ignore[assignment]

            if wordle_wrapper_cls is not None and isinstance(wrapper, wordle_wrapper_cls):
                # Rebuild Wordle env with the same configuration; if we have a preserved
                # initial_engine_snapshot, prefer constructing the instance directly.
                try:
                    from synth_ai.environments.examples.wordle.environment import (
                        WordleEnvironment,
                    )
                    from synth_ai.environments.examples.wordle.taskset import (
                        WordleTaskInstance,
                        WordleTaskInstanceMetadata,
                        create_wordle_taskset,
                    )
                except Exception as e:
                    raise HTTPException(
                        status_code=500, detail=f"Wordle modules unavailable: {e}"
                    ) from e

                init_snap = getattr(wrapper, "initial_engine_snapshot", None)
                if init_snap is not None:
                    metadata = WordleTaskInstanceMetadata(
                        word_length=int(wrapper.word_length),
                        max_guesses=int(wrapper.max_guesses),
                    )
                    instance = WordleTaskInstance(
                        id=uuid4(),
                        impetus=Impetus(instructions="Reset"),
                        intent=Intent(
                            rubric={"goal": "Reset"},
                            gold_trajectories=None,
                            gold_state_diff={},
                        ),
                        metadata=metadata,
                        is_reproducible=True,
                        initial_engine_snapshot=init_snap,
                    )
                    new_base_env = WordleEnvironment(task_instance=instance)
                else:
                    ts = await create_wordle_taskset(
                        sample_size=1,
                        word_length=int(wrapper.word_length),
                        max_guesses=int(wrapper.max_guesses),
                    )
                    instance = ts.instances[0]
                    new_base_env = WordleEnvironment(task_instance=instance)
                wrapper.env = new_base_env
                if request.seed is not None:
                    wrapper.seed = int(request.seed)
                    handle.seed = int(request.seed)
            else:
                pass
            # Rebuild Wordle env with the same configuration; if we have a preserved
            # initial_engine_snapshot, prefer constructing the instance directly.
            try:
                from synth_ai.environments.examples.wordle.environment import (
                    WordleEnvironment,
                )
                from synth_ai.environments.examples.wordle.taskset import (
                    WordleTaskInstance,
                    WordleTaskInstanceMetadata,
                    create_wordle_taskset,
                )
            except Exception as e:
                raise HTTPException(
                    status_code=500, detail=f"Wordle modules unavailable: {e}"
                ) from e

            init_snap = getattr(wrapper, "initial_engine_snapshot", None)
            if init_snap is not None:
                metadata = WordleTaskInstanceMetadata(
                    word_length=int(wrapper.word_length),
                    max_guesses=int(wrapper.max_guesses),
                )
                instance = WordleTaskInstance(
                    id=uuid4(),
                    impetus=Impetus(instructions="Reset"),
                    intent=Intent(
                        rubric={"goal": "Reset"},
                        gold_trajectories=None,
                        gold_state_diff={},
                    ),
                    metadata=metadata,
                    is_reproducible=True,
                    initial_engine_snapshot=init_snap,
                )
                new_base_env = WordleEnvironment(task_instance=instance)
            else:
                ts = await create_wordle_taskset(
                    sample_size=1,
                    word_length=int(wrapper.word_length),
                    max_guesses=int(wrapper.max_guesses),
                )
                instance = ts.instances[0]
                new_base_env = WordleEnvironment(task_instance=instance)
            wrapper.env = new_base_env
            if request.seed is not None:
                wrapper.seed = int(request.seed)
                handle.seed = int(request.seed)

        elif True:
            # Try to dynamically import Sokoban wrapper and check instance safely
            sokoban_wrapper_cls = None
            with contextlib.suppress(Exception):
                from .envs.sokoban.environment import SokobanEnvironmentWrapper

                sokoban_wrapper_cls = SokobanEnvironmentWrapper  # type: ignore[assignment]

            if sokoban_wrapper_cls is not None and isinstance(wrapper, sokoban_wrapper_cls):
                # Rebuild Sokoban env using stored config snapshot
                try:
                    from synth_ai.environments.examples.sokoban.environment import (
                        SokobanEnvironment,
                    )
                    from synth_ai.environments.examples.sokoban.taskset import (
                        SokobanTaskInstance,
                        SokobanTaskInstanceMetadata,
                    )
                except Exception as e:
                    raise HTTPException(
                        status_code=500, detail=f"Sokoban modules unavailable: {e}"
                    ) from e

                cfg = dict(wrapper.config or {})
                metadata = SokobanTaskInstanceMetadata(
                    difficulty=cfg.get("difficulty", "easy"),
                )
                instance = SokobanTaskInstance(
                    id=uuid4(),
                    impetus=Impetus(instructions="Reset"),
                    intent=Intent(
                        rubric={"goal": "Reset"}, gold_trajectories=None, gold_state_diff={}
                    ),
                    metadata=metadata,
                    is_reproducible=True,
                    initial_engine_snapshot=cfg.get("initial_state"),
                )
                new_base_env = SokobanEnvironment(task_instance=instance)
                wrapper.env = new_base_env
                if request.seed is not None:
                    wrapper.seed = int(request.seed)
                    handle.seed = int(request.seed)
            else:
                pass
            # Rebuild Sokoban env using stored config snapshot
            try:
                from synth_ai.environments.examples.sokoban.environment import (
                    SokobanEnvironment,
                )
                from synth_ai.environments.examples.sokoban.taskset import (
                    SokobanTaskInstance,
                    SokobanTaskInstanceMetadata,
                )
            except Exception as e:
                raise HTTPException(
                    status_code=500, detail=f"Sokoban modules unavailable: {e}"
                ) from e

            cfg = dict(wrapper.config or {})
            metadata = SokobanTaskInstanceMetadata(
                difficulty=cfg.get("difficulty", "easy"),
            )
            instance = SokobanTaskInstance(
                id=uuid4(),
                impetus=Impetus(instructions="Reset"),
                intent=Intent(rubric={"goal": "Reset"}, gold_trajectories=None, gold_state_diff={}),
                metadata=metadata,
                is_reproducible=True,
                initial_engine_snapshot=cfg.get("initial_state"),
            )
            new_base_env = SokobanEnvironment(task_instance=instance)
            wrapper.env = new_base_env
            if request.seed is not None:
                wrapper.seed = int(request.seed)
                handle.seed = int(request.seed)

        # Reset the environment regardless of type
        result = await wrapper.initialize()

        # Log a world signature after reset for sanity
        try:
            base_env = handle.env.env  # type: ignore[attr-defined]
            pub_state = base_env.engine._get_public_state_from_env()  # type: ignore[attr-defined]
            import hashlib
            import json as _json

            sig_src = {
                "player_position": list(pub_state.player_position),
                "player_direction": pub_state.player_direction,
                "semantic_map": pub_state.semantic_map,
                "inventory": {k: v for k, v in pub_state.inventory.items() if v},
            }
            sig_str = _json.dumps(sig_src, sort_keys=True)
            sig = hashlib.md5(sig_str.encode("utf-8")).hexdigest()[:12]
            logger.info(
                "Crafter reset signature: seed=%s sig=%s pos=%s inv=%s",
                str(handle.seed),
                sig,
                list(pub_state.player_position),
                {k: v for k, v in pub_state.inventory.items() if v},
            )
        except Exception as _:
            pass

        # Update registry
        handle.step_idx = result["step_idx"]
        handle.last_observation = result["observation"]
        handle.last_info = result.get("info")

        return EnvResetResponse(
            observation=result["observation"],
            info=result.get("info"),
            step_idx=result["step_idx"],
        )

    except Exception as e:
        logger.error(f"Failed to reset environment {request.env_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.post("/step", response_model=EnvStepResponse)
async def step_environment(request: EnvStepRequest) -> EnvStepResponse:
    """Execute a step in the environment."""
    handle = registry.get_env(request.env_id)
    if not handle:
        raise HTTPException(status_code=404, detail=f"Environment {request.env_id} not found")

    try:
        # Execute the step, pre-normalizing invalid Wordle guesses to avoid hard failures
        wrapper = handle.env
        wordle_wrapper_cls = None
        with contextlib.suppress(Exception):
            from .envs.wordle.environment import WordleEnvironmentWrapper

            wordle_wrapper_cls = WordleEnvironmentWrapper  # type: ignore[assignment]

        if wordle_wrapper_cls is not None and isinstance(wrapper, wordle_wrapper_cls):
            expected_len = int(getattr(wrapper, "word_length", 5))
            normalized: list[dict[str, Any]] = []
            for tc in request.tool_calls or []:
                tool = tc.get("tool") or tc.get("tool_name") or tc.get("name") or "interact"
                args = tc.get("arguments") or tc.get("args") or {}
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except Exception:
                        args = {}
                guess = None
                if isinstance(args, dict):
                    guess = args.get("guess") or args.get("word")
                if isinstance(guess, str):
                    g = guess.strip().lower()
                    if (not g.isalpha()) or (len(g) != expected_len):
                        normalized.append(
                            {"tool": "invalid_guess", "args": {"original_guess": guess}}
                        )
                    else:
                        # Preserve the original tool name (interact or submit) for the environment to handle
                        normalized.append({"tool": tool, "args": {"guess": g}})
                else:
                    normalized.append({"tool": "invalid_guess", "args": {"original_guess": guess}})
            result = await wrapper.step(normalized)
        else:
            result = await handle.env.step(request.tool_calls)

            # Validate observation structure for Wordle environments
        env_name = getattr(handle.env, "env", None)
        if (
            env_name
            and hasattr(env_name, "__class__")
            and "wordle" in env_name.__class__.__name__.lower()
        ):
            # Extract observation fields from the flat result structure for validation
            observation_for_validation = result.copy()
            # Remove step_idx, done, info, reward, truncated from the observation since they're separate fields
            for key in ["step_idx", "done", "info", "reward", "truncated"]:
                if key in observation_for_validation:
                    del observation_for_validation[key]
            await validate_environment_observation(observation_for_validation, "step")

        # Update registry
        handle.step_idx = result["step_idx"]

        # Extract the observation fields from the result structure (handle both old nested and new flat)
        if isinstance(result, dict) and "observation" in result:
            # Old nested structure - extract the inner observation
            observation_for_registry = result["observation"].copy()
        else:
            # New flat structure - remove non-observation fields
            observation_for_registry = result.copy()
            for key in ["step_idx", "done", "info", "reward", "truncated"]:
                if key in observation_for_registry:
                    del observation_for_registry[key]

        handle.last_observation = observation_for_registry
        handle.last_info = result.get("info")

        return EnvStepResponse(
            observation=observation_for_registry,
            done=result["done"],
            info=result.get("info"),
            reward=result.get("reward"),
            truncated=result.get("truncated"),
            step_idx=result["step_idx"],
        )

    except Exception as e:
        logger.error(f"Failed to step environment {request.env_id}: {e}")
        # Fallback for Wordle: convert invalid guesses into 'invalid_guess' tool calls and retry once
        try:
            wordle_wrapper_cls = None
            with contextlib.suppress(Exception):
                from .envs.wordle.environment import WordleEnvironmentWrapper

                wordle_wrapper_cls = WordleEnvironmentWrapper  # type: ignore[assignment]

            wrapper = handle.env
            if wordle_wrapper_cls is not None and isinstance(wrapper, wordle_wrapper_cls):
                expected_len = int(getattr(wrapper, "word_length", 5))
                normalized: list[dict[str, Any]] = []
                for tc in request.tool_calls or []:
                    tool = tc.get("tool") or tc.get("tool_name") or tc.get("name") or "interact"
                    args = tc.get("arguments") or tc.get("args") or {}
                    if isinstance(args, str):
                        try:
                            args = json.loads(args)
                        except Exception:
                            args = {}
                    guess = None
                    if isinstance(args, dict):
                        guess = args.get("guess") or args.get("word")
                    if isinstance(guess, str):
                        g = guess.strip().lower()
                        if (not g.isalpha()) or (len(g) != expected_len):
                            normalized.append(
                                {
                                    "tool": "invalid_guess",
                                    "args": {"original_guess": guess},
                                }
                            )
                        else:
                            normalized.append({"tool": "interact", "args": {"guess": g}})
                    else:
                        normalized.append(
                            {"tool": "invalid_guess", "args": {"original_guess": guess}}
                        )

                # Retry with normalized calls, allowing the wrapper to synthesize an observation
                result = await wrapper.step(normalized)

                # Update registry and return as usual
                handle.step_idx = result["step_idx"]
                if isinstance(result, dict) and "observation" in result:
                    observation_for_registry = result["observation"].copy()
                else:
                    observation_for_registry = result.copy()
                    for key in ["step_idx", "done", "info", "reward", "truncated"]:
                        if key in observation_for_registry:
                            del observation_for_registry[key]
                handle.last_observation = observation_for_registry
                handle.last_info = result.get("info")
                return EnvStepResponse(
                    observation=observation_for_registry,
                    done=result["done"],
                    info=result.get("info"),
                    reward=result.get("reward"),
                    truncated=result.get("truncated"),
                    step_idx=result["step_idx"],
                )
        except Exception:
            # Ignore fallback errors; fall through to generic error
            pass

        raise HTTPException(status_code=500, detail=f"{type(e).__name__}: {e}") from e


@router.post("/snapshot", response_model=EnvSnapshotResponse)
async def snapshot_environment(request: EnvSnapshotRequest) -> EnvSnapshotResponse:
    """Create a snapshot of the environment state."""
    handle = registry.get_env(request.env_id)
    if not handle:
        raise HTTPException(status_code=404, detail=f"Environment {request.env_id} not found")

    try:
        # Serialize environment state
        state_dict = await handle.env.serialize()

        # Save to volume
        snapshot_id, path, size = storage.save_snapshot(
            rl_run_id=handle.rl_run_id,
            kind="env",
            state_dict=state_dict,
            config={"seed": handle.seed},
        )

        # Register snapshot
        registry.register_snapshot(
            kind="env",
            rl_run_id=handle.rl_run_id,
            size=size,
            path=path,
        )

        return EnvSnapshotResponse(
            snapshot_id=snapshot_id,
            path=path,
            rl_run_id=handle.rl_run_id,
            size=size,
        )

    except Exception as e:
        logger.error(f"Failed to snapshot environment {request.env_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.post("/restore", response_model=EnvRestoreResponse)
async def restore_environment(request: EnvRestoreRequest) -> EnvRestoreResponse:
    """Restore an environment from a snapshot."""
    snapshot = registry.get_snapshot(request.snapshot_id)
    if not snapshot:
        raise HTTPException(status_code=404, detail=f"Snapshot {request.snapshot_id} not found")

    if snapshot.kind != "env":
        raise HTTPException(
            status_code=422,
            detail=f"Snapshot {request.snapshot_id} is not an environment snapshot",
        )

    try:
        # Load snapshot from volume
        state_dict, meta = storage.load_snapshot(
            rl_run_id=snapshot.rl_run_id,
            kind="env",
            snapshot_id=request.snapshot_id,
        )

        # Recreate environment
        env_name = state_dict.get("name", "crafter")
        name_lower = str(env_name).lower()
        if name_lower == "crafter":
            # Create base environment
            # Recreate classic env from snapshot metadata
            seed_value = state_dict["config"]["seed"]
            metadata = CrafterTaskInstanceMetadata(
                difficulty="normal",
                seed=seed_value,
                num_trees_radius=0,
                num_cows_radius=0,
                num_hostiles_radius=0,
            )
            instance = CrafterTaskInstance(
                id=uuid4(),
                impetus=Impetus(instructions="Restore"),
                intent=Intent(
                    rubric={"goal": "Restore"},
                    gold_trajectories=None,
                    gold_state_diff={},
                ),
                metadata=metadata,
                is_reproducible=True,
                initial_engine_snapshot=None,
            )
            base_env = CrafterClassicEnvironment(task_instance=instance)

            # Deserialize into wrapper
            wrapper = await CrafterEnvironmentWrapper.deserialize(
                payload=state_dict,
                env=base_env,
            )

            # Register new instance
            env_id = registry.register_env(
                env=wrapper,
                seed=wrapper.seed,
                rl_run_id=snapshot.rl_run_id,
                last_observation=wrapper.last_observation,
                last_info=wrapper.last_info,
            )

            # Update step index
            handle = registry.get_env(env_id)
            if handle:
                handle.step_idx = wrapper.step_idx

            return EnvRestoreResponse(
                env_id=env_id,
                observation=wrapper.last_observation or {},
                info=wrapper.last_info,
                step_idx=wrapper.step_idx,
            )
        elif name_lower == "wordle":
            try:
                from synth_ai.environments.examples.wordle.environment import (
                    WordleEnvironment,
                )
                from synth_ai.environments.examples.wordle.taskset import (
                    WordleTaskInstance,
                    WordleTaskInstanceMetadata,
                    create_wordle_taskset,
                )
            except Exception as e:
                raise HTTPException(
                    status_code=500, detail=f"Wordle modules unavailable: {e}"
                ) from e

            cfg = state_dict.get("config", {}) or {}
            word_length = int(cfg.get("word_length", 5))
            max_guesses = int(cfg.get("max_guesses", 6))
            init_snap = cfg.get("initial_engine_snapshot")
            if init_snap is not None:
                metadata = WordleTaskInstanceMetadata(
                    word_length=word_length, max_guesses=max_guesses
                )
                instance = WordleTaskInstance(
                    id=uuid4(),
                    impetus=Impetus(instructions="Restore"),
                    intent=Intent(
                        rubric={"goal": "Restore"},
                        gold_trajectories=None,
                        gold_state_diff={},
                    ),
                    metadata=metadata,
                    is_reproducible=True,
                    initial_engine_snapshot=init_snap,
                )
                base_env = WordleEnvironment(task_instance=instance)
            else:
                ts = await create_wordle_taskset(
                    sample_size=1, word_length=word_length, max_guesses=max_guesses
                )
                instance = ts.instances[0]
                base_env = WordleEnvironment(task_instance=instance)
            # Lazy import of wrapper only when needed
            try:
                from .envs.wordle.environment import WordleEnvironmentWrapper
            except Exception as e:
                raise HTTPException(
                    status_code=500, detail=f"Wordle wrapper unavailable: {e}"
                ) from e
            wrapper = await WordleEnvironmentWrapper.deserialize(payload=state_dict, env=base_env)

            env_id = registry.register_env(
                env=wrapper,
                seed=wrapper.seed,
                rl_run_id=snapshot.rl_run_id,
                last_observation=wrapper.last_observation,
                last_info=wrapper.last_info,
            )
            handle = registry.get_env(env_id)
            if handle:
                handle.step_idx = wrapper.step_idx
            return EnvRestoreResponse(
                env_id=env_id,
                observation=wrapper.last_observation or {},
                info=wrapper.last_info,
                step_idx=wrapper.step_idx,
            )

        elif name_lower == "sokoban":
            try:
                from synth_ai.environments.examples.sokoban.environment import (
                    SokobanEnvironment,
                )
                from synth_ai.environments.examples.sokoban.taskset import (
                    SokobanTaskInstance,
                    SokobanTaskInstanceMetadata,
                )
            except Exception as e:
                raise HTTPException(
                    status_code=500, detail=f"Sokoban modules unavailable: {e}"
                ) from e

            cfg = state_dict.get("config", {}) or {}
            metadata = SokobanTaskInstanceMetadata(difficulty=cfg.get("difficulty", "easy"))
            instance = SokobanTaskInstance(
                id=uuid4(),
                impetus=Impetus(instructions="Restore"),
                intent=Intent(
                    rubric={"goal": "Restore"},
                    gold_trajectories=None,
                    gold_state_diff={},
                ),
                metadata=metadata,
                is_reproducible=True,
                initial_engine_snapshot=cfg.get("initial_state"),
            )
            base_env = SokobanEnvironment(task_instance=instance)
            # Lazy import of wrapper only when needed
            try:
                from .envs.sokoban.environment import SokobanEnvironmentWrapper
            except Exception as e:
                raise HTTPException(
                    status_code=500, detail=f"Sokoban wrapper unavailable: {e}"
                ) from e
            wrapper = await SokobanEnvironmentWrapper.deserialize(payload=state_dict, env=base_env)

            env_id = registry.register_env(
                env=wrapper,
                seed=wrapper.seed,
                rl_run_id=snapshot.rl_run_id,
                last_observation=wrapper.last_observation,
                last_info=wrapper.last_info,
            )
            handle = registry.get_env(env_id)
            if handle:
                handle.step_idx = wrapper.step_idx
            return EnvRestoreResponse(
                env_id=env_id,
                observation=wrapper.last_observation or {},
                info=wrapper.last_info,
                step_idx=wrapper.step_idx,
            )

        else:
            raise HTTPException(
                status_code=422,
                detail=f"Unknown environment name in snapshot: {env_name}",
            )

    except Exception as e:
        logger.error(f"Failed to restore environment from snapshot {request.snapshot_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.post("/terminate", response_model=EnvTerminateResponse)
async def terminate_environment(request: EnvTerminateRequest) -> EnvTerminateResponse:
    """Terminate an environment and clean up resources."""
    handle = registry.get_env(request.env_id)
    if not handle:
        raise HTTPException(status_code=404, detail=f"Environment {request.env_id} not found")

    try:
        # Call terminate on the environment
        await handle.env.terminate()

        # Remove from registry
        registry.remove_env(request.env_id)

        return EnvTerminateResponse(ok=True)

    except Exception as e:
        logger.error(f"Failed to terminate environment {request.env_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e
