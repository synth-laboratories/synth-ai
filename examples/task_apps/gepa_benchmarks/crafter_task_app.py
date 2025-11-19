"""Simplified Crafter task app for GEPA prompt learning benchmarks."""

from __future__ import annotations

import contextlib
import os
import uuid
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any, Mapping, cast
from uuid import uuid4

from fastapi import APIRouter, HTTPException, Request

from synth_ai.environments.examples.crafter_classic.environment import (
    CrafterClassicEnvironment,
)
from synth_ai.environments.examples.crafter_classic.taskset import (
    CrafterTaskInstance,
    CrafterTaskInstanceMetadata,
)
from synth_ai.environments.environment.tools import EnvToolCall
from synth_ai.environments.tasks.core import Impetus, Intent
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

from .common import call_chat_completion

REPO_ROOT = Path(__file__).resolve().parents[3]

crafter_router = APIRouter()

# Import task app code extraction utility (from monorepo, but we'll use a local version)
try:
    from app.routes.prompt_learning.utils.task_app_code_extraction import get_current_module_code
except ImportError:
    # Fallback for synth-ai repo (not in monorepo)
    import inspect
    
    def get_current_module_code():
        """Extract source code for the caller's module using inspect."""
        frame = inspect.currentframe()
        try:
            if frame is None:
                return None
            caller_frame = frame.f_back
            if caller_frame is None:
                return None
            module = inspect.getmodule(caller_frame)
            if module is None:
                return None
            try:
                return inspect.getsource(module)
            except (OSError, TypeError, IOError):
                return None
        finally:
            del frame

@crafter_router.get("/metadata")
async def get_metadata():
    """Return program code and metadata for proposer use.
    
    This endpoint allows task apps to self-extract their own code using inspect,
    keeping the architecture self-contained.
    """
    # Extract code using inspect
    program_code = get_current_module_code()
    
    # Get module path
    import inspect
    frame = inspect.currentframe()
    try:
        if frame is None:
            module_path = None
        else:
            caller_frame = frame.f_back
            if caller_frame is None:
                module_path = None
            else:
                module = inspect.getmodule(caller_frame)
                module_path = module.__name__ if module else None
    finally:
        del frame
    
    return {
        "program_code": program_code,  # Full source code of task app
        "module_path": module_path,    # Module path (e.g., "examples.task_apps.gepa_benchmarks.crafter_task_app")
        "extraction_method": "inspect", # How code was extracted
    }

CRAFTER_DATASET_SPEC = TaskDatasetSpec(
    id="crafter",
    name="Crafter Survival Game",
    version="1.0.0",
    splits=["train", "test"],
    default_split="train",
    description="Crafter survival game with crafting and resource management.",
)

# Common Crafter actions
CRAFTER_ACTIONS = [
    "noop", "move_left", "move_right", "move_up", "move_down",
    "do", "sleep", "place_stone", "place_table", "place_furnace",
    "place_plant", "make_wood_pickaxe", "make_stone_pickaxe", "make_iron_pickaxe",
    "make_wood_sword", "make_stone_sword", "make_iron_sword",
]

# Action mapping: string names to action indices
CRAFTER_ACTION_MAP: dict[str, int] = {
    "noop": 0,
    "move_left": 1,
    "move_right": 2,
    "move_up": 3,
    "move_down": 4,
    "do": 5,
    "sleep": 6,
    "place_stone": 7,
    "place_table": 8,
    "place_furnace": 9,
    "place_plant": 10,
    "make_wood_pickaxe": 11,
    "make_stone_pickaxe": 12,
    "make_iron_pickaxe": 13,
    "make_wood_sword": 14,
    "make_stone_sword": 15,
    "make_iron_sword": 16,
}


def format_crafter_observation(obs: dict[str, Any]) -> str:
    """Format Crafter observation as text for LLM."""
    health = obs.get("health") or obs.get("inventory", {}).get("health", 0)
    inventory = obs.get("inventory", {})
    pos = obs.get("player_position", [0, 0])
    achievements_status = obs.get("achievements_status", {})
    
    # Format inventory (skip health)
    inv_items = [f"{k}:{v}" for k, v in inventory.items() if v > 0 and k != "health"]
    inventory_str = ", ".join(inv_items) if inv_items else "empty"
    
    # Format achievements
    achieved_list = [k for k, v in achievements_status.items() if v]
    achievements_str = ", ".join(achieved_list) if achieved_list else "none"
    
    return f"""Crafter Game State:
- Health: {health}/10
- Hunger: {inventory.get('hunger', 0)}/10
- Position: {pos}
- Inventory: {inventory_str}
- Achievements unlocked: {len(achieved_list)}/22
- Achievements: {achievements_str}

What actions should we take?"""


class CrafterDataset:
    """Lazy loader and sampler for Crafter game states using real environment."""

    def __init__(self) -> None:
        self._cache: dict[str, Any] = {}

    def ensure_ready(self, required_splits: Sequence[str]) -> None:
        pass  # No-op for Crafter

    def size(self, split: str) -> int:
        return 100  # Approximate size

    def sample(self, *, split: str, index: int) -> dict[str, Any]:
        """Create a real Crafter task instance for the given seed/index."""
        # Create task instance with proper seed
        difficulty = "normal"
        seed_value = index
        
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
        task_instance = CrafterTaskInstance(
            id=uuid4(),
            impetus=impetus,
            intent=intent,
            metadata=metadata,
            is_reproducible=True,
            initial_engine_snapshot=None,
        )
        
        # Attach config
        task_instance.config = {"seed": seed_value, "length": 256, "area": [64, 64]}
        
        # Create environment
        env = CrafterClassicEnvironment(task_instance=task_instance)
        
        return {
            "index": index,
            "split": split,
            "task_instance": task_instance,
            "env": env,
        }


def _parse_action(response_text: str) -> str | None:
    """Extract action from response text."""
    if not response_text:
        return None
    
    lower = response_text.lower().strip()
    
    # Remove common prefixes/suffixes
    lower = lower.replace("action:", "").replace("action is", "").replace("action:", "")
    lower = lower.strip()
    
    # Try exact match first (common case: "move_left" or "do")
    for action in CRAFTER_ACTIONS:
        if lower == action.lower() or lower == action:
            return action
    
    # Try to find action tokens (substring match)
    for action in CRAFTER_ACTIONS:
        action_lower = action.lower()
        # Check if action appears as a word boundary
        import re
        pattern = r'\b' + re.escape(action_lower) + r'\b'
        if re.search(pattern, lower):
            return action
    
    # Fallback: try first word if it matches an action
    first_word = lower.split()[0] if lower.split() else ""
    for action in CRAFTER_ACTIONS:
        if action.lower() == first_word:
            return action
    
    # Last resort: return first valid action found (even if partial match)
    for action in CRAFTER_ACTIONS:
        if action.lower() in lower:
            return action
    
    return None


async def rollout_executor(request: RolloutRequest, fastapi_request: Request) -> RolloutResponse:
    dataset: CrafterDataset = fastapi_request.app.state.crafter_dataset

    split = str(((request.env.config or {}).get("split")) or "train")
    seed = request.env.seed or 0
    max_steps = int((request.policy.config or {}).get("max_steps") or 10)

    sample = dataset.sample(split=split, index=seed)
    env: CrafterClassicEnvironment = sample["env"]
    task_instance: CrafterTaskInstance = sample["task_instance"]
    
    # Initialize environment
    raw_obs = await env.initialize()
    observation = raw_obs if isinstance(raw_obs, dict) else (getattr(raw_obs, "observation", raw_obs) if hasattr(raw_obs, "observation") else {})
    obs_dict = observation if isinstance(observation, dict) else {}
    
    # Track achievements
    prev_achievements: set[str] = set()
    if isinstance(obs_dict.get("achievements_status"), dict):
        prev_achievements = {
            k for k, v in obs_dict.get("achievements_status", {}).items() if v
        }
    
    steps: list[RolloutStep] = []
    episode_returns: list[float] = []
    current_obs = obs_dict
    
    # Multi-step rollout loop
    for step_idx in range(max_steps):
        # Format observation for LLM
        observation_text = format_crafter_observation(current_obs)
        
        observation = {
            **current_obs,
            "index": sample["index"],
            "split": sample["split"],
            "step": step_idx,
        }

        placeholders = {
            "observation": observation_text,
        }

        default_messages = [
            {
                "role": "system",
                "pattern": (
                    "You are an agent playing Crafter, a survival game. "
                    "Your goal is to survive, gather resources, craft tools, and unlock achievements. "
                    "Respond with a single action from: noop, move_left, move_right, move_up, move_down, "
                    "do, sleep, place_stone, place_table, place_furnace, place_plant, "
                    "make_wood_pickaxe, make_stone_pickaxe, make_iron_pickaxe, "
                    "make_wood_sword, make_stone_sword, make_iron_sword."
                ),
            },
            {
                "role": "user",
                "pattern": (
                    "{observation}\n\n"
                    "What is the best next action? Respond with a single action name."
                ),
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
        except HTTPException as http_err:
            error_info = {"error": str(http_err.detail), "code": http_err.status_code}
        except Exception as exc:
            error_info = {"error": str(exc)}

        predicted_action = _parse_action(response_text)
        
        # Execute action through the real environment
        reward = 0.0
        env_obs = current_obs
        new_achievements: set[str] = set()
        if predicted_action:
            try:
                # Map action string to action index
                action_idx = CRAFTER_ACTION_MAP.get(predicted_action, 0)
                
                # Execute through environment using EnvToolCall
                tool_call = EnvToolCall(tool="interact", args={"action": action_idx})
                step_result = await env.step(tool_call)
                
                # Get observation from step result (should be a dict)
                env_obs = step_result if isinstance(step_result, dict) else {}
                
                # Debug: Print step details
                with contextlib.suppress(Exception):
                    print(
                        f"[CRAFTER_STEP] step={step_idx} action={predicted_action} (idx={action_idx}) "
                        f"obs_keys={list(env_obs.keys())[:5]} "
                        f"num_steps_taken={env_obs.get('num_steps_taken', 'N/A')}",
                        flush=True,
                    )
                
                # Check for new achievements
                current_achievements: set[str] = set()
                if isinstance(env_obs.get("achievements_status"), dict):
                    current_achievements = {
                        k for k, v in env_obs.get("achievements_status", {}).items() if v
                    }
                    
                    # Debug: Print achievement changes
                    with contextlib.suppress(Exception):
                        if current_achievements != prev_achievements:
                            print(
                                f"[CRAFTER_ACH] step={step_idx} prev={len(prev_achievements)} "
                                f"current={len(current_achievements)} new={current_achievements - prev_achievements}",
                                flush=True,
                            )
                
                new_achievements = current_achievements - prev_achievements
                
                # Reward = number of new achievements unlocked
                reward = float(len(new_achievements))
                
                prev_achievements = current_achievements
                current_obs = env_obs
            except Exception as env_exc:
                # If environment execution fails, log but continue
                error_info["env_error"] = str(env_exc)
                reward = 0.0
        else:
            # No valid action parsed
            reward = 0.0

        episode_returns.append(reward)

        info_payload = {
            "predicted_action": predicted_action,
            "response_text": response_text,
            "response_json": response_json,
            "step": step_idx,
            "new_achievements": list(new_achievements),
            **error_info,
        }

        # Check termination: done if max steps reached OR environment terminated
        is_done = False
        if step_idx >= max_steps - 1:
            is_done = True
        elif env_obs.get("terminated", False) or env_obs.get("truncated", False):
            is_done = True
        
        step = RolloutStep(
            obs=observation,
            tool_calls=[],
            reward=reward,
            done=is_done,
            info=info_payload,
        )
        steps.append(step)
        
        # Break early if terminated
        if is_done:
            break
    
    # Calculate final unique achievements count (this is the return)
    final_achievements: set[str] = set()
    if isinstance(current_obs.get("achievements_status"), dict):
        final_achievements = {
            k for k, v in current_obs.get("achievements_status", {}).items() if v
        }
    num_unique_achievements = len(final_achievements)

    with contextlib.suppress(Exception):
        print(
            f"[CRAFTER_ROLLOUT] run_id={request.run_id} split={sample['split']} "
            f"index={sample['index']} steps={len(steps)} unique_achievements={num_unique_achievements}",
            flush=True,
    )

    inference_url = (request.policy.config or {}).get("inference_url")
    trajectory = RolloutTrajectory(
        env_id=f"crafter::{sample['split']}::{sample['index']}",
        policy_id=request.policy.policy_id or request.policy.policy_name or "policy",
        steps=steps,
        final={"observation": current_obs, "reward": num_unique_achievements},
        length=len(steps),
        inference_url=str(inference_url or ""),
    )

    # Mean return = number of unique achievements (this is what we optimize for)
    metrics = RolloutMetrics(
        episode_returns=episode_returns,
        mean_return=float(num_unique_achievements),  # Return = unique achievements count
        num_steps=len(steps),
        num_episodes=1,
        outcome_score=float(num_unique_achievements),
        events_score=sum(episode_returns),  # Sum of achievement deltas
        details={
            "unique_achievements": num_unique_achievements,
            "achievements": list(final_achievements),
            "steps": len(steps),
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
            "events_count": len(steps),
            "decision_rewards": episode_returns,
            "metadata": {
                "env": "crafter",
                "split": sample["split"],
                "index": sample["index"],
                "total_steps": len(steps),
            },
        }

    return RolloutResponse(
        run_id=request.run_id,
        trajectories=[trajectory],
        branches={},
        metrics=metrics,
        aborted=False,
        ops_executed=len(steps) * 2,
        trace=trace_payload,
    )


def build_dataset() -> tuple[TaskDatasetRegistry, CrafterDataset]:
    registry = TaskDatasetRegistry()
    dataset = CrafterDataset()
    dataset.ensure_ready(["train"])
    registry.register(CRAFTER_DATASET_SPEC, lambda _spec: dataset, cache=True)
    return registry, dataset


def _base_task_info() -> TaskInfo:
    return TaskInfo(
        task={
            "id": "crafter",
            "name": "Crafter Survival Game",
            "version": "1.0.0",
            "action_space": {
                "type": "discrete",
                "description": f"Discrete action space with {len(CRAFTER_ACTIONS)} actions.",
                "actions": CRAFTER_ACTIONS,
            },
        },
        environment="crafter",
        dataset={
            **CRAFTER_DATASET_SPEC.model_dump(),
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
        limits={"max_turns": 15},
        task_metadata={"format": "Single action name"},
    )


def describe_taskset(dataset: CrafterDataset) -> Mapping[str, Any]:
    return {
        **CRAFTER_DATASET_SPEC.model_dump(),
        "sizes": {split: dataset.size(split) for split in ["train", "test"]},
    }


def provide_task_instances(dataset: CrafterDataset, seeds: Sequence[int]) -> Iterable[TaskInfo]:
    base_info = _base_task_info()
    for seed in seeds:
        sample = dataset.sample(split="train", index=seed)
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
                "seed": seed,
            },
        )


OUTCOME_RUBRIC: Rubric = cast(
    Rubric,
    load_rubric(
        {
            "version": "1",
            "goal_text": "Select the best next action for survival and progress in Crafter.",
            "aggregation": "weighted_sum",
            "criteria": [
                {
                    "id": "action_quality",
                    "description": "Action is valid and appropriate for the current game state.",
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
            "goal_text": "Respond with a valid action name.",
            "aggregation": "weighted_sum",
            "criteria": [
                {
                    "id": "action_validity",
                    "description": "Response contains a valid Crafter action.",
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
        system_hint="Respond with a single action name.",
    )

    config = TaskAppConfig(
        app_id="crafter-gepa",
        name="Crafter Survival Game Task (GEPA)",
        description="Simplified Crafter task app for GEPA prompt learning.",
        base_task_info=base_info,
        describe_taskset=lambda: describe_taskset(dataset),
        provide_task_instances=lambda seeds: provide_task_instances(dataset, seeds),
        rollout=rollout_executor,
        dataset_registry=registry,
        rubrics=RubricBundle(outcome=OUTCOME_RUBRIC, events=EVENTS_RUBRIC),
        proxy=proxy_config,
        routers=(crafter_router,),
        app_state={"crafter_dataset": dataset},
        cors_origins=["*"],
    )
    return config


register_task_app(
    entry=TaskAppEntry(
        app_id="crafter",
        description="Simplified Crafter task app for GEPA prompt learning benchmarks.",
        config_factory=build_config,
        aliases=("crafter-gepa", "crafter-gepa-benchmark"),
        modal=ModalDeploymentConfig(
            app_name="synth-crafter-gepa",
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


if __name__ == "__main__":
    import argparse
    from synth_ai.task.server import run_task_app

    parser = argparse.ArgumentParser(description="Run the Crafter GEPA task app locally")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8116)
    parser.add_argument("--reload", action="store_true", help="Enable uvicorn autoreload")
    parser.add_argument(
        "--env-file",
        action="append",
        default=[],
        help="Additional .env files to load before startup",
    )
    args = parser.parse_args()

    default_env = Path(__file__).resolve().parents[3] / ".env"
    env_files = [str(default_env)] if default_env.exists() else []
    env_files.extend(args.env_file or [])

    run_task_app(
        build_config,
        host=args.host,
        port=args.port,
        reload=args.reload,
        env_files=env_files,
    )



