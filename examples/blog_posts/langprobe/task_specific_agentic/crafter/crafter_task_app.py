"""Simplified Crafter task app for GEPA prompt learning benchmarks."""

from __future__ import annotations

import contextlib
import os
import uuid
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any, Mapping, cast

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

import sys
from pathlib import Path

# Add gepa_benchmarks to path
_gepa_benchmarks_path = Path(__file__).resolve().parents[4] / "task_apps" / "gepa_benchmarks"
sys.path.insert(0, str(_gepa_benchmarks_path))
from common import call_chat_completion

REPO_ROOT = Path(__file__).resolve().parents[5]

crafter_router = APIRouter()

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


class CrafterDataset:
    """Lazy loader and sampler for Crafter game states."""

    def __init__(self) -> None:
        self._cache: dict[str, Any] = {}

    def ensure_ready(self, required_splits: Sequence[str]) -> None:
        pass  # No-op for Crafter

    def size(self, split: str) -> int:
        return 100  # Approximate size

    def sample(self, *, split: str, index: int) -> dict[str, Any]:
        # For GEPA, we'll use a simplified observation format
        # In a real implementation, this would initialize a Crafter environment
        # and sample an observation
        
        # Simplified observation format
        observation_text = (
            f"Game state at step {index}:\n"
            f"- Health: 8/9\n"
            f"- Food: 7/9\n"
            f"- Drink: 6/9\n"
            f"- Inventory: wood=5, stone=3, coal=2, iron=1\n"
            f"- Achievements unlocked: collect_wood, collect_stone\n"
            f"- Nearby resources: tree, stone, water\n"
        )
        
        return {
            "index": index,
            "split": split,
            "observation_text": observation_text,
            "health": 8,
            "food": 7,
            "drink": 6,
            "inventory": {"wood": 5, "stone": 3, "coal": 2, "iron": 1},
            "achievements": ["collect_wood", "collect_stone"],
            "nearby_resources": ["tree", "stone", "water"],
        }


def _parse_action(response_text: str) -> str | None:
    """Extract action from response text."""
    if not response_text:
        return None
    
    lower = response_text.lower().strip()
    
    # Try to find action tokens
    for action in CRAFTER_ACTIONS:
        if action.lower() in lower:
            return action
    
    return None


async def rollout_executor(request: RolloutRequest, fastapi_request: Request) -> RolloutResponse:
    dataset: CrafterDataset = fastapi_request.app.state.crafter_dataset

    split = str(((request.env.config or {}).get("split")) or "train")
    seed = request.env.seed or 0

    sample = dataset.sample(split=split, index=seed)
    observation_text = sample["observation_text"]
    inventory = sample["inventory"]
    achievements = sample["achievements"]
    
    observation = {
        "observation_text": observation_text,
        "inventory": inventory,
        "achievements": achievements,
        "index": sample["index"],
        "split": sample["split"],
    }

    placeholders = {
        "observation": observation_text,
        "inventory": str(inventory),
        "achievements": ", ".join(achievements),
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
    
    # Evaluate by checking if action is valid and appropriate
    reward = 0.0
    if predicted_action:
        # Reward valid actions based on context
        if predicted_action in ["make_wood_pickaxe", "make_stone_pickaxe", "make_iron_pickaxe"]:
            # Crafting tools is good if we have resources
            if predicted_action == "make_wood_pickaxe" and inventory.get("wood", 0) >= 1:
                reward = 0.8
            elif predicted_action == "make_stone_pickaxe" and inventory.get("stone", 0) >= 1:
                reward = 0.9
            elif predicted_action == "make_iron_pickaxe" and inventory.get("iron", 0) >= 1:
                reward = 1.0
            else:
                reward = 0.3  # Valid but can't craft
        elif predicted_action in ["do", "move_left", "move_right", "move_up", "move_down"]:
            reward = 0.5  # Basic actions
        elif predicted_action == "noop":
            reward = 0.1  # Idle
        else:
            reward = 0.4  # Other valid actions
    else:
        reward = 0.0

    info_payload = {
        "predicted_action": predicted_action,
        "response_text": response_text,
        "response_json": response_json,
        **error_info,
    }

    with contextlib.suppress(Exception):
        print(
            f"[CRAFTER_ROLLOUT] run_id={request.run_id} split={sample['split']} "
            f"index={sample['index']} action={predicted_action} reward={reward}",
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
        env_id=f"crafter::{sample['split']}::{sample['index']}",
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
        details={"action": predicted_action},
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
                "env": "crafter",
                "split": sample["split"],
                "index": sample["index"],
                "action": predicted_action,
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
        limits={"max_turns": 1},
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
                "observation": sample["observation_text"][:200],
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
        app_id="crafter-gepa",
        description="Simplified Crafter task app for GEPA prompt learning benchmarks.",
        config_factory=build_config,
        aliases=("crafter-gepa-benchmark",),
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

    default_env = Path(__file__).resolve().parents[5] / ".env"
    env_files = [str(default_env)] if default_env.exists() else []
    env_files.extend(args.env_file or [])

    run_task_app(
        build_config,
        host=args.host,
        port=args.port,
        reload=args.reload,
        env_files=env_files,
    )


