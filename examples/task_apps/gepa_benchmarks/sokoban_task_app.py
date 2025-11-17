"""Simplified Sokoban task app for GEPA prompt learning benchmarks."""

from __future__ import annotations

import contextlib
import os
import uuid
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any, Mapping, cast

from fastapi import APIRouter, HTTPException, Request

from synth_ai.environments.examples.sokoban.taskset import (
    SokobanTaskInstance,
    SokobanTaskSet,
    create_task_instance_from_seed,
)
from synth_ai.environments.examples.sokoban.environment import SokobanEnvironment
from synth_ai.environments.environment.tools import EnvToolCall
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
import asyncio

from .common import call_chat_completion

REPO_ROOT = Path(__file__).resolve().parents[3]

sokoban_router = APIRouter()

SOKOBAN_DATASET_SPEC = TaskDatasetSpec(
    id="sokoban",
    name="Sokoban Puzzle",
    version="1.0.0",
    splits=["train", "test"],
    default_split="train",
    description="Sokoban puzzle solving with grid-based navigation.",
)

ACTION_ID_TO_NAME = {0: "left", 1: "up", 2: "right", 3: "down"}
ACTION_TOKEN_TO_ID = {
    "0": 0, "1": 1, "2": 2, "3": 3,
    "left": 0, "move_left": 0, "west": 0, "l": 0,
    "up": 1, "move_up": 1, "north": 1, "u": 1,
    "right": 2, "move_right": 2, "east": 2, "r": 2,
    "down": 3, "move_down": 3, "south": 3, "d": 3,
}


class SokobanDataset:
    """Lazy loader and sampler for Sokoban puzzles."""

    def __init__(self) -> None:
        self._taskset = SokobanTaskSet()

    def ensure_ready(self, required_splits: Sequence[str]) -> None:
        pass  # No-op for Sokoban

    def size(self, split: str) -> int:
        return 100  # Approximate size

    async def sample(self, *, split: str, index: int) -> dict[str, Any]:
        difficulty = "easy"
        instance: SokobanTaskInstance = await create_task_instance_from_seed(difficulty, int(index))
        env = SokobanEnvironment(instance)
        obs = await env.initialize()
        
        grid = obs.get("room_text", "")
        boxes_on_target = obs.get("boxes_on_target", 0)
        num_boxes = obs.get("num_boxes", boxes_on_target)
        
        return {
            "index": index,
            "split": split,
            "grid": grid,
            "boxes_on_target": boxes_on_target,
            "num_boxes": num_boxes,
            "difficulty": difficulty,
            "instance": instance,
            "env": env,
        }


def _parse_action(response_text: str) -> int | None:
    """Extract action from response text."""
    if not response_text:
        return None
    
    lower = response_text.lower().strip()
    
    # Remove common prefixes
    lower = lower.replace("action:", "").replace("action is", "").replace("move", "").strip()
    
    # Try exact word match first
    for token, action_id in ACTION_TOKEN_TO_ID.items():
        import re
        pattern = r'\b' + re.escape(token) + r'\b'
        if re.search(pattern, lower):
            return action_id
    
    # Try substring match
    for token, action_id in ACTION_TOKEN_TO_ID.items():
        if token in lower:
            return action_id
    
    # Try to find digits (0-3)
    import re
    digits = re.findall(r'\b[0-3]\b', lower)
    if digits:
        return int(digits[0])
    
    # Fallback: try first word
    first_word = lower.split()[0] if lower.split() else ""
    for token, action_id in ACTION_TOKEN_TO_ID.items():
        if token == first_word:
            return action_id
    
    return None


async def rollout_executor(request: RolloutRequest, fastapi_request: Request) -> RolloutResponse:
    dataset: SokobanDataset = fastapi_request.app.state.sokoban_dataset

    split = str(((request.env.config or {}).get("split")) or "train")
    seed = request.env.seed or 0
    max_steps = int((request.policy.config or {}).get("max_steps") or 15)

    sample = await dataset.sample(split=split, index=seed)
    env = sample["env"]
    
    # Initialize state
    initial_obs = await env.initialize()
    grid = initial_obs.get("grid", sample["grid"])
    boxes_on_target = initial_obs.get("boxes_on_target", sample["boxes_on_target"])
    num_boxes = initial_obs.get("num_boxes", sample["num_boxes"])
    
    steps: list[RolloutStep] = []
    total_reward = 0.0
    episode_returns: list[float] = []
    current_obs = initial_obs
    
    # Multi-step rollout loop
    for step_idx in range(max_steps):
        observation = {
            "grid": grid,
            "boxes_on_target": boxes_on_target,
            "num_boxes": num_boxes,
            "index": sample["index"],
            "split": sample["split"],
            "step": step_idx,
        }

        placeholders = {
            "grid": grid,
            "boxes_on_target": boxes_on_target,
            "num_boxes": num_boxes,
        }

        default_messages = [
            {
                "role": "system",
                "pattern": (
                    "You are an agent playing Sokoban. The grid uses characters: "
                    "'#' wall, '_' floor, 'O' box, '√' box on target, 'X' target, 'P' player. "
                    "Respond with a single action: left (0), up (1), right (2), or down (3)."
                ),
            },
            {
                "role": "user",
                "pattern": (
                    "Grid:\n{grid}\n\n"
                    "Boxes on target: {boxes_on_target} / {num_boxes}\n\n"
                    "What is the best next action? Respond with a single action (left/up/right/down or 0/1/2/3)."
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
        
        # Execute action through the real environment to get actual rewards
        reward = 0.0
        env_obs = current_obs
        if predicted_action is not None:
            try:
                # Execute through environment - this will check actual correctness
                env_obs = await env.step(EnvToolCall(tool="interact", args={"action": predicted_action}))
                
                # Extract reward from environment observation
                # Sokoban gives: -0.01 per step (penalty) + 1.0 when solved
                reward = float(env_obs.get("reward_last", 0.0))
                
                # Update state from environment observation
                boxes_on_target = env_obs.get("boxes_on_target", boxes_on_target)
                grid = env_obs.get("room_text", grid)  # room_text is the grid representation
                current_obs = env_obs
                
            except Exception as env_exc:
                # If environment execution fails, log but continue
                error_info["env_error"] = str(env_exc)
                reward = 0.0
        else:
            # No valid action parsed
            reward = 0.0

        total_reward += reward
        episode_returns.append(reward)

        info_payload = {
            "predicted_action": predicted_action,
            "action_name": ACTION_ID_TO_NAME.get(predicted_action) if predicted_action is not None else None,
            "response_text": response_text,
            "response_json": response_json,
            "step": step_idx,
            **error_info,
        }

        # Check termination: done if puzzle solved OR max steps reached
        is_done = False
        if step_idx >= max_steps - 1:
            is_done = True
        elif env_obs.get("terminated", False):
            # Puzzle solved (boxes_on_target == num_boxes)
            is_done = True
        
        step = RolloutStep(
            obs=observation,
            tool_calls=[],
            reward=reward,
            done=is_done,
            info=info_payload,
        )
        steps.append(step)
        
        # Break early if puzzle solved
        if is_done and env_obs.get("terminated", False):
            break

    with contextlib.suppress(Exception):
        print(
            f"[SOKOBAN_ROLLOUT] run_id={request.run_id} split={sample['split']} "
            f"index={sample['index']} steps={len(steps)} total_reward={total_reward:.2f}",
            flush=True,
        )

    inference_url = (request.policy.config or {}).get("inference_url")
    trajectory = RolloutTrajectory(
        env_id=f"sokoban::{sample['split']}::{sample['index']}",
        policy_id=request.policy.policy_id or request.policy.policy_name or "policy",
        steps=steps,
        final={"observation": current_obs, "reward": total_reward},
        length=len(steps),
        inference_url=str(inference_url or ""),
    )

    metrics = RolloutMetrics(
        episode_returns=episode_returns,
        mean_return=total_reward / len(steps) if steps else 0.0,
        num_steps=len(steps),
        num_episodes=1,
        outcome_score=total_reward,
        events_score=total_reward,
        details={"total_reward": total_reward, "steps": len(steps), "final_boxes_on_target": boxes_on_target},
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
                "env": "sokoban",
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


def build_dataset() -> tuple[TaskDatasetRegistry, SokobanDataset]:
    registry = TaskDatasetRegistry()
    dataset = SokobanDataset()
    dataset.ensure_ready(["train"])
    registry.register(SOKOBAN_DATASET_SPEC, lambda _spec: dataset, cache=True)
    return registry, dataset


def _base_task_info() -> TaskInfo:
    return TaskInfo(
        task={
            "id": "sokoban",
            "name": "Sokoban Puzzle",
            "version": "1.0.0",
            "action_space": {
                "type": "discrete",
                "description": "Single action: left (0), up (1), right (2), or down (3).",
            },
        },
        environment="sokoban",
        dataset={
            **SOKOBAN_DATASET_SPEC.model_dump(),
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
        task_metadata={"format": "Single action (left/up/right/down or 0/1/2/3)"},
    )


def describe_taskset(dataset: SokobanDataset) -> Mapping[str, Any]:
    return {
        **SOKOBAN_DATASET_SPEC.model_dump(),
        "sizes": {split: dataset.size(split) for split in ["train", "test"]},
    }


async def _provide_task_instances_async(dataset: SokobanDataset, seeds: Sequence[int]) -> list[TaskInfo]:
    base_info = _base_task_info()
    infos = []
    for seed in seeds:
        sample = await dataset.sample(split="train", index=seed)
        infos.append(TaskInfo(
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
                "grid": sample["grid"],
            },
        ))
    return infos


OUTCOME_RUBRIC: Rubric = cast(
    Rubric,
    load_rubric(
        {
            "version": "1",
            "goal_text": "Select the best next action to solve the Sokoban puzzle.",
            "aggregation": "weighted_sum",
            "criteria": [
                {
                    "id": "action_quality",
                    "description": "Action improves puzzle state (more boxes on target).",
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
            "goal_text": "Respond with a valid action.",
            "aggregation": "weighted_sum",
            "criteria": [
                {
                    "id": "action_validity",
                    "description": "Response contains a valid action (left/up/right/down or 0/1/2/3).",
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
        system_hint="Respond with a single action: left, up, right, or down.",
    )

    config = TaskAppConfig(
        app_id="sokoban-gepa",
        name="Sokoban Puzzle Task (GEPA)",
        description="Simplified Sokoban task app for GEPA prompt learning.",
        base_task_info=base_info,
        describe_taskset=lambda: describe_taskset(dataset),
        provide_task_instances=lambda seeds: asyncio.run(_provide_task_instances_async(dataset, seeds)),
        rollout=rollout_executor,
        dataset_registry=registry,
        rubrics=RubricBundle(outcome=OUTCOME_RUBRIC, events=EVENTS_RUBRIC),
        proxy=proxy_config,
        routers=(sokoban_router,),
        app_state={"sokoban_dataset": dataset},
        cors_origins=["*"],
    )
    return config


register_task_app(
    entry=TaskAppEntry(
        app_id="sokoban",
        description="Simplified Sokoban task app for GEPA prompt learning benchmarks.",
        config_factory=build_config,
        aliases=("sokoban-gepa", "sokoban-gepa-benchmark"),
        modal=ModalDeploymentConfig(
            app_name="synth-sokoban-gepa",
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

    parser = argparse.ArgumentParser(description="Run the Sokoban GEPA task app locally")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8114)
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

