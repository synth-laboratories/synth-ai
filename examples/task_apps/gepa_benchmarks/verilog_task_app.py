"""Simplified Verilog task app for GEPA prompt learning benchmarks."""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
import re
import uuid
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any, Mapping, cast

from fastapi import APIRouter, HTTPException, Request

from synth_ai.environments.examples.verilog.environment import VerilogEnvironment
from synth_ai.environments.examples.verilog.taskset import (
    VerilogTaskInstance,
    create_verilog_taskset,
)
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

verilog_router = APIRouter()

VERILOG_DATASET_SPEC = TaskDatasetSpec(
    id="verilog_eval_v2",
    name="VerilogEval Spec-to-RTL",
    version="1.0.0",
    splits=["train", "val", "test"],
    default_split="val",
    description="Spec-to-RTL problems sourced from the VerilogEval v2 benchmark.",
)

MAX_INSTANCES = 10
TOOLS = ["write_file", "compile", "simulate", "submit"]


def _load_taskset_blocking(max_instances: int):
    try:
        return asyncio.run(create_verilog_taskset(max_instances=max_instances))
    except RuntimeError:
        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(create_verilog_taskset(max_instances=max_instances))
        finally:
            loop.close()


class VerilogDataset:
    """Lazy loader and sampler for Verilog problems."""

    def __init__(self) -> None:
        self._taskset = _load_taskset_blocking(MAX_INSTANCES)
        self.instances: list[VerilogTaskInstance] = list(self._taskset.instances)

    def ensure_ready(self, required_splits: Sequence[str]) -> None:
        pass

    def size(self, split: str) -> int:
        return len(self.instances)

    async def sample(self, *, split: str, index: int) -> dict[str, Any]:
        if not self.instances:
            raise RuntimeError("Verilog dataset is empty")
        idx = int(index) % len(self.instances)
        instance = self.instances[idx]
        
        instructions = getattr(getattr(instance, "impetus", None), "instructions", "")
        env = VerilogEnvironment(task_instance=instance)
        obs = await env.initialize()
        
        files = obs.get("files", {})
        file_preview = "\n\n".join(
            f"{name}:\n{(content[:600] + '...' if len(content) > 600 else content)}"
            for name, content in sorted(files.items())
        )
        
        return {
            "index": idx,
            "split": split,
            "instructions": instructions,
            "files": files,
            "file_preview": file_preview,
            "instance": instance,
            "env": env,
        }


JSON_BLOCK_PATTERN = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL)


def _parse_tool_json(text: str) -> dict[str, Any] | None:
    """Parse tool call from JSON response."""
    candidates = []
    
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            candidates.append(parsed)
    except Exception:
        pass
    
    if not candidates:
        for match in JSON_BLOCK_PATTERN.finditer(text):
            snippet = match.group(1)
            try:
                parsed = json.loads(snippet)
            except Exception:
                continue
            if isinstance(parsed, dict):
                candidates.append(parsed)
    
    if not candidates:
        brace_match = re.search(r"\{.*\}", text, re.DOTALL)
        if brace_match:
            try:
                parsed = json.loads(brace_match.group(0))
                if isinstance(parsed, dict):
                    candidates.append(parsed)
            except Exception:
                pass
    
    for candidate in candidates:
        tool_name = candidate.get("tool") if isinstance(candidate, dict) else None
        if isinstance(tool_name, str) and tool_name.strip() in TOOLS:
            raw_args = candidate.get("args") if isinstance(candidate, dict) else None
            args = raw_args if isinstance(raw_args, dict) else {}
            return {"tool": tool_name.strip(), "args": args}
    
    return None


async def rollout_executor(request: RolloutRequest, fastapi_request: Request) -> RolloutResponse:
    dataset: VerilogDataset = fastapi_request.app.state.verilog_dataset

    split = str(((request.env.config or {}).get("split")) or "val")
    seed = request.env.seed or 0
    max_steps = int((request.policy.config or {}).get("max_steps") or 6)

    sample = await dataset.sample(split=split, index=seed)
    env = sample["env"]
    
    # Initialize state
    initial_obs = await env.initialize()
    instructions = sample["instructions"]
    files = dict(initial_obs.get("files", sample["files"]))
    
    steps: list[RolloutStep] = []
    total_reward = 0.0
    episode_returns: list[float] = []
    current_obs = initial_obs
    files_written = 0
    compiled = False
    simulated = False
    
    # Multi-step rollout loop
    for step_idx in range(max_steps):
        # Build file preview from current files
        file_preview = "\n\n".join(
            f"{name}:\n{(content[:600] + '...' if len(content) > 600 else content)}"
            for name, content in sorted(files.items())
        )
        
        # Build status messages from environment observation
        compile_status = current_obs.get("compile_status", "")
        simulate_status = current_obs.get("simulate_status", "")
        status_summary = ""
        if compile_status:
            status_summary += f"{compile_status}\n"
        if simulate_status:
            status_summary += f"{simulate_status}\n"
        
        observation = {
            "instructions": instructions,
            "files": dict(files),
            "index": sample["index"],
            "split": sample["split"],
            "step": step_idx,
            "files_written": files_written,
            "compiled": compiled,
            "simulated": simulated,
            "compile_status": compile_status,
            "simulate_status": simulate_status,
            "task_completed": current_obs.get("task_completed", False),
        }

        placeholders = {
            "instructions": instructions,
            "files": file_preview,
            "status": status_summary.strip() or "No previous operations.",
        }

        default_messages = [
            {
                "role": "system",
                "pattern": (
                    "You are an expert digital design engineer helping with Verilog spec-to-RTL tasks. "
                    "Choose between these tools: write_file, compile, simulate, submit. "
                    "Respond with a JSON object: {\"tool\": \"<tool_name>\", \"args\": {...}}."
                ),
            },
            {
                "role": "user",
                "pattern": (
                    "Task instructions:\n{instructions}\n\n"
                    "Workspace files:\n{files}\n\n"
                    "Previous operations:\n{status}\n\n"
                    "What is the best next tool call? Respond with JSON: {\"tool\": \"<tool_name>\", \"args\": {...}}."
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

        tool_call = _parse_tool_json(response_text)
        
        # Execute tool call through the real environment
        reward = 0.0
        env_obs = current_obs
        if tool_call:
            try:
                from synth_ai.environments.environment.tools import EnvToolCall
                
                # Convert tool_call dict to EnvToolCall
                env_tool_call = EnvToolCall(
                    tool=tool_call.get("tool", ""),
                    args=tool_call.get("args", {})
                )
                
                # Execute through environment
                env_obs = await env.step([env_tool_call])
                
                # ONLY reward successful submission (tests passed)
                # Ignore compile and simulate rewards
                tool_name = tool_call.get("tool", "")
                if tool_name == "submit":
                    # Only give reward if submission passed (tests passed)
                    if env_obs.get("task_completed", False):
                        reward = 1.0  # Full reward only for correct submission
                    else:
                        reward = 0.0  # No reward if tests failed
                else:
                    # No reward for write_file, compile, or simulate
                    reward = 0.0
                
                # Update state tracking based on tool execution
                if tool_name == "write_file":
                    files_written += 1
                    # Update files dict from environment observation
                    env_files = env_obs.get("files", {})
                    if env_files:
                        files.update(env_files)
                elif tool_name == "compile":
                    compiled = env_obs.get("compile_status", "").startswith("Last compile: Success")
                elif tool_name == "simulate":
                    simulated = env_obs.get("simulate_status", "").startswith("Last simulation: Passed")
                
                # Check if task completed (tests passed)
                task_completed = env_obs.get("task_completed", False)
                
            except Exception as env_exc:
                # If environment execution fails, log but continue
                error_info["env_error"] = str(env_exc)
                reward = 0.0
        else:
            reward = 0.0

        total_reward += reward
        episode_returns.append(reward)

        # Update current observation with environment state
        current_obs = env_obs
        files = env_obs.get("files", files)  # Update files from env

        info_payload = {
            "tool_call": tool_call,
            "response_text": response_text,
            "response_json": response_json,
            "step": step_idx,
            "reward_last": reward,
            "total_reward": env_obs.get("total_reward", total_reward),
            "task_completed": env_obs.get("task_completed", False),
            **error_info,
        }

        # Check termination: done if task completed OR max steps reached
        is_done = False
        if step_idx >= max_steps - 1:
            is_done = True
        elif env_obs.get("task_completed", False):
            # Task completed successfully (tests passed)
            is_done = True
        elif env_obs.get("terminated", False):
            # Environment terminated (e.g., submitted)
            is_done = True
        
        step = RolloutStep(
            obs=observation,
            tool_calls=[],
            reward=reward,
            done=is_done,
            info=info_payload,
        )
        steps.append(step)
        
        # Break early if done
        if is_done:
            break

    with contextlib.suppress(Exception):
        print(
            f"[VERILOG_ROLLOUT] run_id={request.run_id} split={sample['split']} "
            f"index={sample['index']} steps={len(steps)} total_reward={total_reward:.2f}",
            flush=True,
    )

    inference_url = (request.policy.config or {}).get("inference_url")
    trajectory = RolloutTrajectory(
        env_id=f"verilog::{sample['split']}::{sample['index']}",
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
        details={"total_reward": total_reward, "steps": len(steps), "files_written": files_written},
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
                "env": "verilog",
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


def build_dataset() -> tuple[TaskDatasetRegistry, VerilogDataset]:
    registry = TaskDatasetRegistry()
    dataset = VerilogDataset()
    dataset.ensure_ready(["val"])
    registry.register(VERILOG_DATASET_SPEC, lambda _spec: dataset, cache=True)
    return registry, dataset


def _base_task_info() -> TaskInfo:
    return TaskInfo(
        task={
            "id": "verilog_eval_v2",
            "name": "VerilogEval Spec-to-RTL",
            "version": "1.0.0",
            "action_space": {
                "type": "tool_calls",
                "tools": TOOLS,
                "description": "Tools: write_file, compile, simulate, submit.",
            },
        },
        environment="verilog",
        dataset={
            **VERILOG_DATASET_SPEC.model_dump(),
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
        limits={"max_turns": 6},
        task_metadata={"format": "JSON: {\"tool\": \"<tool_name>\", \"args\": {...}}"},
    )


def describe_taskset(dataset: VerilogDataset) -> Mapping[str, Any]:
    return {
        **VERILOG_DATASET_SPEC.model_dump(),
        "sizes": {split: dataset.size(split) for split in ["train", "val", "test"]},
    }


async def _provide_task_instances_async(dataset: VerilogDataset, seeds: Sequence[int]) -> list[TaskInfo]:
    base_info = _base_task_info()
    infos = []
    for seed in seeds:
        sample = await dataset.sample(split="val", index=seed)
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
                "instructions": sample["instructions"][:200],
            },
        ))
    return infos


OUTCOME_RUBRIC: Rubric = cast(
    Rubric,
    load_rubric(
        {
            "version": "1",
            "goal_text": "Select the best next tool call for the Verilog task.",
            "aggregation": "weighted_sum",
            "criteria": [
                {
                    "id": "tool_selection",
                    "description": "Tool call is valid and appropriate for the current state.",
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
            "goal_text": "Respond with valid JSON tool call.",
            "aggregation": "weighted_sum",
            "criteria": [
                {
                    "id": "json_validity",
                    "description": "Response contains valid JSON with tool and args.",
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
        system_hint="Respond with JSON: {\"tool\": \"<tool_name>\", \"args\": {...}}.",
    )

    config = TaskAppConfig(
        app_id="verilog-gepa",
        name="Verilog Spec-to-RTL Task (GEPA)",
        description="Simplified Verilog task app for GEPA prompt learning.",
        base_task_info=base_info,
        describe_taskset=lambda: describe_taskset(dataset),
        provide_task_instances=lambda seeds: asyncio.run(_provide_task_instances_async(dataset, seeds)),
        rollout=rollout_executor,
        dataset_registry=registry,
        rubrics=RubricBundle(outcome=OUTCOME_RUBRIC, events=EVENTS_RUBRIC),
        proxy=proxy_config,
        routers=(verilog_router,),
        app_state={"verilog_dataset": dataset},
        cors_origins=["*"],
    )
    return config


register_task_app(
    entry=TaskAppEntry(
        app_id="verilog",
        description="Simplified Verilog task app for GEPA prompt learning benchmarks.",
        config_factory=build_config,
        aliases=("verilog-gepa", "verilog-gepa-benchmark"),
        modal=ModalDeploymentConfig(
            app_name="synth-verilog-gepa",
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

    parser = argparse.ArgumentParser(description="Run the Verilog GEPA task app locally")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8115)
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

