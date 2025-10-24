"""Task App configuration for the GRPO Verilog spec-to-RTL example."""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence

import httpx

from synth_ai.environments.environment.tools import EnvToolCall
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
    RolloutStep,
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


def _resolve_repo_root() -> Path:
    """Find synth-ai repo root, checking env var and parent traversal."""
    candidates: list[Path] = []
    env_root = os.getenv("SYNTH_AI_REPO_ROOT")
    if env_root:
        candidates.append(Path(env_root).expanduser())
    
    # Try Modal mount point
    candidates.append(Path("/opt/synth_ai_repo"))
    
    # Traverse up from current file
    current = _HERE
    for _ in range(6):
        current = current.parent
        candidates.append(current)
        if (current / "synth_ai").is_dir() and (current / "examples").is_dir():
            return current
    
    # Return first existing candidate
    for candidate in candidates:
        if candidate.is_dir() and (candidate / "synth_ai").exists():
            return candidate
    
    # Fallback to current parent structure (may not work in Modal)
    return _HERE.parent.parent.parent.parent


REPO_ROOT = _resolve_repo_root()

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
DEFAULT_INFERENCE_URL = os.getenv(
    "VERILOG_INFERENCE_URL", "https://api.groq.com/openai/v1/chat/completions"
)
DEFAULT_MODEL = os.getenv("VERILOG_DEFAULT_MODEL", "qwen/qwen3-32b")
DEFAULT_TEMPERATURE = float(os.getenv("VERILOG_DEFAULT_TEMPERATURE", "0.2"))
DEFAULT_MAX_TOKENS = int(os.getenv("VERILOG_DEFAULT_MAX_TOKENS", "768"))
DEFAULT_MAX_STEPS = int(os.getenv("VERILOG_DEFAULT_MAX_STEPS", "10"))
FILE_PREVIEW_CHARS = int(os.getenv("VERILOG_FILE_PREVIEW_CHARS", "600"))
HTTP_TIMEOUT_SECONDS = float(os.getenv("VERILOG_INFERENCE_TIMEOUT", "90"))

VERILOG_SYSTEM_PROMPT = (
    "You are an expert digital design engineer helping with Verilog spec-to-RTL tasks. "
    "Choose between these tools: write_file, compile, simulate, submit. "
    "Always respond with a JSON object describing exactly one tool call in the form "
    "{\"tool\": \"<tool_name>\", \"args\": { ... }}. "
    "You may wrap the JSON inside a ```json``` block but MUST NOT include any other prose outside it. "
    "When editing files, rewrite the entire file content. Compile after code changes, simulate to verify behavior, "
    "and submit only after the tests pass. If compilation reports errors (missing ports, mismatched interfaces, etc.), "
    "fix the design with write_file before compiling again—never repeat compile without modifying the source first."
)


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
        environment="verilog",
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
            "supports_proxy": True,
            "endpoints": {
                "openai": "/proxy/v1/chat/completions",
                "groq": "/proxy/groq/v1/chat/completions",
            },
            "tool": {"name": "verilog_tools", "parallel_tool_calls": False},
        },
        limits={"max_ops": 0, "max_time_s": 3600},
    )


def _normalize_inference_url(url: str | None) -> str:
    candidate = (url or DEFAULT_INFERENCE_URL).strip()
    if not candidate:
        candidate = DEFAULT_INFERENCE_URL
    if candidate.endswith("/v1/chat/completions"):
        return candidate
    if candidate.endswith("/chat/completions"):
        return candidate
    if candidate.endswith("/v1"):
        return f"{candidate.rstrip('/')}/chat/completions"
    if candidate.endswith("/v1/"):
        return f"{candidate.rstrip('/')}/chat/completions"
    if candidate.endswith("/chat"):
        return f"{candidate.rstrip('/')}/completions"
    if candidate.endswith("/chat/"):
        return f"{candidate.rstrip('/')}/completions"
    return f"{candidate.rstrip('/')}/v1/chat/completions"


def _format_file_previews(files: dict[str, str]) -> str:
    if not files:
        return "No files in the workspace yet."

    sections: list[str] = []
    for name in sorted(files.keys()):
        content = files[name] or ""
        snippet = content.strip()
        if len(snippet) > FILE_PREVIEW_CHARS:
            snippet = snippet[:FILE_PREVIEW_CHARS] + "\n..."
        sections.append(f"{name}:\n{snippet}")
    return "\n\n".join(sections)


def _format_observation_text(
    *,
    observation: dict[str, Any],
    step_index: int,
    instructions: str | None,
    action_feedback: str | None,
    guidance: str | None = None,
) -> str:
    lines: list[str] = []
    if step_index == 0 and instructions:
        lines.append("Task instructions:")
        lines.append(instructions.strip())
        lines.append("")

    lines.append(f"Step {step_index} status:")
    reward_last = observation.get("reward_last")
    total_reward = observation.get("total_reward")
    if reward_last is not None or total_reward is not None:
        lines.append(
            f"- reward_last={reward_last!r}, total_reward={total_reward!r}"
        )
    lines.append(f"- task_completed={bool(observation.get('task_completed'))}")
    compile_status = observation.get("compile_status")
    if compile_status:
        lines.append(f"- compile_status: {compile_status}")
    simulate_status = observation.get("simulate_status")
    if simulate_status:
        lines.append(f"- simulate_status: {simulate_status}")
    build_dir = observation.get("build_dir")
    if build_dir:
        lines.append(f"- build_directory: {build_dir}")

    if action_feedback:
        lines.append("")
        lines.append(action_feedback)

    files = observation.get("files")
    lines.append("")
    lines.append("Workspace files:")
    lines.append(_format_file_previews(files or {}))

    lines.append("")
    lines.append(
        "Select the single most helpful tool for the next step (write_file, compile, simulate, submit)."
    )
    lines.append(
        "Respond with JSON only: {\"tool\": \"<tool_name>\", \"args\": {...}}."
    )
    if guidance:
        lines.append("")
        lines.append(guidance.strip())
    return "\n".join(lines)


def _summarize_action_feedback(
    tool_name: str, args: dict[str, Any], observation: dict[str, Any], reward: float
) -> str:
    argument_preview = json.dumps(args, ensure_ascii=False)
    parts = [
        f"Previous action: {tool_name}({argument_preview})",
        f"Reward delta: {reward:.4f}",
    ]
    compile_status = observation.get("compile_status")
    if compile_status:
        parts.append(f"Compile status: {compile_status}")
    simulate_status = observation.get("simulate_status")
    if simulate_status:
        parts.append(f"Simulation status: {simulate_status}")
    if observation.get("task_completed"):
        parts.append("Task completed ✅")
    total_reward = observation.get("total_reward")
    if total_reward is not None:
        parts.append(f"Total reward: {total_reward}")
    return "\n".join(parts)


JSON_BLOCK_PATTERN = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL)


def _parse_tool_json(text: str) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []

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
        if not isinstance(tool_name, str):
            continue
        raw_args = candidate.get("args") if isinstance(candidate, dict) else None
        args = raw_args if isinstance(raw_args, dict) else {}
        tool_name = tool_name.strip()
        normalized_args: dict[str, Any] = dict(args)
        if tool_name == "write_file":
            if "file_path" in normalized_args and "path" not in normalized_args:
                normalized_args["path"] = normalized_args.pop("file_path")
            if "file" in normalized_args and "path" not in normalized_args:
                normalized_args["path"] = normalized_args.pop("file")
            if "contents" in normalized_args and "content" not in normalized_args:
                normalized_args["content"] = normalized_args.pop("contents")
        return [{"tool": tool_name, "args": normalized_args}]

    return []


class VerilogLLMAgent:
    """Minimal ReAct-style agent that communicates with a chat-completions API."""

    def __init__(
        self,
        *,
        instructions: str,
        inference_url: str | None,
        model: str | None,
        temperature: float,
        max_tokens: int,
    ) -> None:
        self.instructions = instructions.strip()
        self.inference_url = _normalize_inference_url(inference_url)
        self.model = model or DEFAULT_MODEL
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.messages: list[dict[str, Any]] = [{"role": "system", "content": VERILOG_SYSTEM_PROMPT}]
        self.headers: dict[str, str] = {"Content-Type": "application/json"}

        lowered = self.inference_url.lower()
        if "groq" in lowered:
            api_key = os.getenv("GROQ_API_KEY")
            if not api_key:
                raise RuntimeError("GROQ_API_KEY is not configured for Verilog inference.")
            self.headers["Authorization"] = f"Bearer {api_key.strip()}"
        elif "openai" in lowered:
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise RuntimeError("OPENAI_API_KEY is not configured for Verilog inference.")
            self.headers["Authorization"] = f"Bearer {api_key.strip()}"

        self.history: list[dict[str, Any]] = []

    def append_observation(
        self,
        *,
        observation: dict[str, Any],
        step_index: int,
        action_feedback: str | None,
        guidance: str | None = None,
    ) -> str:
        text = _format_observation_text(
            observation=observation,
            step_index=step_index,
            instructions=self.instructions if step_index == 0 else None,
            action_feedback=action_feedback,
            guidance=guidance,
        )
        self.messages.append({"role": "user", "content": text})
        self.history.append({"role": "user", "content": text})
        return text

    async def invoke(
        self, client: httpx.AsyncClient
    ) -> tuple[str, list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": self.messages,
            "temperature": self.temperature,
        }
        if self.max_tokens > 0:
            payload["max_tokens"] = self.max_tokens

        try:
            response = await client.post(self.inference_url, json=payload, headers=self.headers)
        except Exception as exc:  # pragma: no cover - network failure
            raise RuntimeError(f"Failed to reach inference endpoint: {exc}") from exc

        try:
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:  # pragma: no cover - inference error
            preview = exc.response.text[:2000]
            raise RuntimeError(
                f"Inference call failed with status {exc.response.status_code}: {preview}"
            ) from exc

        data = response.json()
        choices = data.get("choices") or []
        message = choices[0].get("message", {}) if choices else {}
        assistant_text = message.get("content") or ""
        self.messages.append({"role": "assistant", "content": assistant_text})
        self.history.append({"role": "assistant", "content": assistant_text})

        parsed_calls = _parse_tool_json(assistant_text)

        return assistant_text, parsed_calls, data, payload

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
    base_observation = getattr(base_info, "observation", None)
    if hasattr(base_observation, "model_dump"):
        observation_template = base_observation.model_dump()
    elif isinstance(base_observation, dict):
        observation_template = dict(base_observation)
    else:
        observation_template = {}

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
                environment=base_info.environment,
                action_space=base_info.action_space,
                observation={
                    **observation_template,
                    "problem_name": meta_dict["problem_name"],
                    "difficulty": meta_dict["difficulty"],
                },
                dataset={
                    **base_info.dataset.model_dump(),
                    "instance_id": str(instance.id),
                    "metadata": meta_dict,
                },
                rubric=base_info.rubric,
                inference=base_info.inference,
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

    policy_config_raw = getattr(request.policy, "config", {}) if request.policy else {}
    policy_config = dict(policy_config_raw) if isinstance(policy_config_raw, dict) else {}

    policy_model = policy_config.get("model")
    if not isinstance(policy_model, str) or not policy_model.strip():
        policy_model = getattr(request.policy, "policy_name", None) or DEFAULT_MODEL
    policy_model = policy_model.strip()

    temperature = policy_config.get("temperature", DEFAULT_TEMPERATURE)
    try:
        temperature = float(temperature)
    except (TypeError, ValueError):
        temperature = DEFAULT_TEMPERATURE

    max_tokens = policy_config.get("max_tokens", DEFAULT_MAX_TOKENS)
    try:
        max_tokens = int(max_tokens)
    except (TypeError, ValueError):
        max_tokens = DEFAULT_MAX_TOKENS

    max_steps_candidate = (
        policy_config.get("max_steps")
        or policy_config.get("max_llm_calls")
        or DEFAULT_MAX_STEPS
    )
    try:
        max_steps = int(max_steps_candidate)
    except (TypeError, ValueError):
        max_steps = DEFAULT_MAX_STEPS
    max_steps = max(1, min(25, max_steps))

    inference_url = policy_config.get("inference_url")
    if isinstance(inference_url, str) and inference_url.strip():
        resolved_inference = inference_url.strip()
    else:
        resolved_inference = os.getenv("VERILOG_INFERENCE_URL", DEFAULT_INFERENCE_URL)

    instructions = getattr(getattr(instance, "impetus", None), "instructions", "")
    agent = VerilogLLMAgent(
        instructions=getattr(getattr(instance, "impetus", None), "instructions", ""),
        inference_url=resolved_inference,
        model=policy_model,
        temperature=temperature,
        max_tokens=max_tokens,
    )

    policy_id = (
        getattr(request.policy, "policy_id", None)
        or getattr(request.policy, "policy_name", None)
        or policy_model
    )
    env_id = getattr(request.env, "env_id", None) or getattr(request.env, "env_name", None) or "verilog"

    steps: list[RolloutStep] = []
    total_reward = 0.0
    final_observation: dict[str, Any] | None = None
    truncated_due_to_limit = False
    code_dirty = False
    last_compile_success = False
    simulate_since_last_compile = False
    last_compile_failed = False
    needs_design_update = False

    def _build_guidance(step_idx: int) -> str | None:
        hints: list[str] = []
        if step_idx == 0 and not last_compile_success:
            hints.append("Begin by using write_file to implement TopModule according to the problem instructions before compiling.")
        if last_compile_failed or needs_design_update:
            hints.append("Compilation failed; update the design with write_file to match the required ports and behavior before compiling again.")
        if code_dirty and not last_compile_success:
            hints.append("Source was modified; run compile before simulate or submit.")
        if (not code_dirty) and last_compile_success and not simulate_since_last_compile:
            hints.append("Compilation succeeded; run simulate to verify before other actions.")
        if (not code_dirty) and last_compile_success and simulate_since_last_compile:
            hints.append("Simulation already ran after the latest compile; submit if the checks passed or make new edits first.")
        return " ".join(hints) if hints else None

    try:
        initial_raw_observation = await env.initialize()
        current_observation = _normalise_observation(initial_raw_observation)
        final_observation = current_observation
        agent.append_observation(
            observation=current_observation,
            step_index=0,
            action_feedback=None,
            guidance=_build_guidance(0),
        )

        total_reward = float(current_observation.get("total_reward") or 0.0)
        already_done = bool(
            current_observation.get("terminated") or current_observation.get("task_completed")
        )

        timeout = httpx.Timeout(
            HTTP_TIMEOUT_SECONDS,
            connect=HTTP_TIMEOUT_SECONDS,
            read=HTTP_TIMEOUT_SECONDS,
        )

        async with httpx.AsyncClient(timeout=timeout) as client:
            if not already_done:
                for step_index in range(1, max_steps + 1):
                    assistant_text, tool_calls, raw_response, request_payload = await agent.invoke(client)
                    override_info: dict[str, Any] | None = None
                    if not tool_calls:
                        fallback_tool = (
                            "submit" if current_observation.get("task_completed") else "compile"
                        )
                        tool_calls = [{"tool": fallback_tool, "args": {}}]

                    primary_call = dict(tool_calls[0])
                    tool_name_raw = str(primary_call.get("tool", ""))
                    normalized_tool = tool_name_raw.strip().lower()
                    if normalized_tool == "compile":
                        if (not code_dirty) and last_compile_success and not simulate_since_last_compile:
                            override_info = {
                                "from": dict(primary_call),
                                "reason": "compile_after_success_without_changes",
                            }
                            primary_call = {"tool": "simulate", "args": {}}
                            tool_calls = [primary_call]
                            override_info["to"] = dict(primary_call)
                    env_call = EnvToolCall(tool=primary_call["tool"], args=primary_call["args"])

                    try:
                        skip_env_step = (
                            normalized_tool == "compile"
                            and needs_design_update
                            and not code_dirty
                        )
                        if skip_env_step:
                            reward_last = -0.01
                            total_reward += reward_last
                            current_observation = dict(current_observation)
                            current_observation["reward_last"] = reward_last
                            current_observation["total_reward"] = total_reward
                            final_observation = current_observation
                            done_flag = False
                            truncated_flag = False
                        else:
                            step_observation = await env.step(env_call)
                            current_observation = _normalise_observation(step_observation)
                            final_observation = current_observation
                            reward_last = float(current_observation.get("reward_last") or 0.0)
                            total_reward = float(
                                current_observation.get("total_reward") or (total_reward + reward_last)
                            )
                            done_flag = bool(
                                current_observation.get("terminated")
                                or current_observation.get("task_completed")
                            )
                            truncated_flag = bool(current_observation.get("truncated"))

                        executed_tool_name = str(primary_call["tool"])
                        normalized_executed_tool = executed_tool_name.strip().lower()

                        if normalized_executed_tool == "write_file":
                            code_dirty = True
                            last_compile_success = False
                            simulate_since_last_compile = False
                            last_compile_failed = False
                            needs_design_update = False
                        elif normalized_executed_tool == "compile":
                            compile_status_text = str(current_observation.get("compile_status") or "")
                            if "success" in compile_status_text.lower():
                                code_dirty = False
                                last_compile_success = True
                                simulate_since_last_compile = False
                                last_compile_failed = False
                                needs_design_update = False
                            else:
                                last_compile_success = False
                                last_compile_failed = True
                                needs_design_update = True
                        elif normalized_executed_tool == "simulate":
                            simulate_since_last_compile = True

                        tool_call_records = [
                            {"tool_name": call["tool"], "arguments": call["args"]}
                            for call in tool_calls
                        ]
                        step_info = {
                            "assistant_message": assistant_text,
                            "model_response": raw_response,
                            "llm_request": request_payload,
                        }
                        if override_info:
                            step_info["auto_override"] = override_info
                        if normalized_tool == "compile" and skip_env_step:
                            step_info["compile_blocked"] = {
                                "reason": "design_requires_update_before_compile",
                                "hint": "Use write_file to match required ports/behavior before compiling again.",
                            }
                        steps.append(
                            RolloutStep(
                                obs=current_observation,
                                tool_calls=tool_call_records,
                                reward=reward_last,
                                done=done_flag,
                                truncated=truncated_flag,
                                info=step_info,
                            )
                        )

                        if normalized_tool == "compile" and skip_env_step:
                            action_feedback = (
                                "Compilation blocked: update the design with write_file (declare required ports and logic) before compiling again."
                            )
                        else:
                            action_feedback = _summarize_action_feedback(
                                primary_call["tool"], primary_call["args"], current_observation, reward_last
                            )
                        agent.append_observation(
                            observation=current_observation,
                            step_index=step_index,
                            action_feedback=action_feedback,
                            guidance=_build_guidance(step_index),
                        )

                        if done_flag:
                            break

                        if step_index == max_steps:
                            truncated_due_to_limit = True
                            break
                    except Exception as exc:  # pragma: no cover - defensive path
                        error_text = str(exc)
                        logger.exception("Verilog environment step failed: %s", exc)
                        failure_observation = dict(current_observation)
                        failure_observation["error"] = error_text
                        final_observation = failure_observation
                        tool_call_records = [
                            {"tool_name": primary_call["tool"], "arguments": primary_call["args"]}
                        ]
                        step_info = {
                            "assistant_message": assistant_text,
                            "model_response": raw_response,
                            "llm_request": request_payload,
                            "error": error_text,
                        }
                        steps.append(
                            RolloutStep(
                                obs=failure_observation,
                                tool_calls=tool_call_records,
                                reward=0.0,
                                done=True,
                                truncated=True,
                                info=step_info,
                            )
                        )
                        truncated_due_to_limit = True
                        break
    finally:
        with contextlib.suppress(Exception):
            await env.terminate()

    if final_observation is None:
        final_observation = {}

    final_total_reward = float(final_observation.get("total_reward") or total_reward)
    final_done = bool(
        final_observation.get("terminated") or final_observation.get("task_completed")
    )
    final_truncated = truncated_due_to_limit or bool(final_observation.get("truncated"))

    metrics = RolloutMetrics(
        episode_returns=[final_total_reward],
        mean_return=final_total_reward,
        num_steps=len(steps),
        num_episodes=1,
        outcome_score=final_total_reward,
        events_score=None,
        details={
            "task_completed": bool(final_observation.get("task_completed")),
            "total_reward": final_total_reward,
            "steps": len(steps),
            "truncated": final_truncated,
        },
    )

    # Extract inference_url from policy config (REQUIRED for RL trace correlation)
    # The trainer injects this with ?cid=trace_xxxxx parameter for trace linking
    final_inference_url = policy_config.get("inference_url")
    if not isinstance(final_inference_url, str) or not final_inference_url.strip():
        # Fallback to agent's inference_url if not in policy config
        final_inference_url = agent.inference_url
        logger.warning(
            "VERILOG_ROLLOUT: inference_url not found in policy_config, using agent.inference_url run_id=%s url=%s",
            request.run_id,
            final_inference_url,
        )
    else:
        logger.info(
            "VERILOG_ROLLOUT: using inference_url from policy_config run_id=%s url=%s has_cid=%s",
            request.run_id,
            final_inference_url,
            "?cid=" in final_inference_url,
        )
    
    trajectory = RolloutTrajectory(
        env_id=str(env_id),
        policy_id=str(policy_id),
        steps=steps,
        final={
            "observation": final_observation,
            "reward": final_total_reward,
            "done": final_done,
            "truncated": final_truncated,
            "info": {
                "total_reward": final_total_reward,
                "task_completed": bool(final_observation.get("task_completed")),
                "policy_model": policy_model,
                "inference_url": final_inference_url,
            },
        },
        length=len(steps),
        inference_url=final_inference_url,  # CRITICAL: Must contain ?cid=... for trace correlation
        decision_samples=None,
    )

    # Build trace payload
    trace_payload = {
        "session_trace": {
            "session_id": request.run_id,
            "created_at": None,
            "metadata": {
                "task": "verilog",
                "provider": "groq",
                "model": policy_model,
                "total_reward": final_total_reward,
                "task_completed": bool(final_observation.get("task_completed")),
            },
            "session_time_steps": [],
            "event_history": [],
            "markov_blanket_message_history": [],
        }
    }
    
    # Build pipeline_metadata (required for RL training)
    pipeline_metadata = {
        "reward_score": final_total_reward,
        "policy_id": policy_id,
        "inference": {
            "provider": "groq",
            "model": policy_model,
            "url": final_inference_url,  # Use final_inference_url (has ?cid=...)
        },
        "env_name": env_id,
        "task_id": getattr(instance, "problem_id", None),
        "task_split": getattr(instance, "split", "val"),
        "inference_url": final_inference_url,  # CRITICAL: Used by trainer to extract trace_correlation_id
    }
    
    # Log for debugging RL training
    logger.info(
        "VERILOG_ROLLOUT: pipeline_metadata run_id=%s reward=%.3f inference_url=%s",
        request.run_id,
        final_total_reward,
        final_inference_url,
    )
    
    return RolloutResponse(
        run_id=request.run_id,
        trajectories=[trajectory],
        branches={},
        metrics=metrics,
        aborted=False,
        ops_executed=len(steps),
        trace=trace_payload,
        pipeline_metadata=pipeline_metadata,
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
        proxy=ProxyConfig(
            enable_openai=True,
            enable_groq=True,
            system_hint=VERILOG_SYSTEM_PROMPT,
        ),
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
