from __future__ import annotations

import json
import os
import re
import contextlib
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Mapping, Optional, Sequence

import httpx

from fastapi import APIRouter, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from synth_ai.environments.environment.tools import EnvToolCall
from synth_ai.environments.examples.sokoban.environment import SokobanEnvironment
from synth_ai.environments.examples.sokoban.taskset import (
    SokobanTaskInstance,
    SokobanTaskSet,
    create_task_instance_from_seed,
)
from synth_ai.task.apps import TaskAppEntry, register_task_app
from synth_ai.task.contracts import (
    RolloutMetrics,
    RolloutRequest,
    RolloutResponse,
    RolloutStep,
    RolloutTrajectory,
    TaskInfo,
)
from synth_ai.task.auth import is_api_key_header_authorized, normalize_environment_api_key
from synth_ai.task.server import TaskAppConfig, create_task_app


ACTION_ID_TO_NAME = {0: "left", 1: "up", 2: "right", 3: "down"}
ACTION_TOKEN_TO_ID = {
    "0": 0,
    "1": 1,
    "2": 2,
    "3": 3,
    "left": 0,
    "move_left": 0,
    "west": 0,
    "l": 0,
    "up": 1,
    "move_up": 1,
    "north": 1,
    "u": 1,
    "right": 2,
    "move_right": 2,
    "east": 2,
    "r": 2,
    "down": 3,
    "move_down": 3,
    "south": 3,
    "d": 3,
}

SOKOBAN_SYSTEM_PROMPT = """You are an agent playing Sokoban.
The grid uses characters: '#' wall, '_' floor, 'O' box, '√' box on target, 'X' target, 'P' player.
Always respond with a single tool call named interact_many containing 1-5 actions.
Valid action tokens are digits 0/1/2/3 or their direction words (left/up/right/down).
Mapping: 0=left, 1=up, 2=right, 3=down. Avoid undoing progress and focus on pushing boxes onto targets."""


def _short_text(value: Any, *, limit: int = 280) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        text = value.strip()
        return text if len(text) <= limit else text[: limit - 1] + "…"
    if isinstance(value, (int, float, bool)):
        return str(value)
    try:
        text = json.dumps(value, ensure_ascii=False)
    except Exception:
        text = str(value)
    text = text.strip()
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _summarize_observation(observation: Any) -> str:
    if isinstance(observation, dict):
        for key in ("room_text", "observation", "grid"):
            value = observation.get(key)
            if isinstance(value, str) and value.strip():
                return _short_text(value, limit=512)
        preview = {
            key: observation.get(key)
            for key in ("player_position", "boxes_on_target", "num_boxes", "steps_taken")
            if key in observation
        }
        if preview:
            return _short_text(preview, limit=512)
    return _short_text(observation, limit=512)


def _format_tool_calls(tool_calls: Sequence[Dict[str, Any]] | None) -> str:
    if not tool_calls:
        return "<noop>"
    formatted: list[str] = []
    for call in tool_calls:
        args = call.get("args") if isinstance(call, dict) else None
        if not isinstance(args, dict):
            continue
        if "actions" in args and isinstance(args["actions"], list):
            parts: list[str] = []
            for item in args["actions"]:
                try:
                    val = int(item)
                except Exception:
                    token = str(item).strip().lower()
                    val = ACTION_TOKEN_TO_ID.get(token)
                name = ACTION_ID_TO_NAME.get(val, str(item)) if val is not None else str(item)
                parts.append(str(name))
            if parts:
                formatted.append("[" + ", ".join(parts) + "]")
            continue
        action = args.get("action")
        if action is None:
            continue
        try:
            action = int(action)
        except Exception:
            token = str(action).strip().lower()
            action = ACTION_TOKEN_TO_ID.get(token, action)
        name = ACTION_ID_TO_NAME.get(action, str(action))
        formatted.append(str(name))
    return ", ".join(formatted) if formatted else "<noop>"


def _build_trace_payload(
    request: RolloutRequest,
    steps: Sequence[RolloutStep],
    metrics: RolloutMetrics,
    *,
    difficulty: str,
    initial_observation: Any,
    provider: str = "local",
) -> Dict[str, Any]:
    created_at = datetime.now(timezone.utc)
    base_time = time.time()
    event_history: list[dict[str, Any]] = []
    markov_messages: list[dict[str, Any]] = []
    session_steps: list[dict[str, Any]] = []

    if not steps:
        observation_text = _summarize_observation(initial_observation)
        event_time = base_time
        observation_msg = {
            "content": {"text": observation_text},
            "message_type": "observation",
            "time_record": {"event_time": event_time},
            "metadata": {"step_index": 0},
        }
        markov_messages.append(observation_msg)
        event_history.append(
            {
                "system_instance_id": "sokoban.step.0",
                "time_record": {"event_time": event_time},
                "reward": 0.0,
                "terminated": True,
                "truncated": False,
                "metadata": {
                    "tool_calls": [],
                },
            }
        )
        session_steps.append(
            {
                "step_id": "step_0",
                "step_index": 0,
                "events": [event_history[-1]],
                "markov_blanket_messages": markov_messages[-1:],
                "step_metadata": {"reward": 0.0, "done": True, "truncated": False},
            }
        )
    else:
        for idx, step in enumerate(steps):
            event_time = base_time + idx * 0.01
            observation_text = _summarize_observation(step.obs)
            action_text = _format_tool_calls(step.tool_calls)
            observation_msg = {
                "content": {"text": observation_text},
                "message_type": "observation",
                "time_record": {"event_time": event_time},
                "metadata": {"step_index": idx},
            }
            action_msg = {
                "content": {"text": action_text},
                "message_type": "action",
                "time_record": {"event_time": event_time + 0.0005},
                "metadata": {"step_index": idx},
            }
            markov_messages.extend([observation_msg, action_msg])
            reward_val = float(step.reward or 0.0)
            event_history.append(
                {
                    "system_instance_id": f"sokoban.step.{idx}",
                    "time_record": {"event_time": event_time},
                    "reward": reward_val,
                    "terminated": bool(step.done),
                    "truncated": bool(step.truncated),
                    "metadata": {
                        "tool_calls": step.tool_calls,
                        "info": step.info or {},
                    },
                }
            )
            session_steps.append(
                {
                    "step_id": f"step_{idx}",
                    "step_index": idx,
                    "events": [event_history[-1]],
                    "markov_blanket_messages": [observation_msg, action_msg],
                    "step_metadata": {
                        "reward": reward_val,
                        "done": bool(step.done),
                        "truncated": bool(step.truncated),
                    },
                }
            )

    session_trace = {
        "session_id": str(request.run_id),
        "created_at": created_at.isoformat(),
        "metadata": {
            "task": "sokoban",
            "difficulty": difficulty,
            "seed": request.env.seed,
            "provider": provider,
            "env": request.env.model_dump(),
            "policy": request.policy.model_dump(),
        },
        "session_time_steps": session_steps,
        "event_history": event_history,
        "markov_blanket_message_history": markov_messages,
    }

    return {
        "version": 3,
        "session_trace": session_trace,
        "run_id": request.run_id,
        "policy_id": request.policy.policy_id or request.policy.policy_name,
        "reward": metrics.mean_return,
        "episode_returns": metrics.episode_returns,
        "mean_return": metrics.mean_return,
        "num_steps": metrics.num_steps,
    }



def _task_info() -> TaskInfo:
    return TaskInfo(
        task={"id": "sokoban", "name": "Sokoban", "version": "1.0.0"},
        environment="sokoban",
        action_space={
            "type": "tool_call",
            "tools": [{"name": "interact", "schema": {"action": "int"}}],
            "max_calls": 1,
        },
        observation={"summary": "Sokoban grid observation", "keys": ["grid", "player"]},
        dataset={"id": "sokoban", "name": "Sokoban", "version": "1.0.0"},
        rubric={"version": "1", "criteria_count": 1, "source": "inline"},
        inference={"supports_proxy": False},
        limits={"max_turns": 200},
    )


router = APIRouter()


async def rollout_executor(request: RolloutRequest, fastapi_request: Request) -> RolloutResponse:
    policy_cfg = dict(request.policy.config or {})
    provider = str(policy_cfg.get("provider") or "").strip().lower()
    if provider == "groq":
        return await _rollout_with_groq(request, fastapi_request, policy_cfg)
    if provider == "openai":
        return await _rollout_with_openai(request, fastapi_request, policy_cfg)

    taskset: SokobanTaskSet = fastapi_request.app.state.sokoban_taskset
    seed = request.env.seed or 0
    difficulty = (request.env.config or {}).get("difficulty") or "easy"
    # Create deterministic instance from seed
    instance: SokobanTaskInstance = await create_task_instance_from_seed(str(difficulty), int(seed))
    env = SokobanEnvironment(instance)
    obs = await env.initialize()
    initial_observation = obs

    tool_calls: List[Dict[str, Any]] = []
    # If a predefined action sequence is provided, execute it (evaluation-style)
    actions: Optional[Sequence[int]] = None
    try:
        cfg = request.policy.config or {}
        if isinstance(cfg.get("actions"), list):
            actions = [int(a) for a in cfg["actions"]]
    except Exception:
        actions = None

    last_obs: Any = obs
    steps: List[RolloutStep] = []
    max_steps = int((request.env.config or {}).get("max_steps") or 50)
    executed = 0
    if actions:
        for a in actions[:max_steps]:
            last_obs = await env.step(EnvToolCall(tool="interact", args={"action": int(a)}))
            executed += 1
            steps.append(
                RolloutStep(obs=last_obs, tool_calls=[{"tool": "interact", "args": {"action": int(a)}}], reward=0.0, done=False, info={})
            )
    # Mark episode end (single-episode trajectory)
    final = {"observation": last_obs, "reward": 0.0}
    if not steps:
        steps = [RolloutStep(obs=last_obs, tool_calls=[], reward=0.0, done=True, info={})]
    
    # Extract inference_url from policy config (None for manual rollouts)
    inference_url = policy_cfg.get("inference_url")
    
    traj = RolloutTrajectory(
        env_id="sokoban",
        policy_id=request.policy.policy_id or "policy",
        steps=steps,
        final=final,
        length=len(steps),
        inference_url=inference_url,  # NEW: Required for trace correlation
    )
    metrics = RolloutMetrics(
        episode_returns=[final.get("reward", 0.0) or 0.0],
        mean_return=final.get("reward", 0.0) or 0.0,
        num_steps=len(steps),
        num_episodes=1,
        outcome_score=None,
        events_score=None,
        details={},
    )
    trace_payload = _build_trace_payload(
        request,
        steps,
        metrics,
        difficulty=str(difficulty),
        initial_observation=initial_observation,
    )
    return RolloutResponse(
        run_id=request.run_id,
        trajectories=[traj],
        branches={},
        metrics=metrics,
        aborted=False,
        ops_executed=1 + executed,
        trace=trace_payload,
    )


def _format_sokoban_prompt(observation: dict[str, Any], last_actions: list[int]) -> str:
    grid = observation.get("room_text", "")
    boxes = observation.get("boxes_on_target", 0)
    total_boxes = observation.get("num_boxes", boxes)
    position = observation.get("player_position", ())
    reward_last = observation.get("reward_last", 0.0)
    steps_taken = observation.get("steps_taken", 0)
    max_steps = observation.get("max_steps", 0)
    last_str = (
        ", ".join(ACTION_ID_TO_NAME.get(a, str(a)) for a in last_actions) if last_actions else "none"
    )
    return (
        f"Step {steps_taken} / {max_steps}\n"
        f"Player position: {position}\n"
        f"Boxes on target: {boxes} / {total_boxes}\n"
        f"Last reward: {reward_last}\n"
        f"Previous actions: {last_str}\n"
        "Grid:\n"
        f"{grid}\n"
        "Select up to five next actions via the interact_many tool."
    )


def _extract_actions_from_response(
    response: dict[str, Any], max_actions: int
) -> list[int]:
    import json as json_lib
    print(f"[extract] FULL RESPONSE:", flush=True)
    print(json_lib.dumps(response, indent=2)[:2000], flush=True)
    
    actions: list[int] = []
    choices = response.get("choices") or []
    print(f"[extract] {len(choices)} choices", flush=True)
    if choices:
        msg = choices[0].get("message", {})
        print(f"[extract] tool_calls: {msg.get('tool_calls')}", flush=True)
        print(f"[extract] content: {msg.get('content')}", flush=True)
        print(f"[extract] finish_reason: {choices[0].get('finish_reason')}", flush=True)
    for choice in choices:
        message = choice.get("message") or {}
        tool_calls = message.get("tool_calls") or []
        for tool_call in tool_calls:
            function = tool_call.get("function") or {}
            arguments = function.get("arguments")
            payload: dict[str, Any] | None = None
            if isinstance(arguments, str):
                try:
                    payload = json.loads(arguments)
                except json.JSONDecodeError:
                    payload = None
            elif isinstance(arguments, dict):
                payload = arguments
            if not payload:
                continue
            raw_actions = payload.get("actions")
            if isinstance(raw_actions, list):
                for item in raw_actions:
                    if isinstance(item, int) and item in ACTION_ID_TO_NAME:
                        actions.append(int(item))
                        continue
                    if isinstance(item, str):
                        token = item.strip().lower()
                        if token in ACTION_TOKEN_TO_ID:
                            actions.append(ACTION_TOKEN_TO_ID[token])
            if actions:
                break
        if actions:
            break

    if not actions and choices:
        # Fallback: parse tokens from assistant text
        text = choices[0].get("message", {}).get("content") or ""
        tokens = re.findall(r"[0-3a-zA-Z_]+", text)
        for tok in tokens:
            token = tok.strip().lower()
            if token in ACTION_TOKEN_TO_ID:
                actions.append(ACTION_TOKEN_TO_ID[token])

    if len(actions) > max_actions:
        return actions[:max_actions]
    return actions


async def _call_groq_chat(
    client: httpx.AsyncClient,
    api_key: str,
    payload: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    try:
        response = await client.post(
            "https://api.groq.com/openai/v1/chat/completions",
            json=payload,
            headers={"Authorization": f"Bearer {api_key}"},
        )
        response.raise_for_status()
        data = response.json()
        return data, {
            "status": response.status_code,
            "headers": dict(response.headers),
            "body": data,
        }
    except httpx.HTTPStatusError as exc:
        try:
            body = exc.response.json()
        except Exception:
            body = {"raw": exc.response.text}
        error_detail = {
            "status": exc.response.status_code,
            "body": body,
            "headers": dict(exc.response.headers),
        }
        raise HTTPException(status_code=exc.response.status_code, detail=error_detail) from exc
    except httpx.RequestError as exc:
        raise HTTPException(status_code=502, detail=f"Groq request error: {exc}") from exc


async def _call_openai_chat(
    client: httpx.AsyncClient,
    api_key: str,
    payload: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    try:
        response = await client.post(
            "https://api.openai.com/v1/chat/completions",
            json=payload,
            headers={"Authorization": f"Bearer {api_key}"},
        )
        response.raise_for_status()
        data = response.json()
        return data, {
            "status": response.status_code,
            "headers": dict(response.headers),
            "body": data,
        }
    except httpx.HTTPStatusError as exc:
        try:
            body = exc.response.json()
        except Exception:
            body = {"raw": exc.response.text}
        error_detail = {
            "status": exc.response.status_code,
            "body": body,
            "headers": dict(exc.response.headers),
        }
        try:
            print("[openai:error]", error_detail, flush=True)
        except Exception:
            pass
        raise HTTPException(status_code=exc.response.status_code, detail=error_detail) from exc
    except httpx.RequestError as exc:
        raise HTTPException(status_code=502, detail=f"OpenAI request error: {exc}") from exc


async def _rollout_with_groq(
    request: RolloutRequest,
    fastapi_request: Request,
    config: dict[str, Any],
) -> RolloutResponse:
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise HTTPException(
            status_code=503,
            detail="GROQ_API_KEY environment variable is required for Groq rollouts.",
        )

    seed = request.env.seed or 0
    difficulty = (request.env.config or {}).get("difficulty") or "easy"
    instance: SokobanTaskInstance = await create_task_instance_from_seed(str(difficulty), int(seed))
    env = SokobanEnvironment(instance)
    observation = await env.initialize()
    initial_observation = observation

    model = config.get("model") or "qwen/qwen3-32b"
    temperature = float(config.get("temperature", 0.0) or 0.0)
    top_p = float(config.get("top_p", 0.95) or 0.95)
    max_tokens = int(config.get("max_tokens", 128) or 128)
    actions_per_call = int(config.get("max_actions_per_call", 4) or 4)
    actions_per_call = max(1, min(8, actions_per_call))

    max_steps = int((request.env.config or {}).get("max_steps") or 50)

    steps: List[RolloutStep] = []
    last_actions: list[int] = []
    total_reward = float(observation.get("total_reward") or 0.0)
    executed = 0

    tool_items_enum = sorted(set(ACTION_TOKEN_TO_ID.keys()))
    tool_schema = {
        "type": "function",
        "function": {
            "name": "interact_many",
            "description": "Execute a short sequence of Sokoban moves in order.",
            "parameters": {
                "type": "object",
                "properties": {
                    "actions": {
                        "type": "array",
                        "items": {"type": "string", "enum": tool_items_enum},
                        "minItems": 1,
                        "maxItems": actions_per_call,
                    }
                },
                "required": ["actions"],
                "additionalProperties": False,
            },
        },
    }

    async with httpx.AsyncClient(timeout=httpx.Timeout(30.0)) as client:
        for _ in range(max_steps):
            user_prompt = _format_sokoban_prompt(observation, last_actions)
            messages = [
                {"role": "system", "content": SOKOBAN_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ]
            payload = {
                "model": model,
                "messages": messages,
                "temperature": temperature,
                "top_p": top_p,
                "max_tokens": max_tokens,
                "tools": [tool_schema],
                "tool_choice": {"type": "function", "function": {"name": "interact_many"}},
            }
            vendor_attempts: list[dict[str, Any]] = []
            try:
                response, response_meta = await _call_groq_chat(client, api_key, payload)
                vendor_attempts.append({"request": payload, "response": response_meta})
            except HTTPException as exc:
                detail = exc.detail
                if isinstance(detail, dict):
                    vendor_attempts.append({"request": payload, "error": detail})
                else:
                    vendor_attempts.append({"request": payload, "error": {"message": str(detail)}})
                raise

            actions = _extract_actions_from_response(response, actions_per_call)
            if not actions:
                break

            aggregated_actions: list[int] = []
            aggregated_reward = 0.0
            done = False
            truncated = False
            intermediate_rewards: list[float] = []
            if executed >= max_steps:
                break

            for action in actions:
                if executed >= max_steps:
                    break
                aggregated_actions.append(int(action))
                observation = await env.step(
                    EnvToolCall(tool="interact", args={"action": int(action)})
                )
                current_total = float(observation.get("total_reward") or total_reward)
                reward_delta = current_total - total_reward
                total_reward = current_total
                aggregated_reward += reward_delta
                intermediate_rewards.append(reward_delta)
                done = bool(observation.get("terminated"))
                truncated = bool(observation.get("truncated"))
                executed += 1
                if done or truncated:
                    break

            if not aggregated_actions:
                continue

            last_actions = aggregated_actions
            step = RolloutStep(
                obs=observation,
                tool_calls=[
                    {
                        "tool": "interact_many",
                        "args": {"actions": [int(a) for a in aggregated_actions]},
                        "source": "groq",
                    }
                ],
                reward=aggregated_reward,
                done=done,
                truncated=truncated if truncated else None,
                info={
                    "provider": "groq",
                    "model": model,
                    "actions_executed": aggregated_actions,
                    "prompt": user_prompt,
                    "reward_deltas": intermediate_rewards,
                    "vendor_attempts": vendor_attempts,
                    "groq_attempts": vendor_attempts,
                },
            )
            steps.append(step)

            if step.done or (step.truncated or False):
                break

    final = {"observation": observation, "reward": total_reward}
    inference_url_groq = "https://api.groq.com/openai/v1/chat/completions"
    
    trajectory = RolloutTrajectory(
        env_id=request.env.env_id or request.env.env_name or "sokoban",
        policy_id=request.policy.policy_id or request.policy.policy_name or "sokoban-groq",
        steps=steps,
        final=final,
        length=len(steps),
        inference_url=inference_url_groq,  # NEW: Required for trace correlation
    )
    metrics = RolloutMetrics(
        episode_returns=[total_reward],
        mean_return=total_reward if steps else 0.0,
        num_steps=len(steps),
        num_episodes=1,
        outcome_score=None,
        events_score=None,
        details={"provider": "groq", "model": model},
    )
    trace_payload = _build_trace_payload(
        request,
        steps,
        metrics,
        difficulty=str(difficulty),
        initial_observation=initial_observation,
        provider="groq",
    )
    return RolloutResponse(
        run_id=request.run_id,
        trajectories=[trajectory],
        branches={},
        metrics=metrics,
        aborted=False,
        ops_executed=executed,
        trace=trace_payload,
    )


async def _rollout_with_openai(
    request: RolloutRequest,
    fastapi_request: Request,
    config: dict[str, Any],
) -> RolloutResponse:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise HTTPException(
            status_code=503,
            detail="OPENAI_API_KEY environment variable is required for OpenAI rollouts.",
        )

    seed = request.env.seed or 0
    difficulty = (request.env.config or {}).get("difficulty") or "easy"
    instance: SokobanTaskInstance = await create_task_instance_from_seed(str(difficulty), int(seed))
    env = SokobanEnvironment(instance)
    observation = await env.initialize()
    initial_observation = observation

    model = config.get("model") or "gpt-5"
    temperature_cfg = config.get("temperature")
    top_p_cfg = config.get("top_p")
    completion_tokens = int(
        config.get("max_completion_tokens")
        or config.get("max_tokens")
        or 4000
    )
    actions_per_call = int(config.get("max_actions_per_call", 4) or 4)
    actions_per_call = max(1, min(8, actions_per_call))

    max_steps = int((request.env.config or {}).get("max_steps") or 50)

    steps: List[RolloutStep] = []
    last_actions: list[int] = []
    total_reward = float(observation.get("total_reward") or 0.0)
    executed = 0

    tool_items_enum = sorted(set(ACTION_TOKEN_TO_ID.keys()))
    tool_schema = {
        "type": "function",
        "function": {
            "name": "interact_many",
            "description": "Execute a short sequence of Sokoban moves in order.",
            "parameters": {
                "type": "object",
                "properties": {
                    "actions": {
                        "type": "array",
                        "items": {"type": "string", "enum": tool_items_enum},
                        "minItems": 1,
                        "maxItems": actions_per_call,
                    }
                },
                "required": ["actions"],
                "additionalProperties": False,
            },
        },
    }

    async with httpx.AsyncClient(timeout=httpx.Timeout(120.0)) as client:
        # Process ops array - each "policy" op triggers one LLM call
        ops_to_process = request.ops or []
        if not ops_to_process:
            # If no ops provided, default to max_steps policy calls
            ops_to_process = ["policy"] * max_steps
        
        for op_idx, op in enumerate(ops_to_process):
            # Only process "policy" ops, skip explicit actions for now
            if op != "policy" and not (isinstance(op, str) and op.lower() == "policy"):
                continue
                
            user_prompt = _format_sokoban_prompt(observation, last_actions)
            messages = [
                {"role": "system", "content": SOKOBAN_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ]
            payload_base: dict[str, Any] = {
                "model": model,
                "messages": messages,
                "max_completion_tokens": completion_tokens,
                "tools": [tool_schema],
                "tool_choice": {"type": "function", "function": {"name": "interact_many"}},
            }
            # GPT-5 models don't support temperature/top_p (only default value of 1)
            is_gpt5 = "gpt-5" in model.lower()
            if temperature_cfg is not None and not is_gpt5:
                with contextlib.suppress(Exception):
                    payload_base["temperature"] = float(temperature_cfg)
            if top_p_cfg is not None and not is_gpt5:
                with contextlib.suppress(Exception):
                    payload_base["top_p"] = float(top_p_cfg)

            vendor_attempts: list[dict[str, Any]] = []
            attempt_payload = dict(payload_base)
            while True:
                attempt_record: dict[str, Any] = {"request": dict(attempt_payload)}
                try:
                    response, response_meta = await _call_openai_chat(client, api_key, attempt_payload)
                    attempt_record["response"] = response_meta
                    vendor_attempts.append(attempt_record)
                    break
                except HTTPException as exc:
                    detail = exc.detail
                    attempt_record["error"] = detail if isinstance(detail, dict) else {"message": str(detail)}
                    vendor_attempts.append(attempt_record)
                    handled = False
                    body = detail.get("body") if isinstance(detail, dict) else None
                    error_info = body.get("error") if isinstance(body, dict) else None
                    code = error_info.get("code") if isinstance(error_info, dict) else None
                    param = error_info.get("param") if isinstance(error_info, dict) else None
                    if code in {"unsupported_parameter", "unsupported_value"}:
                        if param == "temperature" and "temperature" in attempt_payload:
                            attempt_payload = dict(attempt_payload)
                            attempt_payload.pop("temperature", None)
                            handled = True
                        elif param == "top_p" and "top_p" in attempt_payload:
                            attempt_payload = dict(attempt_payload)
                            attempt_payload.pop("top_p", None)
                            handled = True
                    if handled:
                        continue
                    raise

            actions = _extract_actions_from_response(response, actions_per_call)
            if not actions:
                break

            aggregated_actions: list[int] = []
            aggregated_reward = 0.0
            done = False
            truncated = False
            intermediate_rewards: list[float] = []
            if executed >= max_steps:
                break

            print(f"[debug] Processing {len(actions)} actions from LLM", flush=True)
            for action in actions:
                if executed >= max_steps:
                    break
                aggregated_actions.append(int(action))
                observation = await env.step(
                    EnvToolCall(tool="interact", args={"action": int(action)})
                )
                current_total = float(observation.get("total_reward") or total_reward)
                reward_delta = current_total - total_reward
                total_reward = current_total
                aggregated_reward += reward_delta
                intermediate_rewards.append(reward_delta)
                done = bool(observation.get("terminated"))
                truncated = bool(observation.get("truncated"))
                executed += 1
                if done or truncated:
                    break

                print(f"[debug] After action {action}: done={done}, trunc={truncated}, exec={executed}", flush=True)
            if not aggregated_actions:
                continue

            last_actions = aggregated_actions
            step = RolloutStep(
                obs=observation,
                tool_calls=[
                    {
                        "tool": "interact_many",
                        "args": {"actions": [int(a) for a in aggregated_actions]},
                        "source": "openai",
                    }
                ],
                reward=aggregated_reward,
                done=done,
                truncated=truncated if truncated else None,
                info={
                    "provider": "openai",
                    "model": model,
                    "actions_executed": aggregated_actions,
                    "prompt": user_prompt,
                    "reward_deltas": intermediate_rewards,
                    "vendor_attempts": vendor_attempts,
                    "openai_attempts": vendor_attempts,
                    "max_completion_tokens": completion_tokens,
                    "temperature_requested": temperature_cfg,
                    "top_p_requested": top_p_cfg,
                },
            )
            steps.append(step)

            if step.done or (step.truncated or False):
                break

    final = {"observation": observation, "reward": total_reward}
    inference_url_openai = "https://api.openai.com/v1/chat/completions"
    
    trajectory = RolloutTrajectory(
        env_id=request.env.env_id or request.env.env_name or "sokoban",
        policy_id=request.policy.policy_id or request.policy.policy_name or "sokoban-openai",
        steps=steps,
        final=final,
        length=len(steps),
        inference_url=inference_url_openai,  # NEW: Required for trace correlation
    )
    metrics = RolloutMetrics(
        episode_returns=[total_reward],
        mean_return=total_reward if steps else 0.0,
        num_steps=len(steps),
        num_episodes=1,
        outcome_score=None,
        events_score=None,
        details={"provider": "openai", "model": model},
    )
    trace_payload = _build_trace_payload(
        request,
        steps,
        metrics,
        difficulty=str(difficulty),
        initial_observation=initial_observation,
        provider="openai",
    )
    return RolloutResponse(
        run_id=request.run_id,
        trajectories=[trajectory],
        branches={},
        metrics=metrics,
        aborted=False,
        ops_executed=executed,
        trace=trace_payload,
    )


def build_config() -> TaskAppConfig:
    taskset = SokobanTaskSet()
    base = _task_info()
    app_state: dict[str, Any] = {"sokoban_taskset": taskset, "sokoban_envs": {}}
    config = TaskAppConfig(
        app_id="sokoban",
        name="Sokoban Task App",
        description="Sokoban environment exposed as a Synth task app.",
        base_task_info=base,
        describe_taskset=lambda: {"id": "sokoban", "name": "Sokoban"},
        provide_task_instances=lambda seeds: taskset.provide_task_instances(seeds),
        rollout=rollout_executor,
        dataset_registry=None,
        rubrics=None,
        proxy=None,
        routers=(router,),
        app_state=app_state,
        cors_origins=["*"],
    )
    return config


# --- Health routes (auth-tolerant) ---
def fastapi_app():
    app = create_task_app(build_config())

    # Replace default health handlers to log expected ENVIRONMENT_API_KEY when unauthorized
    filtered_routes = []
    for route in app.router.routes:
        path = getattr(route, "path", None)
        methods = getattr(route, "methods", set()) or set()
        if path in {"/health", "/health/rollout"} and "GET" in methods:
            continue
        filtered_routes.append(route)
    app.router.routes = filtered_routes

    def _key_prefix() -> Optional[str]:
        key = normalize_environment_api_key()
        return key[: max(1, len(key) // 2)] if key else None

    @app.get("/health")
    async def health(request: Request):
        env_key = normalize_environment_api_key()
        if not env_key:
            return JSONResponse(status_code=503, content={"status": "unhealthy", "detail": "Missing ENVIRONMENT_API_KEY"})
        if not is_api_key_header_authorized(request):
            content: Dict[str, Any] = {"status": "healthy", "authorized": False}
            prefix = _key_prefix()
            if prefix:
                content["expected_api_key_prefix"] = prefix
            return JSONResponse(status_code=200, content=content)
        return {"status": "healthy", "authorized": True}

    @app.get("/health/rollout")
    async def health_rollout(request: Request):
        env_key = normalize_environment_api_key()
        if not env_key:
            return JSONResponse(status_code=503, content={"status": "unhealthy", "detail": "Missing ENVIRONMENT_API_KEY"})
        if not is_api_key_header_authorized(request):
            content: Dict[str, Any] = {"status": "healthy", "authorized": False}
            prefix = _key_prefix()
            if prefix:
                content["expected_api_key_prefix"] = prefix
            return JSONResponse(status_code=200, content=content)
        return {"ok": True, "authorized": True}

    # Basic env lifecycle routes (for local eval only)
    @app.post("/env/sokoban/initialize")
    async def initialize_env(request: Request, payload: Dict[str, Any]):
        difficulty = str((payload.get("config") or {}).get("difficulty") or "easy")
        seed = payload.get("seed")
        try:
            instance: SokobanTaskInstance = await create_task_instance_from_seed(difficulty, int(seed) if seed is not None else 0)
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc))
        env = SokobanEnvironment(instance)
        obs = await env.initialize()
        envs: Dict[str, SokobanEnvironment] = request.app.state.sokoban_envs
        env_id = f"{difficulty}:{seed or 0}"
        envs[env_id] = env
        return {"env_id": env_id, "observation": obs}

    @app.post("/env/sokoban/step")
    async def step_env(request: Request, payload: Dict[str, Any]):
        env_id = str(payload.get("env_id") or "")
        if not env_id:
            raise HTTPException(status_code=400, detail="env_id required")
        envs: Dict[str, SokobanEnvironment] = request.app.state.sokoban_envs
        env = envs.get(env_id)
        if not env:
            raise HTTPException(status_code=404, detail="Unknown env_id")

        action = None
        tool_calls = payload.get("tool_calls") or []
        if tool_calls:
            try:
                first = tool_calls[0] or {}
                args = first.get("args") or {}
                action = int(args.get("action")) if "action" in args else None
            except Exception:
                action = None
        if action is None and "action" in payload:
            try:
                action = int(payload.get("action"))
            except Exception:
                action = None
        if action is None:
            raise HTTPException(status_code=400, detail="action required")
        obs = await env.step(EnvToolCall(tool="interact", args={"action": int(action)}))
        return {"observation": obs}

    @app.post("/env/sokoban/terminate")
    async def terminate_env(request: Request, payload: Dict[str, Any]):
        env_id = str(payload.get("env_id") or "")
        envs: Dict[str, SokobanEnvironment] = request.app.state.sokoban_envs
        env = envs.pop(env_id, None)
        if env:
            obs = await env.terminate()
        else:
            obs = {"terminated": True}
        return {"ok": True, "observation": obs}

    @app.exception_handler(RequestValidationError)
    async def _on_validation_error(_request: Request, exc: RequestValidationError):
        return JSONResponse(status_code=422, content={"status": "invalid", "detail": exc.errors()[:5]})

    return app


register_task_app(
    entry=TaskAppEntry(
        app_id="sokoban",
        description="Sokoban task app",
        config_factory=build_config,
        aliases=("sokoban-rl",),
        env_files=(),
        modal=None,
    )
)
