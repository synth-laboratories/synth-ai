from __future__ import annotations

import contextlib
import json
import logging
import os
import time as _time
from datetime import datetime
from typing import Any, Mapping

from fastapi import APIRouter, HTTPException, Request, status
from pydantic import BaseModel, Field
from synth_ai.lm.vendors.base import BaseLMResponse
from synth_ai.task.tracing_utils import unique_sft_path
from synth_ai.tracing_v3.abstractions import EnvironmentEvent, LMCAISEvent, TimeRecord
from synth_ai.tracing_v3.llm_call_record_helpers import create_llm_call_record_from_response
from synth_ai.tracing_v3.session_tracer import SessionTracer

from .registry import registry

logger = logging.getLogger(__name__)


# --- Seeding utilities (robust, optional deps) ---
def _set_global_seed(seed_value: int) -> dict[str, Any]:
    """Set global RNG seeds across common libraries; return details for logging/restoration.

    Returns a dict containing which libraries were seeded and prior states if obtainable.
    """
    seeded: dict[str, Any] = {"seed": int(seed_value), "libs": []}
    with contextlib.suppress(Exception):
        import random as _random  # type: ignore

        _random.seed(seed_value)
        seeded["libs"].append("random")
    with contextlib.suppress(Exception):
        import numpy as _np  # type: ignore

        _np.random.seed(seed_value)
        seeded["libs"].append("numpy")
    with contextlib.suppress(Exception):
        import torch as _torch  # type: ignore

        if hasattr(_torch, "manual_seed"):
            _torch.manual_seed(seed_value)
            seeded["libs"].append("torch")
        # Make CUDA deterministic if present (best-effort)
        with contextlib.suppress(Exception):
            if getattr(_torch, "cuda", None) and _torch.cuda.is_available():
                _torch.cuda.manual_seed_all(seed_value)
                seeded.setdefault("cuda", True)
        # CUDNN deterministic flags (optional)
        with contextlib.suppress(Exception):
            if getattr(_torch, "backends", None) and getattr(_torch.backends, "cudnn", None):
                _torch.backends.cudnn.deterministic = True  # type: ignore[attr-defined]
                _torch.backends.cudnn.benchmark = False  # type: ignore[attr-defined]
    return seeded


def _clear_seed_side_effects() -> None:
    """Best-effort cleanup to avoid global deterministic side-effects between requests."""
    # We cannot truly restore prior RNG states without capturing them; we just avoid
    # leaving aggressive deterministic flags enabled where it matters.
    with contextlib.suppress(Exception):
        import torch as _torch  # type: ignore

        with contextlib.suppress(Exception):
            if getattr(_torch, "backends", None) and getattr(_torch.backends, "cudnn", None):
                # Re-enable cudnn.benchmark default True only if it was True; safest is False -> leave as is.
                # We'll keep deterministic False to avoid global impact; benchmark left False for stability.
                _torch.backends.cudnn.deterministic = False  # type: ignore[attr-defined]


router = APIRouter()


class RolloutEnvSpec(BaseModel):
    env_id: str | None = None
    env_name: str | None = None
    config: dict[str, Any] = {}
    seed: int | None = None


class RolloutPolicySpec(BaseModel):
    policy_id: str | None = None
    policy_name: str | None = None
    config: dict[str, Any] = {}


class RolloutBranchConfig(BaseModel):
    branch_every_n_steps: int = 0
    branch_on_condition: str | None = None
    max_branches: int = 0
    branch_policy: bool = False
    branch_env: bool = False


class RolloutRecordConfig(BaseModel):
    trajectories: bool = True
    logprobs: bool = False
    value: bool = False
    return_trace: bool = False
    trace_format: str = "compact"


class RolloutSafetyConfig(BaseModel):
    max_ops: int = 100000
    max_time_s: float = 3600.0


class RolloutRequest(BaseModel):
    run_id: str
    env: RolloutEnvSpec
    policy: RolloutPolicySpec
    ops: list[str]  # ["agent", "env", ...]
    record: RolloutRecordConfig = RolloutRecordConfig()
    on_done: str = "reset"  # "reset" | "terminate"
    branch: RolloutBranchConfig | None = None
    safety: RolloutSafetyConfig = RolloutSafetyConfig()
    # Optional run/session context
    training_session_id: str | None = None
    synth_base_url: str | None = None


class RolloutStep(BaseModel):
    obs: dict[str, Any]
    tool_calls: list[dict[str, Any]]
    reward: float | None = None
    done: bool = False
    truncated: bool | None = None
    logprob: float | None = None
    value: float | None = None
    info: dict[str, Any] | None = None


class RolloutTrajectory(BaseModel):
    env_id: str
    policy_id: str
    steps: list[RolloutStep]
    final: dict[str, Any] | None = None
    length: int
    decision_samples: list[dict[str, Any]] | None = None


def _normalize_step_strategy(raw_strategy: Any) -> str:
    if not isinstance(raw_strategy, str):
        return "consistent"
    candidate = raw_strategy.strip().lower()
    if not candidate:
        return "consistent"
    mapping = {
        "simple": "consistent",
        "consistent": "consistent",
        "consistent_stepwise": "consistent",
        "decision_consistent": "consistent",
        "per_achievement": "per_achievement",
        "per-achievement": "per_achievement",
        "perachievement": "per_achievement",
        "achievement_weighted": "per_achievement",
        "complex": "per_achievement",
    }
    return mapping.get(candidate, "consistent")


def _coerce_weights(raw_weights: Any) -> dict[str, float]:
    weights: dict[str, float] = {}
    if isinstance(raw_weights, dict):
        for key, value in raw_weights.items():
            try:
                weights[str(key)] = float(value)
            except Exception:
                continue
    return weights


def _coerce_k_limits(raw_limits: Any) -> dict[str, int]:
    limits: dict[str, int] = {}
    if isinstance(raw_limits, dict):
        for key, value in raw_limits.items():
            try:
                limits[str(key)] = int(value)
            except Exception:
                continue
    return limits


def _coerce_int_value(value: Any) -> int | None:
    if isinstance(value, bool):
        return int(value)
    try:
        return int(value)  # type: ignore[arg-type]
    except Exception:
        try:
            return int(float(value))  # type: ignore[arg-type]
        except Exception:
            return None


def _compute_resource_reward(
    prev_inventory: Mapping[str, Any] | None,
    new_inventory: Mapping[str, Any] | None,
    prev_counts: Mapping[str, Any] | None,
    new_counts: Mapping[str, Any] | None,
) -> tuple[float, list[dict[str, Any]], dict[str, int], dict[str, int]]:
    reward_total = 0.0
    components: list[dict[str, Any]] = []
    inventory_deltas: dict[str, int] = {}
    achievement_deltas: dict[str, int] = {}

    resource_weights = {
        "wood": 0.10,
        "sapling": 0.08,
        "stone": 0.15,
        "coal": 0.18,
        "iron": 0.22,
        "plant": 0.06,
        "meat": 0.12,
        "drink": 0.07,
        "food": 0.07,
        "water": 0.07,
        "energy": 0.04,
    }
    tool_weights = {
        "wood_pickaxe": 0.40,
        "stone_pickaxe": 0.55,
        "iron_pickaxe": 0.75,
        "wood_sword": 0.35,
        "stone_sword": 0.50,
        "iron_sword": 0.70,
        "furnace": 0.45,
        "table": 0.30,
        "bow": 0.45,
    }
    achievement_weights = {
        "collect_wood": 0.08,
        "collect_sapling": 0.06,
        "collect_stone": 0.10,
        "collect_coal": 0.12,
        "collect_iron": 0.14,
        "collect_drink": 0.06,
        "collect_food": 0.06,
        "collect_plant": 0.06,
    }
    default_resource_weight = 0.05
    default_achievement_weight = 0.05

    prev_inv = prev_inventory or {}
    new_inv = new_inventory or {}
    for key, raw_value in new_inv.items():
        new_val = _coerce_int_value(raw_value)
        if new_val is None:
            continue
        prev_val = _coerce_int_value(prev_inv.get(key, 0)) or 0
        delta = new_val - prev_val
        if delta <= 0:
            continue
        weight = resource_weights.get(key)
        if weight is None and key in tool_weights:
            weight = tool_weights[key]
        if weight is None:
            weight = default_resource_weight
        gain = weight * delta
        reward_total += gain
        inventory_deltas[str(key)] = delta
        components.append(
            {
                "type": "inventory",
                "item": str(key),
                "delta": delta,
                "weight": weight,
                "reward": gain,
            }
        )

    prev_ct = prev_counts or {}
    new_ct = new_counts or {}
    for key, raw_value in new_ct.items():
        new_val = _coerce_int_value(raw_value)
        if new_val is None:
            continue
        prev_val = _coerce_int_value(prev_ct.get(key, 0)) or 0
        delta = new_val - prev_val
        if delta <= 0:
            continue
        weight = achievement_weights.get(key, default_achievement_weight)
        gain = weight * delta
        reward_total += gain
        achievement_deltas[str(key)] = delta
        components.append(
            {
                "type": "achievement_count",
                "name": str(key),
                "delta": delta,
                "weight": weight,
                "reward": gain,
            }
        )

    return reward_total, components, inventory_deltas, achievement_deltas


def compute_stepwise_reward(
    prev_achievements: dict[str, bool],
    new_achievements: dict[str, bool],
    decision_index: int,
    actions_summary: list[dict[str, Any]],
    indicator_lambda: float,
    *,
    strategy: str | None = None,
    weights: dict[str, float] | None = None,
    k_limits: dict[str, int] | None = None,
    episode_counts: dict[str, int] | None = None,
    prev_inventory: dict[str, int] | None = None,
    new_inventory: dict[str, int] | None = None,
    prev_counts: dict[str, int] | None = None,
    new_counts: dict[str, int] | None = None,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, float]]:
    """Compute stepwise reward metadata given achievement states before/after a decision."""

    prev_map = prev_achievements or {}
    next_map = new_achievements or {}

    unlocked = [name for name, value in next_map.items() if value and not prev_map.get(name, False)]
    indicator_from_achievements = 1 if unlocked else 0
    normalized_strategy = _normalize_step_strategy(strategy)
    base_reward = 0.0
    reward_components: list[dict[str, Any]] = []
    credited: list[str] = []

    if indicator_from_achievements:
        if normalized_strategy == "per_achievement":
            weight_map = weights or {}
            limit_map = k_limits or {}
            counts = episode_counts if isinstance(episode_counts, dict) else {}
            for name in unlocked:
                try:
                    limit_val = int(limit_map.get(name, 1))
                except Exception:
                    limit_val = 1
                # limit_val <= 0 implies unlimited rewards
                unlimited = limit_val <= 0
                try:
                    prev_count = int(counts.get(name, 0))
                except Exception:
                    prev_count = 0
                should_credit = unlimited or (prev_count < max(limit_val, 0))
                if should_credit:
                    try:
                        weight_val = float(weight_map.get(name, 1.0))
                    except Exception:
                        weight_val = 1.0
                    base_reward += weight_val
                    reward_components.append(
                        {
                            "achievement": name,
                            "weight": weight_val,
                            "count_prior": prev_count,
                            "count_limit": limit_val,
                        }
                    )
                    credited.append(name)
                    if episode_counts is not None:
                        episode_counts[name] = prev_count + 1
        else:
            base_reward = 1.0
            reward_components.append(
                {
                    "achievement": "__indicator__",
                    "weight": 1.0,
                    "count_prior": 0,
                    "count_limit": 1,
                }
            )

    resource_reward = 0.0
    resource_components: list[dict[str, Any]] = []
    inventory_deltas: dict[str, int] = {}
    achievement_deltas: dict[str, int] = {}
    if normalized_strategy == "per_achievement":
        (
            resource_reward,
            resource_components,
            inventory_deltas,
            achievement_deltas,
        ) = _compute_resource_reward(prev_inventory, new_inventory, prev_counts, new_counts)
        if resource_components:
            reward_components.extend(resource_components)
        base_reward += resource_reward

    indicator = 1 if base_reward > 0 else 0
    if indicator == 0 and indicator_from_achievements:
        indicator = indicator_from_achievements
    lambda_effective = indicator_lambda if indicator_lambda not in (None, 0) else 1.0
    reward_value = float(lambda_effective) * float(base_reward)

    stepwise_info = {
        "decision_index": decision_index,
        "indicator": indicator,
        "new_achievements": unlocked,
        "reward": reward_value,
        "strategy": normalized_strategy,
        "base_reward": float(base_reward),
    }
    if indicator_from_achievements and not unlocked:
        stepwise_info["indicator_from_achievements"] = indicator_from_achievements
    if reward_components:
        stepwise_info["components"] = reward_components
    if credited:
        stepwise_info["credited_achievements"] = credited
    if resource_reward:
        stepwise_info["resource_reward"] = float(resource_reward)
    if inventory_deltas:
        stepwise_info["inventory_deltas"] = inventory_deltas
    if achievement_deltas:
        stepwise_info["achievement_count_deltas"] = achievement_deltas

    decision_sample = {
        "decision_index": decision_index,
        "indicator": indicator,
        "r_i": reward_value,
        "base": float(base_reward),
        "strategy": normalized_strategy,
        "actions": actions_summary,
    }
    if reward_components:
        decision_sample["components"] = reward_components
    if resource_reward:
        decision_sample["resource_reward"] = float(resource_reward)

    stats = {
        "indicator": float(indicator),
        "reward": reward_value,
        "new_achievements_count": float(len(unlocked)),
        "base_reward": float(base_reward),
        "credited_achievements_count": float(len(credited)),
    }
    if resource_reward:
        stats["resource_reward"] = float(resource_reward)
    return stepwise_info, decision_sample, stats


class RolloutMetrics(BaseModel):
    episode_returns: list[float]
    mean_return: float
    num_steps: int
    num_episodes: int = 0
    outcome_score: float | None = None
    events_score: float | None = None
    details: dict[str, Any] = Field(default_factory=dict)


class RolloutResponse(BaseModel):
    run_id: str
    trajectories: list[RolloutTrajectory]
    branches: dict[str, list[str]] = {}
    metrics: RolloutMetrics
    aborted: bool = False
    ops_executed: int = 0
    trace: dict[str, Any] | None = None


class RolloutTracingContext:
    """Helper managing tracing_v3 recording and optional SFT dumps for a rollout."""

    def __init__(
        self,
        tracer: SessionTracer | None,
        request: RolloutRequest,
        fastapi_request: Request,
    ) -> None:
        self.tracer = tracer
        self.enabled = tracer is not None
        self.request = request
        self.fastapi_request = fastapi_request
        self.run_id = request.run_id
        self.current_step_id: str | None = None
        self.current_turn: int | None = None
        self.lm_calls_summary: list[dict[str, Any]] = []
        self.decision_rewards: list[dict[str, Any]] = []
        self.sft_records: list[dict[str, Any]] = []
        self.latest_system_messages: list[str] = []
        self.latest_user_messages: list[str] = []
        self.latest_system_prompt_content: list[Any] = []
        self.latest_user_prompt_content: list[Any] = []
        self.trace_format = (
            getattr(request.record, "trace_format", "compact") or "compact"
        ).lower()
        self.return_trace = bool(getattr(request.record, "return_trace", False))
        self.sft_output_dir = getattr(fastapi_request.app.state, "sft_output_dir", None)
        self.session_trace = None
        self.metadata_updates: dict[str, Any] = {}
        self.policy_name = request.policy.policy_name or ""
        self.env_name = request.env.env_name or ""
        self.metadata_base: dict[str, Any] = {
            "run_id": self.run_id,
            "policy_name": self.policy_name,
            "policy_id": request.policy.policy_id,
            "env_name": self.env_name,
            "env_id": request.env.env_id,
            "seed": request.env.seed,
            "training_session_id": request.training_session_id,
            "synth_base_url": request.synth_base_url,
        }

        # Expose context for downstream calls inside this request lifecycle
        fastapi_request.state.rollout_tracing = self
        fastapi_request.state.rollout_run_id = self.run_id

    async def start_session(self) -> None:
        if not self.enabled or self.tracer is None:
            return
        try:
            await self.tracer.initialize()
        except Exception as exc:
            logger.debug("TRACING_INIT_FAIL: %s", exc)
        try:
            await self.tracer.start_session(
                session_id=self.run_id, metadata=dict(self.metadata_base)
            )
        except Exception as exc:
            logger.info("TRACING_START_FAIL: %s", exc)
            self.enabled = False
            self.tracer = None

    async def start_decision(self, turn_number: int) -> None:
        self.current_turn = turn_number
        self.current_step_id = f"decision_{turn_number}"
        if not self.enabled or self.tracer is None:
            return
        try:
            await self.tracer.start_timestep(step_id=self.current_step_id, turn_number=turn_number)
        except Exception as exc:
            logger.debug("TRACING_STEP_START_FAIL: %s", exc)

    async def end_decision(self) -> None:
        if not self.enabled or self.tracer is None:
            return
        try:
            await self.tracer.end_timestep(step_id=self.current_step_id)
        except Exception as exc:
            logger.debug("TRACING_STEP_END_FAIL: %s", exc)
        finally:
            self.current_step_id = None

    def _message_metadata(self) -> dict[str, Any]:
        return {
            "turn": self.current_turn,
            "step_id": self.current_step_id,
        }

    async def record_policy_prompts(
        self,
        system_messages: list[Any],
        user_messages: list[Any],
    ) -> None:
        self.latest_system_messages = [self._prompt_text(entry) for entry in system_messages]
        self.latest_user_messages = [self._prompt_text(entry) for entry in user_messages]
        self.latest_system_prompt_content = [
            self._prompt_content(entry, role="system") for entry in system_messages
        ]
        self.latest_user_prompt_content = [
            self._prompt_content(entry, role="user") for entry in user_messages
        ]
        if not self.enabled or self.tracer is None:
            return
        for entry in system_messages:
            try:
                await self.tracer.record_message(
                    content=self._prompt_payload(entry, role="system"),
                    message_type="policy_system_prompt",
                    metadata=self._message_metadata(),
                )
            except Exception as exc:
                logger.debug("TRACING_SYSTEM_MSG_FAIL: %s", exc)
        for entry in user_messages:
            try:
                await self.tracer.record_message(
                    content=self._prompt_payload(entry, role="user"),
                    message_type="policy_user_prompt",
                    metadata=self._message_metadata(),
                )
            except Exception as exc:
                logger.debug("TRACING_USER_MSG_FAIL: %s", exc)

    def _content_to_text(self, content: Any) -> str:
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts: list[str] = []
            for seg in content:
                if isinstance(seg, dict):
                    text_val = seg.get("text") or seg.get("content")
                    if isinstance(text_val, str):
                        parts.append(text_val)
            return "".join(parts)
        if content is None:
            return ""
        return str(content)

    def _prompt_text(self, entry: Any) -> str:
        if isinstance(entry, dict):
            text = entry.get("text")
            if isinstance(text, str):
                return text
            content = entry.get("content")
            return self._content_to_text(content)
        return self._content_to_text(entry)

    def _prompt_payload(self, entry: Any, *, role: str) -> dict[str, Any]:
        if isinstance(entry, dict):
            payload = dict(entry)
            payload.setdefault("role", role)
            return payload
        return {
            "role": role,
            "text": self._prompt_text(entry),
            "content": entry,
        }

    def _prompt_content(self, entry: Any, *, role: str) -> Any:
        payload = self._prompt_payload(entry, role=role)
        return payload.get("content", payload.get("text"))

    def _content_has_image(self, content: Any) -> bool:
        if isinstance(content, list):
            return any(
                isinstance(seg, dict)
                and seg.get("type") in {"image", "image_url"}
                for seg in content
            )
        if isinstance(content, dict):
            if content.get("type") in {"image", "image_url"}:
                return True
            inner = content.get("content")
            if isinstance(inner, list):
                return any(
                    isinstance(seg, dict)
                    and seg.get("type") in {"image", "image_url"}
                    for seg in inner
                )
        return False

    def _safe_json(self, payload: Any, limit: int = 4000) -> str:
        try:
            text = json.dumps(payload, ensure_ascii=False)
        except Exception:
            text = str(payload)
        if len(text) > limit:
            return text[:limit] + "…"
        return text

    async def record_tool_invocation(self, tool_calls: list[dict[str, Any]] | None) -> None:
        if tool_calls is None:
            return
        if self.enabled and self.tracer is not None:
            try:
                await self.tracer.record_message(
                    content=self._safe_json(tool_calls),
                    message_type="policy_tool_call",
                    metadata=self._message_metadata(),
                )
            except Exception as exc:
                logger.debug("TRACING_TOOL_MSG_FAIL: %s", exc)

    async def _record_event(self, event: Any) -> int | None:
        if not self.enabled or self.tracer is None:
            return None
        try:
            return await self.tracer.record_event(event)
        except Exception as exc:
            logger.debug("TRACING_EVENT_FAIL: %s", exc)
            return None

    async def record_llm_call(
        self,
        *,
        inference_request: dict[str, Any],
        inference_response: dict[str, Any],
        tool_calls: list[dict[str, Any]] | None,
        provider: str,
        model_name: str,
        started_at: datetime,
        completed_at: datetime,
        latency_ms: int | None,
    ) -> None:
        usage = inference_response.get("usage") or {}
        input_tokens = usage.get("input_tokens") or usage.get("prompt_tokens")
        output_tokens = usage.get("output_tokens") or usage.get("completion_tokens")
        total_tokens = usage.get("total_tokens")
        cost_usd = usage.get("cost_usd") or usage.get("cost") or usage.get("total_cost")

        assistant_message = None
        choices = inference_response.get("choices") or []
        if choices:
            assistant_message = choices[0].get("message") or {}
        assistant_content = (
            assistant_message.get("content") if isinstance(assistant_message, dict) else None
        )

        raw_response = self._content_to_text(assistant_content)
        if not raw_response:
            raw_response = self._safe_json(inference_response, limit=2000)

        base_response = BaseLMResponse(
            raw_response=raw_response,
            tool_calls=assistant_message.get("tool_calls")
            if isinstance(assistant_message, dict)
            else None,
            usage=usage or None,
            api_type="chat_completions",
        )

        request_messages = inference_request.get("messages") or []
        try:
            temperature = float(inference_request.get("temperature"))
        except Exception:
            temperature = 0.0

        call_record = create_llm_call_record_from_response(
            response=base_response,
            model_name=model_name,
            provider=provider,
            messages=request_messages,
            temperature=temperature,
            request_params=inference_request,
            tools=inference_request.get("tools"),
            started_at=started_at,
            completed_at=completed_at,
            latency_ms=latency_ms,
        )

        event_metadata = {
            "policy_id": self.request.policy.policy_id,
            "turn": self.current_turn,
            "run_id": self.run_id,
        }

        event = LMCAISEvent(
            system_instance_id=f"policy:{self.policy_name or 'unknown'}",
            time_record=TimeRecord(event_time=completed_at.timestamp()),
            model_name=model_name,
            provider=provider,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=total_tokens,
            cost_usd=cost_usd,
            latency_ms=latency_ms,
            call_records=[call_record],
            metadata=event_metadata,
        )

        await self._record_event(event)

        self.lm_calls_summary.append(
            {
                "turn": self.current_turn,
                "model": model_name,
                "provider": provider,
                "total_tokens": total_tokens,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "latency_ms": latency_ms,
                "tool_calls": len(tool_calls or []),
            }
        )

        if self.sft_output_dir is not None:
            assistant_structured = assistant_content if assistant_content is not None else ""
            assistant_text = self._content_to_text(assistant_content)
            dialogue_structured: list[dict[str, Any]] = []
            for content in self.latest_system_prompt_content:
                if content is None:
                    continue
                dialogue_structured.append({"role": "system", "content": content})
            for content in self.latest_user_prompt_content:
                if content is None:
                    continue
                dialogue_structured.append({"role": "user", "content": content})
            dialogue_text = (
                [{"role": "system", "content": s} for s in self.latest_system_messages]
                + [{"role": "user", "content": u} for u in self.latest_user_messages]
            )
            user_has_image = any(
                self._content_has_image(content) for content in self.latest_user_prompt_content
            )
            assistant_has_image = self._content_has_image(assistant_structured)
            record = {
                "run_id": self.run_id,
                "turn": self.current_turn,
                "model": model_name,
                "provider": provider,
                "dialogue": dialogue_structured,
                "dialogue_text": dialogue_text,
                "assistant": {
                    "content": assistant_structured,
                    "content_text": assistant_text,
                    "tool_calls": assistant_message.get("tool_calls")
                    if isinstance(assistant_message, dict)
                    else [],
                    "has_image": assistant_has_image,
                },
                "metadata": {
                    "user_has_image": user_has_image,
                    "assistant_has_image": assistant_has_image,
                    "has_image": user_has_image or assistant_has_image,
                },
                "timestamp": datetime.utcnow().isoformat(),
            }
            self.sft_records.append(record)

    async def record_environment_event(
        self,
        *,
        env_handle: Any,
        prev_obs: dict[str, Any] | None,
        env_response: Any,
        next_obs: dict[str, Any] | None,
        metadata: dict[str, Any] | None = None,
    ) -> int | None:
        if not self.enabled or self.tracer is None:
            return None

        try:
            prev_summary = (
                _summarize_observation_for_storage(env_handle, prev_obs or {})
                if prev_obs is not None
                else None
            )
        except Exception:
            prev_summary = None
        try:
            next_summary = (
                _summarize_observation_for_storage(env_handle, next_obs or {})
                if next_obs is not None
                else None
            )
        except Exception:
            next_summary = None

        reward_val = getattr(env_response, "reward", None)
        try:
            reward_float = float(reward_val) if reward_val is not None else 0.0
        except Exception:
            reward_float = 0.0

        event = EnvironmentEvent(
            system_instance_id=f"environment:{self.env_name or 'unknown'}",
            time_record=TimeRecord(event_time=datetime.utcnow().timestamp()),
            reward=reward_float,
            terminated=bool(getattr(env_response, "done", False)),
            truncated=bool(getattr(env_response, "truncated", False)),
            system_state_before=prev_summary,
            system_state_after=next_summary,
            metadata={
                "turn": self.current_turn,
                "run_id": self.run_id,
                **(metadata or {}),
            },
        )

        return await self._record_event(event)

    async def record_decision_reward(
        self,
        *,
        event_id: int | None,
        decision_meta: dict[str, Any] | None,
    ) -> None:
        decision_meta = decision_meta or {}
        ach_delta = int(decision_meta.get("ach_delta", 0))
        unique_delta = int(decision_meta.get("unique_delta", 0))
        all_ach = list(decision_meta.get("all") or [])
        unique_ach = list(decision_meta.get("unique") or [])

        self.decision_rewards.append(
            {
                "turn": self.current_turn,
                "ach_delta": ach_delta,
                "unique_delta": unique_delta,
                "achievements": all_ach,
                "unique_achievements": unique_ach,
            }
        )

        if not self.enabled or self.tracer is None or event_id is None:
            return
        try:
            await self.tracer.record_event_reward(
                event_id=event_id,
                turn_number=self.current_turn,
                reward_value=float(ach_delta),
                reward_type="achievement_delta",
                annotation={"achievements": all_ach},
                source="environment",
            )
            if unique_delta:
                await self.tracer.record_event_reward(
                    event_id=event_id,
                    turn_number=self.current_turn,
                    reward_value=float(unique_delta),
                    reward_type="unique_achievement_delta",
                    annotation={"achievements": unique_ach},
                    source="environment",
                )
        except Exception as exc:
            logger.debug("TRACING_REWARD_FAIL: %s", exc)

    def update_metadata(self, **kwargs: Any) -> None:
        self.metadata_updates.update({k: v for k, v in kwargs.items() if v is not None})

    async def finalize(
        self,
        *,
        total_reward: float,
        achievement_state: dict[str, bool] | None,
        total_steps: int,
    ) -> Any:
        final_achievements = [key for key, val in (achievement_state or {}).items() if val]
        self.metadata_updates.setdefault("final_achievements", final_achievements)
        if self.enabled and self.tracer is not None:
            try:
                await self.tracer.record_outcome_reward(
                    total_reward=int(total_reward),
                    achievements_count=len(final_achievements),
                    total_steps=int(total_steps),
                    reward_metadata=dict(self.metadata_updates),
                )
            except Exception as exc:
                logger.debug("TRACING_OUTCOME_FAIL: %s", exc)
            try:
                self.session_trace = await self.tracer.end_session()
                if self.session_trace is not None:
                    self.session_trace.metadata.update(self.metadata_updates)
            except Exception as exc:
                logger.debug("TRACING_END_SESSION_FAIL: %s", exc)
                self.session_trace = None
            with contextlib.suppress(Exception):
                await self.tracer.close()

        if self.sft_records and self.sft_output_dir:
            self.write_sft_records()

        # Clear context from request state to avoid leaks
        self.fastapi_request.state.rollout_tracing = None

        return self.session_trace

    def write_sft_records(self) -> None:
        if not self.sft_output_dir or not self.sft_records:
            return
        try:
            path = unique_sft_path(self.sft_output_dir, run_id=self.run_id)
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("w", encoding="utf-8") as fh:
                for record in self.sft_records:
                    json.dump(record, fh, ensure_ascii=False)
                    fh.write("\n")
            logger.info(f"SFT_WRITTEN: {path}")
        except Exception as exc:
            logger.warning(f"SFT_WRITE_FAIL: {exc}")
        finally:
            self.sft_records.clear()

    def build_trace_payload(self, session_trace: Any) -> dict[str, Any] | None:
        if not self.return_trace or session_trace is None:
            return None
        if self.trace_format == "full":
            payload = session_trace.to_dict()
            payload.setdefault("metadata", {}).update(self.metadata_updates)
            return payload
        metadata = dict(session_trace.metadata)
        metadata.update(self.metadata_updates)
        return {
            "session_id": session_trace.session_id,
            "created_at": session_trace.created_at.isoformat(),
            "metadata": metadata,
            "events_count": len(session_trace.event_history),
            "messages_count": len(session_trace.markov_blanket_message_history),
            "lm_calls": self.lm_calls_summary,
            "decision_rewards": self.decision_rewards,
        }


def _summarize_observation_for_storage(
    env_handle: Any, observation: dict[str, Any]
) -> dict[str, Any]:
    """Return a compact dict for trajectory storage instead of the raw observation.

    - For Crafter, use the same summary used for the policy user prompt
    - For others, keep a minimal subset or plain text preview
    """
    # Try Crafter-specific formatter
    crafter_wrapper = None
    with contextlib.suppress(Exception):
        from .envs.crafter.environment import (
            CrafterEnvironmentWrapper as _CrafterWrapper,  # type: ignore
        )

        crafter_wrapper = _CrafterWrapper  # type: ignore[assignment]

    if crafter_wrapper is not None and isinstance(
        getattr(env_handle, "env", None), crafter_wrapper
    ):
        with contextlib.suppress(Exception):
            from .envs.crafter.shared import format_observation as _fmt  # type: ignore

            text = _fmt(observation or {})
            return {"text": text}

    # Generic fallback: extract a few small fields if present; avoid huge arrays
    with contextlib.suppress(Exception):
        inv = observation.get("inventory") if isinstance(observation, dict) else None
        ach = observation.get("achievements_status") if isinstance(observation, dict) else None
        pos = observation.get("player_position") if isinstance(observation, dict) else None
        health = None
        if isinstance(inv, dict):
            health = inv.get("health")
        summary = {
            "position": pos,
            "health": health,
            "inventory_keys": sorted(k for k, v in (inv or {}).items() if v)[:10]
            if isinstance(inv, dict)
            else None,
            "achievements_unlocked": sorted(k for k, v in (ach or {}).items() if v)[:10]
            if isinstance(ach, dict)
            else None,
        }
        return {"text": json.dumps(summary, ensure_ascii=False)}

    # Last resort: plain string preview
    try:
        return {"text": str(observation)[:10000]}
    except Exception:
        return {"text": ""}


class RunAbortRequest(BaseModel):
    run_id: str


class RunAbortResponse(BaseModel):
    ok: bool
    run_id: str


class RunStatusResponse(BaseModel):
    run_id: str
    status: str
    started_at: datetime
    finished_at: datetime | None = None


@router.post("/rollout", response_model=RolloutResponse)
async def execute_rollout(
    request: RolloutRequest,
    req: Request,
) -> RolloutResponse:
    """Execute a rollout with coordinated environment and policy steps."""
    # Emit rollout identifier early for correlation
    with contextlib.suppress(Exception):
        _rid = getattr(request, "run_id", None)
        _pol = getattr(request.policy, "policy_name", None) or getattr(request.policy, "policy_id", None)
        _env = getattr(request.env, "env_name", None) or getattr(request.env, "env_id", None)
        logger.info("ROLLOUT_BEGIN: run_id=%s policy=%s env=%s", _rid, _pol, _env)
        print(f"[rollout] begin run_id={_rid} policy={_pol} env={_env}", flush=True)
    # Enforce per-episode step cap via env-specific parameters; default to 20 if omitted
    try:
        _env_params = {}
        if isinstance(request.env, RolloutEnvSpec) and isinstance(request.env.config, dict):
            _env_params = dict(request.env.config.get("env_params") or {})
        max_steps_per_episode = int(_env_params.get("max_steps_per_episode") or 20)
        assert max_steps_per_episode > 0, "max_steps_per_episode must be a positive integer"
    except Exception as _mse:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={
                "error": "invalid_env_params",
                "message": f"Invalid or missing env_params.max_steps_per_episode: {_mse}",
            },
        ) from _mse
    # Truncate incoming ops to the enforced cap (each step is [agent, env])
    ops_seq: list[str] = list(request.ops or [])
    allowed_ops = max(0, int(max_steps_per_episode) * 2)
    if len(ops_seq) > allowed_ops:
        with contextlib.suppress(Exception):
            logger.info(
                "ROLL_OUT: truncating ops to cap: requested_ops=%s allowed_ops=%s",
                str(len(ops_seq)),
                str(allowed_ops),
            )
        ops_seq = ops_seq[:allowed_ops]
    # Simple API key auth for inbound rollout
    header_key = req.headers.get("x-api-key")
    env_key = os.getenv("ENVIRONMENT_API_KEY")
    dev_key = os.getenv("DEV_ENVIRONMENT_API_KEY")
    # Accept either ENVIRONMENT_API_KEY or DEV_ENVIRONMENT_API_KEY
    expected_keys = [k for k in (env_key, dev_key) if k]
    if not expected_keys:
        missing = []
        if not env_key:
            missing.append("ENVIRONMENT_API_KEY")
        if not dev_key:
            missing.append("DEV_ENVIRONMENT_API_KEY")
        msg = f"Auth not configured: missing {', '.join(missing)} in task service environment"
        logger.error(msg)
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=msg)
    if not header_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API key: X-API-Key header not provided",
        )
    if header_key not in expected_keys:
        # Do not leak secrets; include short prefix for diagnostics
        exp_src = env_key if env_key else (dev_key or "")
        exp_prefix = (exp_src[:7] + "…") if len(exp_src) >= 7 else "set"
        got_prefix = (header_key[:7] + "…") if len(header_key) >= 7 else "set"
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Invalid API key: header does not match expected (got={got_prefix}, expected_prefix={exp_prefix})",
        )

    # Log contextual fields for traceability
    if request.training_session_id:
        logger.info(f"ROLL_OUT: training_session_id={request.training_session_id}")
    if request.synth_base_url:
        logger.info(f"ROLL_OUT: synth_base_url={request.synth_base_url}")

    # Log masked OpenAI API key presence for diagnostics
    with contextlib.suppress(Exception):
        _oa = os.getenv("OPENAI_API_KEY")
        if _oa:
            _pref = (_oa[:6] + "…") if len(_oa) >= 6 else "set"
            logger.info(f"ROLL_OUT: OPENAI_API_KEY present (prefix={_pref})")
        else:
            logger.warning("ROLL_OUT: OPENAI_API_KEY missing")

    # Make synth_base_url available for outbound calls in this app
    with contextlib.suppress(Exception):
        task_app = req.app.state.task_app
        if request.synth_base_url:
            task_app.synth_base_url = request.synth_base_url

    tracer_factory = getattr(req.app.state, "session_tracer_factory", None)
    tracer_instance: SessionTracer | None = None
    if callable(tracer_factory):
        try:
            inst = tracer_factory()
            tracer_instance = inst if isinstance(inst, SessionTracer) else None
        except Exception as exc:
            logger.debug(f"TRACER_FACTORY_FAIL: {exc}")
    tracing_context = RolloutTracingContext(tracer_instance, request, req)
    await tracing_context.start_session()
    # Print whether tracing is active for this rollout
    try:
        print(
            f"[rollout] tracing enabled={bool(tracing_context.enabled)} run_id={request.run_id}",
            flush=True,
        )
    except Exception:
        pass

    # Register run
    registry.register_run(request.run_id)

    # Track resources created during this rollout so we can guarantee cleanup
    created_env_id: str | None = None
    created_policy_id: str | None = None
    env_seed_used: int | None = None
    trajectory_steps: list[RolloutStep] = []
    decision_samples: list[dict[str, Any]] = []
    pending_tool_calls: Any = None
    current_obs: Any = {}
    total_reward: float = 0.0
    ops_executed = 0
    last_agent_response_ts: float | None = None
    last_policy_meta: dict[str, Any] | None = None
    last_env_step_ms: float | None = None
    last_env_step_completed_ts: float | None = None
    decision_open = False
    finalized = False
    prev_achievements: dict[str, bool] = {}
    session_trace = None
    step_rewards_active = False

    try:
        # Initialize deterministic seed early for the entire rollout
        seed_value: int | None = None
        try:
            if request.env and request.env.seed is not None:
                seed_value = int(request.env.seed)
            else:
                # Derive a stable seed from run_id
                import hashlib as _hashlib  # local import to avoid global deps

                _digest = _hashlib.sha256(request.run_id.encode("utf-8")).hexdigest()
                # Use lower 32 bits to fit common RNG ranges
                seed_value = int(_digest[:8], 16)
        except Exception:
            # Fallback to time-based seed if anything goes wrong
            try:
                seed_value = int((_time.time_ns() // 1_000_000) % (2**31 - 1))
            except Exception:
                seed_value = 42

        _seed_info = _set_global_seed(int(seed_value))
        with contextlib.suppress(Exception):
            logger.info(
                "ROLL_OUT: RNG seeded seed=%s libs=%s",
                str(_seed_info.get("seed")),
                ",".join(_seed_info.get("libs", [])),
            )
        # Resolve or create environment
        if request.env.env_id:
            env_handle = registry.get_env(request.env.env_id)
            if not env_handle:
                raise HTTPException(
                    status_code=404,
                    detail=f"Environment {request.env.env_id} not found",
                )
            env_id = request.env.env_id
        else:
            # Create new environment
            from .environment_routes import EnvCreateRequest, create_environment

            if not request.env.env_name:
                raise ValueError("FATAL: env_name is required - NO FALLBACKS!")

            # Propagate training_session_id via env config for downstream usage
            _env_config = dict(request.env.config or {})
            if request.training_session_id is not None:
                _env_config.setdefault("training_session_id", request.training_session_id)
            env_response = await create_environment(
                EnvCreateRequest(
                    env_name=request.env.env_name,
                    config=_env_config,
                    seed=request.env.seed,
                    rl_run_id=request.run_id,
                )
            )
            env_id = env_response.env_id
            env_handle = registry.get_env(env_id)
            created_env_id = env_id

        tracing_context.update_metadata(env_id=env_id)

        # Resolve or create policy
        if request.policy.policy_id:
            policy_handle = registry.get_policy(request.policy.policy_id)
            if not policy_handle:
                raise HTTPException(
                    status_code=404,
                    detail=f"Policy {request.policy.policy_id} not found",
                )
            policy_id = request.policy.policy_id
        else:
            # Create new policy
            from .policy_routes import PolicyCreateRequest, create_policy

            if not request.policy.policy_name:
                raise ValueError("FATAL: policy_name is required - NO FALLBACKS!")

            # Propagate training_session_id and synth_base_url via policy config
            _policy_config = dict(request.policy.config or {})
            if request.training_session_id is not None:
                _policy_config.setdefault("training_session_id", request.training_session_id)
            if request.synth_base_url is not None:
                _policy_config.setdefault("synth_base_url", request.synth_base_url)
            policy_response = await create_policy(
                PolicyCreateRequest(
                    policy_name=request.policy.policy_name,
                    config=_policy_config,
                    rl_run_id=request.run_id,
                    bound_env_id=env_id,
                ),
                req,
            )
            policy_id = policy_response.policy_id
            policy_handle = registry.get_policy(policy_id)
            created_policy_id = policy_id

        tracing_context.update_metadata(policy_id=policy_id)

        # Bind policy to environment if not already bound
        if policy_handle and not policy_handle.bound_env_id:
            policy_handle.bound_env_id = env_id

        # Record seed bound to environment for end-of-rollout verification/logging
        try:
            env_seed_used = int(getattr(env_handle, "seed", 0) or 0)
        except Exception:
            env_seed_used = None
        tracing_context.update_metadata(env_seed=env_seed_used)
        # Initialize trajectory
        trajectory_steps = []
        pending_tool_calls = None
        current_obs = env_handle.last_observation
        total_reward = 0.0
        ops_executed = 0
        last_agent_response_ts = None
        last_policy_meta = None
        last_env_step_ms = None
        last_env_step_completed_ts = None

        # Stepwise reward configuration (Crafter shaping; gate on explicit enable)
        step_rewards_cfg_raw: dict[str, Any] = {}
        try:
            if isinstance(request.policy.config, dict):
                step_rewards_cfg_raw = dict(request.policy.config.get("step_rewards") or {})
        except Exception:
            step_rewards_cfg_raw = {}
        if not step_rewards_cfg_raw:
            try:
                if isinstance(request.env.config, dict):
                    step_rewards_cfg_raw = dict(request.env.config.get("step_rewards") or {})
            except Exception:
                step_rewards_cfg_raw = {}

        step_rewards_enabled = bool(step_rewards_cfg_raw.get("enabled", False))
        step_rewards_mode = str(step_rewards_cfg_raw.get("mode") or "off").lower()
        step_rewards_strategy = _normalize_step_strategy(step_rewards_cfg_raw.get("strategy"))
        step_rewards_weights = _coerce_weights(step_rewards_cfg_raw.get("weights"))
        step_rewards_k_limits = _coerce_k_limits(step_rewards_cfg_raw.get("k_limits"))
        try:
            step_rewards_indicator_lambda = float(
                step_rewards_cfg_raw.get("indicator_lambda") or 0.0
            )
        except Exception:
            step_rewards_indicator_lambda = 0.0
        try:
            step_rewards_beta = float(step_rewards_cfg_raw.get("step_beta") or 0.0)
        except Exception:
            step_rewards_beta = 0.0
        step_rewards_active = step_rewards_enabled and step_rewards_mode == "decision_stepwise"

        def _extract_achievements(obs: Any) -> dict[str, bool]:
            if not isinstance(obs, dict):
                return {}
            ach = obs.get("achievements_status")
            if isinstance(ach, dict):
                return {str(k): bool(v) for k, v in ach.items()}
            return {}

        def _extract_inventory(obs: Any) -> dict[str, int]:
            if not isinstance(obs, dict):
                return {}
            inv = obs.get("inventory")
            if not isinstance(inv, dict):
                return {}
            cleaned: dict[str, int] = {}
            for key, value in inv.items():
                coerced = _coerce_int_value(value)
                if coerced is None:
                    continue
                cleaned[str(key)] = coerced
            return cleaned

        def _extract_achievement_counts(obs: Any) -> dict[str, int]:
            if not isinstance(obs, dict):
                return {}
            counts = obs.get("achievements_counts")
            if not isinstance(counts, dict):
                return {}
            cleaned: dict[str, int] = {}
            for key, value in counts.items():
                coerced = _coerce_int_value(value)
                if coerced is None:
                    continue
                cleaned[str(key)] = coerced
            return cleaned

        def _summarize_tool_calls(tool_calls: Any) -> list[dict[str, Any]]:
            if not tool_calls:
                return []
            try:
                items = (
                    tool_calls
                    if isinstance(tool_calls, list)
                    else list(tool_calls)  # tolerates tuples or pydantic lists
                )
            except Exception:
                return []
            summary: list[dict[str, Any]] = []
            for tc in items:
                tool_name = None
                args: Any = {}
                if isinstance(tc, dict):
                    tool_name = tc.get("tool") or tc.get("tool_name") or tc.get("name")
                    raw_args = tc.get("arguments") or tc.get("args") or {}
                else:
                    tool_name = getattr(tc, "tool", None) or getattr(tc, "tool_name", None)
                    raw_args = getattr(tc, "arguments", None) or getattr(tc, "args", None) or {}
                args = raw_args
                if isinstance(raw_args, str):
                    try:
                        args = json.loads(raw_args)
                    except Exception:
                        args = raw_args
                summary.append({"tool": tool_name, "args": args})
            return summary

        decision_samples: list[dict[str, Any]] = []
        decision_index = 0
        decision_open = False
        session_trace = None
        finalized = False
        prev_achievements = _extract_achievements(current_obs)
        prev_inventory_state = _extract_inventory(current_obs)
        prev_achievement_counts_state = _extract_achievement_counts(current_obs)
        # Track episode-level achievements that have been seen as true at any point so far
        episode_seen_achievements: set[str] = {
            k for k, v in (prev_achievements or {}).items() if bool(v)
        }
        episode_achievement_counts: dict[str, int] = {}
        stepwise_indicator_sum = 0.0
        stepwise_reward_sum = 0.0
        stepwise_resource_reward_sum = 0.0
        stepwise_new_achievements_total = 0
        final_achievement_count = sum(1 for v in prev_achievements.values() if v)

        # Execute ops sequence (capped by env_params.max_steps_per_episode)
        for op_idx, op in enumerate(ops_seq):
            # Check for abort
            if registry.is_run_aborted(request.run_id):
                logger.info(f"Run {request.run_id} aborted at op {op_idx}")
                break

            # Check safety limits
            if ops_executed >= request.safety.max_ops:
                logger.warning(f"Reached max_ops limit ({request.safety.max_ops})")
                break

            if op == "agent":
                # Policy step
                from .policy_routes import PolicyStepRequest, step_policy

                if not decision_open:
                    await tracing_context.start_decision(decision_index)
                    decision_open = True

                agent_request_start = _time.perf_counter()
                if last_agent_response_ts is not None and last_policy_meta is not None:
                    with contextlib.suppress(Exception):
                        timing_prev = last_policy_meta.setdefault("timing", {})
                        decision_ms = max(
                            0.0,
                            (agent_request_start - float(last_agent_response_ts)) * 1000.0,
                        )
                        # Update timing on prior policy meta (kept by previous env step)
                        timing_prev["decision_ms"] = decision_ms
                        if last_env_step_ms is not None:
                            timing_prev["env_step_ms"] = float(last_env_step_ms)
                            timing_prev["overhead_ms"] = max(
                                0.0, decision_ms - float(last_env_step_ms)
                            )
                        else:
                            timing_prev.setdefault("overhead_ms", 0.0)
                        timing_prev["decision_ready_s"] = agent_request_start
                        # Also backfill the last appended trajectory step so the trainer
                        # can always see decision_ms without relying on shared dict refs.
                        if trajectory_steps:
                            with contextlib.suppress(Exception):
                                _last = trajectory_steps[-1]
                                _info = dict(_last.info or {})
                                _meta = dict(_info.get("meta") or {})
                                _timing = dict(_meta.get("timing") or {})
                                _timing["decision_ms"] = decision_ms
                                if last_env_step_ms is not None:
                                    _timing.setdefault("env_step_ms", float(last_env_step_ms))
                                    _timing.setdefault(
                                        "overhead_ms",
                                        max(0.0, decision_ms - float(last_env_step_ms)),
                                    )
                                else:
                                    _timing.setdefault("overhead_ms", 0.0)
                                _meta["timing"] = _timing
                                _info["meta"] = _meta
                                _last.info = _info
                last_env_step_ms = None
                last_env_step_completed_ts = None

                # Build metadata for policy (carry previous tool_calls and env result)
                metadata = {}
                if pending_tool_calls:
                    metadata["prev_tool_calls"] = pending_tool_calls
                if len(trajectory_steps) > 0:
                    last_step = trajectory_steps[-1]
                    # Prefer the last executed tool calls to seed history
                    if last_step.tool_calls:
                        metadata["prev_tool_calls"] = last_step.tool_calls
                    # Provide a compact env result snapshot
                    metadata["prev_env_result"] = {
                        "observation": last_step.obs,
                        "reward": last_step.reward,
                        "done": last_step.done,
                        "truncated": last_step.truncated,
                        "info": last_step.info,
                    }

                # Log compact metadata summary to confirm history threading
                with contextlib.suppress(Exception):
                    _prev_calls = metadata.get("prev_tool_calls")
                    _count = len(_prev_calls) if isinstance(_prev_calls, list) else 0
                    _first_guess = None
                    if _count > 0 and isinstance(_prev_calls[0], dict):
                        _args = _prev_calls[0].get("arguments", None)
                        if isinstance(_args, str):
                            import json as _json
                            with contextlib.suppress(Exception):
                                _args = _json.loads(_args)
                        if not isinstance(_args, dict):
                            _args = {}
                        _first_guess = _args.get("guess") or _args.get("word")
                    logger.info(
                        "POLICY_METADATA: prev_tool_calls=%d first_guess=%r has_prev_env_result=%s",
                        _count,
                        _first_guess,
                        str("prev_env_result" in metadata),
                    )

                try:
                    policy_response = await step_policy(
                        PolicyStepRequest(
                            policy_id=policy_id,
                            observation=current_obs,
                            metadata=metadata,
                        ),
                        req,
                    )
                except Exception as _pe:
                    # Hard fail the rollout on policy step error (e.g., inference auth 4xx)
                    logger.error(
                        "POLICY_STEP_HARD_FAIL: run_id=%s op_idx=%s err=%s",
                        request.run_id,
                        str(op_idx),
                        str(_pe),
                    )
                    raise HTTPException(status_code=500, detail=f"policy_step_failed: {str(_pe)}")

                agent_response_ts = _time.perf_counter()
                if isinstance(policy_response.meta, dict):
                    with contextlib.suppress(Exception):
                        timing_cur = policy_response.meta.setdefault("timing", {})
                        timing_cur["agent_request_start_s"] = agent_request_start
                        timing_cur["agent_response_s"] = agent_response_ts
                        if "inference_ms" in policy_response.meta:
                            with contextlib.suppress(Exception):
                                timing_cur.setdefault(
                                    "inference_ms",
                                    float(policy_response.meta["inference_ms"]),
                                )
                                timing_cur.setdefault(
                                    "inference_s",
                                    float(policy_response.meta["inference_ms"]) / 1000.0,
                                )
                    last_policy_meta = policy_response.meta
                else:
                    last_policy_meta = None
                last_agent_response_ts = agent_response_ts

                # Diagnostic: summarize policy step target and tool calls
                try:
                    model_name = None
                    target_url = None
                    if isinstance(policy_response.meta, dict):
                        req_body = policy_response.meta.get("inference_request") or {}
                        model_name = req_body.get("model")
                        target_url = policy_response.meta.get("inference_url")
                    _tc = policy_response.tool_calls or []
                    print(
                        {
                            "rollout.policy_step": True,
                            "run_id": request.run_id,
                            "model": model_name,
                            "inference_url": target_url,
                            "tool_calls_count": len(_tc) if isinstance(_tc, list) else 0,
                        },
                        flush=True,
                    )
                except Exception:
                    pass

                pending_tool_calls = policy_response.tool_calls
                # Log summarized agent tool calls
                with contextlib.suppress(Exception):
                    _tc = pending_tool_calls or []
                    _summary = []
                    for _item in (_tc if isinstance(_tc, list) else []):
                        try:
                            if isinstance(_item, dict):
                                _tool = _item.get("tool")
                                _args = _item.get("args")
                                _keys = list(_args.keys()) if isinstance(_args, dict) else []
                                _summary.append({"tool": _tool, "args_keys": _keys})
                        except Exception:
                            continue
                    _rid = getattr(request, "run_id", None)
                    logger.info("AGENT_TOOL_CALLS: run_id=%s count=%d summary=%s", _rid, len(_tc), _summary)
                    print(f"[rollout] agent tool_calls run_id={_rid} count={len(_tc)} summary={_summary}", flush=True)
                await tracing_context.record_tool_invocation(pending_tool_calls)
                ops_executed += 1

            elif op == "env":
                if not pending_tool_calls:
                    with contextlib.suppress(Exception):
                        logger.warning(
                            "POLICY_STEP_FAIL: missing tool_calls; failing rollout run_id=%s op_idx=%s",
                            request.run_id,
                            str(op_idx),
                        )
                    raise HTTPException(
                        status_code=500,
                        detail="policy_step_failed: missing tool_calls (no_tool_calls)",
                    )

                # Environment step
                from .environment_routes import EnvStepRequest, step_environment

                env_step_error: Exception | None = None
                env_response = None
                env_step_start = _time.perf_counter()
                try:
                    env_response = await step_environment(
                        EnvStepRequest(
                            env_id=env_id,
                            tool_calls=pending_tool_calls,
                        )
                    )
                except Exception as _ee:
                    env_step_error = _ee
                env_step_end = _time.perf_counter()
                env_step_duration_ms = (env_step_end - env_step_start) * 1000.0
                last_env_step_ms = env_step_duration_ms
                last_env_step_completed_ts = env_step_end
                if last_policy_meta is not None:
                    with contextlib.suppress(Exception):
                        timing_env = last_policy_meta.setdefault("timing", {})
                        timing_env["env_step_ms"] = env_step_duration_ms
                        timing_env["env_step_end_s"] = env_step_end

                if env_step_error is not None:
                    with contextlib.suppress(Exception):
                        logger.warning(
                            "ENV_STEP_FAIL: failing rollout run_id=%s op_idx=%s err=%s",
                            request.run_id,
                            str(op_idx),
                            str(env_step_error),
                        )
                    raise HTTPException(
                        status_code=500,
                        detail=f"env_step_failed: {str(env_step_error)}",
                    )

                # Reaching here means env step succeeded
                assert env_response is not None

                # Record step, including policy meta if present for timing/tokens observability
                _info = env_response.info if isinstance(env_response.info, dict) else {}
                # Attach policy meta from the immediately preceding agent step
                with contextlib.suppress(Exception):
                    prev_meta = {}
                    if "policy_response" in locals() and isinstance(policy_response.meta, dict):  # type: ignore[name-defined]
                        prev_meta = policy_response.meta
                    if prev_meta:
                        _info = dict(_info)
                        _info["meta"] = prev_meta

                event_metadata = {
                    "op_index": op_idx,
                }
                event_id = await tracing_context.record_environment_event(
                    env_handle=env_handle,
                    prev_obs=current_obs,
                    env_response=env_response,
                    next_obs=getattr(env_response, "observation", None),
                    metadata=event_metadata,
                )

                decision_index += 1
                next_obs = env_response.observation
                new_achievement_state = _extract_achievements(next_obs)
                new_inventory_state = _extract_inventory(next_obs)
                new_achievement_counts_state = _extract_achievement_counts(next_obs)
                final_achievement_count = sum(
                    1 for _, unlocked in new_achievement_state.items() if unlocked
                )
                indicator_val = 0
                reward_stepwise = 0.0
                decision_rewards_meta: dict[str, Any] | None = None
                decision_record = None
                _info = {} if not isinstance(_info, dict) else dict(_info)
                if step_rewards_active:
                    decision_actions = _summarize_tool_calls(pending_tool_calls)
                    stepwise_info, decision_record, stats = compute_stepwise_reward(
                        prev_achievements or {},
                        new_achievement_state,
                        decision_index,
                        decision_actions,
                        step_rewards_indicator_lambda,
                        strategy=step_rewards_strategy,
                        weights=step_rewards_weights,
                        k_limits=step_rewards_k_limits,
                        episode_counts=episode_achievement_counts,
                        prev_inventory=prev_inventory_state,
                        new_inventory=new_inventory_state,
                        prev_counts=prev_achievement_counts_state,
                        new_counts=new_achievement_counts_state,
                    )
                    indicator_val = int(stats.get("indicator", 0.0))
                    reward_stepwise = float(stats.get("reward", 0.0))
                    stepwise_indicator_sum += float(stats.get("indicator", 0.0))
                    stepwise_reward_sum += reward_stepwise
                    stepwise_new_achievements_total += int(stats.get("new_achievements_count", 0.0))
                    with contextlib.suppress(Exception):
                        resource_component = stats.get("resource_reward")
                        if resource_component is not None:
                            stepwise_resource_reward_sum += float(resource_component)
                    _info["stepwise"] = stepwise_info
                    # Compute decision-level rewards (absolute vs unique) and attach to metadata
                    with contextlib.suppress(Exception):
                        turned_true = set(stepwise_info.get("new_achievements") or [])
                        seen_before = set(episode_seen_achievements)
                        new_unique = sorted(turned_true - seen_before)
                        ach_delta = int(len(turned_true))
                        unique_delta = int(len(new_unique))
                        # Prepare stable lists for logging/metadata
                        all_list = sorted(turned_true)
                        # Ensure nested meta exists
                        meta_block = (
                            _info.get("meta") if isinstance(_info.get("meta"), dict) else {}
                        )
                        decision_rewards = {
                            "turn": int(decision_index),
                            "ach_delta": ach_delta,
                            "unique_delta": unique_delta,
                            "all": all_list,
                            "unique": new_unique,
                        }
                    decision_rewards_meta = decision_rewards
                    meta_block["decision_rewards"] = decision_rewards
                    _info["meta"] = meta_block
                    # Update episode-level seen set after attributing uniqueness to this decision
                    episode_seen_achievements.update(turned_true)
                if decision_record is not None:
                    decision_samples.append(decision_record)
                prev_achievements = new_achievement_state
                prev_inventory_state = new_inventory_state
                prev_achievement_counts_state = new_achievement_counts_state

                await tracing_context.record_decision_reward(
                    event_id=event_id,
                    decision_meta=decision_rewards_meta,
                )

                step = RolloutStep(
                    obs=_summarize_observation_for_storage(env_handle, current_obs),
                    tool_calls=pending_tool_calls,
                    reward=env_response.reward,
                    done=env_response.done,
                    truncated=env_response.truncated,
                    info=_info,
                )
                # Log summarized env application of tool calls and immediate reward/done
                with contextlib.suppress(Exception):
                    _tc = pending_tool_calls or []
                    _summary = []
                    for _item in (_tc if isinstance(_tc, list) else []):
                        try:
                            if isinstance(_item, dict):
                                _tool = _item.get("tool")
                                _args = _item.get("args")
                                _keys = list(_args.keys()) if isinstance(_args, dict) else []
                                _summary.append({"tool": _tool, "args_keys": _keys})
                        except Exception:
                            continue
                    _rid = getattr(request, "run_id", None)
                    logger.info(
                        "ENV_APPLY: run_id=%s tool_calls=%d reward=%s done=%s summary=%s",
                        _rid,
                        len(_tc),
                        str(env_response.reward),
                        str(env_response.done),
                        _summary,
                    )
                    print(
                        f"[rollout] env apply run_id={_rid} tool_calls={len(_tc)} reward={env_response.reward} done={env_response.done} summary={_summary}",
                        flush=True,
                    )
                trajectory_steps.append(step)

                if env_response.reward is not None:
                    total_reward += env_response.reward

                # Update state
                current_obs = next_obs
                pending_tool_calls = None
                ops_executed += 1

                # Handle episode end
                if env_response.done:
                    if request.on_done == "reset":
                        # Reset environment
                        from .environment_routes import (
                            EnvResetRequest,
                            reset_environment,
                        )

                        reset_response = await reset_environment(EnvResetRequest(env_id=env_id))
                        current_obs = reset_response.observation
                        prev_achievements = _extract_achievements(current_obs)
                        episode_seen_achievements = {
                            k for k, v in (prev_achievements or {}).items() if bool(v)
                        }
                        episode_achievement_counts.clear()
                    elif request.on_done == "terminate":
                        break

                if decision_open:
                    await tracing_context.end_decision()
                    decision_open = False

            else:
                logger.warning(f"Unknown op: {op}")

        if (
            last_policy_meta is not None
            and last_agent_response_ts is not None
            and "timing" in last_policy_meta
            and isinstance(last_policy_meta["timing"], dict)
            and "decision_ms" not in last_policy_meta["timing"]
        ):
            with contextlib.suppress(Exception):
                final_now = last_env_step_completed_ts or _time.perf_counter()
                final_decision_ms = max(0.0, (final_now - float(last_agent_response_ts)) * 1000.0)
                timing_final = last_policy_meta.setdefault("timing", {})
                timing_final["decision_ms"] = final_decision_ms
                if last_env_step_ms is not None:
                    timing_final.setdefault("env_step_ms", float(last_env_step_ms))
                    timing_final.setdefault(
                        "overhead_ms",
                        max(0.0, final_decision_ms - float(last_env_step_ms)),
                    )
                else:
                    timing_final.setdefault("overhead_ms", 0.0)

        # Build trajectory
        trajectory = RolloutTrajectory(
            env_id=env_id,
            policy_id=policy_id,
            steps=trajectory_steps,
            final={"observation": _summarize_observation_for_storage(env_handle, current_obs)},
            length=len(trajectory_steps),
            decision_samples=decision_samples if step_rewards_active else None,
        )

        # Build metrics
        metrics = RolloutMetrics(
            episode_returns=[total_reward],
            mean_return=total_reward,
            num_steps=len(trajectory_steps),
            num_episodes=1,
        )
        if step_rewards_active:
            stepwise_summary: dict[str, Any] = {
                "indicator_sum": float(stepwise_indicator_sum),
                "reward_sum": float(stepwise_reward_sum),
                "resource_reward": float(stepwise_resource_reward_sum),
                "new_achievements_total": int(stepwise_new_achievements_total),
                "mode": step_rewards_mode,
                "strategy": step_rewards_strategy,
                "indicator_lambda": float(step_rewards_indicator_lambda),
            }
            if step_rewards_beta:
                stepwise_summary["step_beta"] = float(step_rewards_beta)
            if step_rewards_strategy == "per_achievement":
                if step_rewards_weights:
                    stepwise_summary["weights"] = dict(step_rewards_weights)
                if step_rewards_k_limits:
                    stepwise_summary["k_limits"] = dict(step_rewards_k_limits)
            final_achievements_list = sorted(
                key for key, val in (prev_achievements or {}).items() if bool(val)
            )
            stepwise_summary["unique_achievements_total"] = int(len(episode_seen_achievements))
            stepwise_summary["unique_achievements"] = sorted(episode_seen_achievements)
            stepwise_summary["final_achievements"] = final_achievements_list
            metrics.details["stepwise"] = stepwise_summary

        # Environment-specific: Log summary if available
        try:
            # Check if this is a Wordle environment and use Wordle helpers (lazy import)
            wordle_wrapper_cls = None
            try:
                from .envs.wordle.environment import WordleEnvironmentWrapper
                from .envs.wordle.helpers import (
                    get_wordle_rollout_summary,
                    log_wordle_rollout_summary,
                )

                wordle_wrapper_cls = WordleEnvironmentWrapper
            except Exception:
                wordle_wrapper_cls = None  # type: ignore[assignment]
                get_wordle_rollout_summary = None  # type: ignore
                log_wordle_rollout_summary = None  # type: ignore

            is_wordle = wordle_wrapper_cls is not None and isinstance(
                env_handle.env,
                wordle_wrapper_cls,  # type: ignore[arg-type]
            )
            if is_wordle:
                # Convert trajectory steps to expected format
                formatted_steps = []
                for step in trajectory_steps:
                    formatted_steps.append({"tool_calls": step.tool_calls or []})

                if (
                    get_wordle_rollout_summary is not None
                    and log_wordle_rollout_summary is not None
                ):
                    summary = get_wordle_rollout_summary(formatted_steps, current_obs, env_handle)
                    log_wordle_rollout_summary(request.run_id, summary)
        except ImportError:
            # Wordle helpers not available, skip Wordle-specific logging
            pass
        except Exception as e:
            logger.warning(f"Failed to generate environment-specific summary: {e}")

        # Mark run as completed
        aborted = registry.is_run_aborted(request.run_id)
        if not aborted:
            registry.complete_run(request.run_id)
        if decision_open:
            await tracing_context.end_decision()
            decision_open = False
        if not finalized:
            session_trace = await tracing_context.finalize(
                total_reward=total_reward,
                achievement_state=prev_achievements,
                total_steps=len(trajectory_steps),
            )
            finalized = True
        trace_payload = tracing_context.build_trace_payload(session_trace)

        # Hard-fail if no steps executed (avg_turns == 0 scenario)
        if metrics.num_steps <= 0:
            raise HTTPException(status_code=500, detail="no_steps_executed: avg_turns == 0")

        return RolloutResponse(
            run_id=request.run_id,
            trajectories=[trajectory],
            branches={},
            metrics=metrics,
            aborted=aborted,
            ops_executed=ops_executed,
            trace=trace_payload,
        )

    except Exception as e:
        logger.error(f"Rollout failed for run {request.run_id}: {e}")
        registry.abort_run(request.run_id)
        if decision_open:
            with contextlib.suppress(Exception):
                await tracing_context.end_decision()
            decision_open = False
        if not finalized:
            session_trace = None
            with contextlib.suppress(Exception):
                session_trace = await tracing_context.finalize(
                    total_reward=total_reward,
                    achievement_state=prev_achievements,
                    total_steps=len(trajectory_steps),
                )
            finalized = True
        raise HTTPException(status_code=500, detail=str(e)) from e
    finally:
        # Ensure any environment created for this rollout is terminated (no reuse across rollouts)
        try:
            if created_env_id:
                from .environment_routes import EnvTerminateRequest, terminate_environment

                await terminate_environment(EnvTerminateRequest(env_id=created_env_id))
                logger.info(
                    "ROLL_OUT: terminated environment env_id=%s seed=%s",
                    str(created_env_id),
                    str(env_seed_used) if env_seed_used is not None else "unknown",
                )
                # Verify removal from registry
                with contextlib.suppress(Exception):
                    _post = registry.get_env(created_env_id)
                    logger.info(
                        "ROLL_OUT: env_killed=%s (post_lookup=%s)",
                        str(_post is None),
                        str(_post),
                    )
        except Exception as _te:
            logger.warning(f"ROLL_OUT: failed to terminate environment {created_env_id}: {_te}")

        # Best-effort policy cleanup if we created one (avoid reuse across rollouts)
        with contextlib.suppress(Exception):
            if created_policy_id:
                from .policy_routes import PolicyTerminateRequest, terminate_policy

                await terminate_policy(PolicyTerminateRequest(policy_id=created_policy_id))
                logger.info("ROLL_OUT: terminated policy policy_id=%s", str(created_policy_id))

        if not finalized:
            session_trace = None
            with contextlib.suppress(Exception):
                session_trace = await tracing_context.finalize(
                    total_reward=total_reward,
                    achievement_state=prev_achievements,
                    total_steps=len(trajectory_steps),
                )
            finalized = True

        with contextlib.suppress(Exception):
            _clear_seed_side_effects()
            logger.info("ROLL_OUT: RNG seed terminated/cleared before conclusion")


@router.post("/run/abort", response_model=RunAbortResponse)
async def abort_run(request: RunAbortRequest) -> RunAbortResponse:
    """Abort a running rollout."""
    success = registry.abort_run(request.run_id)

    if not success:
        raise HTTPException(
            status_code=404,
            detail=f"Run {request.run_id} not found",
        )

    return RunAbortResponse(
        ok=True,
        run_id=request.run_id,
    )


@router.get("/run/status/{run_id}", response_model=RunStatusResponse)
async def get_run_status(run_id: str) -> RunStatusResponse:
    """Get the status of a run."""
    run_handle = registry.get_run(run_id)

    if not run_handle:
        raise HTTPException(
            status_code=404,
            detail=f"Run {run_id} not found",
        )

    return RunStatusResponse(
        run_id=run_id,
        status=run_handle.status,
        started_at=run_handle.started_at,
        finished_at=run_handle.finished_at,
    )
