"""Compatibility helpers for legacy reward fields."""

from __future__ import annotations

import warnings
from typing import Any, Dict, List, Mapping, Optional, Sequence

from synth_ai.data.objectives import (
    EventObjectiveAssignment,
    InstanceObjectiveAssignment,
    OutcomeObjectiveAssignment,
)

_LEGACY_WARNED_FIELDS: set[str] = set()


def _warn_legacy_field(field_name: str, guidance: str) -> None:
    if field_name in _LEGACY_WARNED_FIELDS:
        return
    _LEGACY_WARNED_FIELDS.add(field_name)
    warnings.warn(
        f"{field_name} is deprecated; use {guidance}.",
        DeprecationWarning,
        stacklevel=2,
    )


def _coerce_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, bool):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _is_sequence(value: Any) -> bool:
    return isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray))


def _coerce_str(value: Any) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, str):
        return value
    return str(value)


def extract_outcome_reward(payload: Mapping[str, Any]) -> Optional[float]:
    """Extract outcome reward from legacy payloads with a precedence chain."""
    if not isinstance(payload, Mapping):
        return None

    outcome_objectives = payload.get("outcome_objectives")
    if isinstance(outcome_objectives, Mapping):
        reward_val = _coerce_float(outcome_objectives.get("reward"))
        if reward_val is not None:
            return reward_val
    objectives = payload.get("objectives")
    if isinstance(objectives, Mapping):
        reward_val = _coerce_float(objectives.get("reward"))
        if reward_val is not None:
            return reward_val

    for key in (
        "outcome_reward",
        "total_reward",
        "outcome_score",
        "score",
        "accuracy",
        "reward_mean",
    ):
        reward_val = _coerce_float(payload.get(key))
        if reward_val is not None:
            if key == "outcome_score":
                _warn_legacy_field("outcome_score", "outcome_reward or objectives['reward']")
            elif key == "accuracy":
                _warn_legacy_field("accuracy", "objectives['reward']")
            return reward_val

    episode_rewards = payload.get("episode_rewards")
    if _is_sequence(episode_rewards) and episode_rewards:
        return _coerce_float(episode_rewards[0])

    return None


def extract_instance_rewards(payload: Mapping[str, Any]) -> Optional[List[float]]:
    """Extract per-instance rewards from legacy payloads."""
    if not isinstance(payload, Mapping):
        return None

    instance_objectives = payload.get("instance_objectives")
    if _is_sequence(instance_objectives):
        values: List[float] = []
        for item in instance_objectives:
            reward_val = None
            if isinstance(item, InstanceObjectiveAssignment):
                reward_val = _coerce_float(item.objectives.get("reward"))
            elif isinstance(item, Mapping):
                objectives = item.get("objectives")
                if isinstance(objectives, Mapping):
                    reward_val = _coerce_float(objectives.get("reward"))
                else:
                    reward_val = _coerce_float(item.get("reward"))
            if reward_val is None:
                return None
            values.append(reward_val)
        return values if values else None

    for key in ("instance_rewards", "instance_scores"):
        values_raw = payload.get(key)
        if _is_sequence(values_raw):
            values: List[float] = []
            for val in values_raw:
                reward_val = _coerce_float(val)
                if reward_val is None:
                    return None
                values.append(reward_val)
            if key == "instance_scores":
                _warn_legacy_field("instance_scores", "instance_rewards or instance_objectives")
            return values if values else None

    return None


def normalize_to_outcome_objectives(payload: Mapping[str, Any]) -> OutcomeObjectiveAssignment:
    """Normalize legacy reward payloads into an OutcomeObjectiveAssignment."""
    objectives: Dict[str, float] = {}

    raw_objectives = payload.get("objectives")
    if isinstance(raw_objectives, Mapping):
        for key, value in raw_objectives.items():
            reward_val = _coerce_float(value)
            if reward_val is not None:
                objectives[str(key)] = reward_val

    raw_outcome_objectives = payload.get("outcome_objectives")
    if isinstance(raw_outcome_objectives, Mapping):
        for key, value in raw_outcome_objectives.items():
            reward_val = _coerce_float(value)
            if reward_val is not None:
                objectives[str(key)] = reward_val

    if "reward" not in objectives:
        reward_val = extract_outcome_reward(payload)
        if reward_val is not None:
            objectives["reward"] = reward_val

    for key in ("latency_ms", "cost_usd"):
        if key in objectives:
            continue
        reward_val = _coerce_float(payload.get(key))
        if reward_val is not None:
            objectives[key] = reward_val

    metadata = payload.get("metadata") if isinstance(payload.get("metadata"), Mapping) else {}

    return OutcomeObjectiveAssignment(
        objectives=objectives,
        session_id=_coerce_str(payload.get("session_id")),
        trace_id=_coerce_str(payload.get("trace_id")),
        metadata=dict(metadata),
    )


def normalize_to_event_objectives(payload: Mapping[str, Any]) -> EventObjectiveAssignment:
    """Normalize legacy reward payloads into an EventObjectiveAssignment."""
    event_id = payload.get("event_id") if isinstance(payload, Mapping) else None
    if event_id is None:
        event_id = payload.get("id") if isinstance(payload, Mapping) else None
    if event_id is None:
        raise ValueError("event_id is required to normalize event objectives")

    objectives: Dict[str, float] = {}
    raw_objectives = payload.get("objectives")
    if isinstance(raw_objectives, Mapping):
        for key, value in raw_objectives.items():
            reward_val = _coerce_float(value)
            if reward_val is not None:
                objectives[str(key)] = reward_val

    if "reward" not in objectives:
        for key in ("reward_value", "value", "reward"):
            reward_val = _coerce_float(payload.get(key))
            if reward_val is not None:
                objectives["reward"] = reward_val
                break

    metadata = payload.get("metadata") if isinstance(payload.get("metadata"), Mapping) else {}

    return EventObjectiveAssignment(
        event_id=event_id,
        objectives=objectives,
        metadata=dict(metadata),
    )


def to_legacy_format(assignment: OutcomeObjectiveAssignment) -> Dict[str, Any]:
    """Convert OutcomeObjectiveAssignment to legacy payload fields."""
    result: Dict[str, Any] = {
        "objectives": dict(assignment.objectives),
    }

    reward_val = assignment.objectives.get("reward")
    if reward_val is not None:
        result.update(
            {
                "outcome_reward": reward_val,
                "total_reward": reward_val,
                "outcome_score": reward_val,
                "score": reward_val,
                "accuracy": reward_val,
                "reward_mean": reward_val,
                "episode_rewards": [reward_val],
            }
        )

    if assignment.session_id is not None:
        result["session_id"] = assignment.session_id
    if assignment.trace_id is not None:
        result["trace_id"] = assignment.trace_id
    if assignment.metadata:
        result["metadata"] = dict(assignment.metadata)

    return result
