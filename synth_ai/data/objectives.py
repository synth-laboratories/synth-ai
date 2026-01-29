"""Objective specifications and reward observations."""

from __future__ import annotations

import contextlib
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from synth_ai.data.enums import (
    ObjectiveDirection,
    ObjectiveKey,
    RewardScope,
    RewardSource,
    RewardType,
)

try:
    from . import rust as _rust_data
except Exception as exc:  # pragma: no cover
    raise RuntimeError("synth_ai_py is required for data.objectives.") from exc


@dataclass(frozen=True)
class ObjectiveSpec:
    """Specification for a canonical objective."""

    key: ObjectiveKey
    direction: ObjectiveDirection
    units: Optional[str] = None
    description: Optional[str] = None
    target: Optional[float] = None
    min_value: Optional[float] = None
    max_value: Optional[float] = None

    @classmethod
    def from_dict(cls, data: dict) -> ObjectiveSpec:
        if _rust_data is not None:
            with contextlib.suppress(Exception):
                data = _rust_data.normalize_objective_spec(data)  # noqa: F811
        return cls(**data)


OBJECTIVE_REGISTRY: Dict[ObjectiveKey, ObjectiveSpec] = {
    ObjectiveKey.REWARD: ObjectiveSpec(
        key=ObjectiveKey.REWARD,
        direction=ObjectiveDirection.MAXIMIZE,
        units=None,
        description="Task performance reward",
    ),
    ObjectiveKey.LATENCY_MS: ObjectiveSpec(
        key=ObjectiveKey.LATENCY_MS,
        direction=ObjectiveDirection.MINIMIZE,
        units="milliseconds",
        description="Execution latency",
    ),
    ObjectiveKey.COST_USD: ObjectiveSpec(
        key=ObjectiveKey.COST_USD,
        direction=ObjectiveDirection.MINIMIZE,
        units="USD",
        description="LLM API cost",
    ),
}


@dataclass
class RewardObservation:
    """Normalized reward observation for storage or optimization."""

    value: float
    reward_type: RewardType = RewardType.SHAPED
    scope: RewardScope = RewardScope.OUTCOME
    source: RewardSource = RewardSource.TASK_APP
    objective_key: ObjectiveKey = ObjectiveKey.REWARD
    event_id: Optional[str | int] = None
    turn_number: Optional[int] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict) -> RewardObservation:
        if _rust_data is not None:
            with contextlib.suppress(Exception):
                data = _rust_data.normalize_reward_observation(data)  # noqa: F811
        return cls(**data)


@dataclass
class OutcomeObjectiveAssignment:
    """Objective values scoped to a full trace/session."""

    objectives: Dict[str, float]
    session_id: Optional[str] = None
    trace_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict) -> OutcomeObjectiveAssignment:
        return cls(**data)


@dataclass
class EventObjectiveAssignment:
    """Objective values scoped to a single event."""

    event_id: str | int
    objectives: Dict[str, float]
    turn_number: Optional[int] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict) -> EventObjectiveAssignment:
        return cls(**data)


@dataclass
class InstanceObjectiveAssignment:
    """Objective values scoped to a single task instance."""

    instance_id: str | int
    objectives: Dict[str, float]
    split: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict) -> InstanceObjectiveAssignment:
        return cls(**data)


try:  # Require Rust-backed classes
    import synth_ai_py as _rust_models  # type: ignore
except Exception as exc:  # pragma: no cover
    raise RuntimeError("synth_ai_py is required for data.objectives.") from exc

with contextlib.suppress(AttributeError):
    ObjectiveSpec = _rust_models.ObjectiveSpec  # noqa: F811
    RewardObservation = _rust_models.RewardObservation  # noqa: F811
    OutcomeObjectiveAssignment = _rust_models.OutcomeObjectiveAssignment  # noqa: F811
    EventObjectiveAssignment = _rust_models.EventObjectiveAssignment  # noqa: F811
    InstanceObjectiveAssignment = _rust_models.InstanceObjectiveAssignment  # noqa: F811


__all__ = [
    "ObjectiveSpec",
    "OBJECTIVE_REGISTRY",
    "RewardObservation",
    "OutcomeObjectiveAssignment",
    "EventObjectiveAssignment",
    "InstanceObjectiveAssignment",
]
