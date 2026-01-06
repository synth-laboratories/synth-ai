"""Objective specifications and reward observations."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Union

from synth_ai.data.enums import (
    ObjectiveDirection,
    ObjectiveKey,
    RewardScope,
    RewardSource,
    RewardType,
)


@dataclass(frozen=True)
class ObjectiveSpec:
    """Specification for a canonical objective."""

    key: ObjectiveKey
    direction: ObjectiveDirection
    units: Optional[str] = None
    description: Optional[str] = None


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
    event_id: Optional[Union[str, int]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class OutcomeObjectiveAssignment:
    """Objective values scoped to a full trace/session."""

    objectives: Dict[str, float]
    session_id: Optional[str] = None
    trace_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EventObjectiveAssignment:
    """Objective values scoped to a single event."""

    event_id: Union[str, int]
    objectives: Dict[str, float]
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class InstanceObjectiveAssignment:
    """Objective values scoped to a single task instance."""

    instance_id: Union[str, int]
    objectives: Dict[str, float]
    metadata: Dict[str, Any] = field(default_factory=dict)


__all__ = [
    "ObjectiveSpec",
    "OBJECTIVE_REGISTRY",
    "RewardObservation",
    "OutcomeObjectiveAssignment",
    "EventObjectiveAssignment",
    "InstanceObjectiveAssignment",
]
