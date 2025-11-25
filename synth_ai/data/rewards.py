"""Reward data structures.

This module defines pure data types for representing rewards in training
and evaluation contexts. These are actual data records, not API abstractions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Literal


@dataclass
class RewardRecord:
    """A single reward observation.

    Represents a reward signal at a specific point in a trajectory,
    with metadata about its source and scope.
    """

    value: float
    reward_type: Literal["shaped", "sparse", "achievement", "penalty", "evaluator", "human"] = "shaped"
    scope: Literal["step", "event", "outcome"] = "step"
    source: Literal["environment", "runner", "evaluator", "human"] | None = None
    key: str | None = None  # e.g., achievement name
    turn: int | None = None
    timestamp: datetime | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class OutcomeRewardRecord:
    """Episode-level reward summary.

    Aggregates reward information for a complete episode/session,
    including total reward, achievements, and step counts.
    """

    session_id: str
    total_reward: float
    achievements_count: int = 0
    total_steps: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: datetime | None = None


@dataclass
class EventRewardRecord:
    """Event-level reward annotation.

    Links a reward to a specific event in a trace, with optional
    annotations and source information.
    """

    event_id: str
    session_id: str
    reward_value: float
    reward_type: str | None = None
    key: str | None = None
    turn_number: int | None = None
    source: str | None = None
    annotation: dict[str, Any] = field(default_factory=dict)
    created_at: datetime | None = None


@dataclass
class RewardAggregates:
    """Aggregated statistics for a set of rewards."""

    mean: float
    median: float = 0.0
    std: float = 0.0
    n: int = 0
    min_value: float | None = None
    max_value: float | None = None


__all__ = [
    "RewardRecord",
    "OutcomeRewardRecord",
    "EventRewardRecord",
    "RewardAggregates",
]


