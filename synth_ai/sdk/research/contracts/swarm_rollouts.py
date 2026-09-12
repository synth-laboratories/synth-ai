"""Container-pool rollouts a swarm launched.

Mirrors backend ``SwarmRolloutPage`` (``app/api/v1/managed_research/schemas/
swarm_rollouts.py``). A rollout appears here only when its verified budget
parent is the swarm; caller-supplied metadata cannot place it in this list.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime


def _text(payload: Mapping[str, object], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"swarm rollout {key} is required")
    return value


def _optional_text(payload: Mapping[str, object], key: str) -> str | None:
    value = payload.get(key)
    if value is None or isinstance(value, str):
        return value
    raise ValueError(f"swarm rollout {key} must be a string or null")


def _optional_datetime(payload: Mapping[str, object], key: str) -> datetime | None:
    value = _optional_text(payload, key)
    return datetime.fromisoformat(value) if value is not None else None


@dataclass(frozen=True)
class SwarmRollout:
    rollout_id: str
    pool_id: str
    adapter: str
    status: str
    seed: int
    trace_correlation_id: str
    budget_parent_run_id: str
    task_id: str | None = None
    success: bool | None = None
    score: float | None = None
    error: str | None = None
    created_at: datetime | None = None
    started_at: datetime | None = None
    completed_at: datetime | None = None
    cancelled_at: datetime | None = None

    @classmethod
    def from_wire(cls, payload: object) -> SwarmRollout:
        if not isinstance(payload, Mapping):
            raise ValueError("swarm rollout must be an object")
        seed = payload.get("seed")
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise ValueError("swarm rollout seed must be an integer")
        success = payload.get("success")
        if success is not None and not isinstance(success, bool):
            raise ValueError("swarm rollout success must be a boolean or null")
        score = payload.get("score")
        if score is not None and (isinstance(score, bool) or not isinstance(score, (int, float))):
            raise ValueError("swarm rollout score must be a number or null")
        return cls(
            rollout_id=_text(payload, "rollout_id"),
            pool_id=_text(payload, "pool_id"),
            adapter=_text(payload, "adapter"),
            status=_text(payload, "status"),
            seed=seed,
            trace_correlation_id=_text(payload, "trace_correlation_id"),
            budget_parent_run_id=_text(payload, "budget_parent_run_id"),
            task_id=_optional_text(payload, "task_id"),
            success=success,
            score=float(score) if score is not None else None,
            error=_optional_text(payload, "error"),
            created_at=_optional_datetime(payload, "created_at"),
            started_at=_optional_datetime(payload, "started_at"),
            completed_at=_optional_datetime(payload, "completed_at"),
            cancelled_at=_optional_datetime(payload, "cancelled_at"),
        )


def swarm_rollouts_from_wire(payload: object, *, swarm_id: str) -> tuple[SwarmRollout, ...]:
    if not isinstance(payload, Mapping) or payload.get("run_id") != swarm_id:
        raise ValueError("swarm rollout page identity drifted")
    items = payload.get("items")
    if not isinstance(items, list):
        raise ValueError("swarm rollout page items must be a list")
    rollouts = tuple(SwarmRollout.from_wire(item) for item in items)
    if any(rollout.budget_parent_run_id != swarm_id for rollout in rollouts):
        raise ValueError("swarm rollout budget parent drifted")
    return rollouts


__all__ = ["SwarmRollout", "swarm_rollouts_from_wire"]
