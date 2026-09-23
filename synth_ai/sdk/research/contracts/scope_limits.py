"""Typed read model for the per-scope limit rows (`/smr/projects/.../limits`).

These routes address one unselected cap per `(scope, dimension)`: the total
spend, tokens, wall-clock and GPU-hour caps of a run or an objective.
Selector-scoped caps (per provider/model/resource) are set through
`SwarmSpec(spend=...)` and raised through resource-limit extensions.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from typing import Any


class LimitScope(StrEnum):
    """The path segment naming the kind of scope a limit belongs to."""

    RUN = "runs"
    OBJECTIVE = "objectives"


def _number(value: object) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    raise ValueError(f"expected a number, got {type(value).__name__}")


def _text(payload: Mapping[str, Any], name: str) -> str:
    value = payload.get(name)
    if not isinstance(value, str) or not value:
        raise ValueError(f"scope limit is missing {name}")
    return value


@dataclass(frozen=True, slots=True)
class ScopeLimit:
    scope_kind: str
    scope_id: str
    dimension: str
    cap_amount: float | None
    used_amount: float
    remaining_amount: float | None
    fraction_used: float | None
    unit: str
    cap_revision: int
    cap_source: str
    alert_at_fraction: float
    enforce_at_fraction: float
    exhaustion_action: str
    notify_audience: str
    policy: Mapping[str, Any] = field(default_factory=dict)
    last_used_writer: str | None = None
    updated_at: datetime | None = None

    @classmethod
    def from_wire(cls, payload: object) -> ScopeLimit:
        if not isinstance(payload, Mapping):
            raise ValueError("scope limit must be an object")
        updated_at = payload.get("updated_at")
        return cls(
            scope_kind=_text(payload, "scope_kind"),
            scope_id=_text(payload, "scope_id"),
            dimension=_text(payload, "dimension"),
            cap_amount=_number(payload.get("cap_amount")),
            used_amount=_number(payload.get("used_amount")) or 0.0,
            remaining_amount=_number(payload.get("remaining_amount")),
            fraction_used=_number(payload.get("fraction_used")),
            unit=_text(payload, "unit"),
            cap_revision=int(payload.get("cap_revision") or 1),
            cap_source=_text(payload, "cap_source"),
            alert_at_fraction=float(payload.get("alert_at_fraction") or 0.9),
            enforce_at_fraction=float(payload.get("enforce_at_fraction") or 1.0),
            exhaustion_action=_text(payload, "exhaustion_action"),
            notify_audience=_text(payload, "notify_audience"),
            policy=dict(payload.get("policy") or {}),
            last_used_writer=(
                str(payload["last_used_writer"]) if payload.get("last_used_writer") else None
            ),
            updated_at=(
                datetime.fromisoformat(updated_at) if isinstance(updated_at, str) else None
            ),
        )


__all__ = ["LimitScope", "ScopeLimit"]
