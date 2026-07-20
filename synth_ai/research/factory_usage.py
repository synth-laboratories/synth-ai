"""Typed Factory usage and event bindings for ``client.research.factories``.

Backend routes (backend PR #747):

- ``GET /smr/factories/{factory_id}/usage`` → ``SmrFactoryUsageResponse``
- ``GET /smr/factories/{factory_id}/events`` → ``SmrFactoryEventsResponse``

Both are read via the session client's internal ``_request_json`` layer, the
same transport every other ``/smr`` binding uses.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from synth_ai.managed_research.models.factories import _optional_datetime
from synth_ai.managed_research.sdk.client import ManagedResearchClient


def _require_usage_mapping(payload: object, *, label: str) -> dict[str, object]:
    if not isinstance(payload, Mapping):
        raise ValueError(f"{label} payload must be an object")
    return {str(key): value for key, value in payload.items()}


def _optional_int(mapping: Mapping[str, object], key: str) -> int | None:
    value = mapping.get(key)
    return int(value) if value is not None else None  # type: ignore[arg-type]


def _optional_float(mapping: Mapping[str, object], key: str) -> float | None:
    value = mapping.get(key)
    return float(value) if value is not None else None  # type: ignore[arg-type]


def _optional_str(mapping: Mapping[str, object], key: str) -> str | None:
    value = mapping.get(key)
    return str(value) if value is not None else None


@dataclass(frozen=True)
class FactoryUsageCost:
    """Typed ``SmrFactoryUsageCostTotals`` (nominal ledger cost for the window)."""

    total_cents: int
    total_pico_usd: int
    total_usd: float
    raw: dict[str, object] = field(default_factory=dict)

    @classmethod
    def from_wire(cls, payload: object) -> FactoryUsageCost:
        mapping = _require_usage_mapping(payload, label="factory usage cost")
        return cls(
            total_cents=int(mapping.get("total_cents") or 0),
            total_pico_usd=int(mapping.get("total_pico_usd") or 0),
            total_usd=float(mapping.get("total_usd") or 0.0),
            raw=dict(mapping),
        )


@dataclass(frozen=True)
class FactoryUsageBudget:
    """Typed ``SmrFactoryUsageBudgetResponse`` (budget policy status; all optional)."""

    run_limit: int | None = None
    run_count: int | None = None
    remaining_runs: int | None = None
    run_budget_scope: str | None = None
    run_budget_window_start: datetime | None = None
    limit_pico_usd: int | None = None
    used_pico_usd: int | None = None
    remaining_pico_usd: int | None = None
    limit_usd: float | None = None
    used_usd: float | None = None
    remaining_usd: float | None = None
    period_start: datetime | None = None
    period_end: datetime | None = None
    raw: dict[str, object] = field(default_factory=dict)

    @classmethod
    def from_wire(cls, payload: object) -> FactoryUsageBudget:
        mapping = _require_usage_mapping(payload, label="factory usage budget")
        return cls(
            run_limit=_optional_int(mapping, "run_limit"),
            run_count=_optional_int(mapping, "run_count"),
            remaining_runs=_optional_int(mapping, "remaining_runs"),
            run_budget_scope=_optional_str(mapping, "run_budget_scope"),
            run_budget_window_start=_optional_datetime(
                mapping, "run_budget_window_start"
            ),
            limit_pico_usd=_optional_int(mapping, "limit_pico_usd"),
            used_pico_usd=_optional_int(mapping, "used_pico_usd"),
            remaining_pico_usd=_optional_int(mapping, "remaining_pico_usd"),
            limit_usd=_optional_float(mapping, "limit_usd"),
            used_usd=_optional_float(mapping, "used_usd"),
            remaining_usd=_optional_float(mapping, "remaining_usd"),
            period_start=_optional_datetime(mapping, "period_start"),
            period_end=_optional_datetime(mapping, "period_end"),
            raw=dict(mapping),
        )


@dataclass(frozen=True)
class FactoryEffortUsage:
    """Typed ``SmrFactoryEffortUsageResponse`` (per-effort drawdown in the window)."""

    effort_id: str
    name: str
    cost_microcents: int
    cost_cents: int
    cost_usd: float
    debit_count: int
    raw: dict[str, object] = field(default_factory=dict)

    @classmethod
    def from_wire(cls, payload: object) -> FactoryEffortUsage:
        mapping = _require_usage_mapping(payload, label="factory effort usage")
        return cls(
            effort_id=str(mapping.get("effort_id") or ""),
            name=str(mapping.get("name") or ""),
            cost_microcents=int(mapping.get("cost_microcents") or 0),
            cost_cents=int(mapping.get("cost_cents") or 0),
            cost_usd=float(mapping.get("cost_usd") or 0.0),
            debit_count=int(mapping.get("debit_count") or 0),
            raw=dict(mapping),
        )


@dataclass(frozen=True)
class FactoryUsage:
    """Typed ``SmrFactoryUsageResponse``.

    Backend route: ``GET /smr/factories/{factory_id}/usage``
    (query ``window``: ``month_to_date`` | ``last_7_days``).
    """

    factory_id: str
    org_id: str
    window: str
    window_start: datetime | None
    window_end: datetime | None
    cost: FactoryUsageCost
    budget: FactoryUsageBudget | None = None
    run_count: int = 0
    effort_count: int = 0
    by_effort: tuple[FactoryEffortUsage, ...] = ()
    updated_at: datetime | None = None
    raw: dict[str, object] = field(default_factory=dict)

    @classmethod
    def from_wire(cls, payload: object) -> FactoryUsage:
        mapping = _require_usage_mapping(payload, label="factory usage")
        budget = mapping.get("budget")
        return cls(
            factory_id=str(mapping.get("factory_id") or ""),
            org_id=str(mapping.get("org_id") or ""),
            window=str(mapping.get("window") or ""),
            window_start=_optional_datetime(mapping, "window_start"),
            window_end=_optional_datetime(mapping, "window_end"),
            cost=FactoryUsageCost.from_wire(mapping.get("cost") or {}),
            budget=FactoryUsageBudget.from_wire(budget) if budget is not None else None,
            run_count=int(mapping.get("run_count") or 0),
            effort_count=int(mapping.get("effort_count") or 0),
            by_effort=tuple(
                FactoryEffortUsage.from_wire(item)
                for item in list(mapping.get("by_effort") or [])
            ),
            updated_at=_optional_datetime(mapping, "updated_at"),
            raw=dict(mapping),
        )


@dataclass(frozen=True)
class FactoryEvent:
    """Typed ``SmrFactoryEventResponse`` (one durable factory event row)."""

    event_id: str
    occurred_at: datetime | None
    kind: str
    source: str
    payload: dict[str, object] = field(default_factory=dict)
    raw: dict[str, object] = field(default_factory=dict)

    @classmethod
    def from_wire(cls, payload: object) -> FactoryEvent:
        mapping = _require_usage_mapping(payload, label="factory event")
        event_payload = mapping.get("payload")
        return cls(
            event_id=str(mapping.get("event_id") or ""),
            occurred_at=_optional_datetime(mapping, "occurred_at"),
            kind=str(mapping.get("kind") or ""),
            source=str(mapping.get("source") or ""),
            payload=_require_usage_mapping(event_payload, label="factory event payload")
            if isinstance(event_payload, Mapping)
            else {},
            raw=dict(mapping),
        )


@dataclass(frozen=True)
class FactoryEventsPage:
    """Typed ``SmrFactoryEventsResponse`` (newest-first cursor page).

    Backend route: ``GET /smr/factories/{factory_id}/events``
    (query ``limit`` 1-500, ``cursor``; ``next_cursor`` is null when drained).
    """

    factory_id: str
    events: tuple[FactoryEvent, ...] = ()
    next_cursor: str | None = None
    raw: dict[str, object] = field(default_factory=dict)

    @classmethod
    def from_wire(cls, payload: object) -> FactoryEventsPage:
        mapping = _require_usage_mapping(payload, label="factory events page")
        return cls(
            factory_id=str(mapping.get("factory_id") or ""),
            events=tuple(
                FactoryEvent.from_wire(item)
                for item in list(mapping.get("events") or [])
            ),
            next_cursor=_optional_str(mapping, "next_cursor"),
            raw=dict(mapping),
        )


def fetch_factory_usage(
    session: ManagedResearchClient,
    factory_id: str,
    *,
    window: str = "month_to_date",
) -> FactoryUsage:
    """Read the factory usage aggregate.

    Backend route: ``GET /smr/factories/{factory_id}/usage`` (query ``window``).
    """
    payload = session._request_json(
        "GET",
        f"/smr/factories/{factory_id}/usage",
        params={"window": str(window)},
    )
    return FactoryUsage.from_wire(payload)


def fetch_factory_events(
    session: ManagedResearchClient,
    factory_id: str,
    *,
    limit: int | None = None,
    cursor: str | None = None,
) -> FactoryEventsPage:
    """Read one page of the durable factory event projection.

    Backend route: ``GET /smr/factories/{factory_id}/events``
    (query ``limit``, ``cursor``).
    """
    params: dict[str, Any] = {}
    if limit is not None:
        params["limit"] = int(limit)
    if cursor is not None:
        params["cursor"] = str(cursor)
    payload = session._request_json(
        "GET",
        f"/smr/factories/{factory_id}/events",
        params=params or None,
    )
    return FactoryEventsPage.from_wire(payload)


__all__ = [
    "FactoryEffortUsage",
    "FactoryEvent",
    "FactoryEventsPage",
    "FactoryUsage",
    "FactoryUsageBudget",
    "FactoryUsageCost",
    "fetch_factory_events",
    "fetch_factory_usage",
]
