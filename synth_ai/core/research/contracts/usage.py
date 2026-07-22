"""Typed usage and attribution evidence for Research swarms.

# See: specifications/sdk/core_research_migration.md
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from types import MappingProxyType

from synth_ai.core.contracts.json_value import JsonObject, JsonValue
from synth_ai.core.research.contracts._wire import object_value, required_datetime, required_text
from synth_ai.core.research.contracts.common import ProjectId, SwarmId


def _exact_object(value: JsonValue, *, label: str, fields: frozenset[str]) -> JsonObject:
    payload = object_value(value, operation_id=label)
    missing = fields - payload.keys()
    extra = payload.keys() - fields
    if missing or extra:
        raise ValueError(
            f"{label} fields drifted: missing={sorted(missing)!r} extra={sorted(extra)!r}"
        )
    return payload


def _non_negative_int(payload: JsonObject, name: str) -> int:
    value = payload.get(name)
    if type(value) is not int or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value


def _required_bool(payload: JsonObject, name: str) -> bool:
    value = payload.get(name)
    if not isinstance(value, bool):
        raise ValueError(f"{name} must be a boolean")
    return value


def _optional_datetime(payload: JsonObject, name: str) -> datetime | None:
    if payload.get(name) is None:
        return None
    return required_datetime(payload, name)


@dataclass(frozen=True, slots=True)
class UsageMoney:
    nominal_cents: int
    billed_cents: int
    internal_cost_cents: int
    nominal_pico_usd: int
    billed_pico_usd: int
    internal_cost_pico_usd: int

    @classmethod
    def from_wire(cls, value: JsonValue) -> UsageMoney:
        fields = frozenset(
            {
                "nominal_cents",
                "billed_cents",
                "internal_cost_cents",
                "nominal_pico_usd",
                "billed_pico_usd",
                "internal_cost_pico_usd",
            }
        )
        payload = _exact_object(value, label="swarm usage money", fields=fields)
        return cls(**{name: _non_negative_int(payload, name) for name in fields})

    def to_wire(self) -> JsonObject:
        return {
            "nominal_cents": self.nominal_cents,
            "billed_cents": self.billed_cents,
            "internal_cost_cents": self.internal_cost_cents,
            "nominal_pico_usd": self.nominal_pico_usd,
            "billed_pico_usd": self.billed_pico_usd,
            "internal_cost_pico_usd": self.internal_cost_pico_usd,
        }


@dataclass(frozen=True, slots=True)
class TokenCounts:
    input_tokens: int
    cached_input_tokens: int
    non_cached_input_tokens: int
    output_tokens: int
    reasoning_output_tokens: int
    non_reasoning_output_tokens: int
    snapshots: int

    def __post_init__(self) -> None:
        if self.cached_input_tokens > self.input_tokens:
            raise ValueError("cached_input_tokens cannot exceed input_tokens")
        if self.non_cached_input_tokens != self.input_tokens - self.cached_input_tokens:
            raise ValueError("non_cached_input_tokens must equal input minus cached input")
        if self.reasoning_output_tokens > self.output_tokens:
            raise ValueError("reasoning_output_tokens cannot exceed output_tokens")
        if self.non_reasoning_output_tokens != (
            self.output_tokens - self.reasoning_output_tokens
        ):
            raise ValueError(
                "non_reasoning_output_tokens must equal output minus reasoning output"
            )

    @classmethod
    def from_wire(cls, value: JsonValue) -> TokenCounts:
        fields = frozenset(
            {
                "input_tokens",
                "cached_input_tokens",
                "non_cached_input_tokens",
                "output_tokens",
                "reasoning_output_tokens",
                "non_reasoning_output_tokens",
                "snapshots",
            }
        )
        payload = _exact_object(value, label="swarm token counts", fields=fields)
        return cls(**{name: _non_negative_int(payload, name) for name in fields})

    def to_wire(self) -> JsonObject:
        return {
            "input_tokens": self.input_tokens,
            "cached_input_tokens": self.cached_input_tokens,
            "non_cached_input_tokens": self.non_cached_input_tokens,
            "output_tokens": self.output_tokens,
            "reasoning_output_tokens": self.reasoning_output_tokens,
            "non_reasoning_output_tokens": self.non_reasoning_output_tokens,
            "snapshots": self.snapshots,
        }


@dataclass(frozen=True, slots=True)
class TokenUsage:
    sessions_seen: int
    session_snapshots_count: int
    totals: TokenCounts
    by_model: Mapping[str, TokenCounts]

    def __post_init__(self) -> None:
        if self.session_snapshots_count != self.totals.snapshots:
            raise ValueError("session_snapshots_count must equal totals.snapshots")

    @classmethod
    def from_wire(cls, value: JsonValue) -> TokenUsage:
        payload = _exact_object(
            value,
            label="swarm token usage",
            fields=frozenset(
                {"sessions_seen", "session_snapshots_count", "totals", "by_model"}
            ),
        )
        models = object_value(payload["by_model"], operation_id="swarm token usage.by_model")
        return cls(
            sessions_seen=_non_negative_int(payload, "sessions_seen"),
            session_snapshots_count=_non_negative_int(payload, "session_snapshots_count"),
            totals=TokenCounts.from_wire(payload["totals"]),
            by_model=MappingProxyType(
                {name: TokenCounts.from_wire(model) for name, model in models.items()}
            ),
        )

    def to_wire(self) -> JsonObject:
        return {
            "sessions_seen": self.sessions_seen,
            "session_snapshots_count": self.session_snapshots_count,
            "totals": self.totals.to_wire(),
            "by_model": {name: usage.to_wire() for name, usage in self.by_model.items()},
        }


@dataclass(frozen=True, slots=True)
class ActorTokenUsage:
    input_tokens: int
    cached_input_tokens: int
    output_tokens: int
    reasoning_output_tokens: int
    total_tokens: int

    @classmethod
    def from_wire(cls, value: JsonValue) -> ActorTokenUsage:
        fields = frozenset(
            {
                "input_tokens",
                "cached_input_tokens",
                "output_tokens",
                "reasoning_output_tokens",
                "total_tokens",
            }
        )
        payload = _exact_object(value, label="swarm actor token usage", fields=fields)
        return cls(**{name: _non_negative_int(payload, name) for name in fields})

    def to_wire(self) -> JsonObject:
        return {
            "input_tokens": self.input_tokens,
            "cached_input_tokens": self.cached_input_tokens,
            "output_tokens": self.output_tokens,
            "reasoning_output_tokens": self.reasoning_output_tokens,
            "total_tokens": self.total_tokens,
        }


@dataclass(frozen=True, slots=True)
class ActorUsageMoney:
    """Cent-granular money attributed to one actor."""

    nominal_cents: int
    billed_cents: int
    internal_cost_cents: int

    @classmethod
    def from_wire(cls, value: JsonValue) -> ActorUsageMoney:
        fields = frozenset(
            {"nominal_cents", "billed_cents", "internal_cost_cents"}
        )
        payload = _exact_object(value, label="swarm actor usage money", fields=fields)
        return cls(**{name: _non_negative_int(payload, name) for name in fields})

    def to_wire(self) -> JsonObject:
        return {
            "nominal_cents": self.nominal_cents,
            "billed_cents": self.billed_cents,
            "internal_cost_cents": self.internal_cost_cents,
        }


def _int_mapping(value: JsonValue, *, label: str) -> Mapping[str, int]:
    payload = object_value(value, operation_id=label)
    values: dict[str, int] = {}
    for name, amount in payload.items():
        if type(amount) is not int or amount < 0:
            raise ValueError(f"{label}.{name} must be a non-negative integer")
        values[name] = amount
    return MappingProxyType(values)


@dataclass(frozen=True, slots=True)
class ActorUsage:
    actor_id: str
    task_id: str | None
    task_key: str | None
    worker_id: str | None
    participant_role: str | None
    money: ActorUsageMoney
    event_count: int
    latest_usage_at: datetime | None
    nominal_cents_by_provider: Mapping[str, int]
    nominal_cents_by_model: Mapping[str, int]
    tokens: ActorTokenUsage

    @classmethod
    def from_wire(cls, value: JsonValue) -> ActorUsage:
        payload = _exact_object(
            value,
            label="swarm actor usage",
            fields=frozenset(
                {
                    "actor_id",
                    "task_id",
                    "task_key",
                    "worker_id",
                    "participant_role",
                    "money",
                    "event_count",
                    "latest_usage_at",
                    "nominal_cents_by_provider",
                    "nominal_cents_by_model",
                    "tokens",
                }
            ),
        )
        return cls(
            actor_id=required_text(payload, "actor_id"),
            task_id=_optional_text(payload, "task_id"),
            task_key=_optional_text(payload, "task_key"),
            worker_id=_optional_text(payload, "worker_id"),
            participant_role=_optional_text(payload, "participant_role"),
            money=ActorUsageMoney.from_wire(payload["money"]),
            event_count=_non_negative_int(payload, "event_count"),
            latest_usage_at=_optional_datetime(payload, "latest_usage_at"),
            nominal_cents_by_provider=_int_mapping(
                payload["nominal_cents_by_provider"],
                label="swarm actor usage.nominal_cents_by_provider",
            ),
            nominal_cents_by_model=_int_mapping(
                payload["nominal_cents_by_model"],
                label="swarm actor usage.nominal_cents_by_model",
            ),
            tokens=ActorTokenUsage.from_wire(payload["tokens"]),
        )

    def to_wire(self) -> JsonObject:
        return {
            "actor_id": self.actor_id,
            "task_id": self.task_id,
            "task_key": self.task_key,
            "worker_id": self.worker_id,
            "participant_role": self.participant_role,
            "money": self.money.to_wire(),
            "event_count": self.event_count,
            "latest_usage_at": (
                self.latest_usage_at.isoformat() if self.latest_usage_at is not None else None
            ),
            "nominal_cents_by_provider": dict(self.nominal_cents_by_provider),
            "nominal_cents_by_model": dict(self.nominal_cents_by_model),
            "tokens": self.tokens.to_wire(),
        }


def _optional_text(payload: JsonObject, name: str) -> str | None:
    value = payload.get(name)
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string when present")
    return value.strip()


class UsageSource(StrEnum):
    NONE = "none"
    USAGE_FACTS = "usage_facts"
    SPEND_LEDGER = "spend_ledger"


@dataclass(frozen=True, slots=True)
class UsageFreshness:
    source: UsageSource
    as_of: datetime | None
    record_count: int
    run_is_terminal: bool

    def __post_init__(self) -> None:
        if self.source is UsageSource.NONE and self.record_count != 0:
            raise ValueError("usage source none requires record_count=0")
        if self.source is not UsageSource.NONE and self.record_count == 0:
            raise ValueError("a usage source requires at least one record")

    @classmethod
    def from_wire(cls, value: JsonValue) -> UsageFreshness:
        payload = _exact_object(
            value,
            label="swarm usage freshness",
            fields=frozenset({"source", "as_of", "record_count", "run_is_terminal"}),
        )
        return cls(
            source=UsageSource(required_text(payload, "source")),
            as_of=_optional_datetime(payload, "as_of"),
            record_count=_non_negative_int(payload, "record_count"),
            run_is_terminal=_required_bool(payload, "run_is_terminal"),
        )

    def to_wire(self) -> JsonObject:
        return {
            "source": self.source.value,
            "as_of": self.as_of.isoformat() if self.as_of is not None else None,
            "record_count": self.record_count,
            "run_is_terminal": self.run_is_terminal,
        }


@dataclass(frozen=True, slots=True)
class SwarmUsage:
    swarm_id: SwarmId
    project_id: ProjectId
    money: UsageMoney
    tokens: TokenUsage
    actors: tuple[ActorUsage, ...]
    freshness: UsageFreshness

    @classmethod
    def from_wire(cls, value: JsonValue) -> SwarmUsage:
        payload = _exact_object(
            value,
            label="retrieve_swarm_usage",
            fields=frozenset(
                {
                    "schema_version",
                    "run_id",
                    "project_id",
                    "money",
                    "tokens",
                    "actors",
                    "freshness",
                }
            ),
        )
        if payload["schema_version"] != 1:
            raise ValueError("retrieve_swarm_usage.schema_version must be 1")
        actor_values = payload["actors"]
        if not isinstance(actor_values, list):
            raise ValueError("retrieve_swarm_usage.actors must be an array")
        return cls(
            swarm_id=SwarmId(required_text(payload, "run_id")),
            project_id=ProjectId(required_text(payload, "project_id")),
            money=UsageMoney.from_wire(payload["money"]),
            tokens=TokenUsage.from_wire(payload["tokens"]),
            actors=tuple(ActorUsage.from_wire(actor) for actor in actor_values),
            freshness=UsageFreshness.from_wire(payload["freshness"]),
        )

    def to_wire(self) -> JsonObject:
        return {
            "schema_version": 1,
            "run_id": self.swarm_id,
            "project_id": self.project_id,
            "money": self.money.to_wire(),
            "tokens": self.tokens.to_wire(),
            "actors": [actor.to_wire() for actor in self.actors],
            "freshness": self.freshness.to_wire(),
        }


__all__ = [
    "ActorTokenUsage",
    "ActorUsage",
    "ActorUsageMoney",
    "SwarmUsage",
    "TokenCounts",
    "TokenUsage",
    "UsageFreshness",
    "UsageMoney",
    "UsageSource",
]
