"""Typed canonical Factory runtime-policy contracts.

# See: specifications/tanha/factory_runtime_orchestration.md
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Literal, Mapping

from synth_ai.core.contracts.json_value import JsonObject


def _wire(value: object) -> JsonObject:
    to_wire = getattr(value, "to_wire", None)
    if callable(to_wire):
        return dict(to_wire())
    if isinstance(value, Mapping):
        return {str(key): item for key, item in value.items()}
    raise TypeError("runtime policy values must be mappings or support to_wire()")


@dataclass(frozen=True, slots=True)
class FactoryRuntimeCapacity:
    max_active_efforts: int | None = None
    max_active_runs: int | None = None
    max_actors_per_run: int | None = None

    def to_wire(self) -> JsonObject:
        return {
            key: value
            for key, value in (
                ("max_active_efforts", self.max_active_efforts),
                ("max_active_runs", self.max_active_runs),
                ("max_actors_per_run", self.max_actors_per_run),
            )
            if value is not None
        }


@dataclass(frozen=True, slots=True)
class FactoryRuntimeWindowBudget:
    hard_limit_usd: float
    target_usd: float | None = None
    pacing: Literal["none", "even"] = "none"

    def to_wire(self) -> JsonObject:
        payload: JsonObject = {
            "hard_limit_usd": self.hard_limit_usd,
            "pacing": self.pacing,
        }
        if self.target_usd is not None:
            payload["target_usd"] = self.target_usd
        return payload


@dataclass(frozen=True, slots=True)
class FactoryRuntimeSpendConstraint:
    name: str
    hard_limit_usd: float
    period: Literal["all_time", "day", "week", "month"] = "all_time"
    target_usd: float | None = None
    pacing: Literal["none", "even"] = "none"
    rollover: Literal["discard"] = "discard"

    def to_wire(self) -> JsonObject:
        payload: JsonObject = {
            "name": self.name,
            "period": self.period,
            "hard_limit_usd": self.hard_limit_usd,
            "pacing": self.pacing,
            "rollover": self.rollover,
        }
        if self.target_usd is not None:
            payload["target_usd"] = self.target_usd
        return payload


@dataclass(frozen=True, slots=True)
class FactoryRuntimeScheduleWindow:
    name: str
    cron: str
    duration_seconds: int
    timezone: str | None = None
    budget: FactoryRuntimeWindowBudget | Mapping[str, Any] | None = None
    capacity: FactoryRuntimeCapacity | Mapping[str, Any] | None = None
    runtime_profiles: tuple[str, ...] = ()
    on_close: Literal["drain_then_sleep", "pause_safe", "stop"] = "drain_then_sleep"

    def to_wire(self) -> JsonObject:
        payload: JsonObject = {
            "name": self.name,
            "cron": self.cron,
            "duration_seconds": self.duration_seconds,
            "runtime_profiles": list(self.runtime_profiles),
            "on_close": self.on_close,
        }
        if self.timezone is not None:
            payload["timezone"] = self.timezone
        if self.budget is not None:
            payload["budget"] = _wire(self.budget)
        if self.capacity is not None:
            payload["capacity"] = _wire(self.capacity)
        return payload


@dataclass(frozen=True, slots=True)
class FactoryRuntimeProfile:
    name: str
    run_kind: Literal["research", "maintenance", "verification"]
    max_wallclock_seconds: int
    provider: str | None = None
    model: str | None = None
    target_spend_usd: float | None = None
    hard_spend_limit_usd: float | None = None
    max_tokens: int | None = None
    max_gpu_seconds: int | None = None
    max_actors: int | None = None

    def to_wire(self) -> JsonObject:
        payload: JsonObject = {
            "name": self.name,
            "run_kind": self.run_kind,
            "max_wallclock_seconds": self.max_wallclock_seconds,
        }
        for key in (
            "provider",
            "model",
            "target_spend_usd",
            "hard_spend_limit_usd",
            "max_tokens",
            "max_gpu_seconds",
            "max_actors",
        ):
            value = getattr(self, key)
            if value is not None:
                payload[key] = value
        return payload


@dataclass(frozen=True, slots=True)
class FactoryRuntimeDuration:
    max_elapsed_seconds: int | None = None
    max_active_compute_seconds: int | None = None
    max_worker_seconds: int | None = None
    on_limit: Literal["drain", "pause", "stop"] = "drain"

    def to_wire(self) -> JsonObject:
        payload: JsonObject = {"on_limit": self.on_limit}
        for key in (
            "max_elapsed_seconds",
            "max_active_compute_seconds",
            "max_worker_seconds",
        ):
            value = getattr(self, key)
            if value is not None:
                payload[key] = value
        return payload


@dataclass(frozen=True, slots=True)
class FactoryRuntimeMaintenance:
    enabled: bool = False
    every_n_research_runs: int | None = None
    counter_boundary: Literal[
        "next_maintenance", "next_research_terminal"
    ] = "next_maintenance"

    def to_wire(self) -> JsonObject:
        payload: JsonObject = {
            "enabled": self.enabled,
            "counter_boundary": self.counter_boundary,
        }
        if self.every_n_research_runs is not None:
            payload["every_n_research_runs"] = self.every_n_research_runs
        return payload


@dataclass(frozen=True, slots=True)
class FactoryRuntimeRenewal:
    mode: Literal["none", "evidence_gated"] = "none"
    lease_seconds: int | None = None
    trusted_score_min_delta: float | None = None
    allow_information_gain: bool = False
    require_evidence_receipt: bool = True
    max_consecutive_stalled_cycles: int = 2
    on_first_stall: Literal["continue", "redirect", "targeted_probe"] = "redirect"
    on_repeated_stall: Literal["drain", "pause", "human_review"] = "pause"

    def to_wire(self) -> JsonObject:
        payload: JsonObject = {
            "mode": self.mode,
            "allow_information_gain": self.allow_information_gain,
            "require_evidence_receipt": self.require_evidence_receipt,
            "max_consecutive_stalled_cycles": self.max_consecutive_stalled_cycles,
            "on_first_stall": self.on_first_stall,
            "on_repeated_stall": self.on_repeated_stall,
        }
        if self.lease_seconds is not None:
            payload["lease_seconds"] = self.lease_seconds
        if self.trusted_score_min_delta is not None:
            payload["trusted_score_min_delta"] = self.trusted_score_min_delta
        return payload


@dataclass(frozen=True, slots=True)
class FactoryRuntimeControllerAuthority:
    delegated_fields: tuple[str, ...] = ()
    may_pause_factory: bool = True
    may_increase_owner_envelope: Literal[False] = False

    def to_wire(self) -> JsonObject:
        return {
            "delegated_fields": list(self.delegated_fields),
            "may_pause_factory": self.may_pause_factory,
            "may_increase_owner_envelope": self.may_increase_owner_envelope,
        }


@dataclass(frozen=True, slots=True)
class FactoryRuntimePolicy:
    timezone: str = "UTC"
    schedules: tuple[FactoryRuntimeScheduleWindow | Mapping[str, Any], ...] = ()
    spend: tuple[FactoryRuntimeSpendConstraint | Mapping[str, Any], ...] = ()
    duration: FactoryRuntimeDuration | Mapping[str, Any] = field(
        default_factory=FactoryRuntimeDuration
    )
    capacity: FactoryRuntimeCapacity | Mapping[str, Any] = field(
        default_factory=FactoryRuntimeCapacity
    )
    runtime_profiles: tuple[FactoryRuntimeProfile | Mapping[str, Any], ...] = ()
    maintenance: FactoryRuntimeMaintenance | Mapping[str, Any] = field(
        default_factory=FactoryRuntimeMaintenance
    )
    renewal: FactoryRuntimeRenewal | Mapping[str, Any] = field(
        default_factory=FactoryRuntimeRenewal
    )
    controller_authority: FactoryRuntimeControllerAuthority | Mapping[str, Any] = field(
        default_factory=FactoryRuntimeControllerAuthority
    )
    schema_version: Literal["factory-runtime-policy.v1"] = "factory-runtime-policy.v1"

    def to_wire(self) -> JsonObject:
        return {
            "schema_version": self.schema_version,
            "timezone": self.timezone,
            "schedules": [_wire(item) for item in self.schedules],
            "spend": [_wire(item) for item in self.spend],
            "duration": _wire(self.duration),
            "capacity": _wire(self.capacity),
            "runtime_profiles": [_wire(item) for item in self.runtime_profiles],
            "maintenance": _wire(self.maintenance),
            "renewal": _wire(self.renewal),
            "controller_authority": _wire(self.controller_authority),
        }


@dataclass(frozen=True, slots=True)
class FactoryRuntimePolicyReadback:
    factory_id: str
    revision: int
    etag: str
    policy: JsonObject
    effective_at: datetime
    expires_at: datetime | None
    updated_at: datetime
    raw: JsonObject = field(default_factory=dict)

    @classmethod
    def from_wire(cls, payload: Mapping[str, Any]) -> FactoryRuntimePolicyReadback:
        return cls(
            factory_id=str(payload["factory_id"]),
            revision=int(payload["revision"]),
            etag=str(payload["etag"]),
            policy=_wire(payload["policy"]),
            effective_at=datetime.fromisoformat(str(payload["effective_at"])),
            expires_at=(
                datetime.fromisoformat(str(payload["expires_at"]))
                if payload.get("expires_at")
                else None
            ),
            updated_at=datetime.fromisoformat(str(payload["updated_at"])),
            raw=dict(payload),
        )


@dataclass(frozen=True, slots=True)
class FactoryRuntimePolicyMutationReceipt:
    factory_id: str
    previous_revision: int
    revision: int
    etag: str
    policy: JsonObject
    audit_receipt_id: str
    effective_at: datetime
    expires_at: datetime | None
    effects: tuple[str, ...]
    warnings: tuple[str, ...]
    affected_active_work: tuple[str, ...]
    affected_future_work: tuple[str, ...]
    raw: JsonObject = field(default_factory=dict)

    @classmethod
    def from_wire(
        cls, payload: Mapping[str, Any]
    ) -> FactoryRuntimePolicyMutationReceipt:
        return cls(
            factory_id=str(payload["factory_id"]),
            previous_revision=int(payload["previous_revision"]),
            revision=int(payload["revision"]),
            etag=str(payload["etag"]),
            policy=_wire(payload["policy"]),
            audit_receipt_id=str(payload["audit_receipt_id"]),
            effective_at=datetime.fromisoformat(str(payload["effective_at"])),
            expires_at=(
                datetime.fromisoformat(str(payload["expires_at"]))
                if payload.get("expires_at")
                else None
            ),
            effects=tuple(str(item) for item in payload.get("effects", [])),
            warnings=tuple(str(item) for item in payload.get("warnings", [])),
            affected_active_work=tuple(
                str(item) for item in payload.get("affected_active_work", [])
            ),
            affected_future_work=tuple(
                str(item) for item in payload.get("affected_future_work", [])
            ),
            raw=dict(payload),
        )


__all__ = [name for name in globals() if name.startswith("FactoryRuntime")]
