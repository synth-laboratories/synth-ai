"""Public Managed Research runbook contracts."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from typing import TypeAlias

from synth_ai.managed_research.models.smr_host_kinds import SmrHostKind, coerce_smr_host_kind
from synth_ai.managed_research.models.smr_work_modes import SmrWorkMode, coerce_smr_work_mode

JsonObject: TypeAlias = Mapping[str, object]


class SmrRunbookKind(StrEnum):
    LITE = "lite"
    STANDARD = "standard"
    HEAVY = "heavy"
    OVERNIGHT = "overnight"
    CONTINUOUS = "continuous"


SMR_RUNBOOK_KIND_VALUES: tuple[str, ...] = tuple(kind.value for kind in SmrRunbookKind)


def coerce_smr_runbook_kind(
    value: SmrRunbookKind | str | None,
    *,
    field_name: str = "runbook",
) -> SmrRunbookKind | None:
    if value is None:
        return None
    if isinstance(value, SmrRunbookKind):
        return value
    normalized = str(value).strip().lower()
    if not normalized:
        return None
    if normalized == "runbook_lite":
        normalized = SmrRunbookKind.LITE.value
    if normalized == "runbook_standard":
        normalized = SmrRunbookKind.STANDARD.value
    if normalized == "runbook_heavy":
        normalized = SmrRunbookKind.HEAVY.value
    if normalized == "runbook_overnight":
        normalized = SmrRunbookKind.OVERNIGHT.value
    if normalized == "runbook_continuous":
        normalized = SmrRunbookKind.CONTINUOUS.value
    try:
        return SmrRunbookKind(normalized)
    except ValueError as exc:
        allowed = ", ".join(SMR_RUNBOOK_KIND_VALUES)
        raise ValueError(f"{field_name} must be one of: {allowed}") from exc


@dataclass(frozen=True, slots=True)
class SmrRunbookLimitSummary:
    max_spend_usd: float | None = None
    max_wallclock_seconds: int | None = None
    max_gpu_hours: float | None = None
    max_tokens: int | None = None

    @classmethod
    def from_wire(cls, payload: JsonObject | None) -> SmrRunbookLimitSummary:
        if payload is None:
            return cls()
        return cls(
            max_spend_usd=_optional_float(payload, "max_spend_usd"),
            max_wallclock_seconds=_optional_int(payload, "max_wallclock_seconds"),
            max_gpu_hours=_optional_float(payload, "max_gpu_hours"),
            max_tokens=_optional_int(payload, "max_tokens"),
        )

    def to_wire(self) -> dict[str, object]:
        payload: dict[str, object] = {}
        if self.max_spend_usd is not None:
            payload["max_spend_usd"] = self.max_spend_usd
        if self.max_wallclock_seconds is not None:
            payload["max_wallclock_seconds"] = self.max_wallclock_seconds
        if self.max_gpu_hours is not None:
            payload["max_gpu_hours"] = self.max_gpu_hours
        if self.max_tokens is not None:
            payload["max_tokens"] = self.max_tokens
        return payload


@dataclass(frozen=True, slots=True)
class SmrRunbookPreset:
    preset_id: str
    label: str
    description: str
    public: bool
    runbook_kind: SmrRunbookKind
    default_work_mode: SmrWorkMode
    host_kind: SmrHostKind
    visibility: str = "public"
    limit_summary: SmrRunbookLimitSummary = field(default_factory=SmrRunbookLimitSummary)
    capabilities: tuple[str, ...] = ()

    @classmethod
    def from_wire(cls, payload: JsonObject) -> SmrRunbookPreset:
        preset_id = _required_text(payload, "id")
        raw_runbook = _optional_value(payload, "runbook_kind")
        if raw_runbook is None:
            raw_runbook = _optional_value(payload, "runbook")
        runbook_kind = coerce_smr_runbook_kind(
            _optional_text_value(raw_runbook),
            field_name="runbook_kind",
        )
        if runbook_kind is None:
            raise ValueError("runbook_kind is required")
        work_mode = coerce_smr_work_mode(
            _optional_text(payload, "default_work_mode"),
            field_name="default_work_mode",
        )
        if work_mode is None:
            raise ValueError("default_work_mode is required")
        host_kind = coerce_smr_host_kind(
            _optional_text(payload, "host_kind"),
            field_name="host_kind",
        )
        if host_kind is None:
            raise ValueError("host_kind is required")
        return cls(
            preset_id=preset_id,
            label=_required_text(payload, "label"),
            description=_required_text(payload, "description"),
            public=_required_bool(payload, "public"),
            runbook_kind=runbook_kind,
            default_work_mode=work_mode,
            host_kind=host_kind,
            visibility=_optional_text(payload, "visibility") or "public",
            limit_summary=SmrRunbookLimitSummary.from_wire(
                _optional_mapping(payload, "limit_summary")
            ),
            capabilities=_optional_string_tuple(payload, "capabilities"),
        )

    def to_wire(self) -> dict[str, object]:
        return {
            "id": self.preset_id,
            "label": self.label,
            "description": self.description,
            "public": self.public,
            "runbook": self.runbook_kind.value,
            "runbook_kind": self.runbook_kind.value,
            "default_work_mode": self.default_work_mode.value,
            "host_kind": self.host_kind.value,
            "visibility": self.visibility,
            "limit_summary": self.limit_summary.to_wire(),
            "capabilities": list(self.capabilities),
        }


def _required_text(payload: JsonObject, key: str) -> str:
    value = _optional_text(payload, key)
    if value is None:
        raise ValueError(f"{key} is required")
    return value


def _optional_text(payload: JsonObject, key: str) -> str | None:
    return _optional_text_value(_optional_value(payload, key))


def _optional_text_value(value: object) -> str | None:
    if isinstance(value, StrEnum):
        value = value.value
    text = str(value or "").strip()
    return text or None


def _required_bool(payload: JsonObject, key: str) -> bool:
    value = _optional_value(payload, key)
    if isinstance(value, bool):
        return value
    raise ValueError(f"{key} must be a boolean")


def _optional_mapping(payload: JsonObject, key: str) -> JsonObject | None:
    value = _optional_value(payload, key)
    if value is None:
        return None
    if isinstance(value, Mapping):
        return value
    raise ValueError(f"{key} must be an object when provided")


def _optional_string_tuple(payload: JsonObject, key: str) -> tuple[str, ...]:
    value = _optional_value(payload, key)
    if value is None:
        return ()
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise ValueError(f"{key} must be an array when provided")
    return tuple(_required_sequence_text(item, field_name=key) for item in value)


def _required_sequence_text(value: object, *, field_name: str) -> str:
    text = _optional_text_value(value)
    if text is None:
        raise ValueError(f"{field_name} entries must be non-empty strings")
    return text


def _optional_float(payload: JsonObject, key: str) -> float | None:
    value = _optional_value(payload, key)
    if value is None:
        return None
    return float(value)


def _optional_int(payload: JsonObject, key: str) -> int | None:
    value = _optional_value(payload, key)
    if value is None:
        return None
    return int(value)


def _optional_value(payload: JsonObject, key: str) -> object | None:
    return payload.get(key)


__all__ = [
    "SMR_RUNBOOK_KIND_VALUES",
    "SmrRunbookKind",
    "SmrRunbookLimitSummary",
    "SmrRunbookPreset",
    "coerce_smr_runbook_kind",
]
