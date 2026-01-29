"""Event type definitions and enums for prompt learning jobs.

Rust-backed enum values for consistency across SDKs.
"""

from __future__ import annotations

import json
from enum import Enum
from pathlib import Path
from typing import Set

try:
    import synth_ai_py
except Exception as exc:  # pragma: no cover
    raise RuntimeError("synth_ai_py is required for orchestration.events.types.") from exc


def _load_enum_values_fallback() -> dict[str, dict[str, str]]:
    asset = (
        Path(__file__).resolve().parents[5] / "synth_ai_core" / "assets" / "event_enum_values.json"
    )
    if asset.exists():
        return json.loads(asset.read_text())
    return {}


_RUST = synth_ai_py if synth_ai_py is not None else None
if _RUST is None or not hasattr(_RUST, "orchestration_event_enum_values"):
    _ENUM_VALUES = _load_enum_values_fallback()
else:
    _ENUM_VALUES = _RUST.orchestration_event_enum_values()


class _StrEnum(str, Enum):
    """Base class for string enums."""

    pass


class _JobStatusMixin(_StrEnum):
    """JobStatus mixin with helper classmethods."""

    @classmethod
    def terminal_statuses(cls) -> Set[JobStatus]:
        return {cls.SUCCEEDED, cls.COMPLETED, cls.FAILED, cls.CANCELLED}

    @classmethod
    def active_statuses(cls) -> Set[JobStatus]:
        return {cls.QUEUED, cls.PENDING, cls.RUNNING, cls.IN_PROGRESS}


def _build_enum(name: str, values: dict[str, str], mixin: type[Enum] | None = None):
    base = mixin or _StrEnum
    return Enum(name, values, type=base, module=__name__)


JobStatus = _build_enum("JobStatus", _ENUM_VALUES.get("JobStatus", {}), _JobStatusMixin)
Phase = _build_enum("Phase", _ENUM_VALUES.get("Phase", {}))
CandidateStatus = _build_enum("CandidateStatus", _ENUM_VALUES.get("CandidateStatus", {}))
MutationType = _build_enum("MutationType", _ENUM_VALUES.get("MutationType", {}))
TerminationReason = _build_enum("TerminationReason", _ENUM_VALUES.get("TerminationReason", {}))
ErrorType = _build_enum("ErrorType", _ENUM_VALUES.get("ErrorType", {}))
EventType = _build_enum("EventType", _ENUM_VALUES.get("EventType", {}))
EventLevel = _build_enum("EventLevel", _ENUM_VALUES.get("EventLevel", {}))
JobEventType = _build_enum("JobEventType", _ENUM_VALUES.get("JobEventType", {}))


# =============================================================================
# Event Type Validation
# =============================================================================


def is_valid_event_type(event_type: str) -> bool:
    """Check if an event type string is a known event type."""
    if _RUST is not None and hasattr(_RUST, "orchestration_is_valid_event_type"):
        return bool(_RUST.orchestration_is_valid_event_type(event_type))
    return event_type in _ENUM_VALUES.get("EventType", {}) or event_type in _ENUM_VALUES.get(
        "JobEventType", {}
    )


def validate_event_type(event_type: str) -> str:
    """Validate and return the event type string.

    Raises ValueError if the event type is not recognized.
    """
    if _RUST is not None and hasattr(_RUST, "orchestration_validate_event_type"):
        return _RUST.orchestration_validate_event_type(event_type)
    if not is_valid_event_type(event_type):
        raise ValueError(f"Unknown event type: {event_type}")
    return event_type


__all__ = [
    "JobStatus",
    "Phase",
    "CandidateStatus",
    "MutationType",
    "TerminationReason",
    "ErrorType",
    "EventType",
    "EventLevel",
    "JobEventType",
    "is_valid_event_type",
    "validate_event_type",
]
