"""Event type definitions and enums for prompt learning jobs.

Rust-backed enum values for consistency across SDKs.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Set

try:
    import synth_ai_py
except Exception as exc:  # pragma: no cover
    raise RuntimeError("synth_ai_py is required for orchestration.events.types.") from exc


def _require_rust() -> Any:
    if synth_ai_py is None or not hasattr(synth_ai_py, "orchestration_event_enum_values"):
        raise RuntimeError("Rust core event enums required; synth_ai_py is unavailable.")
    return synth_ai_py


_rust = _require_rust()
_ENUM_VALUES = _rust.orchestration_event_enum_values()


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
    return bool(_rust.orchestration_is_valid_event_type(event_type))


def validate_event_type(event_type: str) -> str:
    """Validate and return the event type string.

    Raises ValueError if the event type is not recognized.
    """
    return _rust.orchestration_validate_event_type(event_type)


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
