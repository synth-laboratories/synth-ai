"""Event parser for converting raw backend events to typed OpenResponses events.

This module now requires the Rust core event parser for canonical behavior.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any, Dict, Optional

from .base import (
    BaseJobEvent,
    CandidateEvent,
    CandidateStatus,
    EventLevel,
    JobEvent,
    JobStatus,
)

try:
    import synth_ai_py
except Exception as exc:  # pragma: no cover
    raise RuntimeError("synth_ai_py is required for orchestration.events.parser.") from exc


def _require_rust() -> Any:
    if synth_ai_py is None or not hasattr(synth_ai_py, "parse_job_event"):
        raise RuntimeError("Rust core event parser required; synth_ai_py is unavailable.")
    return synth_ai_py


_TERMINAL_JOB_STATUSES = {
    "succeeded",
    "completed",
    "failed",
    "cancelled",
    "canceled",
}
_SUCCESS_JOB_STATUSES = {"succeeded", "completed"}
_FAILURE_JOB_STATUSES = {"failed", "cancelled", "canceled"}


def _status_value(status: Any) -> str | None:
    if status is None:
        return None
    if isinstance(status, (JobStatus, CandidateStatus)):
        return str(status.value)
    if isinstance(status, str):
        return status
    return str(status)


def get_event_type(event: BaseJobEvent | Dict[str, Any]) -> Optional[str]:
    """Return the event type string from a parsed event or raw dict."""
    if isinstance(event, dict):
        return event.get("type") or event.get("event_type")
    return getattr(event, "type", None)


def is_terminal_event(event: BaseJobEvent | Dict[str, Any]) -> bool:
    """Return True if the event represents a terminal job state."""
    status = _status_value(
        getattr(event, "status", None) if not isinstance(event, dict) else event.get("status")
    )
    if not status:
        return False
    return status.lower() in _TERMINAL_JOB_STATUSES


def is_success_event(event: BaseJobEvent | Dict[str, Any]) -> bool:
    """Return True if the event represents a successful terminal state."""
    status = _status_value(
        getattr(event, "status", None) if not isinstance(event, dict) else event.get("status")
    )
    if not status:
        return False
    return status.lower() in _SUCCESS_JOB_STATUSES


def is_failure_event(event: BaseJobEvent | Dict[str, Any]) -> bool:
    """Return True if the event represents a failed terminal state."""
    status = _status_value(
        getattr(event, "status", None) if not isinstance(event, dict) else event.get("status")
    )
    if not status:
        return False
    return status.lower() in _FAILURE_JOB_STATUSES


def parse_event(raw: Dict[str, Any], job_id: Optional[str] = None) -> Optional[BaseJobEvent]:
    """Parse a raw event dictionary into a typed event object.

    Args:
        raw: Raw event dictionary from backend (SSE data or polling response).
             Expected keys: type, job_id, ts/timestamp, data, seq, level, message, run_id
        job_id: Fallback job_id if not present in the raw event

    Returns:
        A JobEvent or CandidateEvent if the event type is recognized, else None.
    """
    rust = _require_rust()
    parsed = rust.parse_job_event(raw, job_id)
    if not parsed:
        return None

    data = dict(parsed.get("data") or {})
    ts = parsed.get("ts") or datetime.now(UTC).isoformat()
    level = EventLevel(parsed.get("level", "info"))
    message = parsed.get("message", "")

    if parsed.get("event_kind") == "candidate":
        return CandidateEvent(
            job_id=parsed.get("job_id", ""),
            seq=int(parsed.get("seq", 0)),
            ts=ts,
            type=parsed.get("type", ""),
            level=level,
            message=message,
            data=data,
            run_id=parsed.get("run_id"),
            candidate_id=parsed.get("candidate_id"),
            status=CandidateStatus(parsed["status"]) if parsed.get("status") else None,
        )

    return JobEvent(
        job_id=parsed.get("job_id", ""),
        seq=int(parsed.get("seq", 0)),
        ts=ts,
        type=parsed.get("type", ""),
        level=level,
        message=message,
        data=data,
        run_id=parsed.get("run_id"),
        status=JobStatus(parsed["status"]) if parsed.get("status") else None,
    )


__all__ = [
    "parse_event",
    "get_event_type",
    "is_terminal_event",
    "is_success_event",
    "is_failure_event",
]
