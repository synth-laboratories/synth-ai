"""Event parser for converting raw backend events to typed OpenResponses events.

This module provides utilities to parse raw event dictionaries (from SSE or polling)
into the typed event structures defined in base.py. The parser handles:

- Job lifecycle events (started, in_progress, completed, failed, cancelled)
- Candidate lifecycle events (added, evaluated, completed)
- Mapping legacy event type strings to canonical JobEventType enums
- Converting between timestamp formats (float/ISO 8601)
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
    JobEventType,
    JobStatus,
)

# Mapping from raw backend event type strings to canonical JobEventType
# Supports both legacy and canonical event naming conventions
_EVENT_TYPE_MAP: Dict[str, JobEventType] = {
    # Job lifecycle - canonical format
    "job.started": JobEventType.JOB_STARTED,
    "job.in_progress": JobEventType.JOB_IN_PROGRESS,
    "job.completed": JobEventType.JOB_COMPLETED,
    "job.failed": JobEventType.JOB_FAILED,
    "job.cancelled": JobEventType.JOB_CANCELLED,
    # Job lifecycle - legacy format (learning.policy.*.job.*)
    "learning.policy.gepa.job.started": JobEventType.JOB_STARTED,
    "learning.policy.gepa.job.completed": JobEventType.JOB_COMPLETED,
    "learning.policy.gepa.job.failed": JobEventType.JOB_FAILED,
    "learning.policy.mipro.job.started": JobEventType.JOB_STARTED,
    "learning.policy.mipro.job.completed": JobEventType.JOB_COMPLETED,
    "learning.policy.mipro.job.failed": JobEventType.JOB_FAILED,
    "learning.graph.gepa.job.started": JobEventType.JOB_STARTED,
    "learning.graph.gepa.job.completed": JobEventType.JOB_COMPLETED,
    "learning.graph.gepa.job.failed": JobEventType.JOB_FAILED,
    # Eval lifecycle
    "eval.policy.job.started": JobEventType.JOB_STARTED,
    "eval.policy.job.completed": JobEventType.JOB_COMPLETED,
    "eval.policy.job.failed": JobEventType.JOB_FAILED,
    "eval.verifier.rlm.job.started": JobEventType.JOB_STARTED,
    "eval.verifier.rlm.job.completed": JobEventType.JOB_COMPLETED,
    "eval.verifier.rlm.job.failed": JobEventType.JOB_FAILED,
    # Candidate lifecycle - canonical format
    "candidate.added": JobEventType.CANDIDATE_ADDED,
    "candidate.evaluated": JobEventType.CANDIDATE_EVALUATED,
    "candidate.completed": JobEventType.CANDIDATE_COMPLETED,
    # Candidate lifecycle - legacy format
    "learning.policy.gepa.candidate.new_best": JobEventType.CANDIDATE_EVALUATED,
    "learning.policy.gepa.candidate.evaluated": JobEventType.CANDIDATE_EVALUATED,
    "learning.policy.mipro.candidate.new_best": JobEventType.CANDIDATE_EVALUATED,
    "learning.graph.gepa.candidate.evaluated": JobEventType.CANDIDATE_EVALUATED,
}

# Mapping from JobEventType to default EventLevel
_EVENT_LEVEL_MAP: Dict[JobEventType, EventLevel] = {
    JobEventType.JOB_STARTED: EventLevel.INFO,
    JobEventType.JOB_IN_PROGRESS: EventLevel.INFO,
    JobEventType.JOB_COMPLETED: EventLevel.INFO,
    JobEventType.JOB_FAILED: EventLevel.ERROR,
    JobEventType.JOB_CANCELLED: EventLevel.WARN,
    JobEventType.CANDIDATE_ADDED: EventLevel.INFO,
    JobEventType.CANDIDATE_EVALUATED: EventLevel.INFO,
    JobEventType.CANDIDATE_COMPLETED: EventLevel.INFO,
}

# Event types that represent candidate events (used to decide JobEvent vs CandidateEvent)
_CANDIDATE_EVENT_TYPES = {
    JobEventType.CANDIDATE_ADDED,
    JobEventType.CANDIDATE_EVALUATED,
    JobEventType.CANDIDATE_COMPLETED,
}

# Event types that represent job events
_JOB_EVENT_TYPES = {
    JobEventType.JOB_STARTED,
    JobEventType.JOB_IN_PROGRESS,
    JobEventType.JOB_COMPLETED,
    JobEventType.JOB_FAILED,
    JobEventType.JOB_CANCELLED,
}


def _parse_timestamp(raw: Dict[str, Any]) -> str:
    """Extract and normalize timestamp to ISO 8601 format."""
    ts = raw.get("ts") or raw.get("timestamp") or raw.get("created_at")
    if ts is None:
        return datetime.now(UTC).isoformat()

    # Already ISO 8601 string
    if isinstance(ts, str):
        return ts

    # Unix timestamp (float or int)
    try:
        ts_float = float(ts)
        return datetime.fromtimestamp(ts_float, tz=UTC).isoformat()
    except (TypeError, ValueError):
        return datetime.now(UTC).isoformat()


def _infer_job_status(event_type: JobEventType) -> Optional[JobStatus]:
    """Infer JobStatus from event type."""
    status_map = {
        JobEventType.JOB_STARTED: JobStatus.IN_PROGRESS,
        JobEventType.JOB_IN_PROGRESS: JobStatus.IN_PROGRESS,
        JobEventType.JOB_COMPLETED: JobStatus.COMPLETED,
        JobEventType.JOB_FAILED: JobStatus.FAILED,
        JobEventType.JOB_CANCELLED: JobStatus.CANCELLED,
    }
    return status_map.get(event_type)


def _infer_candidate_status(
    event_type: JobEventType, data: Dict[str, Any]
) -> Optional[CandidateStatus]:
    """Infer CandidateStatus from event type and data."""
    # Check explicit status in data first
    if "status" in data:
        try:
            return CandidateStatus(data["status"])
        except ValueError:
            pass

    # Infer from event type
    status_map = {
        JobEventType.CANDIDATE_ADDED: CandidateStatus.IN_PROGRESS,
        JobEventType.CANDIDATE_EVALUATED: CandidateStatus.COMPLETED,
        JobEventType.CANDIDATE_COMPLETED: CandidateStatus.COMPLETED,
    }
    return status_map.get(event_type)


def parse_event(raw: Dict[str, Any], job_id: Optional[str] = None) -> Optional[BaseJobEvent]:
    """Parse a raw event dictionary into a typed event object.

    Args:
        raw: Raw event dictionary from backend (SSE data or polling response).
             Expected keys: type, job_id, ts/timestamp, data, seq, level, message, run_id
        job_id: Fallback job_id if not present in the raw event

    Returns:
        A JobEvent or CandidateEvent if the event type is recognized, else None.
        Returns None for event types that don't map to job/candidate lifecycle.

    Example:
        >>> raw = {"type": "job.completed", "job_id": "abc123", "seq": 5, "data": {"status": "completed"}}
        >>> event = parse_event(raw)
        >>> event.type
        'job.completed'
    """
    raw_type = str(raw.get("type") or "").lower()
    if not raw_type:
        return None

    # Try direct mapping
    event_type = _EVENT_TYPE_MAP.get(raw_type)

    # If not found, try matching suffix patterns
    if event_type is None:
        for suffix, mapped_type in _EVENT_TYPE_MAP.items():
            if raw_type.endswith(suffix):
                event_type = mapped_type
                break

    if event_type is None:
        # Not a recognized lifecycle event
        return None

    # Extract common fields (OpenResponses-aligned)
    event_job_id = raw.get("job_id") or job_id or ""
    seq = int(raw.get("seq", 0))
    ts = _parse_timestamp(raw)
    level = EventLevel(raw.get("level", _EVENT_LEVEL_MAP.get(event_type, EventLevel.INFO).value))
    message = raw.get("message") or f"Event: {event_type.value}"
    data = dict(raw.get("data") or {})
    run_id = raw.get("run_id")

    # Decide between JobEvent and CandidateEvent
    if event_type in _CANDIDATE_EVENT_TYPES:
        candidate_id = data.get("candidate_id") or data.get("version_id") or raw.get("candidate_id")
        status = _infer_candidate_status(event_type, data)
        return CandidateEvent(
            job_id=event_job_id,
            seq=seq,
            ts=ts,
            type=event_type.value,
            level=level,
            message=message,
            data=data,
            run_id=run_id,
            candidate_id=candidate_id,
            status=status,
        )
    else:
        status = _infer_job_status(event_type)
        return JobEvent(
            job_id=event_job_id,
            seq=seq,
            ts=ts,
            type=event_type.value,
            level=level,
            message=message,
            data=data,
            run_id=run_id,
            status=status,
        )


def is_terminal_event(event: BaseJobEvent) -> bool:
    """Check if an event represents a terminal job state.

    Terminal events indicate the job has finished (success, failure, or cancellation).

    Args:
        event: A typed event object

    Returns:
        True if the event represents job completion, failure, or cancellation
    """
    return event.type in {
        JobEventType.JOB_COMPLETED.value,
        JobEventType.JOB_FAILED.value,
        JobEventType.JOB_CANCELLED.value,
    }


def is_success_event(event: BaseJobEvent) -> bool:
    """Check if an event represents successful job completion.

    Args:
        event: A typed event object

    Returns:
        True if the event represents successful completion
    """
    return event.type == JobEventType.JOB_COMPLETED.value


def is_failure_event(event: BaseJobEvent) -> bool:
    """Check if an event represents job failure.

    Args:
        event: A typed event object

    Returns:
        True if the event represents failure
    """
    return event.type == JobEventType.JOB_FAILED.value


def get_event_type(event: BaseJobEvent) -> Optional[JobEventType]:
    """Get the JobEventType enum from an event's type string.

    Args:
        event: A typed event object

    Returns:
        The JobEventType enum value, or None if not a recognized type
    """
    try:
        return JobEventType(event.type)
    except ValueError:
        return None


__all__ = [
    "parse_event",
    "is_terminal_event",
    "is_success_event",
    "is_failure_event",
    "get_event_type",
]
