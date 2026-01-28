"""Job lifecycle utilities for optimization jobs.

This module provides utilities for tracking and emitting job lifecycle events
using the OpenResponses-aligned event schema. It can be used by SDK clients
to construct canonical events and track job state.

Example:
    >>> from synth_ai.sdk.optimization.job import JobLifecycle, JobStatus
    >>>
    >>> # Create a job lifecycle tracker
    >>> lifecycle = JobLifecycle(job_id="job_123")
    >>>
    >>> # Emit job started event
    >>> event = lifecycle.start()
    >>> print(event)  # {"type": "job.in_progress", "job_id": "job_123", ...}
    >>>
    >>> # Check status
    >>> print(lifecycle.status)  # JobStatus.IN_PROGRESS
    >>>
    >>> # Emit completion event
    >>> event = lifecycle.complete(data={"best_score": 0.95})
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

from .events import JobEventType

try:
    import synth_ai_py  # type: ignore
except Exception as exc:  # pragma: no cover
    raise RuntimeError("synth_ai_py is required for optimization.job.") from exc


def _require_rust() -> Any:
    if synth_ai_py is None or not hasattr(synth_ai_py, "job_status_from_str"):
        raise RuntimeError("Rust core job lifecycle required; synth_ai_py is unavailable.")
    return synth_ai_py


def _map_rust_status(status: str) -> JobStatus:
    mapping = {
        "pending": JobStatus.PENDING,
        "queued": JobStatus.PENDING,
        "running": JobStatus.IN_PROGRESS,
        "succeeded": JobStatus.COMPLETED,
        "failed": JobStatus.FAILED,
        "cancelled": JobStatus.CANCELLED,
    }
    return mapping.get(status, JobStatus.PENDING)


class JobStatus(str, Enum):
    """Job lifecycle status values."""

    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

    @classmethod
    def from_string(cls, status: str) -> JobStatus:
        rust = _require_rust()
        rust_status = rust.job_status_from_str(status)
        if isinstance(rust_status, str):
            return _map_rust_status(rust_status)
        normalized = status.strip().lower().replace(" ", "_")
        if normalized in ("success", "succeeded", "completed", "complete"):
            return cls.COMPLETED
        if normalized in ("cancelled", "canceled", "cancel"):
            return cls.CANCELLED
        if normalized in ("failed", "failure", "error"):
            return cls.FAILED
        if normalized in ("running", "in_progress"):
            return cls.IN_PROGRESS
        if normalized in ("queued", "pending"):
            return cls.PENDING
        return cls.PENDING

    @property
    def is_terminal(self) -> bool:
        return self in (JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED)

    @property
    def is_success(self) -> bool:
        return self == JobStatus.COMPLETED


@dataclass
class JobLifecycle:
    """Track and emit job lifecycle events.

    This class provides a stateful wrapper around job lifecycle, making it easy
    to emit canonical events and track status transitions. It maintains an
    event history that can be used for debugging or persistence.

    Attributes:
        job_id: The unique job identifier
        status: Current job status
        events: History of emitted events
        started_at: Timestamp when job started (set on first start())
        ended_at: Timestamp when job ended (set on complete/fail/cancel)
    """

    job_id: str
    status: JobStatus = JobStatus.PENDING
    events: List[Dict[str, Any]] = field(default_factory=list)
    started_at: Optional[float] = None
    ended_at: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    _rust: Any | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        rust = _require_rust()
        self._rust = rust.JobLifecycle(self.job_id)
        if hasattr(self._rust, "status"):
            self.status = _map_rust_status(self._rust.status)

    def _emit(
        self,
        event_type: JobEventType,
        data: Optional[Dict[str, Any]] = None,
        message: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Emit a job event and add it to history.

        Args:
            event_type: The canonical event type
            data: Optional event data payload
            message: Optional human-readable message

        Returns:
            The constructed event dictionary
        """
        now = time.time()
        event: Dict[str, Any] = {
            "type": event_type.value,
            "job_id": self.job_id,
            "timestamp": now,
            "seq": len(self.events) + 1,
        }
        if data:
            event["data"] = data
        if message:
            event["message"] = message

        self.events.append(event)
        return event

    def start(
        self,
        data: Optional[Dict[str, Any]] = None,
        message: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Emit a job.in_progress event and update status.

        Args:
            data: Optional event data (e.g., config info)
            message: Optional human-readable message

        Returns:
            The job.in_progress event dictionary

        Raises:
            ValueError: If job is not in PENDING status
        """
        if self._rust is not None:
            event = self._rust.start(data, message)
            if isinstance(event, dict):
                self.events.append(event)
                timestamp = event.get("timestamp")
                if isinstance(timestamp, (int, float)):
                    self.started_at = float(timestamp)
            self.status = JobStatus.IN_PROGRESS
            return event

        if self.status != JobStatus.PENDING:
            raise ValueError(f"Cannot start job in {self.status} status")

        self.status = JobStatus.IN_PROGRESS
        self.started_at = time.time()

        return self._emit(
            JobEventType.JOB_IN_PROGRESS,
            data=data,
            message=message or "Job started",
        )

    def complete(
        self,
        data: Optional[Dict[str, Any]] = None,
        message: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Emit a job.completed event and update status.

        Args:
            data: Optional event data (e.g., results, best_score)
            message: Optional human-readable message

        Returns:
            The job.completed event dictionary

        Raises:
            ValueError: If job is not in IN_PROGRESS status
        """
        if self._rust is not None:
            event = self._rust.complete(data, message)
            if isinstance(event, dict):
                self.events.append(event)
                timestamp = event.get("timestamp")
                if isinstance(timestamp, (int, float)):
                    self.ended_at = float(timestamp)
                    if self.started_at is None:
                        self.started_at = self.ended_at
            self.status = JobStatus.COMPLETED
            return event

        if self.status != JobStatus.IN_PROGRESS:
            raise ValueError(f"Cannot complete job in {self.status} status")

        self.status = JobStatus.COMPLETED
        self.ended_at = time.time()

        elapsed = self.ended_at - self.started_at if self.started_at else None
        event_data = dict(data or {})
        if elapsed is not None:
            event_data["elapsed_seconds"] = elapsed

        return self._emit(
            JobEventType.JOB_COMPLETED,
            data=event_data if event_data else None,
            message=message or "Job completed successfully",
        )

    def fail(
        self,
        error: Optional[str] = None,
        data: Optional[Dict[str, Any]] = None,
        message: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Emit a job.failed event and update status.

        Args:
            error: Error message or description
            data: Optional event data (e.g., traceback, error details)
            message: Optional human-readable message

        Returns:
            The job.failed event dictionary

        Raises:
            ValueError: If job is not in IN_PROGRESS status
        """
        if self._rust is not None:
            event = self._rust.fail(error, data)
            if isinstance(event, dict):
                self.events.append(event)
                timestamp = event.get("timestamp")
                if isinstance(timestamp, (int, float)):
                    self.ended_at = float(timestamp)
                    if self.started_at is None:
                        self.started_at = self.ended_at
            self.status = JobStatus.FAILED
            return event

        if self.status != JobStatus.IN_PROGRESS:
            raise ValueError(f"Cannot fail job in {self.status} status")

        self.status = JobStatus.FAILED
        self.ended_at = time.time()

        event_data = dict(data or {})
        if error:
            event_data["error"] = error

        return self._emit(
            JobEventType.JOB_FAILED,
            data=event_data if event_data else None,
            message=message or error or "Job failed",
        )

    def cancel(
        self,
        reason: Optional[str] = None,
        data: Optional[Dict[str, Any]] = None,
        message: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Emit a job.cancelled event and update status.

        Args:
            reason: Cancellation reason
            data: Optional event data
            message: Optional human-readable message

        Returns:
            The job.cancelled event dictionary

        Raises:
            ValueError: If job is already in a terminal status
        """
        if self._rust is not None:
            event = self._rust.cancel(message)
            if isinstance(event, dict):
                self.events.append(event)
                timestamp = event.get("timestamp")
                if isinstance(timestamp, (int, float)):
                    self.ended_at = float(timestamp)
                    if self.started_at is None:
                        self.started_at = self.ended_at
            self.status = JobStatus.CANCELLED
            return event

        if self.status in {JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED}:
            raise ValueError(f"Cannot cancel job in {self.status} status")

        self.status = JobStatus.CANCELLED
        self.ended_at = time.time()

        event_data = dict(data or {})
        if reason:
            event_data["reason"] = reason

        return self._emit(
            JobEventType.JOB_CANCELLED,
            data=event_data if event_data else None,
            message=message or reason or "Job cancelled",
        )

    def add_candidate(
        self,
        candidate_id: str,
        data: Optional[Dict[str, Any]] = None,
        message: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Emit a candidate.added event.

        Args:
            candidate_id: Unique identifier for the candidate
            data: Optional event data (e.g., candidate config)
            message: Optional human-readable message

        Returns:
            The candidate.added event dictionary
        """
        event_data = dict(data or {})
        event_data["candidate_id"] = candidate_id

        return self._emit(
            JobEventType.CANDIDATE_ADDED,
            data=event_data,
            message=message or f"Candidate {candidate_id} added",
        )

    def complete_candidate(
        self,
        candidate_id: str,
        score: Optional[float] = None,
        data: Optional[Dict[str, Any]] = None,
        message: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Emit a candidate.completed event.

        Args:
            candidate_id: Unique identifier for the candidate
            score: Optional evaluation score
            data: Optional event data (e.g., metrics, evaluation results)
            message: Optional human-readable message

        Returns:
            The candidate.completed event dictionary
        """
        event_data = dict(data or {})
        event_data["candidate_id"] = candidate_id
        if score is not None:
            event_data["score"] = score

        return self._emit(
            JobEventType.CANDIDATE_COMPLETED,
            data=event_data,
            message=message or f"Candidate {candidate_id} completed",
        )

    @property
    def is_terminal(self) -> bool:
        """Check if the job is in a terminal status."""
        return self.status in {
            JobStatus.COMPLETED,
            JobStatus.FAILED,
            JobStatus.CANCELLED,
        }

    @property
    def is_successful(self) -> bool:
        """Check if the job completed successfully."""
        return self.status == JobStatus.COMPLETED

    @property
    def elapsed_seconds(self) -> Optional[float]:
        """Get elapsed time in seconds (None if not started)."""
        if self._rust is not None:
            try:
                elapsed = self._rust.elapsed_seconds
                if isinstance(elapsed, (int, float)):
                    return float(elapsed)
            except Exception:
                pass
        if self.started_at is None:
            return None
        end = self.ended_at or time.time()
        return end - self.started_at


__all__ = [
    "JobLifecycle",
    "JobStatus",
]
