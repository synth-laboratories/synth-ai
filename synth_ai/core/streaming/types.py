from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from synth_ai.sdk.shared.orchestration.events import BaseJobEvent


class StreamType(Enum):
    """Categories of streaming payloads emitted by training jobs."""

    STATUS = auto()
    EVENTS = auto()
    METRICS = auto()
    TIMELINE = auto()

    @property
    def endpoint_path(self) -> str:
        """Return the endpoint suffix used when polling this stream."""
        return {
            StreamType.STATUS: "",
            StreamType.EVENTS: "/events",
            StreamType.METRICS: "/metrics",
            StreamType.TIMELINE: "/timeline",
        }[self]


@dataclass(slots=True)
class StreamMessage:
    """Unified representation of a streaming payload.

    Attributes:
        stream_type: Category of the payload (STATUS, EVENTS, METRICS, TIMELINE)
        timestamp: ISO timestamp or empty string
        job_id: The job identifier
        data: Raw event data dictionary
        seq: Sequence number for events
        step: Step number for metrics
        phase: Phase name for timeline events
        _typed_event: Cached typed event (lazily parsed)
    """

    stream_type: StreamType
    timestamp: str
    job_id: str
    data: dict[str, Any]
    seq: int | None = None
    step: int | None = None
    phase: str | None = None
    _typed_event: Any = field(default=None, repr=False, compare=False)

    def typed_event(self) -> BaseJobEvent | None:
        """Return the typed event for EVENTS stream messages.

        Lazily parses the raw data into a JobEvent or CandidateEvent using
        the OpenResponses-aligned event parser. Returns None for non-event
        stream types or if the event type is not recognized.

        Returns:
            A BaseJobEvent subclass (JobEvent or CandidateEvent) if parseable,
            otherwise None.
        """
        if self.stream_type is not StreamType.EVENTS:
            return None

        if self._typed_event is None:
            # Lazy import to avoid circular dependencies
            from synth_ai.sdk.shared.orchestration.events import parse_event

            self._typed_event = parse_event(self.data, self.job_id)

        return self._typed_event

    @property
    def key(self) -> str:
        """Return a unique identifier used for deduplication."""
        if self.stream_type is StreamType.EVENTS:
            return f"event:{self.seq}"
        if self.stream_type is StreamType.METRICS:
            name = self.data.get("name", "")
            if self.step is not None:
                return f"metric:{name}:{self.step}"
            ts = (
                self.timestamp
                or self.data.get("created_at")
                or self.data.get("updated_at")
                or self.data.get("timestamp")
            )
            if ts:
                return f"metric:{name}:{ts}"
            try:
                fingerprint = json.dumps(self.data, sort_keys=True, default=str)
            except Exception:
                fingerprint = repr(self.data)
            return f"metric:{name}:{fingerprint}"
        if self.stream_type is StreamType.TIMELINE:
            return f"timeline:{self.phase}:{self.timestamp}"
        # Include status value in key so status changes are always shown
        status = str(self.data.get("status") or self.data.get("state") or "")
        return f"status:{status}:{self.timestamp}"

    @classmethod
    def from_status(cls, job_id: str, status_data: dict[str, Any]) -> StreamMessage:
        """Create a message representing a job status payload."""
        return cls(
            stream_type=StreamType.STATUS,
            timestamp=status_data.get("updated_at", "") or status_data.get("created_at", ""),
            job_id=job_id,
            data=status_data,
        )

    @classmethod
    def from_event(cls, job_id: str, event_data: dict[str, Any]) -> StreamMessage:
        """Create a message describing a job event."""
        return cls(
            stream_type=StreamType.EVENTS,
            timestamp=event_data.get("created_at", ""),
            job_id=job_id,
            data=event_data,
            seq=event_data.get("seq"),
        )

    @classmethod
    def from_metric(cls, job_id: str, metric_data: dict[str, Any]) -> StreamMessage:
        """Create a message describing a metric point."""
        return cls(
            stream_type=StreamType.METRICS,
            timestamp=metric_data.get("created_at", ""),
            job_id=job_id,
            data=metric_data,
            step=metric_data.get("step"),
        )

    @classmethod
    def from_timeline(cls, job_id: str, timeline_data: dict[str, Any]) -> StreamMessage:
        """Create a message describing a status timeline entry."""
        return cls(
            stream_type=StreamType.TIMELINE,
            timestamp=timeline_data.get("created_at", ""),
            job_id=job_id,
            data=timeline_data,
            phase=timeline_data.get("phase"),
        )


__all__ = ["StreamMessage", "StreamType"]
