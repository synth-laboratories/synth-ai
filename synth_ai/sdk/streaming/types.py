from __future__ import annotations

import json
from dataclasses import dataclass
from enum import Enum, auto
from typing import Any


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
    """Unified representation of a streaming payload."""

    stream_type: StreamType
    timestamp: str
    job_id: str
    data: dict[str, Any]
    seq: int | None = None
    step: int | None = None
    phase: str | None = None

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
