"""Data models for TUI monitoring views."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, Optional


@dataclass(slots=True)
class JobEvent:
    seq: int
    event_type: str
    message: str | None = None
    timestamp: str | None = None
    data: Dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class JobSummary:
    job_id: str
    status: str
    training_type: str | None = None
    created_at: str | None = None
    started_at: str | None = None
    finished_at: str | None = None
    best_score: float | None = None
    total_tokens: int | None = None
    total_cost_usd: float | None = None
    error: str | None = None
    raw: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_api(cls, payload: Dict[str, Any]) -> "JobSummary":
        return cls(
            job_id=str(payload.get("job_id") or payload.get("id") or ""),
            status=str(payload.get("status") or "unknown"),
            training_type=payload.get("training_type"),
            created_at=payload.get("created_at"),
            started_at=payload.get("started_at"),
            finished_at=payload.get("finished_at"),
            best_score=_to_float(payload.get("best_score")),
            total_tokens=_to_int(payload.get("total_tokens")),
            total_cost_usd=_to_float(payload.get("total_cost_usd") or payload.get("total_cost")),
            error=payload.get("error"),
            raw=dict(payload),
        )


@dataclass(slots=True)
class JobDetail(JobSummary):
    config: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    progress: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_api(cls, payload: Dict[str, Any]) -> "JobDetail":
        summary = JobSummary.from_api(payload)
        return cls(
            **summary.__dict__,
            config=dict(payload.get("config") or payload.get("config_body") or {}),
            metadata=dict(payload.get("metadata") or {}),
            progress=dict(payload.get("progress") or {}),
        )


@dataclass(slots=True)
class JobListFilter:
    statuses: tuple[str, ...] = ()
    search: str | None = None
    limit: int = 50

    def matches(self, job: JobSummary) -> bool:
        if self.statuses and job.status.lower() not in {s.lower() for s in self.statuses}:
            return False
        if self.search:
            haystack = f"{job.job_id} {job.training_type or ''} {job.status}".lower()
            if self.search.lower() not in haystack:
                return False
        return True


@dataclass(slots=True)
class ActionResult:
    ok: bool
    message: str
    payload: Dict[str, Any] = field(default_factory=dict)


def _to_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _to_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def coerce_events(raw_events: Iterable[Dict[str, Any]]) -> list[JobEvent]:
    events: list[JobEvent] = []
    for idx, event in enumerate(raw_events):
        seq_value = event.get("seq") or event.get("sequence") or event.get("id") or idx
        seq = _to_int(seq_value) or idx
        events.append(
            JobEvent(
                seq=seq,
                event_type=str(event.get("type") or event.get("event_type") or "event"),
                message=event.get("message"),
                timestamp=event.get("timestamp") or event.get("created_at"),
                data=dict(event.get("data") or {}),
            )
        )
    return events

