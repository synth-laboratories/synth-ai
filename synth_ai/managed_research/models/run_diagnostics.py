"""Typed Managed Research run trace and actor-usage read models."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


def _require_text(value: str | None, *, field_name: str) -> str:
    text = str(value or "").strip()
    if not text:
        raise ValueError(f"{field_name} is required")
    return text


def _optional_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _optional_int(value: Any) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _parse_datetime(value: Any, *, field_name: str) -> datetime:
    if isinstance(value, datetime):
        return value
    text = _require_text(str(value) if value is not None else None, field_name=field_name)
    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(f"{field_name} must be an ISO-8601 datetime") from exc


def _coerce_dict(value: Any, *, field_name: str) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError(f"{field_name} must be an object")
    return {str(key): item for key, item in value.items()}


def _coerce_int_dict(value: Any, *, field_name: str) -> dict[str, int]:
    return {
        str(key): int(item or 0) for key, item in _coerce_dict(value, field_name=field_name).items()
    }


def _coerce_nested_int_dict(value: Any, *, field_name: str) -> dict[str, dict[str, int]]:
    out: dict[str, dict[str, int]] = {}
    for key, item in _coerce_dict(value, field_name=field_name).items():
        out[str(key)] = _coerce_int_dict(item, field_name=f"{field_name}.{key}")
    return out


@dataclass(frozen=True, slots=True)
class SmrRunTraceItem:
    trace_id: str
    org_id: str
    project_id: str
    run_id: str
    created_at: datetime
    artifact_id: str | None = None
    artifact_uri: str | None = None
    artifact_content_path: str | None = None
    task_id: str | None = None
    task_key: str | None = None
    actor_id: str | None = None
    worker_id: str | None = None
    participant_session_id: str | None = None
    participant_role: str | None = None
    turn_id: str | None = None
    event_count: int | None = None
    preview: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "trace_id", _require_text(self.trace_id, field_name="trace_id"))
        object.__setattr__(self, "org_id", _require_text(self.org_id, field_name="org_id"))
        object.__setattr__(
            self, "project_id", _require_text(self.project_id, field_name="project_id")
        )
        object.__setattr__(self, "run_id", _require_text(self.run_id, field_name="run_id"))
        object.__setattr__(self, "artifact_id", _optional_text(self.artifact_id))
        object.__setattr__(self, "artifact_uri", _optional_text(self.artifact_uri))
        object.__setattr__(
            self, "artifact_content_path", _optional_text(self.artifact_content_path)
        )
        object.__setattr__(self, "task_id", _optional_text(self.task_id))
        object.__setattr__(self, "task_key", _optional_text(self.task_key))
        object.__setattr__(self, "actor_id", _optional_text(self.actor_id))
        object.__setattr__(self, "worker_id", _optional_text(self.worker_id))
        object.__setattr__(
            self, "participant_session_id", _optional_text(self.participant_session_id)
        )
        object.__setattr__(self, "participant_role", _optional_text(self.participant_role))
        object.__setattr__(self, "turn_id", _optional_text(self.turn_id))
        object.__setattr__(self, "event_count", _optional_int(self.event_count))
        object.__setattr__(self, "preview", _coerce_dict(self.preview, field_name="preview"))

    @classmethod
    def from_wire(cls, payload: dict[str, Any]) -> SmrRunTraceItem:
        return cls(
            trace_id=_require_text(payload.get("trace_id"), field_name="trace_id"),
            org_id=_require_text(payload.get("org_id"), field_name="org_id"),
            project_id=_require_text(payload.get("project_id"), field_name="project_id"),
            run_id=_require_text(payload.get("run_id"), field_name="run_id"),
            created_at=_parse_datetime(payload.get("created_at"), field_name="created_at"),
            artifact_id=_optional_text(payload.get("artifact_id")),
            artifact_uri=_optional_text(payload.get("artifact_uri")),
            artifact_content_path=_optional_text(payload.get("artifact_content_path")),
            task_id=_optional_text(payload.get("task_id")),
            task_key=_optional_text(payload.get("task_key")),
            actor_id=_optional_text(payload.get("actor_id")),
            worker_id=_optional_text(payload.get("worker_id")),
            participant_session_id=_optional_text(payload.get("participant_session_id")),
            participant_role=_optional_text(payload.get("participant_role")),
            turn_id=_optional_text(payload.get("turn_id")),
            event_count=_optional_int(payload.get("event_count")),
            preview=_coerce_dict(payload.get("preview"), field_name="preview"),
        )


@dataclass(frozen=True, slots=True)
class SmrRunTraces:
    org_id: str
    project_id: str
    run_id: str
    count: int
    traces: tuple[SmrRunTraceItem, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "org_id", _require_text(self.org_id, field_name="org_id"))
        object.__setattr__(
            self, "project_id", _require_text(self.project_id, field_name="project_id")
        )
        object.__setattr__(self, "run_id", _require_text(self.run_id, field_name="run_id"))
        object.__setattr__(self, "count", max(0, int(self.count)))

    @classmethod
    def from_wire(cls, payload: dict[str, Any]) -> SmrRunTraces:
        traces = tuple(
            SmrRunTraceItem.from_wire(item)
            for item in (payload.get("traces") or [])
            if isinstance(item, dict)
        )
        return cls(
            org_id=_require_text(payload.get("org_id"), field_name="org_id"),
            project_id=_require_text(payload.get("project_id"), field_name="project_id"),
            run_id=_require_text(payload.get("run_id"), field_name="run_id"),
            count=int(payload.get("count") or len(traces)),
            traces=traces,
        )


@dataclass(frozen=True, slots=True)
class SmrRunParticipant:
    actor_id: str
    role: str
    session_id: str | None = None
    usage_recording_status: str = "missing"
    actor_key: str | None = None
    participant_session_id: str | None = None

    @classmethod
    def from_wire(cls, payload: dict[str, Any]) -> SmrRunParticipant:
        return cls(
            actor_id=_require_text(payload.get("actor_id"), field_name="actor_id"),
            role=_require_text(payload.get("role"), field_name="role"),
            session_id=_optional_text(payload.get("session_id")),
            usage_recording_status=_require_text(
                payload.get("usage_recording_status"),
                field_name="usage_recording_status",
            ),
            actor_key=_optional_text(payload.get("actor_key")),
            participant_session_id=_optional_text(payload.get("participant_session_id")),
        )


@dataclass(frozen=True, slots=True)
class SmrRunParticipants:
    project_id: str
    run_id: str
    participants: tuple[SmrRunParticipant, ...]

    @classmethod
    def from_wire(cls, payload: dict[str, Any]) -> SmrRunParticipants:
        participants = tuple(
            SmrRunParticipant.from_wire(item)
            for item in (payload.get("participants") or [])
            if isinstance(item, dict)
        )
        return cls(
            project_id=_require_text(payload.get("project_id"), field_name="project_id"),
            run_id=_require_text(payload.get("run_id"), field_name="run_id"),
            participants=participants,
        )


@dataclass(frozen=True, slots=True)
class SmrRunArtifactProgress:
    project_id: str
    run_id: str
    staged: int = 0
    required: int = 0
    missing: tuple[str, ...] = ()
    optional_staged: int = 0
    optional_total: int = 0

    @classmethod
    def from_wire(cls, payload: dict[str, Any]) -> SmrRunArtifactProgress:
        return cls(
            project_id=_require_text(payload.get("project_id"), field_name="project_id"),
            run_id=_require_text(payload.get("run_id"), field_name="run_id"),
            staged=int(payload.get("staged") or 0),
            required=int(payload.get("required") or 0),
            missing=tuple(str(item) for item in (payload.get("missing") or [])),
            optional_staged=int(payload.get("optional_staged") or 0),
            optional_total=int(payload.get("optional_total") or 0),
        )


@dataclass(frozen=True, slots=True)
class SmrRunActorLogEvent:
    event_id: str
    run_id: str
    project_id: str
    occurred_at: datetime
    seq: str
    kind: str
    payload_excerpt: str
    actor_id: str | None = None
    participant_session_id: str | None = None
    turn_id: str | None = None
    byte_count: int = 0
    line_count: int = 0
    truncated: bool = False
    redacted: bool = False

    @classmethod
    def from_wire(cls, payload: dict[str, Any]) -> SmrRunActorLogEvent:
        return cls(
            event_id=_require_text(payload.get("event_id"), field_name="event_id"),
            run_id=_require_text(payload.get("run_id"), field_name="run_id"),
            project_id=_require_text(payload.get("project_id"), field_name="project_id"),
            actor_id=_optional_text(payload.get("actor_id")),
            participant_session_id=_optional_text(payload.get("participant_session_id")),
            turn_id=_optional_text(payload.get("turn_id")),
            occurred_at=_parse_datetime(payload.get("occurred_at"), field_name="occurred_at"),
            seq=_require_text(payload.get("seq"), field_name="seq"),
            kind=_require_text(payload.get("kind"), field_name="kind"),
            payload_excerpt=str(payload.get("payload_excerpt") or ""),
            byte_count=int(payload.get("byte_count") or 0),
            line_count=int(payload.get("line_count") or 0),
            truncated=bool(payload.get("truncated")),
            redacted=bool(payload.get("redacted")),
        )


@dataclass(frozen=True, slots=True)
class SmrRunActorLogs:
    project_id: str
    run_id: str
    events: tuple[SmrRunActorLogEvent, ...]
    next_cursor: str | None = None

    @classmethod
    def from_wire(cls, payload: dict[str, Any]) -> SmrRunActorLogs:
        events = tuple(
            SmrRunActorLogEvent.from_wire(item)
            for item in (payload.get("events") or [])
            if isinstance(item, dict)
        )
        return cls(
            project_id=_require_text(payload.get("project_id"), field_name="project_id"),
            run_id=_require_text(payload.get("run_id"), field_name="run_id"),
            events=events,
            next_cursor=_optional_text(payload.get("next_cursor")),
        )


@dataclass(frozen=True, slots=True)
class SmrRunMeterCost:
    meter_kind: str
    billed_amount_cents: int = 0
    billed_amount_pico_usd: int = 0
    billed_amount_usd: float = 0.0
    quantity: float = 0.0

    @classmethod
    def from_wire(cls, payload: dict[str, Any]) -> SmrRunMeterCost:
        return cls(
            meter_kind=_require_text(payload.get("meter_kind"), field_name="meter_kind"),
            billed_amount_cents=int(payload.get("billed_amount_cents") or 0),
            billed_amount_pico_usd=int(payload.get("billed_amount_pico_usd") or 0),
            billed_amount_usd=float(payload.get("billed_amount_usd") or 0.0),
            quantity=float(payload.get("quantity") or 0.0),
        )


@dataclass(frozen=True, slots=True)
class SmrRunCostSummary:
    run_id: str
    total_cents: int = 0
    total_pico_usd: int = 0
    total_usd: float = 0.0
    recording_status: str = "complete"
    missing_meters: tuple[str, ...] = ()
    by_meter: tuple[SmrRunMeterCost, ...] = ()

    @classmethod
    def from_wire(cls, payload: dict[str, Any]) -> SmrRunCostSummary:
        return cls(
            run_id=_require_text(payload.get("run_id"), field_name="run_id"),
            total_cents=int(payload.get("total_cents") or 0),
            total_pico_usd=int(payload.get("total_pico_usd") or 0),
            total_usd=float(payload.get("total_usd") or 0.0),
            recording_status=str(payload.get("recording_status") or "complete"),
            missing_meters=tuple(str(item) for item in (payload.get("missing_meters") or [])),
            by_meter=tuple(
                SmrRunMeterCost.from_wire(item)
                for item in (payload.get("by_meter") or [])
                if isinstance(item, dict)
            ),
        )


@dataclass(frozen=True, slots=True)
class SmrActorUsageSummary:
    actor_id: str
    org_id: str
    project_id: str
    run_id: str
    task_id: str | None = None
    task_key: str | None = None
    worker_id: str | None = None
    participant_role: str | None = None
    nominal_amount_cents: int = 0
    billed_amount_cents: int = 0
    internal_cost_cents: int = 0
    event_count: int = 0
    latest_usage_at: datetime | None = None
    by_provider: dict[str, int] = field(default_factory=dict)
    by_source_type: dict[str, int] = field(default_factory=dict)
    by_source_subtype: dict[str, int] = field(default_factory=dict)
    by_model: dict[str, int] = field(default_factory=dict)
    event_count_by_model: dict[str, int] = field(default_factory=dict)
    token_usage: dict[str, int] = field(default_factory=dict)
    token_usage_by_model: dict[str, dict[str, int]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "actor_id", _require_text(self.actor_id, field_name="actor_id"))
        object.__setattr__(self, "org_id", _require_text(self.org_id, field_name="org_id"))
        object.__setattr__(
            self, "project_id", _require_text(self.project_id, field_name="project_id")
        )
        object.__setattr__(self, "run_id", _require_text(self.run_id, field_name="run_id"))
        object.__setattr__(self, "task_id", _optional_text(self.task_id))
        object.__setattr__(self, "task_key", _optional_text(self.task_key))
        object.__setattr__(self, "worker_id", _optional_text(self.worker_id))
        object.__setattr__(self, "participant_role", _optional_text(self.participant_role))
        object.__setattr__(self, "nominal_amount_cents", int(self.nominal_amount_cents or 0))
        object.__setattr__(self, "billed_amount_cents", int(self.billed_amount_cents or 0))
        object.__setattr__(self, "internal_cost_cents", int(self.internal_cost_cents or 0))
        object.__setattr__(self, "event_count", max(0, int(self.event_count or 0)))
        object.__setattr__(
            self,
            "by_provider",
            {
                str(k): int(v or 0)
                for k, v in _coerce_dict(self.by_provider, field_name="by_provider").items()
            },
        )
        object.__setattr__(
            self,
            "by_source_type",
            {
                str(k): int(v or 0)
                for k, v in _coerce_dict(self.by_source_type, field_name="by_source_type").items()
            },
        )
        object.__setattr__(
            self,
            "by_source_subtype",
            {
                str(k): int(v or 0)
                for k, v in _coerce_dict(
                    self.by_source_subtype, field_name="by_source_subtype"
                ).items()
            },
        )
        object.__setattr__(
            self,
            "by_model",
            _coerce_int_dict(self.by_model, field_name="by_model"),
        )
        object.__setattr__(
            self,
            "event_count_by_model",
            _coerce_int_dict(
                self.event_count_by_model,
                field_name="event_count_by_model",
            ),
        )
        object.__setattr__(
            self,
            "token_usage",
            _coerce_int_dict(self.token_usage, field_name="token_usage"),
        )
        object.__setattr__(
            self,
            "token_usage_by_model",
            _coerce_nested_int_dict(
                self.token_usage_by_model,
                field_name="token_usage_by_model",
            ),
        )

    @classmethod
    def from_wire(cls, payload: dict[str, Any]) -> SmrActorUsageSummary:
        latest_usage_raw = payload.get("latest_usage_at")
        return cls(
            actor_id=_require_text(payload.get("actor_id"), field_name="actor_id"),
            org_id=_require_text(payload.get("org_id"), field_name="org_id"),
            project_id=_require_text(payload.get("project_id"), field_name="project_id"),
            run_id=_require_text(payload.get("run_id"), field_name="run_id"),
            task_id=_optional_text(payload.get("task_id")),
            task_key=_optional_text(payload.get("task_key")),
            worker_id=_optional_text(payload.get("worker_id")),
            participant_role=_optional_text(payload.get("participant_role")),
            nominal_amount_cents=int(payload.get("nominal_amount_cents") or 0),
            billed_amount_cents=int(payload.get("billed_amount_cents") or 0),
            internal_cost_cents=int(payload.get("internal_cost_cents") or 0),
            event_count=int(payload.get("event_count") or 0),
            latest_usage_at=(
                _parse_datetime(latest_usage_raw, field_name="latest_usage_at")
                if latest_usage_raw is not None
                else None
            ),
            by_provider=_coerce_dict(payload.get("by_provider"), field_name="by_provider"),
            by_source_type=_coerce_dict(payload.get("by_source_type"), field_name="by_source_type"),
            by_source_subtype=_coerce_dict(
                payload.get("by_source_subtype"), field_name="by_source_subtype"
            ),
            by_model=_coerce_dict(payload.get("by_model"), field_name="by_model"),
            event_count_by_model=_coerce_dict(
                payload.get("event_count_by_model"),
                field_name="event_count_by_model",
            ),
            token_usage=_coerce_dict(payload.get("token_usage"), field_name="token_usage"),
            token_usage_by_model=_coerce_dict(
                payload.get("token_usage_by_model"),
                field_name="token_usage_by_model",
            ),
        )


@dataclass(frozen=True, slots=True)
class SmrRunActorUsage:
    org_id: str
    project_id: str
    run_id: str
    count: int
    actors: tuple[SmrActorUsageSummary, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "org_id", _require_text(self.org_id, field_name="org_id"))
        object.__setattr__(
            self, "project_id", _require_text(self.project_id, field_name="project_id")
        )
        object.__setattr__(self, "run_id", _require_text(self.run_id, field_name="run_id"))
        object.__setattr__(self, "count", max(0, int(self.count)))

    @classmethod
    def from_wire(cls, payload: dict[str, Any]) -> SmrRunActorUsage:
        actors = tuple(
            SmrActorUsageSummary.from_wire(item)
            for item in (payload.get("actors") or [])
            if isinstance(item, dict)
        )
        return cls(
            org_id=_require_text(payload.get("org_id"), field_name="org_id"),
            project_id=_require_text(payload.get("project_id"), field_name="project_id"),
            run_id=_require_text(payload.get("run_id"), field_name="run_id"),
            count=int(payload.get("count") or len(actors)),
            actors=actors,
        )


__all__ = [
    "SmrActorUsageSummary",
    "SmrRunActorLogEvent",
    "SmrRunActorLogs",
    "SmrRunActorUsage",
    "SmrRunArtifactProgress",
    "SmrRunCostSummary",
    "SmrRunMeterCost",
    "SmrRunParticipant",
    "SmrRunParticipants",
    "SmrRunTraceItem",
    "SmrRunTraces",
]
