"""Typed runtime event stream models."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum

from synth_ai.core.research._legacy.transport.streaming import SseEvent


class RunRuntimeStreamEventKind(StrEnum):
    SNAPSHOT = "snapshot"
    MESSAGE = "message"
    DELIVERY = "delivery"
    TIMELINE = "timeline"
    WORKER = "worker"
    HEARTBEAT = "heartbeat"
    TRANSCRIPT = "transcript"


class TranscriptEventKind(StrEnum):
    OPERATOR_MESSAGE_SENT = "operator.message.sent"
    MESSAGE_DELTA = "message.delta"
    MESSAGE_COMPLETED = "message.completed"
    REASONING_SUMMARY = "reasoning.summary"
    TURN_STARTED = "turn.started"
    TURN_COMPLETED = "turn.completed"
    TURN_INTERRUPTED = "turn.interrupted"
    TURN_FAILED = "turn.failed"
    TOOL_CALL_STARTED = "tool.call.started"
    TOOL_CALL_COMPLETED = "tool.call.completed"
    TOOL_CALL_FAILED = "tool.call.failed"
    TOKEN_USAGE = "token.usage"


def _mapping(payload: object, *, label: str) -> Mapping[str, object]:
    if not isinstance(payload, Mapping):
        raise ValueError(f"{label} must be an object")
    return payload


def _optional_text(payload: Mapping[str, object], key: str) -> str | None:
    value = payload.get(key)
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError(f"{key} must be a string when provided")
    text = value.strip()
    return text or None


def _required_text(payload: Mapping[str, object], key: str, *, label: str) -> str:
    value = _optional_text(payload, key)
    if value is None:
        raise ValueError(f"{label} is required")
    return value


def _optional_datetime(payload: Mapping[str, object], key: str) -> datetime | None:
    value = payload.get(key)
    if value is None:
        return None
    if isinstance(value, datetime):
        return value
    if not isinstance(value, str):
        raise ValueError(f"{key} must be an ISO-8601 datetime when provided")
    text = value.strip()
    if not text:
        return None
    return datetime.fromisoformat(text.replace("Z", "+00:00"))


def _object_payload(payload: object) -> dict[str, object]:
    if payload is None:
        return {}
    return dict(_mapping(payload, label="payload"))


@dataclass(frozen=True, slots=True)
class TranscriptEvent:
    participant_session_id: str
    kind: str
    payload: dict[str, object] = field(default_factory=dict)
    event_id: str | None = None
    live_cursor: str | None = None
    run_id: str | None = None
    participant_role: str | None = None
    thread_id: str | None = None
    turn_id: str | None = None
    occurred_at: datetime | None = None
    redaction_profile: str | None = None
    visibility_decision: str | None = None
    payload_classification: str | None = None

    @classmethod
    def from_wire(cls, payload: object) -> TranscriptEvent:
        mapping = _mapping(payload, label="transcript event")
        return cls(
            event_id=_optional_text(mapping, "event_id"),
            live_cursor=_optional_text(mapping, "live_cursor"),
            run_id=_optional_text(mapping, "run_id"),
            participant_session_id=_required_text(
                mapping,
                "participant_session_id",
                label="participant_session_id",
            ),
            participant_role=_optional_text(mapping, "participant_role"),
            thread_id=_optional_text(mapping, "thread_id"),
            turn_id=_optional_text(mapping, "turn_id"),
            occurred_at=_optional_datetime(mapping, "occurred_at"),
            kind=_required_text(mapping, "kind", label="kind"),
            payload=_object_payload(mapping.get("payload")),
            redaction_profile=_optional_text(mapping, "redaction_profile"),
            visibility_decision=_optional_text(mapping, "visibility_decision"),
            payload_classification=_optional_text(mapping, "payload_classification"),
        )


@dataclass(frozen=True, slots=True)
class RunRuntimeStreamEvent:
    event: str
    event_id: str | None
    sequence: str | None
    run_id: str | None
    occurred_at: datetime | None
    kind: str
    payload: object
    state_version: str | None = None
    transcript_cursor: str | None = None

    @classmethod
    def from_sse(cls, event: SseEvent) -> RunRuntimeStreamEvent:
        data = _mapping(event.json_data(), label="SSE data")
        kind = _required_text(data, "kind", label="kind")
        return cls(
            event=event.event,
            event_id=event.event_id,
            sequence=_optional_text(data, "sequence"),
            run_id=_optional_text(data, "run_id"),
            occurred_at=_optional_datetime(data, "ts"),
            kind=kind,
            payload=data.get("payload"),
            state_version=_optional_text(data, "state_version"),
            transcript_cursor=_optional_text(data, "transcript_cursor"),
        )

    @property
    def transcript_event(self) -> TranscriptEvent | None:
        if self.kind != RunRuntimeStreamEventKind.TRANSCRIPT.value:
            return None
        return TranscriptEvent.from_wire(self.payload)


__all__ = [
    "RunRuntimeStreamEvent",
    "RunRuntimeStreamEventKind",
    "TranscriptEvent",
    "TranscriptEventKind",
]
