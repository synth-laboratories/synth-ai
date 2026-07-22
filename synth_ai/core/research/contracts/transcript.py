"""Typed transcript pages for Research swarms.

# See: specifications/sdk/core_research_migration.md
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from typing import TypeVar

from synth_ai.core.contracts.json_value import JsonObject, JsonValue
from synth_ai.core.research.contracts._transcript_wire import (
    exact_object,
    object_field,
    optional_datetime,
    optional_string,
    required_bool,
    required_datetime,
    required_string,
)
from synth_ai.core.research.contracts.common import (
    ParticipantSessionId,
    SwarmId,
    TranscriptEventId,
)


class TranscriptView(StrEnum):
    """Backend-owned transcript visibility policy."""

    PUBLIC = "public"
    OPERATOR = "operator"
    DEBUG = "debug"


class TranscriptProjectionAuthority(StrEnum):
    """Authority that produced a transcript projection."""

    LIVE_CONTROL_PLANE = "smr_control_plane.redis.live_transcript.v1"
    TERMINAL_ARCHIVE = "smr_transcript_events.terminal_archive.v1"


class TranscriptReplayMode(StrEnum):
    LIVE_TAIL = "live_tail"
    TERMINAL_ARCHIVE = "terminal_archive"


class TranscriptCursorKind(StrEnum):
    REDIS_STREAM_ID = "redis_stream_id"
    DURABLE_TRANSCRIPT_PAGE = "durable_transcript_page"


class TranscriptRedactionProfile(StrEnum):
    OPEN_RESEARCH_PUBLIC = "open_research_public"
    OPERATOR_DEFAULT = "operator_default"
    DEBUG_OPERATOR = "debug_operator"


class TranscriptVisibilityDecision(StrEnum):
    STREAM_REDACTED = "stream_redacted"
    STREAM_SUMMARY_ONLY = "stream_summary_only"


class TranscriptPayloadClassification(StrEnum):
    PUBLIC_SAFE = "public_safe"
    PRIVATE = "private"
    INTERNAL = "internal"
    SECRET = "secret"
    HIDDEN_REASONING = "hidden_reasoning"


_TranscriptEnum = TypeVar("_TranscriptEnum", bound=StrEnum)


def _optional_enum(
    payload: JsonObject,
    name: str,
    enum_type: type[_TranscriptEnum],
) -> _TranscriptEnum | None:
    value = optional_string(payload, name)
    return enum_type(value) if value is not None else None


@dataclass(frozen=True, slots=True)
class SwarmTranscriptEvent:
    """One viewer-safe event from the versioned transcript projection."""

    event_id: TranscriptEventId | None
    live_cursor: str | None
    swarm_id: SwarmId
    participant_session_id: ParticipantSessionId
    participant_role: str | None
    thread_id: str | None
    turn_id: str | None
    occurred_at: datetime
    kind: str
    payload: JsonObject
    run_terminal: bool
    redaction_profile: TranscriptRedactionProfile | None
    visibility_decision: TranscriptVisibilityDecision | None
    payload_classification: TranscriptPayloadClassification | None

    @classmethod
    def from_wire(cls, value: JsonValue) -> SwarmTranscriptEvent:
        payload = exact_object(
            value,
            label="swarm transcript event",
            fields=frozenset(
                {
                    "schema_version",
                    "event_id",
                    "live_cursor",
                    "run_id",
                    "participant_session_id",
                    "participant_role",
                    "thread_id",
                    "turn_id",
                    "occurred_at",
                    "kind",
                    "payload",
                    "semantic_scope",
                    "run_terminal",
                    "redaction_profile",
                    "visibility_decision",
                    "payload_classification",
                }
            ),
        )
        if required_string(payload, "schema_version") != (
            "synth.research.transcript-event.v1"
        ):
            raise ValueError("unsupported transcript event schema_version")
        if required_string(payload, "semantic_scope") != "participant":
            raise ValueError("transcript event semantic_scope must be 'participant'")
        event_id = optional_string(payload, "event_id")
        return cls(
            event_id=TranscriptEventId(event_id) if event_id is not None else None,
            live_cursor=optional_string(payload, "live_cursor"),
            swarm_id=SwarmId(required_string(payload, "run_id")),
            participant_session_id=ParticipantSessionId(
                required_string(payload, "participant_session_id")
            ),
            participant_role=optional_string(payload, "participant_role"),
            thread_id=optional_string(payload, "thread_id"),
            turn_id=optional_string(payload, "turn_id"),
            occurred_at=required_datetime(payload, "occurred_at"),
            kind=required_string(payload, "kind"),
            payload=object_field(payload, "payload"),
            run_terminal=required_bool(payload, "run_terminal"),
            redaction_profile=_optional_enum(
                payload,
                "redaction_profile",
                TranscriptRedactionProfile,
            ),
            visibility_decision=_optional_enum(
                payload,
                "visibility_decision",
                TranscriptVisibilityDecision,
            ),
            payload_classification=_optional_enum(
                payload,
                "payload_classification",
                TranscriptPayloadClassification,
            ),
        )

    def to_wire(self) -> JsonObject:
        return {
            "schema_version": "synth.research.transcript-event.v1",
            "event_id": self.event_id,
            "live_cursor": self.live_cursor,
            "run_id": self.swarm_id,
            "participant_session_id": self.participant_session_id,
            "participant_role": self.participant_role,
            "thread_id": self.thread_id,
            "turn_id": self.turn_id,
            "occurred_at": self.occurred_at.isoformat(),
            "kind": self.kind,
            "payload": dict(self.payload),
            "semantic_scope": "participant",
            "run_terminal": self.run_terminal,
            "redaction_profile": (
                self.redaction_profile.value if self.redaction_profile else None
            ),
            "visibility_decision": (
                self.visibility_decision.value if self.visibility_decision else None
            ),
            "payload_classification": (
                self.payload_classification.value
                if self.payload_classification
                else None
            ),
        }


@dataclass(frozen=True, slots=True)
class TranscriptParticipant:
    participant_session_id: ParticipantSessionId
    participant_id: str | None
    participant_role: str | None
    subscriber_id: str | None
    session_status: str | None
    live_session: bool
    has_active_turn: bool
    thread_id: str | None
    turn_id: str | None
    last_heartbeat_at: datetime | None
    runtime_source: str | None
    worker_id: str | None

    @classmethod
    def from_wire(cls, value: JsonValue) -> TranscriptParticipant:
        payload = exact_object(
            value,
            label="transcript participant",
            fields=frozenset(
                {
                    "participant_session_id",
                    "participant_id",
                    "participant_role",
                    "subscriber_id",
                    "session_status",
                    "live_session",
                    "has_active_turn",
                    "thread_id",
                    "turn_id",
                    "last_heartbeat_at",
                    "runtime_source",
                    "worker_id",
                }
            ),
        )
        return cls(
            participant_session_id=ParticipantSessionId(
                required_string(payload, "participant_session_id")
            ),
            participant_id=optional_string(payload, "participant_id"),
            participant_role=optional_string(payload, "participant_role"),
            subscriber_id=optional_string(payload, "subscriber_id"),
            session_status=optional_string(payload, "session_status"),
            live_session=required_bool(payload, "live_session"),
            has_active_turn=required_bool(payload, "has_active_turn"),
            thread_id=optional_string(payload, "thread_id"),
            turn_id=optional_string(payload, "turn_id"),
            last_heartbeat_at=optional_datetime(payload, "last_heartbeat_at"),
            runtime_source=optional_string(payload, "runtime_source"),
            worker_id=optional_string(payload, "worker_id"),
        )

    def to_wire(self) -> JsonObject:
        return {
            "participant_session_id": self.participant_session_id,
            "participant_id": self.participant_id,
            "participant_role": self.participant_role,
            "subscriber_id": self.subscriber_id,
            "session_status": self.session_status,
            "live_session": self.live_session,
            "has_active_turn": self.has_active_turn,
            "thread_id": self.thread_id,
            "turn_id": self.turn_id,
            "last_heartbeat_at": (
                self.last_heartbeat_at.isoformat() if self.last_heartbeat_at else None
            ),
            "runtime_source": self.runtime_source,
            "worker_id": self.worker_id,
        }


@dataclass(frozen=True, slots=True)
class TranscriptFreshness:
    observed_at: datetime
    projection_authority: TranscriptProjectionAuthority
    replay_mode: TranscriptReplayMode
    live_tail_available: bool

    def __post_init__(self) -> None:
        if self.replay_mode is TranscriptReplayMode.LIVE_TAIL:
            if self.projection_authority is not (
                TranscriptProjectionAuthority.LIVE_CONTROL_PLANE
            ):
                raise ValueError("live_tail requires live control-plane authority")
            if not self.live_tail_available:
                raise ValueError("live_tail requires live_tail_available=true")
        elif self.live_tail_available:
            raise ValueError("terminal archive cannot advertise a live tail")

    @classmethod
    def from_wire(cls, value: JsonValue) -> TranscriptFreshness:
        payload = exact_object(
            value,
            label="transcript freshness",
            fields=frozenset(
                {
                    "observed_at",
                    "projection_authority",
                    "replay_mode",
                    "live_tail_available",
                }
            ),
        )
        return cls(
            observed_at=required_datetime(payload, "observed_at"),
            projection_authority=TranscriptProjectionAuthority(
                required_string(payload, "projection_authority")
            ),
            replay_mode=TranscriptReplayMode(
                required_string(payload, "replay_mode")
            ),
            live_tail_available=required_bool(payload, "live_tail_available"),
        )

    def to_wire(self) -> JsonObject:
        return {
            "observed_at": self.observed_at.isoformat(),
            "projection_authority": self.projection_authority.value,
            "replay_mode": self.replay_mode.value,
            "live_tail_available": self.live_tail_available,
        }


@dataclass(frozen=True, slots=True)
class TranscriptCursor:
    kind: TranscriptCursorKind
    requested_cursor: str | None
    next_cursor: str | None
    live_resume_cursor: str | None

    def __post_init__(self) -> None:
        if (
            self.kind is TranscriptCursorKind.DURABLE_TRANSCRIPT_PAGE
            and self.live_resume_cursor is not None
        ):
            raise ValueError("durable transcript cursor cannot resume a live stream")

    @classmethod
    def from_wire(cls, value: JsonValue) -> TranscriptCursor:
        payload = exact_object(
            value,
            label="transcript cursor",
            fields=frozenset(
                {
                    "kind",
                    "semantics",
                    "requested_cursor",
                    "next_cursor",
                    "live_resume_cursor",
                }
            ),
        )
        if required_string(payload, "semantics") != "exclusive":
            raise ValueError("transcript cursor semantics must be 'exclusive'")
        return cls(
            kind=TranscriptCursorKind(required_string(payload, "kind")),
            requested_cursor=optional_string(payload, "requested_cursor"),
            next_cursor=optional_string(payload, "next_cursor"),
            live_resume_cursor=optional_string(payload, "live_resume_cursor"),
        )

    def to_wire(self) -> JsonObject:
        return {
            "kind": self.kind.value,
            "semantics": "exclusive",
            "requested_cursor": self.requested_cursor,
            "next_cursor": self.next_cursor,
            "live_resume_cursor": self.live_resume_cursor,
        }


@dataclass(frozen=True, slots=True)
class SwarmTranscriptPage:
    swarm_id: SwarmId
    participant_session_id: ParticipantSessionId | None
    events: tuple[SwarmTranscriptEvent, ...]
    participants: tuple[TranscriptParticipant, ...]
    next_cursor: str | None
    live_resume_cursor: str | None
    projection_authority: TranscriptProjectionAuthority
    freshness: TranscriptFreshness
    cursor: TranscriptCursor

    def __post_init__(self) -> None:
        if self.projection_authority is not self.freshness.projection_authority:
            raise ValueError("page and freshness projection authorities must match")
        if self.next_cursor != self.cursor.next_cursor:
            raise ValueError("page and cursor next_cursor values must match")
        if self.live_resume_cursor != self.cursor.live_resume_cursor:
            raise ValueError("page and cursor live_resume_cursor values must match")
        if any(event.swarm_id != self.swarm_id for event in self.events):
            raise ValueError("transcript event run_id must match the page run_id")

    @classmethod
    def from_wire(cls, value: JsonValue) -> SwarmTranscriptPage:
        payload = exact_object(
            value,
            label="swarm transcript page",
            fields=frozenset(
                {
                    "schema_version",
                    "run_id",
                    "participant_session_id",
                    "events",
                    "participants",
                    "next_cursor",
                    "live_resume_cursor",
                    "projection_authority",
                    "freshness",
                    "cursor",
                }
            ),
        )
        if required_string(payload, "schema_version") != (
            "synth.research.transcript-page.v1"
        ):
            raise ValueError("unsupported transcript page schema_version")
        raw_events = payload.get("events")
        raw_participants = payload.get("participants")
        if not isinstance(raw_events, list):
            raise ValueError("events must be an array")
        if not isinstance(raw_participants, list):
            raise ValueError("participants must be an array")
        participant_session_id = optional_string(payload, "participant_session_id")
        return cls(
            swarm_id=SwarmId(required_string(payload, "run_id")),
            participant_session_id=(
                ParticipantSessionId(participant_session_id)
                if participant_session_id is not None
                else None
            ),
            events=tuple(SwarmTranscriptEvent.from_wire(item) for item in raw_events),
            participants=tuple(
                TranscriptParticipant.from_wire(item) for item in raw_participants
            ),
            next_cursor=optional_string(payload, "next_cursor"),
            live_resume_cursor=optional_string(payload, "live_resume_cursor"),
            projection_authority=TranscriptProjectionAuthority(
                required_string(payload, "projection_authority")
            ),
            freshness=TranscriptFreshness.from_wire(payload.get("freshness")),
            cursor=TranscriptCursor.from_wire(payload.get("cursor")),
        )

    def to_wire(self) -> JsonObject:
        return {
            "schema_version": "synth.research.transcript-page.v1",
            "run_id": self.swarm_id,
            "participant_session_id": self.participant_session_id,
            "events": [event.to_wire() for event in self.events],
            "participants": [participant.to_wire() for participant in self.participants],
            "next_cursor": self.next_cursor,
            "live_resume_cursor": self.live_resume_cursor,
            "projection_authority": self.projection_authority.value,
            "freshness": self.freshness.to_wire(),
            "cursor": self.cursor.to_wire(),
        }


__all__ = [
    "SwarmTranscriptEvent",
    "SwarmTranscriptPage",
    "TranscriptCursor",
    "TranscriptCursorKind",
    "TranscriptFreshness",
    "TranscriptParticipant",
    "TranscriptPayloadClassification",
    "TranscriptProjectionAuthority",
    "TranscriptRedactionProfile",
    "TranscriptReplayMode",
    "TranscriptView",
    "TranscriptVisibilityDecision",
]
