"""Catalog definitions, cursor codec, and filter matching for MCP Events (SEP-3415)."""

from __future__ import annotations

import base64
import json
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

if TYPE_CHECKING:
    from pydantic import JsonValue

    from synth_ai._generated.models import RunEvent, RunMilestone, RunSnapshot

__all__ = [
    "CREDENTIAL_REVOKED",
    "DEFAULT_POLL_LIMIT",
    "DEFAULT_STREAM_TIMEOUT_S",
    "EVENT_CATALOG",
    "EVENT_SWARMS_RESEARCH_UPDATED",
    "EVENT_SWARMS_RUN_APPROVAL_REQUESTED",
    "EVENT_SWARMS_RUN_LIMIT_ALERT",
    "EVENT_SWARMS_RUN_MESSAGE_POSTED",
    "EVENT_SWARMS_RUN_MILESTONE",
    "EVENT_SWARMS_RUN_PROGRESS",
    "EXTENSION_CAPABILITIES",
    "EXTENSION_METHODS",
    "EXTENSION_URI",
    "INTERNAL_ERROR",
    "INVALID_PARAMS",
    "KNOWN_EVENT_NAMES",
    "MAX_POLL_LIMIT",
    "MAX_POLL_SCAN_PAGES",
    "MAX_STREAM_TIMEOUT_S",
    "MILESTONE_LEVELS",
    "MILESTONE_ORDER",
    "POLL_INTERVAL_IDLE_MS",
    "POLL_INTERVAL_MORE_MS",
    "POLL_INTERVAL_SCAN_MS",
    "RESOURCE_NOT_FOUND",
    "UPSTREAM_PAGE_LIMIT",
    "EventCursor",
    "EventError",
    "EventFilter",
    "catalog_for_run_event",
    "decode_cursor",
    "encode_cursor",
    "envelope_from_milestone",
    "envelope_from_run_event",
    "list_events_payload",
    "matches_catalog",
    "parse_input",
    "parse_int_param",
    "snapshot_milestone_level",
]

EXTENSION_URI = "io.modelcontextprotocol/events"
EXTENSION_CAPABILITIES: dict[str, dict[str, JsonValue]] = {
    EXTENSION_URI: {"poll": True, "push": True},
}
EXTENSION_METHODS: frozenset[str] = frozenset({"events/list", "events/poll", "events/stream"})

INVALID_PARAMS = -32602
INTERNAL_ERROR = -32603
RESOURCE_NOT_FOUND = -32002
CREDENTIAL_REVOKED = -32001

EVENT_SWARMS_RUN_MILESTONE = "swarms.run.milestone"
EVENT_SWARMS_RUN_PROGRESS = "swarms.run.progress"
EVENT_SWARMS_RUN_APPROVAL_REQUESTED = "swarms.run.approval_requested"
EVENT_SWARMS_RUN_MESSAGE_POSTED = "swarms.run.message_posted"
EVENT_SWARMS_RUN_LIMIT_ALERT = "swarms.run.limit_alert"
EVENT_SWARMS_RESEARCH_UPDATED = "swarms.research.updated"

CURSOR_PREFIX = "mcpevt_v1."
MAX_CURSOR_LEN = 1024
DEFAULT_POLL_LIMIT = 50
MAX_POLL_LIMIT = 100
UPSTREAM_PAGE_LIMIT = 100
MAX_POLL_SCAN_PAGES = 5
POLL_INTERVAL_IDLE_MS = 2000
POLL_INTERVAL_MORE_MS = 250
POLL_INTERVAL_SCAN_MS = 500
DEFAULT_STREAM_TIMEOUT_S = 30
MAX_STREAM_TIMEOUT_S = 300

RUN_ID_PATTERN = re.compile(r"^run_[A-Za-z0-9_-]{22}$")
IDENTITY_OVERRIDE_KEYS = frozenset(
    {
        "org_id",
        "organization_id",
        "tenant_id",
        "user_id",
        "principal_id",
        "account_id",
        "credential_id",
    }
)

MILESTONE_ORDER: tuple[str, ...] = ("execution_closed", "accounting_final", "cleanup_complete")
MILESTONE_LEVELS: dict[str, int] = {name: idx + 1 for idx, name in enumerate(MILESTONE_ORDER)}

PROGRESS_KINDS: tuple[str, ...] = ("objective.progress", "objective.claimed")
APPROVAL_KINDS: tuple[str, ...] = ("approval.requested", "approval.decided")
MESSAGE_KINDS: tuple[str, ...] = (
    "run.message.posted",
    "run.message.accepted",
    "run.message.delivered",
    "run.message.disposed",
    "run.message.expired",
)
LIMIT_ALERT_KINDS: tuple[str, ...] = (
    "limit.warning",
    "limit.reached",
    "limits.revised",
    "run.paused",
    "run.resumed",
)
RESEARCH_KINDS: tuple[str, ...] = (
    "research.artifact_published",
    "research.experiment_created",
    "research.experiment_revised",
    "research.trial_logged",
    "research.hypothesis_created",
    "research.correction_recorded",
    "research.log_appended",
)


class EventError(ValueError):
    """Fail-closed JSON-RPC error for SEP-3415 event operations."""

    def __init__(
        self,
        code: int,
        message: str,
        *,
        data: dict[str, JsonValue] | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.data = data


@dataclass(frozen=True, slots=True)
class EventCursor:
    event: str
    run_id: str
    upstream: str = ""
    skip: int = 0
    milestone_level: int = 0
    stream_resume_id: str = ""


@dataclass(frozen=True, slots=True)
class EventFilter:
    run_id: str
    milestone: str | None = None
    kinds: frozenset[str] = frozenset()
    include_decisions: bool = False


def _run_id_schema() -> dict[str, JsonValue]:
    return {
        "type": "string",
        "pattern": r"^run_[A-Za-z0-9_-]{22}$",
        "description": "Cloud Swarms run identifier.",
    }


def _kinds_filter_schema(allowed: tuple[str, ...], description: str) -> dict[str, JsonValue]:
    enum_values: list[JsonValue] = list(allowed)
    return {
        "type": "array",
        "description": description,
        "uniqueItems": True,
        "items": {"type": "string", "enum": enum_values},
    }


def _run_event_payload_schema(
    allowed_kinds: tuple[str, ...], *, include_approval_id: bool = False
) -> dict[str, JsonValue]:
    kind_enum: list[JsonValue] = list(allowed_kinds)
    props: dict[str, JsonValue] = {
        "run_id": {"type": "string"},
        "sequence": {"type": "integer", "minimum": 0},
        "kind": {"type": "string", "enum": kind_enum},
        "observed_at": {"type": ["string", "null"]},
        "payload": {"type": ["object", "null"]},
    }
    required: list[JsonValue] = ["run_id", "sequence", "kind", "payload"]
    if include_approval_id:
        props["approval_id"] = {"type": ["string", "null"]}
        required.append("approval_id")
    return {
        "type": "object",
        "additionalProperties": False,
        "required": required,
        "properties": props,
    }


EVENT_CATALOG: tuple[dict[str, JsonValue], ...] = (
    {
        "name": EVENT_SWARMS_RUN_MILESTONE,
        "description": (
            "Lifecycle closure milestones for a Cloud Swarms run "
            "(execution_closed, accounting_final, cleanup_complete)."
        ),
        "deliveryModes": ["poll", "push"],
        "inputSchema": {
            "type": "object",
            "additionalProperties": False,
            "required": ["run_id"],
            "properties": {
                "run_id": _run_id_schema(),
                "milestone": {
                    "type": "string",
                    "enum": list(MILESTONE_ORDER),
                    "description": "Optional filter for a specific closure milestone.",
                },
            },
        },
        "payloadSchema": {
            "type": "object",
            "additionalProperties": False,
            "required": [
                "milestone",
                "run_id",
                "state",
                "execution_closed",
                "accounting_final",
                "cleanup_complete",
                "result_available",
                "final",
            ],
            "properties": {
                "milestone": {"type": "string", "enum": list(MILESTONE_ORDER)},
                "run_id": {"type": "string"},
                "state": {"type": "string"},
                "execution_closed": {"type": "boolean"},
                "accounting_final": {"type": "boolean"},
                "cleanup_complete": {"type": "boolean"},
                "result_available": {"type": "boolean"},
                "failure": {"type": ["object", "null"]},
                "final": {"type": "boolean"},
            },
        },
    },
    {
        "name": EVENT_SWARMS_RUN_PROGRESS,
        "description": "Objective progress and claim updates emitted while a Cloud Swarms run executes.",
        "deliveryModes": ["poll", "push"],
        "inputSchema": {
            "type": "object",
            "additionalProperties": False,
            "required": ["run_id"],
            "properties": {
                "run_id": _run_id_schema(),
                "kinds": _kinds_filter_schema(
                    PROGRESS_KINDS, "Optional subset of progress event kinds."
                ),
            },
        },
        "payloadSchema": _run_event_payload_schema(PROGRESS_KINDS),
    },
    {
        "name": EVENT_SWARMS_RUN_APPROVAL_REQUESTED,
        "description": "Human-in-the-loop approval requests (and optional decisions) for a Cloud Swarms run.",
        "deliveryModes": ["poll", "push"],
        "inputSchema": {
            "type": "object",
            "additionalProperties": False,
            "required": ["run_id"],
            "properties": {
                "run_id": _run_id_schema(),
                "include_decisions": {
                    "type": "boolean",
                    "description": "When true, include approval.decided events alongside approval.requested.",
                },
            },
        },
        "payloadSchema": _run_event_payload_schema(APPROVAL_KINDS, include_approval_id=True),
    },
    {
        "name": EVENT_SWARMS_RUN_MESSAGE_POSTED,
        "description": "Run thread message lifecycle events (posted, accepted, delivered, disposed, expired).",
        "deliveryModes": ["poll", "push"],
        "inputSchema": {
            "type": "object",
            "additionalProperties": False,
            "required": ["run_id"],
            "properties": {
                "run_id": _run_id_schema(),
                "kinds": _kinds_filter_schema(
                    MESSAGE_KINDS, "Optional subset of run message event kinds."
                ),
            },
        },
        "payloadSchema": _run_event_payload_schema(MESSAGE_KINDS),
    },
    {
        "name": EVENT_SWARMS_RUN_LIMIT_ALERT,
        "description": "Run budget, wall-time, or token limit alerts, revisions, and pause/resume transitions.",
        "deliveryModes": ["poll", "push"],
        "inputSchema": {
            "type": "object",
            "additionalProperties": False,
            "required": ["run_id"],
            "properties": {
                "run_id": _run_id_schema(),
                "kinds": _kinds_filter_schema(
                    LIMIT_ALERT_KINDS, "Optional subset of limit/pause event kinds."
                ),
            },
        },
        "payloadSchema": _run_event_payload_schema(LIMIT_ALERT_KINDS),
    },
    {
        "name": EVENT_SWARMS_RESEARCH_UPDATED,
        "description": (
            "Track B research mutations bound to a run "
            "(artifacts, experiments, trials, hypotheses, corrections, log entries)."
        ),
        "deliveryModes": ["poll", "push"],
        "inputSchema": {
            "type": "object",
            "additionalProperties": False,
            "required": ["run_id"],
            "properties": {
                "run_id": _run_id_schema(),
                "kinds": _kinds_filter_schema(
                    RESEARCH_KINDS, "Optional subset of research event kinds."
                ),
            },
        },
        "payloadSchema": _run_event_payload_schema(RESEARCH_KINDS),
    },
)

KNOWN_EVENT_NAMES: frozenset[str] = frozenset(str(entry["name"]) for entry in EVENT_CATALOG)


def list_events_payload() -> dict[str, JsonValue]:
    items: list[JsonValue] = list(EVENT_CATALOG)
    return {"events": items}


def encode_cursor(cursor: EventCursor) -> str:
    payload: dict[str, JsonValue] = {"v": 1, "e": cursor.event, "r": cursor.run_id}
    if cursor.upstream:
        payload["u"] = cursor.upstream
    if cursor.skip > 0:
        payload["s"] = cursor.skip
    if cursor.milestone_level > 0:
        payload["m"] = cursor.milestone_level
    if cursor.stream_resume_id:
        payload["id"] = cursor.stream_resume_id
    raw = json.dumps(payload, separators=(",", ":")).encode("utf-8")
    return CURSOR_PREFIX + base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")


def decode_cursor(raw: str, *, expected_event: str, expected_run_id: str) -> EventCursor:
    if not raw:
        return EventCursor(event=expected_event, run_id=expected_run_id)
    if len(raw) > MAX_CURSOR_LEN or not raw.startswith(CURSOR_PREFIX):
        raise EventError(INVALID_PARAMS, "Invalid params: malformed event cursor.")
    encoded = raw[len(CURSOR_PREFIX) :]
    padding = "=" * ((4 - len(encoded) % 4) % 4)
    try:
        decoded = base64.urlsafe_b64decode((encoded + padding).encode("ascii"))
        obj = json.loads(decoded.decode("utf-8"))
    except (ValueError, UnicodeDecodeError) as error:
        raise EventError(INVALID_PARAMS, "Invalid params: malformed event cursor.") from error
    if not isinstance(obj, dict) or set(obj) - {"v", "e", "r", "u", "s", "m", "id"}:
        raise EventError(INVALID_PARAMS, "Invalid params: malformed event cursor.")
    version = obj.get("v")
    event = obj.get("e", "")
    run_id = obj.get("r")
    upstream = obj.get("u", "")
    skip = obj.get("s", 0)
    milestone_level = obj.get("m", 0)
    stream_resume_id = obj.get("id", "")
    if (
        type(version) is not int
        or version != 1
        or not isinstance(event, str)
        or not isinstance(run_id, str)
        or not isinstance(upstream, str)
        or type(skip) is not int
        or skip < 0
        or type(milestone_level) is not int
        or not (0 <= milestone_level <= len(MILESTONE_ORDER))
        or not isinstance(stream_resume_id, str)
    ):
        raise EventError(INVALID_PARAMS, "Invalid params: malformed event cursor.")
    if run_id != expected_run_id:
        raise EventError(INVALID_PARAMS, "Invalid params: event cursor does not match run_id.")
    if expected_event and event and event != expected_event:
        raise EventError(
            INVALID_PARAMS, "Invalid params: event cursor belongs to a different event type."
        )
    return EventCursor(
        event=event,
        run_id=run_id,
        upstream=upstream,
        skip=skip,
        milestone_level=milestone_level,
        stream_resume_id=stream_resume_id,
    )


def _find_identity_overrides(value: JsonValue, depth: int) -> list[str]:
    if depth <= 0:
        return []
    found: list[str] = []
    if isinstance(value, dict):
        for key, item in value.items():
            if key in IDENTITY_OVERRIDE_KEYS:
                found.append(key)
            found.extend(_find_identity_overrides(item, depth - 1))
    elif isinstance(value, list):
        for item in value:
            found.extend(_find_identity_overrides(item, depth - 1))
    return sorted(set(found))


def parse_int_param(
    raw: JsonValue | None, *, field: str, default: int, minimum: int, maximum: int
) -> int:
    if raw is None:
        return default
    if type(raw) is not int:
        raise EventError(INVALID_PARAMS, f"Invalid params: {field} must be an integer.")
    if raw < minimum or raw > maximum:
        raise EventError(
            INVALID_PARAMS, f"Invalid params: {field} must be between {minimum} and {maximum}."
        )
    return raw


def parse_input(events: tuple[str, ...], raw_input: JsonValue | None) -> EventFilter:
    if not isinstance(raw_input, dict):
        raise EventError(INVALID_PARAMS, "Invalid params: input object is required.")
    overrides = _find_identity_overrides(raw_input, 4)
    if overrides:
        override_list: list[JsonValue] = list(overrides)
        raise EventError(
            INVALID_PARAMS,
            "Caller-supplied identity fields are not accepted.",
            data={"fields": override_list},
        )
    raw_run_id = raw_input.get("run_id")
    if not isinstance(raw_run_id, str) or not RUN_ID_PATTERN.fullmatch(raw_run_id.strip()):
        raise EventError(
            INVALID_PARAMS, "Invalid params: input.run_id must be a valid run identifier."
        )
    run_id = raw_run_id.strip()

    allow_milestone = EVENT_SWARMS_RUN_MILESTONE in events
    allow_approval = EVENT_SWARMS_RUN_APPROVAL_REQUESTED in events
    allowed_kinds: set[str] = set()
    for ev in events:
        if ev == EVENT_SWARMS_RUN_PROGRESS:
            allowed_kinds.update(PROGRESS_KINDS)
        elif ev == EVENT_SWARMS_RUN_MESSAGE_POSTED:
            allowed_kinds.update(MESSAGE_KINDS)
        elif ev == EVENT_SWARMS_RUN_LIMIT_ALERT:
            allowed_kinds.update(LIMIT_ALERT_KINDS)
        elif ev == EVENT_SWARMS_RESEARCH_UPDATED:
            allowed_kinds.update(RESEARCH_KINDS)

    milestone: str | None = None
    include_decisions = False
    kinds: frozenset[str] = frozenset()

    for key, val in raw_input.items():
        if key == "run_id":
            continue
        if key == "milestone":
            if not allow_milestone:
                raise EventError(
                    INVALID_PARAMS,
                    "Invalid params: input.milestone is only valid for swarms.run.milestone.",
                )
            if not isinstance(val, str) or val not in MILESTONE_LEVELS:
                raise EventError(INVALID_PARAMS, "Invalid params: unknown milestone filter.")
            milestone = val
        elif key == "include_decisions":
            if not allow_approval:
                raise EventError(
                    INVALID_PARAMS,
                    "Invalid params: input.include_decisions is only valid for swarms.run.approval_requested.",
                )
            if type(val) is not bool:
                raise EventError(
                    INVALID_PARAMS, "Invalid params: input.include_decisions must be a boolean."
                )
            include_decisions = val
        elif key == "kinds":
            if not allowed_kinds:
                raise EventError(
                    INVALID_PARAMS, "Invalid params: input.kinds is not valid for this event."
                )
            if not isinstance(val, list):
                raise EventError(
                    INVALID_PARAMS, "Invalid params: input.kinds must be an array of strings."
                )
            parsed_kinds: set[str] = set()
            for item in val:
                if not isinstance(item, str) or item not in allowed_kinds:
                    raise EventError(INVALID_PARAMS, "Invalid params: disallowed kind in kinds.")
                parsed_kinds.add(item)
            kinds = frozenset(parsed_kinds)
        else:
            raise EventError(
                INVALID_PARAMS, "Invalid params: unknown input field.", data={"field": key}
            )
    return EventFilter(
        run_id=run_id,
        milestone=milestone,
        kinds=kinds,
        include_decisions=include_decisions,
    )


def matches_catalog(event_name: str, kind: str, filt: EventFilter) -> bool:
    if filt.kinds and kind not in filt.kinds:
        return False
    if event_name == EVENT_SWARMS_RUN_PROGRESS:
        return kind in PROGRESS_KINDS
    if event_name == EVENT_SWARMS_RUN_APPROVAL_REQUESTED:
        return kind == "approval.requested" or (
            filt.include_decisions and kind == "approval.decided"
        )
    if event_name == EVENT_SWARMS_RUN_MESSAGE_POSTED:
        return kind in MESSAGE_KINDS
    if event_name == EVENT_SWARMS_RUN_LIMIT_ALERT:
        return kind in LIMIT_ALERT_KINDS
    if event_name == EVENT_SWARMS_RESEARCH_UPDATED:
        return kind in RESEARCH_KINDS
    return False


def catalog_for_run_event(requested: tuple[str, ...], kind: str, filt: EventFilter) -> str | None:
    for event_name in requested:
        if event_name == EVENT_SWARMS_RUN_MILESTONE:
            continue
        if matches_catalog(event_name, kind, filt):
            return event_name
    return None


def envelope_from_run_event(event_name: str, item: RunEvent, now_iso: str) -> dict[str, JsonValue]:
    raw_payload: JsonValue = (
        cast("JsonValue", item.payload.model_dump(mode="json"))
        if item.payload is not None
        else None
    )
    payload: dict[str, JsonValue] = {
        "run_id": item.run_id,
        "sequence": item.sequence,
        "kind": item.kind,
        "observed_at": item.observed_at,
        "payload": raw_payload,
    }
    if event_name == EVENT_SWARMS_RUN_APPROVAL_REQUESTED:
        payload["approval_id"] = item.approval_id
    return {
        "eventId": item.event_id,
        "event": event_name,
        "timestamp": item.observed_at or now_iso,
        "payload": payload,
    }


def envelope_from_milestone(item: RunMilestone, now_iso: str) -> dict[str, JsonValue]:
    ms_name = str(item.milestone)
    return {
        "eventId": f"ms:{item.run_id}:{ms_name}",
        "event": EVENT_SWARMS_RUN_MILESTONE,
        "timestamp": now_iso,
        "payload": {
            "milestone": ms_name,
            "run_id": item.run_id,
            "state": item.state,
            "execution_closed": item.execution_closed,
            "accounting_final": item.accounting_final,
            "cleanup_complete": item.cleanup_complete,
            "result_available": item.result_available,
            "failure": item.failure,
            "final": item.final,
        },
    }


def snapshot_milestone_level(snapshot: RunSnapshot) -> int:
    if not snapshot.execution_closed:
        return 0
    if not snapshot.accounting_final:
        return 1
    if not snapshot.cleanup_complete:
        return 2
    return 3
