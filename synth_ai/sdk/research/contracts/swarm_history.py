"""Backend-authorized journal pages, preserving redacted wire values exactly.

# See: backend/packages/smr/contracts/public_api/v1/run_history.py
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from synth_ai.core.contracts.json_value import JsonObject, JsonValue
from synth_ai.sdk.research.contracts._wire import object_value
from synth_ai.sdk.research.contracts.common import SwarmId

_MAX_SEQUENCE = 2**64 - 1
_DIGEST = re.compile(r"[0-9a-f]{64}\Z")


def _closed(value: JsonValue, fields: set[str]) -> JsonObject:
    payload = object_value(value, operation_id="swarm journal history")
    if set(payload) != fields:
        raise ValueError("journal history fields drifted")
    return payload


def _integer(value: JsonValue, *, natural: bool = True) -> int:
    if type(value) is not int or (natural and not 0 <= value <= _MAX_SEQUENCE):
        raise ValueError("journal history integer is invalid")
    return value


def _text(value: JsonValue) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError("journal history text is invalid")
    return value


def history_parameters(after: int, limit: int) -> JsonObject:
    """Validate the backend's unsigned cursor and bounded page request."""
    _integer(after)
    if type(limit) is not int or not 1 <= limit <= 256:
        raise ValueError("journal history limit must be an integer from 1 to 256")
    return {"after": after, "limit": limit}


@dataclass(frozen=True, slots=True)
class SwarmHistoryDecision:
    sequence: int
    reason: str
    input_digest: str
    state_before_digest: str
    state_after_digest: str
    intents_digest: str
    events_digest: str
    wakes_digest: str

    @classmethod
    def from_wire(cls, value: JsonValue) -> SwarmHistoryDecision:
        digests = {
            "input_digest",
            "state_before_digest",
            "state_after_digest",
            "intents_digest",
            "events_digest",
            "wakes_digest",
        }
        payload = _closed(value, {"sequence", "reason"} | digests)
        values = {name: _text(payload[name]) for name in digests}
        if any(_DIGEST.fullmatch(digest) is None for digest in values.values()):
            raise ValueError("journal history digest is invalid")
        return cls(_integer(payload["sequence"]), _text(payload["reason"]), **values)


@dataclass(frozen=True, slots=True)
class SwarmHistoryEntry:
    sequence: int
    generation: int
    command_id: str
    causation_id: str
    source_id: str
    occurred_at_ms: int
    recorded_at_ms: int
    input: JsonObject
    decision: SwarmHistoryDecision

    @classmethod
    def from_wire(cls, value: JsonValue) -> SwarmHistoryEntry:
        payload = _closed(
            value,
            {
                "sequence",
                "generation",
                "command_id",
                "causation_id",
                "source_id",
                "occurred_at_ms",
                "recorded_at_ms",
                "input",
                "decision",
            },
        )
        return cls(
            _integer(payload["sequence"]),
            _integer(payload["generation"]),
            _text(payload["command_id"]),
            _text(payload["causation_id"]),
            _text(payload["source_id"]),
            _integer(payload["occurred_at_ms"], natural=False),
            _integer(payload["recorded_at_ms"], natural=False),
            object_value(payload["input"], operation_id="journal history input"),
            SwarmHistoryDecision.from_wire(payload["decision"]),
        )


@dataclass(frozen=True, slots=True)
class SwarmHistoryPage:
    swarm_id: SwarmId
    head_sequence: int
    after: int
    entries: tuple[SwarmHistoryEntry, ...]
    next_after: int | None

    @classmethod
    def from_wire(
        cls, value: JsonValue, *, swarm_id: SwarmId, after: int, limit: int
    ) -> SwarmHistoryPage:
        history_parameters(after, limit)
        payload = _closed(
            value, {"schema_version", "run_id", "head_sequence", "after", "entries", "next_after"}
        )
        if (
            payload["schema_version"] != "orchestra.journal_inspection.v1"
            or payload["run_id"] != swarm_id
        ):
            raise ValueError("journal history schema or run differs")
        head, cursor = _integer(payload["head_sequence"]), _integer(payload["after"])
        raw_entries = payload["entries"]
        if cursor != after or not isinstance(raw_entries, list) or len(raw_entries) > limit:
            raise ValueError("journal history page differs from request")
        entries = tuple(SwarmHistoryEntry.from_wire(entry) for entry in raw_entries)
        if [entry.sequence for entry in entries] != list(
            range(after + 1, after + 1 + len(entries))
        ):
            raise ValueError("journal history sequence gap")
        if any(
            entry.sequence > head or entry.decision.sequence != entry.sequence for entry in entries
        ):
            raise ValueError("journal history decision or head differs")
        next_after = None if payload["next_after"] is None else _integer(payload["next_after"])
        expected_next = entries[-1].sequence if entries and entries[-1].sequence < head else None
        if next_after != expected_next:
            raise ValueError("journal history cursor differs")
        return cls(swarm_id, head, cursor, entries, next_after)


__all__ = ["SwarmHistoryDecision", "SwarmHistoryEntry", "SwarmHistoryPage"]
