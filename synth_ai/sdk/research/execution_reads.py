"""Direct Orchestra execution snapshots and journal pages.

# See: testing/specifications/sdk/owner_reads.md
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

from synth_ai.core.contracts.json_value import JsonObject, JsonValue
from synth_ai.sdk.research.contracts._wire import object_value

if TYPE_CHECKING:
    from synth_ai.sdk.research.owner_reads import (
        AsyncOwnerReadClient,
        OwnerReadClient,
        OwnerReadScope,
    )

_SCHEMA = "orchestra.owner.run-execution.v1"
_VIEW = "run_execution"
_DIGEST = re.compile(r"sha256:[0-9a-f]{64}\Z")


def _integer(value: JsonValue, *, maximum: int = 2**64 - 1) -> int:
    if type(value) is not int or not 0 <= value <= maximum:
        raise ValueError("execution owner position outside wire bounds")
    return value


def _object(value: JsonValue, fields: set[str]) -> JsonObject:
    result = object_value(value, operation_id="execution owner read")
    if set(result) != fields:
        raise ValueError("execution owner response fields drifted")
    return result


def _scope(scope: OwnerReadScope) -> JsonObject:
    return {
        "organization_id": scope.organization_id,
        "project_id": scope.project_id,
        "stream_id": scope.run_id,
    }


@dataclass(frozen=True, slots=True)
class ExecutionReadCursor:
    """A position in one Orchestra view, independent of Sublinear publication."""

    scope: OwnerReadScope
    sequence: int

    def __post_init__(self) -> None:
        _integer(self.sequence)

    @classmethod
    def from_wire(cls, value: JsonValue, scope: OwnerReadScope) -> ExecutionReadCursor:
        body = _object(value, {"scope", "sequence", "view", "schema"})
        if body["scope"] != _scope(scope) or body["view"] != _VIEW or body["schema"] != _SCHEMA:
            raise ValueError("execution owner cursor scope or view mismatch")
        return cls(scope, _integer(body["sequence"]))

    def wire(self) -> JsonObject:
        return {"scope": _scope(self.scope), "sequence": self.sequence, "view": _VIEW, "schema": _SCHEMA}

    def event_id(self) -> str:
        """Return the exact canonical cursor accepted by the owner HTTP API."""
        return json.dumps(self.wire(), sort_keys=True, separators=(",", ":"), ensure_ascii=False)


@dataclass(frozen=True, slots=True)
class ExecutionReadRetention:
    floor_sequence: int
    head_sequence: int
    anchor: JsonObject

    @classmethod
    def from_wire(cls, value: JsonValue) -> ExecutionReadRetention:
        body = _object(value, {"floor_sequence", "head_sequence", "anchor"})
        floor, head = _integer(body["floor_sequence"]), _integer(body["head_sequence"])
        anchor = _object(body["anchor"], {"owner", "schema", "reference", "sha256"})
        if floor > head or not all(isinstance(v, str) and v for v in anchor.values()):
            raise ValueError("execution owner retention invalid")
        digest = anchor["sha256"]
        if not isinstance(digest, str) or not _DIGEST.fullmatch(digest):
            raise ValueError("execution owner retention anchor digest invalid")
        return cls(floor, head, anchor)

    def contains(self, position: ExecutionReadCursor) -> None:
        if not self.floor_sequence <= position.sequence <= self.head_sequence:
            raise ValueError("execution owner cursor outside retained range")


@dataclass(frozen=True, slots=True)
class ExecutionReadValue:
    """Owner public view with its original JSON spelling and verified digest."""

    position: ExecutionReadCursor
    raw_json: str
    sha256: str
    value: JsonObject

    @classmethod
    def from_wire(
        cls, payload: JsonValue, position: ExecutionReadCursor, *, kind: str
    ) -> ExecutionReadValue:
        body = _object(payload, {"kind", "schema", "canonical_json", "sha256"})
        raw = body["canonical_json"]
        if body["kind"] != "inline" or body["schema"] != _SCHEMA or not isinstance(raw, str):
            raise ValueError("execution owner public view payload invalid")
        encoded = raw.encode("utf-8")
        digest = "sha256:" + hashlib.sha256(encoded).hexdigest()
        if len(encoded) > 131072 or body["sha256"] != digest:
            raise ValueError("execution owner public view digest mismatch")
        value = object_value(cast(JsonValue, json.loads(raw)), operation_id="execution public view")
        if (
            value.get("schema_version") != _SCHEMA
            or value.get("kind") != kind
            or value.get("scope") != _scope(position.scope)
            or _integer(value.get("sequence")) != position.sequence
        ):
            raise ValueError("execution owner public view binding mismatch")
        return cls(position, raw, digest, value)


@dataclass(frozen=True, slots=True)
class ExecutionSnapshot:
    snapshot: ExecutionReadValue
    retention: ExecutionReadRetention

    @classmethod
    def from_wire(cls, value: JsonValue, scope: OwnerReadScope) -> ExecutionSnapshot:
        body = _object(value, {"scope", "cursor", "snapshot", "retention"})
        if body["scope"] != _scope(scope):
            raise ValueError("execution owner snapshot scope mismatch")
        position = ExecutionReadCursor.from_wire(body["cursor"], scope)
        retention = ExecutionReadRetention.from_wire(body["retention"])
        retention.contains(position)
        return cls(ExecutionReadValue.from_wire(body["snapshot"], position, kind="snapshot"), retention)


@dataclass(frozen=True, slots=True)
class ExecutionEventPage:
    events: tuple[ExecutionReadValue, ...]
    next: ExecutionReadCursor
    retention: ExecutionReadRetention

    @classmethod
    def from_wire(
        cls, value: JsonValue, after: ExecutionReadCursor, limit: int
    ) -> ExecutionEventPage:
        body = _object(value, {"scope", "items", "next", "retention"})
        if body["scope"] != _scope(after.scope):
            raise ValueError("execution owner page scope mismatch")
        rows = body["items"]
        if not isinstance(rows, list) or len(rows) > limit:
            raise ValueError("execution owner page exceeds requested bound")
        retention = ExecutionReadRetention.from_wire(body["retention"])
        retention.contains(after)
        next_cursor = ExecutionReadCursor.from_wire(body["next"], after.scope)
        retention.contains(next_cursor)
        events: list[ExecutionReadValue] = []
        previous = after.sequence
        for row in rows:
            item = _object(row, {"cursor", "body"})
            position = ExecutionReadCursor.from_wire(item["cursor"], after.scope)
            retention.contains(position)
            if position.sequence <= previous:
                raise ValueError("execution owner page did not advance")
            events.append(ExecutionReadValue.from_wire(item["body"], position, kind="position"))
            previous = position.sequence
        if next_cursor.sequence != previous:
            raise ValueError("execution owner continuation differs from last item")
        return cls(tuple(events), next_cursor, retention)


def _query(after: ExecutionReadCursor, scope: OwnerReadScope, limit: int) -> JsonObject:
    if after.scope != scope:
        raise ValueError("execution owner cursor differs from reader scope")
    if _integer(limit, maximum=256) == 0:
        raise ValueError("execution owner page limit must be positive")
    return {"after": after.event_id(), "limit": limit}


class ExecutionReads:
    """Owner execution views; see testing/specifications/sdk/owner_reads.md."""

    def __init__(self, reader: OwnerReadClient) -> None:
        if reader.access.owner != "orchestra":
            raise ValueError("execution reads require Orchestra owner access")
        self._reader = reader

    def snapshot(self) -> ExecutionSnapshot:
        return ExecutionSnapshot.from_wire(
            self._reader.read_json("/snapshot", query={"view": _VIEW}), self._reader.access.scope
        )

    def events(self, after: ExecutionReadCursor, *, limit: int = 128) -> ExecutionEventPage:
        query = _query(after, self._reader.access.scope, limit)
        return ExecutionEventPage.from_wire(self._reader.read_json("/events", query=query), after, limit)


class AsyncExecutionReads:
    """Native async owner views; see testing/specifications/sdk/owner_reads.md."""

    def __init__(self, reader: AsyncOwnerReadClient) -> None:
        if reader.access.owner != "orchestra":
            raise ValueError("execution reads require Orchestra owner access")
        self._reader = reader

    async def snapshot(self) -> ExecutionSnapshot:
        return ExecutionSnapshot.from_wire(
            await self._reader.read_json("/snapshot", query={"view": _VIEW}), self._reader.access.scope
        )

    async def events(self, after: ExecutionReadCursor, *, limit: int = 128) -> ExecutionEventPage:
        query = _query(after, self._reader.access.scope, limit)
        return ExecutionEventPage.from_wire(
            await self._reader.read_json("/events", query=query), after, limit
        )
