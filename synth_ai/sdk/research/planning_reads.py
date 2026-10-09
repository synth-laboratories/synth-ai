"""Direct Sublinear planning reads, with owner positions and exact spec bytes.

# See: testing/specifications/sdk/owner_reads.md
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING
from uuid import UUID

from synth_ai.core.contracts.json_value import JsonObject, JsonValue
from synth_ai.sdk.research.contracts._wire import object_value

if TYPE_CHECKING:
    from synth_ai.sdk.research.owner_reads import (
        AsyncOwnerReadClient,
        OwnerReadClient,
        OwnerReadScope,
    )

_INTEGER_MAX = 2**63 - 1
_DIGEST = re.compile(r"sha256:[0-9a-f]{64}\Z")
_OPERATION = re.compile(r"[0-9a-f]{64}\Z")
_POSITION_FIELDS = {"graph_revision", "owner_committed_cursor", "published_cursor"}


def _integer(value: JsonValue, *, minimum: int = 1, maximum: int = _INTEGER_MAX) -> int:
    if type(value) is not int or not minimum <= value <= maximum:
        raise ValueError("planning read integer outside owner bounds")
    return value


def _entity(value: str) -> str:
    identity = UUID(value)
    if identity.int == 0 or str(identity) != value:
        raise ValueError("planning read requires canonical nonzero UUID")
    return value


def _response(value: JsonValue, scope: OwnerReadScope, schema: str, fields: set[str]) -> JsonObject:
    payload = object_value(value, operation_id=schema)
    if set(payload) != {"schema_version", "scope"} | fields or payload["schema_version"] != schema:
        raise ValueError("planning owner read response fields drifted")
    if payload["scope"] != {
        "organization_id": scope.organization_id,
        "project_id": scope.project_id,
        "stream_id": scope.run_id,
    }:
        raise ValueError("planning owner read scope mismatch")
    return payload


@dataclass(frozen=True, slots=True)
class PlanningReadPosition:
    """Sublinear position only; publication is not Orchestra application."""

    graph_revision: int
    owner_committed_cursor: int
    published_cursor: int

    @classmethod
    def from_wire(cls, payload: JsonObject) -> PlanningReadPosition:
        revision = _integer(payload["graph_revision"])
        committed = _integer(payload["owner_committed_cursor"], minimum=0)
        published = _integer(payload["published_cursor"], minimum=0)
        if published > committed:
            raise ValueError("planning publication cursor exceeds owner commit")
        return cls(revision, committed, published)


@dataclass(frozen=True, slots=True)
class PlanningTaskRead:
    scope: OwnerReadScope
    position: PlanningReadPosition
    metadata: JsonObject

    @classmethod
    def from_wire(
        cls, value: JsonValue, *, scope: OwnerReadScope, task_id: str
    ) -> PlanningTaskRead:
        payload = _response(value, scope, "sublinear.run-task-read.v1", _POSITION_FIELDS | {"task"})
        metadata = object_value(payload["task"], operation_id="planning Task")
        if metadata.get("task_id") != task_id:
            raise ValueError("planning owner returned another Task")
        _integer(metadata.get("task_revision"))
        definition = object_value(metadata.get("definition_ref"), operation_id="Task definition")
        if (
            set(definition) != {"owner", "schema", "reference", "sha256"}
            or definition["owner"] != "sublinear"
            or definition["schema"] != "sublinear.task-definition.v1"
            or not isinstance(definition["reference"], str)
            or not isinstance(definition["sha256"], str)
            or not _DIGEST.fullmatch(definition["sha256"])
        ):
            raise ValueError("planning Task definition reference invalid")
        return cls(scope, PlanningReadPosition.from_wire(payload), metadata)


@dataclass(frozen=True, slots=True)
class PlanningTaskPage:
    scope: OwnerReadScope
    position: PlanningReadPosition
    tasks: tuple[JsonObject, ...]
    next_after_task_id: str | None

    @classmethod
    def from_wire(
        cls, value: JsonValue, *, scope: OwnerReadScope, limit: int, after_task_id: str | None
    ) -> PlanningTaskPage:
        payload = _response(
            value,
            scope,
            "sublinear.run-task-page.v1",
            _POSITION_FIELDS | {"tasks", "next_after_task_id"},
        )
        rows = payload["tasks"]
        if not isinstance(rows, list) or len(rows) > limit:
            raise ValueError("planning owner Task page exceeds requested bound")
        tasks = tuple(object_value(row, operation_id="planning Task page item") for row in rows)
        identities: list[str] = []
        for row in tasks:
            identity = row.get("task_id")
            if not isinstance(identity, str):
                raise ValueError("planning owner Task identity invalid")
            identities.append(_entity(identity))
        if identities != sorted(set(identities)) or (
            after_task_id is not None and any(identity <= after_task_id for identity in identities)
        ):
            raise ValueError("planning owner Task page cursor did not advance")
        next_task = payload["next_after_task_id"]
        if next_task is not None and (
            not isinstance(next_task, str) or not identities or next_task != identities[-1]
        ):
            raise ValueError("planning owner Task continuation differs from final item")
        return cls(scope, PlanningReadPosition.from_wire(payload), tasks, next_task)


@dataclass(frozen=True, slots=True)
class PlanningGraphRead:
    scope: OwnerReadScope
    position: PlanningReadPosition
    current_graph_revision: int
    metadata: JsonObject

    @classmethod
    def from_wire(cls, value: JsonValue, *, scope: OwnerReadScope) -> PlanningGraphRead:
        payload = _response(
            value,
            scope,
            "sublinear.run-graph-read.v1",
            _POSITION_FIELDS
            | {"current_graph_revision", "desired_posture", "graph", "manifest", "receipt"},
        )
        position = PlanningReadPosition.from_wire(payload)
        current = _integer(payload["current_graph_revision"])
        if position.graph_revision > current:
            raise ValueError("planning graph revision exceeds owner head")
        return cls(scope, position, current, payload)


def _task_query(limit: int, after_task_id: str | None) -> dict[str, JsonValue]:
    query: dict[str, JsonValue] = {"limit": _integer(limit, maximum=64)}
    if after_task_id is not None:
        query["after_task_id"] = _entity(after_task_id)
    return query


def _spec_request(
    task: PlanningTaskRead, scope: OwnerReadScope
) -> tuple[str, dict[str, JsonValue], str]:
    if task.scope != scope:
        raise ValueError("planning Task scope differs from reader")
    identity = task.metadata.get("task_id")
    if not isinstance(identity, str):
        raise ValueError("planning Task identity invalid")
    revision = _integer(task.metadata.get("task_revision"))
    reference = object_value(task.metadata.get("definition_ref"), operation_id="Task definition")
    digest = reference.get("sha256")
    if not isinstance(digest, str) or not _DIGEST.fullmatch(digest):
        raise ValueError("planning Task definition digest invalid")
    return f"/tasks/{_entity(identity)}/spec", {"task_revision": revision}, digest


def _spec_bytes(raw: bytes, digest: str) -> bytes:
    if (
        not raw
        or len(raw) > 4 * 1024 * 1024
        or "sha256:" + hashlib.sha256(raw).hexdigest() != digest
    ):
        raise ValueError("planning owner Task definition bytes differ from pinned reference")
    return raw


class PlanningReads:
    """Scoped owner operations; pages retain their own revision, without a merge.

    # See: testing/specifications/sdk/owner_reads.md
    """

    def __init__(self, reader: OwnerReadClient) -> None:
        if reader.access.owner != "sublinear":
            raise ValueError("planning reads require Sublinear owner access")
        self._reader = reader

    def graph(self, *, graph_revision: int | None = None) -> PlanningGraphRead:
        query = {} if graph_revision is None else {"graph_revision": _integer(graph_revision)}
        result = PlanningGraphRead.from_wire(
            self._reader.read_json("/graph", query=query), scope=self._reader.access.scope
        )
        if graph_revision is not None and result.position.graph_revision != graph_revision:
            raise ValueError("planning owner returned another graph revision")
        return result

    def tasks(self, *, limit: int = 32, after_task_id: str | None = None) -> PlanningTaskPage:
        query = _task_query(limit, after_task_id)
        return PlanningTaskPage.from_wire(
            self._reader.read_json("/tasks", query=query),
            scope=self._reader.access.scope,
            limit=limit,
            after_task_id=after_task_id,
        )

    def task(self, task_id: str) -> PlanningTaskRead:
        _entity(task_id)
        return PlanningTaskRead.from_wire(
            self._reader.read_json(f"/tasks/{task_id}"),
            scope=self._reader.access.scope,
            task_id=task_id,
        )

    def task_spec(self, task: PlanningTaskRead) -> bytes:
        resource, query, digest = _spec_request(task, self._reader.access.scope)
        return _spec_bytes(self._reader.read_bytes(resource, query=query), digest)

    def receipt(self, operation_id: str) -> JsonValue:
        if not _OPERATION.fullmatch(operation_id):
            raise ValueError("planning receipt requires original 64-hex operation ID")
        return self._reader.read_json(f"/receipts/{operation_id}")


class AsyncPlanningReads:
    """Native async counterpart; authority and transport stay with the owner."""

    def __init__(self, reader: AsyncOwnerReadClient) -> None:
        if reader.access.owner != "sublinear":
            raise ValueError("planning reads require Sublinear owner access")
        self._reader = reader

    async def graph(self, *, graph_revision: int | None = None) -> PlanningGraphRead:
        query = {} if graph_revision is None else {"graph_revision": _integer(graph_revision)}
        result = PlanningGraphRead.from_wire(
            await self._reader.read_json("/graph", query=query), scope=self._reader.access.scope
        )
        if graph_revision is not None and result.position.graph_revision != graph_revision:
            raise ValueError("planning owner returned another graph revision")
        return result

    async def tasks(self, *, limit: int = 32, after_task_id: str | None = None) -> PlanningTaskPage:
        query = _task_query(limit, after_task_id)
        return PlanningTaskPage.from_wire(
            await self._reader.read_json("/tasks", query=query),
            scope=self._reader.access.scope,
            limit=limit,
            after_task_id=after_task_id,
        )

    async def task(self, task_id: str) -> PlanningTaskRead:
        _entity(task_id)
        return PlanningTaskRead.from_wire(
            await self._reader.read_json(f"/tasks/{task_id}"),
            scope=self._reader.access.scope,
            task_id=task_id,
        )

    async def task_spec(self, task: PlanningTaskRead) -> bytes:
        resource, query, digest = _spec_request(task, self._reader.access.scope)
        return _spec_bytes(await self._reader.read_bytes(resource, query=query), digest)

    async def receipt(self, operation_id: str) -> JsonValue:
        if not _OPERATION.fullmatch(operation_id):
            raise ValueError("planning receipt requires original 64-hex operation ID")
        return await self._reader.read_json(f"/receipts/{operation_id}")
