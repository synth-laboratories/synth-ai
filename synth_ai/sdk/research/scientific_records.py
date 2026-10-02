"""Scientific records through backend research APIs, with sync/async parity.

See Forge docs/contracts.md and SYN-3559. Public clients never call Forge or
execution services directly. Explicit operation IDs survive uncertain responses.
"""

import hashlib
from urllib.parse import quote

from synth_ai.sdk.research.contracts.forge.contracts import contract_digest
from synth_ai.sdk.research.contracts.forge.operations import (
    Event,
    PublicWrite,
    Receipt,
    RecordRevision,
)
from synth_ai.sdk.research.contracts.forge.records import Artifact, Record


def _verified_bytes(record, content):
    if (
        not isinstance(record.payload, Artifact)
        or len(content) != record.payload.byte_count
        or hashlib.sha256(content).hexdigest() != record.payload.object.digest_sha256
    ):
        raise ValueError("artifact bytes differ from exact scientific association")
    return content


def _record(document, project_id, record_id, revision):
    record = RecordRevision.model_validate(document)
    if (
        record.scope.project_id != project_id
        or record.reference.authority != "forge"
        or record.reference.record_id != record_id
        or record.reference.kind != record.payload.kind
        or record.reference.digest_sha256 != contract_digest(record.payload)
        or (revision is not None and record.reference.revision != str(revision))
    ):
        raise ValueError("scientific response differs from requested exact identity")
    return record


def _receipt(
    document, project_id, operation_id, *, payload=None, record_id=None, expected_revision=None
):
    receipt = Receipt.model_validate(document)
    if (
        receipt.scope.project_id != project_id
        or receipt.operation_id != operation_id
        or receipt.reference.authority != "forge"
    ):
        raise ValueError("scientific receipt differs from requested identity")
    if payload is not None and (
        receipt.reference.record_id != record_id
        or receipt.reference.kind != payload.kind
        or receipt.reference.revision != str(expected_revision + 1)
        or receipt.reference.digest_sha256 != contract_digest(payload)
    ):
        raise ValueError("scientific receipt differs from submitted payload")
    return receipt


def _records(documents, project_id, kind, after, limit):
    records = tuple(
        _record(item, project_id, item["reference"]["record_id"], None) for item in documents
    )
    ids = tuple(item.reference.record_id for item in records)
    if (
        len(records) > limit
        or any(item.payload.kind != kind for item in records)
        or any(item <= after for item in ids)
        or tuple(sorted(set(ids))) != ids
    ):
        raise ValueError("scientific page differs from requested cursor/kind")
    return records


def _events(documents, project_id, after, limit):
    events = tuple(Event.model_validate(item) for item in documents)
    cursors = tuple(event.cursor for event in events)
    if (
        len(events) > limit
        or tuple(sorted(set(cursors))) != cursors
        or any(cursor <= after for cursor in cursors)
    ):
        raise ValueError("scientific events differ from requested cursor")
    for event in events:
        _receipt(event.receipt.model_dump(mode="json"), project_id, event.receipt.operation_id)
        if event.cursor != event.receipt.cursor:
            raise ValueError("event and receipt cursors differ")
    return events


def _root(project_id: str) -> str:
    if not project_id:
        raise ValueError("project_id is required")
    return f"/smr/projects/{quote(project_id, safe='')}/research"


def _body(payload: Record, record_id: str, operation_id: str, expected_revision: int):
    values = {
        "operation_id": operation_id,
        "record_id": record_id,
        "expected_revision": expected_revision,
        "payload": payload.model_dump(mode="json"),
    }
    request = PublicWrite(**values, request_digest_sha256=contract_digest(values))
    return request.model_dump(mode="json")


class ScientificRecordsAPI:
    def __init__(self, transport):
        self._transport = transport

    def write(
        self,
        project_id: str,
        payload: Record,
        *,
        record_id: str,
        operation_id: str,
        expected_revision: int = 0,
    ) -> Receipt:
        """Append a revision; retry the same intent with its original operation ID."""
        return _receipt(
            self._transport.request_json(
                "POST",
                _root(project_id) + "/operations",
                json_body=_body(payload, record_id, operation_id, expected_revision),
                operation_id=operation_id,
            ),
            project_id,
            operation_id,
            payload=payload,
            record_id=record_id,
            expected_revision=expected_revision,
        )

    def get(
        self, project_id: str, record_id: str, *, revision: int | None = None
    ) -> RecordRevision:
        return _record(
            self._transport.request_json(
                "GET",
                _root(project_id) + "/records/" + quote(record_id, safe=""),
                params={"revision": revision} if revision is not None else None,
            ),
            project_id,
            record_id,
            revision,
        )

    def list(
        self, project_id: str, kind: str, *, after: str = "", limit: int = 100
    ) -> tuple[RecordRevision, ...]:
        records = self._transport.request_json(
            "GET",
            _root(project_id) + "/records",
            params={"kind": kind, "after": after, "limit": limit},
        )
        return _records(records, project_id, kind, after, limit)

    def receipt(self, project_id: str, operation_id: str) -> Receipt:
        return _receipt(
            self._transport.request_json(
                "GET", _root(project_id) + "/receipts/" + quote(operation_id, safe="")
            ),
            project_id,
            operation_id,
        )

    def download_artifact(self, project_id: str, record_id: str, *, revision: int) -> bytes:
        record = self.get(project_id, record_id, revision=revision)
        if not isinstance(record.payload, Artifact) or record.payload.delivery is None:
            raise ValueError("scientific record has no retained byte locator")
        content = self._transport.request_bytes(
            "GET",
            _root(project_id) + "/records/" + quote(record_id, safe="") + "/artifact",
            params={"revision": revision},
        )
        return _verified_bytes(record, content)

    def events(self, project_id: str, *, after: int = 0, limit: int = 100) -> tuple[Event, ...]:
        events = self._transport.request_json(
            "GET", _root(project_id) + "/events", params={"after": after, "limit": limit}
        )
        return _events(events, project_id, after, limit)


class AsyncScientificRecordsAPI:
    def __init__(self, transport):
        self._transport = transport

    async def write(
        self,
        project_id: str,
        payload: Record,
        *,
        record_id: str,
        operation_id: str,
        expected_revision: int = 0,
    ) -> Receipt:
        return _receipt(
            await self._transport.request_json(
                "POST",
                _root(project_id) + "/operations",
                json_body=_body(payload, record_id, operation_id, expected_revision),
                operation_id=operation_id,
            ),
            project_id,
            operation_id,
            payload=payload,
            record_id=record_id,
            expected_revision=expected_revision,
        )

    async def get(
        self, project_id: str, record_id: str, *, revision: int | None = None
    ) -> RecordRevision:
        return _record(
            await self._transport.request_json(
                "GET",
                _root(project_id) + "/records/" + quote(record_id, safe=""),
                params={"revision": revision} if revision is not None else None,
            ),
            project_id,
            record_id,
            revision,
        )

    async def list(
        self, project_id: str, kind: str, *, after: str = "", limit: int = 100
    ) -> tuple[RecordRevision, ...]:
        records = await self._transport.request_json(
            "GET",
            _root(project_id) + "/records",
            params={"kind": kind, "after": after, "limit": limit},
        )
        return _records(records, project_id, kind, after, limit)

    async def receipt(self, project_id: str, operation_id: str) -> Receipt:
        return _receipt(
            await self._transport.request_json(
                "GET", _root(project_id) + "/receipts/" + quote(operation_id, safe="")
            ),
            project_id,
            operation_id,
        )

    async def download_artifact(self, project_id: str, record_id: str, *, revision: int) -> bytes:
        record = await self.get(project_id, record_id, revision=revision)
        if not isinstance(record.payload, Artifact) or record.payload.delivery is None:
            raise ValueError("scientific record has no retained byte locator")
        content = await self._transport.request_bytes(
            "GET",
            _root(project_id) + "/records/" + quote(record_id, safe="") + "/artifact",
            params={"revision": revision},
        )
        return _verified_bytes(record, content)

    async def events(
        self, project_id: str, *, after: int = 0, limit: int = 100
    ) -> tuple[Event, ...]:
        events = await self._transport.request_json(
            "GET", _root(project_id) + "/events", params={"after": after, "limit": limit}
        )
        return _events(events, project_id, after, limit)
