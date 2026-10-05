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
from synth_ai.sdk.research.contracts.forge_citations import CitationVerification
from synth_ai.sdk.research.contracts.forge_uploads import Uploaded, UploadRequest


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


def _citations(document, record_id, revision):
    verification = CitationVerification.model_validate(document)
    record = verification.record
    if (
        record.authority != "forge"
        or record.record_id != record_id
        or (revision is not None and record.revision != str(revision))
        or any(check.reference.authority == "forge" for check in verification.citations)
        or verification.retained
        != all(check.status == "retained" for check in verification.citations)
    ):
        raise ValueError("citation verification differs from requested exact identity")
    return verification


def _citations_path(project_id, record_id):
    return _root(project_id) + "/records/" + quote(record_id, safe="") + "/citations"


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
        or after in ids
        or len(set(ids)) != len(ids)
    ):
        raise ValueError("scientific page differs from requested cursor/kind")
    # Preserve native database collation; only that producer defines whether
    # an ID follows the opaque cursor. Python ordering is not interchangeable.
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


def _uploaded(document, request):
    uploaded = Uploaded.model_validate(document)
    if (
        uploaded.scope != request.scope
        or uploaded.operation_id != request.operation_id
        or uploaded.request_digest_sha256 != contract_digest(request)
        or uploaded.manifest.authority != "artifact-platform"
        or uploaded.manifest.kind != "manifest"
        or uploaded.manifest.record_id != uploaded.publication_id
        or uploaded.manifest.revision != str(uploaded.revision)
    ):
        raise ValueError("private publication response differs from scoped operation")
    declarations = {item.logical_path: item for item in request.objects}
    actual = {item.logical_path: item for item in uploaded.objects}
    if len(actual) != len(uploaded.objects) or set(actual) != set(declarations):
        raise ValueError("private publication object selection differs")
    for path, item in actual.items():
        expected = declarations[path]
        if (
            item.size_bytes != expected.size_bytes
            or item.media_type != expected.media_type
            or item.reference.authority != "artifact-platform"
            or item.reference.kind != "object"
            or item.reference.record_id != uploaded.publication_id
            or item.reference.revision != str(uploaded.revision)
            or item.reference.digest_sha256 != expected.digest_sha256
        ):
            raise ValueError("private publication byte receipt differs")
    manifest = {
        "schema_version": "synth.artifact-platform.v1",
        "manifest_schema_version": "forge.scientific.evidence.v1",
        "publication_id": uploaded.publication_id,
        "collection_id": uploaded.collection_id,
        "revision": uploaded.revision,
        "objects": [
            {
                "logical_path": item.logical_path,
                "digest_sha256": item.digest_sha256,
                "size_bytes": item.size_bytes,
                "media_type": item.media_type,
            }
            for item in sorted(request.objects, key=lambda item: item.logical_path)
        ],
    }
    if contract_digest(manifest) != uploaded.manifest.digest_sha256:
        raise ValueError("private publication manifest identity differs")
    return uploaded


def _upload_body(project_id, request):
    request = UploadRequest.model_validate(request.model_dump(mode="json"))
    if request.scope.project_id != project_id:
        raise ValueError("private upload belongs to another project")
    return request.model_dump(mode="json")


class ScientificRecordsAPI:
    def __init__(self, transport):
        self._transport = transport

    def upload_artifacts(self, project_id: str, request: UploadRequest) -> Uploaded:
        """Retain bounded private evidence with a retryable operation identity.

        See backend docs/contracts/forge-authority.v1.md. This creates no public
        contribution or release approval and returns no storage credentials.
        """
        body = _upload_body(project_id, request)
        document = self._transport.request_json(
            "POST",
            f"/smr/projects/{quote(project_id, safe='')}/forge-artifacts",
            json_body=body,
            operation_id=request.operation_id,
        )
        return _uploaded(document, request)

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

    def verify_citations(
        self, project_id: str, record_id: str, *, revision: int | None = None
    ) -> CitationVerification:
        """Current custodian state of a retained revision's external citations.

        Read-only; dangling/denied/conflict/unavailable citations are explicit
        and the retained record is unchanged.
        """
        return _citations(
            self._transport.request_json(
                "GET",
                _citations_path(project_id, record_id),
                params={"revision": revision} if revision is not None else None,
            ),
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

    async def upload_artifacts(self, project_id: str, request: UploadRequest) -> Uploaded:
        """Async parity for bounded private evidence retention; see Forge authority v1."""
        body = _upload_body(project_id, request)
        document = await self._transport.request_json(
            "POST",
            f"/smr/projects/{quote(project_id, safe='')}/forge-artifacts",
            json_body=body,
            operation_id=request.operation_id,
        )
        return _uploaded(document, request)

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

    async def verify_citations(
        self, project_id: str, record_id: str, *, revision: int | None = None
    ) -> CitationVerification:
        return _citations(
            await self._transport.request_json(
                "GET",
                _citations_path(project_id, record_id),
                params={"revision": revision} if revision is not None else None,
            ),
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
