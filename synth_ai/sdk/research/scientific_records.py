"""Scientific records through backend research APIs, with sync/async parity.

See Forge docs/contracts.md and SYN-3559. Public clients never call Forge or
execution services directly. Explicit operation IDs survive uncertain responses.
"""

import hashlib
from typing import Literal
from urllib.parse import quote

from pydantic import Field

from synth_ai.sdk.research.contracts.forge.contracts import Contract, contract_digest
from synth_ai.sdk.research.contracts.forge.operations import (
    Event,
    PublicWrite,
    Receipt,
    RecordRevision,
)
from synth_ai.sdk.research.contracts.forge.records import Artifact, Record
from synth_ai.sdk.research.contracts.native_attachment import NativeAttachmentPage
from synth_ai.sdk.research.contracts.scientific_citations import CitationVerification


class ExecutionWriteRequest(Contract):
    """An intent under an existing mloky/Orchestra producer admission.

    Backend ExecutionWrite remains authoritative; this DTO cannot supply or
    replace the producer stamp and native trial identity remains native.
    """

    organization_id: str = Field(
        pattern=r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$"
    )
    admission_id: str = Field(pattern=r"^adm_[0-9A-HJKMNP-TV-Z]{26}$")
    write: PublicWrite


class ScientificWriterState(Contract):
    organization_id: str
    project_id: str
    state: Literal["native", "fenced", "forge_writer", "reverted"]
    transition_id: str | None = None
    fence_id: str | None = None
    forge_identity: str | None = None
    receipt_digest_sha256: str | None = None


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


class ScientificRecordsAPI:
    def __init__(self, transport):
        self._transport = transport

    def execution_write(self, project_id: str, request: ExecutionWriteRequest) -> Receipt:
        """Write under an admitted execution; use its service credential.

        No human stamp is synthesized. Null/negative outcomes remain ordinary
        Result records with their declared evaluator/scorer references.
        """
        write = request.write
        receipt = _receipt(
            self._transport.request_json(
                "POST",
                _root(project_id) + "/execution-operations",
                json_body=request.model_dump(mode="json"),
                operation_id=write.operation_id,
            ),
            project_id,
            write.operation_id,
            payload=write.payload,
            record_id=write.record_id,
            expected_revision=write.expected_revision,
        )
        if receipt.scope.organization_id != request.organization_id:
            raise ValueError("execution receipt belongs to a different organization")
        return receipt

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

    def native_attachments(
        self, project_id: str, record_id: str, *, limit: int = 100, after: str | None = None
    ) -> NativeAttachmentPage:
        """Read native-owned execution evidence with explicit pending/attached state."""
        if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= 1000:
            raise ValueError("native evidence limit must be 1 through 1000")
        return _native_attachment_page(
            self._transport.request_json(
                "GET",
                _root(project_id) + f"/records/{quote(record_id, safe='')}/native-attachments",
                params={"limit": limit, **({"after": after} if after is not None else {})},
            ),
            project_id,
            record_id,
        )

    def writer_state(self, project_id: str) -> ScientificWriterState:
        """Read the canonical scope writer without inferring it from refusals."""
        result = ScientificWriterState.model_validate(
            self._transport.request_json("GET", _root(project_id) + "/writer-state")
        )
        if result.project_id != project_id:
            raise ValueError("writer state differs from requested project")
        return result

    def citations(
        self, project_id: str, record_id: str, *, revision: int | None = None
    ) -> CitationVerification:
        """Read current custody for exact retained citations, without changing history."""
        if revision is not None and (
            isinstance(revision, bool) or not isinstance(revision, int) or revision < 1
        ):
            raise ValueError("citation revision must be a positive integer")
        result = CitationVerification.model_validate(
            self._transport.request_json(
                "GET",
                _root(project_id) + f"/records/{quote(record_id, safe='')}/citations",
                params={"revision": revision} if revision is not None else None,
            )
        )
        if result.record.record_id != record_id or (
            revision is not None and result.record.revision != str(revision)
        ):
            raise ValueError("citation verification differs from requested exact record")
        if result.retained != all(check.status == "retained" for check in result.citations):
            raise ValueError("citation verification retained flag contradicts custody checks")
        return result

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

    async def execution_write(self, project_id: str, request: ExecutionWriteRequest) -> Receipt:
        """Write under an admitted execution; use its service credential.

        No human stamp is synthesized. Null/negative outcomes remain ordinary
        Result records with their declared evaluator/scorer references.
        """
        write = request.write
        receipt = _receipt(
            await self._transport.request_json(
                "POST",
                _root(project_id) + "/execution-operations",
                json_body=request.model_dump(mode="json"),
                operation_id=write.operation_id,
            ),
            project_id,
            write.operation_id,
            payload=write.payload,
            record_id=write.record_id,
            expected_revision=write.expected_revision,
        )
        if receipt.scope.organization_id != request.organization_id:
            raise ValueError("execution receipt belongs to a different organization")
        return receipt

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

    async def native_attachments(
        self, project_id: str, record_id: str, *, limit: int = 100, after: str | None = None
    ) -> NativeAttachmentPage:
        """Read exact native evidence without inferring delivery from accepted intent."""
        if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= 1000:
            raise ValueError("native evidence limit must be 1 through 1000")
        return _native_attachment_page(
            await self._transport.request_json(
                "GET",
                _root(project_id) + f"/records/{quote(record_id, safe='')}/native-attachments",
                params={"limit": limit, **({"after": after} if after is not None else {})},
            ),
            project_id,
            record_id,
        )

    async def writer_state(self, project_id: str) -> ScientificWriterState:
        """Read the canonical scope writer without inferring it from refusals."""
        result = ScientificWriterState.model_validate(
            await self._transport.request_json("GET", _root(project_id) + "/writer-state")
        )
        if result.project_id != project_id:
            raise ValueError("writer state differs from requested project")
        return result

    async def citations(
        self, project_id: str, record_id: str, *, revision: int | None = None
    ) -> CitationVerification:
        """Read current custody for exact retained citations, without changing history."""
        if revision is not None and (
            isinstance(revision, bool) or not isinstance(revision, int) or revision < 1
        ):
            raise ValueError("citation revision must be a positive integer")
        result = CitationVerification.model_validate(
            await self._transport.request_json(
                "GET",
                _root(project_id) + f"/records/{quote(record_id, safe='')}/citations",
                params={"revision": revision} if revision is not None else None,
            )
        )
        if result.record.record_id != record_id or (
            revision is not None and result.record.revision != str(revision)
        ):
            raise ValueError("citation verification differs from requested exact record")
        if result.retained != all(check.status == "retained" for check in result.citations):
            raise ValueError("citation verification retained flag contradicts custody checks")
        return result

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


def _native_attachment_page(document, project_id, record_id):
    from synth_ai.core.errors import (
        RetryDirective,
        SynthErrorCategory,
        SynthErrorCode,
        SynthFailure,
    )

    from .errors import ResearchApiError

    try:
        page = NativeAttachmentPage.model_validate(document)
        if any(
            item.attachment.scope.project_id != project_id
            or item.attachment.experiment.record_id != record_id
            or (
                item.receipt is not None
                and (
                    item.receipt.scope != item.attachment.scope
                    or item.receipt.reference.record_id != record_id
                    or item.receipt.reference.kind != "experiment"
                    or item.receipt.reference.revision
                    != str(int(item.attachment.experiment.revision) + 1)
                )
            )
            for item in page.attachments
        ):
            raise ValueError("native evidence scope/canonical receipt identity differs")
        if page.truncated != (page.next_cursor is not None):
            raise ValueError("native evidence continuation identity differs")
        return page
    except (ValueError, TypeError) as error:
        message = "Native evidence response violates its producer contract"
        raise ResearchApiError(
            message,
            status_code=200,
            operation_id="get_native_scientific_attachments",
            cause=[{"kind": type(error).__name__}],
            failure=SynthFailure(
                code=SynthErrorCode("schema_integrity_conflict"),
                category=SynthErrorCategory.CONTRACT_MISMATCH,
                operation="get_native_scientific_attachments",
                request_id=None,
                correlation_id=None,
                retry=RetryDirective(retryable=False),
                status=200,
                detail=message,
                reason="native_attachment_wire_contract",
            ),
        ) from None
