"""Typed scientific intent and receipt recovery in an independent workspace."""

from uuid import uuid4

from pydantic import TypeAdapter

from synth_ai.contracts import forge as operations
from synth_ai.core.parameters import identifier, verified
from synth_ai.options import MutationOptions

from .contracts.contracts import contract_digest
from .contracts.operations_v2 import PublicWriteV2
from .contracts.projects import ProjectId
from .contracts.scientific_v2 import RecordV2, VersionedRecord
from .contracts.workspace_science import WorkspaceIntent
from .types import ScientificReceiptResponse, ScientificRecordResponse


def project_identity(value):
    return TypeAdapter(ProjectId).validate_python(value)


def intent(record_id, payload, expected_revision, mutation):
    selected = mutation or MutationOptions(idempotency_key=uuid4().hex)
    key = selected.identity.key if selected.identity is not None else selected.idempotency_key
    if key is None:
        key = uuid4().hex
        selected = selected.model_copy(update={"idempotency_key": key})
    values = {
        "operation_id": key,
        "record_id": identifier(record_id),
        "expected_revision": expected_revision,
        "payload": payload,
    }
    # Validate the canonical record before hashing so defaults and tuple/list
    # encoding cannot change the retry identity at the public boundary.
    from .contracts.workspace_science import WorkspaceRecord

    record = TypeAdapter(WorkspaceRecord | RecordV2).validate_python(payload)
    model = PublicWriteV2 if isinstance(record, VersionedRecord) else WorkspaceIntent
    values["payload"] = record.model_dump(mode="json")
    if model is PublicWriteV2:
        values["schema_version"] = model.model_fields["schema_version"].default
    values["request_digest_sha256"] = contract_digest(values)
    return model.model_validate(values), selected


def check_record(result, project_id, record_id, revision):
    record = result.record
    return verified(
        result,
        record.scope.project_id == project_id
        and record.reference.authority == "forge"
        and record.reference.record_id == record_id
        and record.reference.kind == record.payload.kind
        and record.reference.digest_sha256 == contract_digest(record.payload)
        and (revision is None or record.reference.revision == str(revision)),
        "Forge returned a different scientific record",
    )


def check_receipt(result, project_id, operation_id, request=None):
    receipt = result.receipt
    valid = receipt.scope.project_id == project_id and receipt.operation_id == operation_id
    if request is not None:
        valid = valid and (
            receipt.reference.authority == "forge"
            and receipt.reference.record_id == request.record_id
            and receipt.reference.kind == request.payload.kind
            and receipt.reference.digest_sha256 == contract_digest(request.payload)
            and receipt.reference.revision == str(request.expected_revision + 1)
        )
    return verified(result, valid, "Forge returned a different scientific receipt")


class Records:
    def __init__(self, transport):
        self._transport = transport

    def get(self, project_id, record_id, *, revision=None, request=None):
        project_id, record_id = project_identity(project_id), identifier(record_id)
        if revision is not None and (type(revision) is not int or not 1 <= revision <= 2147483647):
            raise ValueError("revision must be a positive bounded integer")
        return self._transport.request(
            operations.RECORD_GET,
            ScientificRecordResponse,
            path=operations.RECORD_GET.path_for(project_id=project_id, record_id=record_id),
            query={} if revision is None else {"revision": revision},
            options=request,
            validate=lambda result: check_record(result, project_id, record_id, revision),
        )

    def write(self, project_id, record_id, payload, *, expected_revision=0, mutation=None):
        project_id = project_identity(project_id)
        request, selected = intent(record_id, payload, expected_revision, mutation)
        return self._transport.request(
            operations.RECORD_WRITE,
            ScientificReceiptResponse,
            path=operations.RECORD_WRITE.path_for(project_id=project_id),
            body=request,
            options=selected,
            validate=lambda result: check_receipt(
                result, project_id, request.operation_id, request
            ),
        )


class Receipts:
    def __init__(self, transport):
        self._transport = transport

    def get(self, project_id, operation_id, *, request=None):
        project_id, operation_id = project_identity(project_id), identifier(operation_id)
        return self._transport.request(
            operations.RECEIPT_GET,
            ScientificReceiptResponse,
            path=operations.RECEIPT_GET.path_for(project_id=project_id, operation_id=operation_id),
            options=request,
            validate=lambda result: check_receipt(result, project_id, operation_id),
        )


class AsyncRecords:
    def __init__(self, transport):
        self._transport = transport

    async def get(self, project_id, record_id, *, revision=None, request=None):
        project_id, record_id = project_identity(project_id), identifier(record_id)
        if revision is not None and (type(revision) is not int or not 1 <= revision <= 2147483647):
            raise ValueError("revision must be a positive bounded integer")
        return await self._transport.request(
            operations.RECORD_GET,
            ScientificRecordResponse,
            path=operations.RECORD_GET.path_for(project_id=project_id, record_id=record_id),
            query={} if revision is None else {"revision": revision},
            options=request,
            validate=lambda result: check_record(result, project_id, record_id, revision),
        )

    async def write(self, project_id, record_id, payload, *, expected_revision=0, mutation=None):
        project_id = project_identity(project_id)
        request, selected = intent(record_id, payload, expected_revision, mutation)
        return await self._transport.request(
            operations.RECORD_WRITE,
            ScientificReceiptResponse,
            path=operations.RECORD_WRITE.path_for(project_id=project_id),
            body=request,
            options=selected,
            validate=lambda result: check_receipt(
                result, project_id, request.operation_id, request
            ),
        )


class AsyncReceipts:
    def __init__(self, transport):
        self._transport = transport

    async def get(self, project_id, operation_id, *, request=None):
        project_id, operation_id = project_identity(project_id), identifier(operation_id)
        return await self._transport.request(
            operations.RECEIPT_GET,
            ScientificReceiptResponse,
            path=operations.RECEIPT_GET.path_for(project_id=project_id, operation_id=operation_id),
            options=request,
            validate=lambda result: check_receipt(result, project_id, operation_id),
        )
