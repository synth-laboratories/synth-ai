"""Project evaluation evidence; see backend mnemosyne-evaluation-evidence.md."""

from __future__ import annotations

import hashlib
import json
import re
from decimal import Decimal
from typing import Annotated, Literal

from pydantic import Field, StrictInt

from synth_ai.contracts import mnemosyne as operations
from synth_ai.core.parameters import verified
from synth_ai.sdk.forge.contracts.contracts import Digest, ExactReference, Scope
from synth_ai.sdk.forge.contracts.records import ArtifactDeliveryBinding
from synth_ai.sdk.forge.records import project_identity
from synth_ai.types.base import Model, ResponseModel


class SourceAssociation(Model):
    source_kind: str
    benchmark_id: str | None = None
    producer_run_id: str | None = None
    task_id: str | None = None
    attempt_id: str | None = None
    rollout_id: str | None = None
    candidate_id: str | None = None
    verification: Literal["producer_asserted"]


class EvidenceLocator(Model):
    schema_version: Literal["synth.forge.evidence.v1"]
    result: Literal["present"]
    scope: Scope
    digest_sha256: Digest
    size_bytes: Annotated[StrictInt, Field(ge=0)]
    media_type: str
    manifest: ExactReference
    object: ExactReference
    delivery: ArtifactDeliveryBinding


class EvidenceObject(Model):
    role: str
    locator: EvidenceLocator


class Receipt(Model):
    schema_version: Literal["synth.mnemosyne.evaluation-evidence-receipt.v1"]
    evidence_id: str
    scope: Scope
    operation_id: str
    request_digest: Digest
    source_association: SourceAssociation
    objects: tuple[EvidenceObject, ...]
    imported_by: str
    custody: Literal["artifact_reference"]
    replayed: bool | None = None


class CapabilitiesData(Model):
    schema_version: Literal["synth.mnemosyne.evaluation-evidence-capabilities.v1"]
    intake: Literal["available", "unavailable"]
    can_read: bool
    can_publish: bool
    request_schema_version: Literal["synth.mnemosyne.evaluation-evidence-request.v1"]
    scope: Scope


class ReceiptResponse(ResponseModel):
    data: Receipt


class EvidencePageData(Model):
    schema_version: Literal["synth.mnemosyne.evaluation-evidence-page.v1"]
    scope: Scope
    items: tuple[Receipt, ...] = Field(max_length=100)
    next_cursor: str | None


class ComputedCost(Model):
    record_id: str
    resource_id: str
    component: str
    currency: str
    reported_amount: Decimal | None
    reported_basis: str
    implied_amount: Decimal | None
    selected_amount: Decimal | None
    selected_basis: Literal["provider_reported", "settled", "estimated", "rate_derived", "unknown"]
    coverage: Literal["complete", "partial", "unknown"]
    accounting_role: Literal["charge", "reference"]


class CostSubtotal(Model):
    currency: str
    component: str
    known_amount: Decimal | None
    line_count: int
    known_line_count: int
    complete: bool


class ComputedSpan(Model):
    span_id: str
    resource_id: str
    phase: str
    elapsed_seconds: Decimal | None
    basis: str


class MeasurementSummary(Model):
    schema_version: Literal["synth.mnemosyne.measurement-summary.v1"]
    recipe_version: Literal["evaluation-measurements.v1"]
    canonical_measurements_sha256: Digest
    producer_run_id: str
    attempt_id: str
    coverage: Literal["complete", "partial", "unknown"]
    costs: tuple[ComputedCost, ...]
    subtotals: tuple[CostSubtotal, ...]
    spans: tuple[ComputedSpan, ...]
    run_wall_seconds: Decimal | None
    limitations: tuple[str, ...]


class MeasurementsData(Model):
    evidence_id: str
    request_digest: Digest
    scope: Scope
    source_association: SourceAssociation
    summary: MeasurementSummary


class CapabilitiesResponse(ResponseModel):
    data: CapabilitiesData


class EvidencePage(ResponseModel):
    data: EvidencePageData


class MeasurementsResponse(ResponseModel):
    data: MeasurementsData


def check_receipt(receipt, project_id, evidence_id=None):
    encoded = json.dumps(
        [receipt.scope.organization_id, project_id, receipt.operation_id],
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode()
    expected = "mev_" + hashlib.sha256(encoded).hexdigest()
    roles = [item.role for item in receipt.objects]
    return verified(
        receipt,
        receipt.scope.project_id == project_id
        and receipt.evidence_id == expected
        and (evidence_id is None or evidence_id == expected)
        and "report" in roles
        and len(set(roles)) == len(roles)
        and all(
            item.locator.scope == receipt.scope
            and item.locator.object.digest_sha256 == item.locator.digest_sha256
            for item in receipt.objects
        ),
        "Mnemosyne returned different evidence identity",
    )


def check_capabilities(result, project_id):
    data = result.data
    return verified(
        result,
        data.scope.project_id == project_id
        and data.can_read
        and (data.intake == "available") == data.can_publish,
        "Mnemosyne returned different evidence capabilities",
    )


def check_page(result, project_id):
    ids = []
    for item in result.data.items:
        check_receipt(item, project_id)
        ids.append(item.evidence_id)
    return verified(
        result,
        result.data.scope.project_id == project_id
        and all(item.scope == result.data.scope for item in result.data.items)
        and ids == sorted(set(ids))
        and (result.data.next_cursor is None or (ids and result.data.next_cursor == ids[-1])),
        "Mnemosyne returned an invalid evidence page",
    )


def check_measurements(result, project_id, evidence_id):
    data = result.data
    source = data.source_association
    return verified(
        result,
        data.scope.project_id == project_id
        and data.evidence_id == evidence_id
        and data.summary.producer_run_id == source.producer_run_id
        and data.summary.attempt_id == source.attempt_id,
        "Mnemosyne returned different measurement identity",
    )


def page_query(after, limit):
    if type(limit) is not int or not 1 <= limit <= 100:
        raise ValueError("limit must be between 1 and 100")
    return {"limit": limit, **({"after": evidence_identifier(after)} if after is not None else {})}


def evidence_identifier(value):
    if not isinstance(value, str) or re.fullmatch(r"mev_[0-9a-f]{64}", value) is None:
        raise ValueError("evidence_id must be a Mnemosyne evidence identifier")
    return value


class EvaluationEvidence:
    def __init__(self, transport):
        self._transport = transport

    def capabilities(self, project_id, *, request=None):
        project_id = project_identity(project_id)
        return self._transport.request(
            operations.CAPABILITIES,
            CapabilitiesResponse,
            path=operations.CAPABILITIES.path_for(project_id=project_id),
            options=request,
            validate=lambda r: check_capabilities(r, project_id),
        )

    def list(self, project_id, *, after=None, limit=50, request=None):
        project_id = project_identity(project_id)
        return self._transport.request(
            operations.LIST,
            EvidencePage,
            path=operations.LIST.path_for(project_id=project_id),
            query=page_query(after, limit),
            options=request,
            validate=lambda r: check_page(r, project_id),
        )

    def get(self, project_id, evidence_id, *, request=None):
        project_id, evidence_id = project_identity(project_id), evidence_identifier(evidence_id)
        return self._transport.request(
            operations.GET,
            ReceiptResponse,
            path=operations.GET.path_for(project_id=project_id, identifier=evidence_id),
            options=request,
            validate=lambda r: (check_receipt(r.data, project_id, evidence_id), r)[1],
        )

    def measurements(self, project_id, evidence_id, *, request=None):
        project_id, evidence_id = project_identity(project_id), evidence_identifier(evidence_id)
        return self._transport.request(
            operations.MEASUREMENTS,
            MeasurementsResponse,
            path=operations.MEASUREMENTS.path_for(project_id=project_id, identifier=evidence_id),
            options=request,
            validate=lambda r: check_measurements(r, project_id, evidence_id),
        )


class AsyncEvaluationEvidence:
    def __init__(self, transport):
        self._transport = transport

    async def capabilities(self, project_id, *, request=None):
        project_id = project_identity(project_id)
        return await self._transport.request(
            operations.CAPABILITIES,
            CapabilitiesResponse,
            path=operations.CAPABILITIES.path_for(project_id=project_id),
            options=request,
            validate=lambda r: check_capabilities(r, project_id),
        )

    async def list(self, project_id, *, after=None, limit=50, request=None):
        project_id = project_identity(project_id)
        return await self._transport.request(
            operations.LIST,
            EvidencePage,
            path=operations.LIST.path_for(project_id=project_id),
            query=page_query(after, limit),
            options=request,
            validate=lambda r: check_page(r, project_id),
        )

    async def get(self, project_id, evidence_id, *, request=None):
        project_id, evidence_id = project_identity(project_id), evidence_identifier(evidence_id)
        return await self._transport.request(
            operations.GET,
            ReceiptResponse,
            path=operations.GET.path_for(project_id=project_id, identifier=evidence_id),
            options=request,
            validate=lambda r: (check_receipt(r.data, project_id, evidence_id), r)[1],
        )

    async def measurements(self, project_id, evidence_id, *, request=None):
        project_id, evidence_id = project_identity(project_id), evidence_identifier(evidence_id)
        return await self._transport.request(
            operations.MEASUREMENTS,
            MeasurementsResponse,
            path=operations.MEASUREMENTS.path_for(project_id=project_id, identifier=evidence_id),
            options=request,
            validate=lambda r: check_measurements(r, project_id, evidence_id),
        )
