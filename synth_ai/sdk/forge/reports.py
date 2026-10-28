"""Publish report versions and revoke inactive shares as the calling user.

SYN-5157. The backend admits the producer and forwards the caller's own API
key to Forge as the acting credential, so evidence custody and sharing never
exceed that user's grants. Callers without a raw API key are refused.
"""

from uuid import uuid4

from pydantic import TypeAdapter

from synth_ai.contracts import forge as operations
from synth_ai.core.parameters import identifier, verified
from synth_ai.options import MutationOptions

from .contracts.report_manifests import ReportVersion, version_digest
from .records import project_identity
from .types import (
    ReportPublicationResponse,
    ReportPublishIntent,
    ReportShareRevocationResponse,
    ReportShareRevokeIntent,
)


def _selected(mutation):
    selected = mutation or MutationOptions(idempotency_key=uuid4().hex)
    key = selected.identity.key if selected.identity is not None else selected.idempotency_key
    if key is None:
        key = uuid4().hex
        selected = selected.model_copy(update={"idempotency_key": key})
    return key, selected


def publish_intent(version, mutation):
    key, selected = _selected(mutation)
    version = TypeAdapter(ReportVersion).validate_python(version)
    return ReportPublishIntent(operation_id=key, version=version), selected


def revoke_intent(lineage_id, mutation):
    key, selected = _selected(mutation)
    return ReportShareRevokeIntent(operation_id=key, lineage_id=identifier(lineage_id)), selected


def check_publication(result, project_id, request):
    publication = result.publication
    revocation = publication.revocation
    return verified(
        result,
        publication.version_sha256 == version_digest(request.version)
        and (
            revocation is None
            or (
                revocation.scope.project_id == project_id
                and revocation.lineage_id == request.version.lineage_id
            )
        ),
        "Forge returned a different report publication",
    )


def check_revocation(result, project_id, request):
    revocation = result.revocation
    return verified(
        result,
        revocation.scope.project_id == project_id
        and revocation.lineage_id == request.lineage_id,
        "Forge returned a different share revocation report",
    )


class Reports:
    def __init__(self, transport):
        self._transport = transport

    def publish(self, project_id, version, *, mutation=None) -> ReportPublicationResponse:
        project_id = project_identity(project_id)
        request, selected = publish_intent(version, mutation)
        return self._transport.request(
            operations.REPORT_VERSION_PUBLISH,
            ReportPublicationResponse,
            path=operations.REPORT_VERSION_PUBLISH.path_for(project_id=project_id),
            body=request,
            options=selected,
            validate=lambda result: check_publication(result, project_id, request),
        )

    def revoke_inactive(
        self, project_id, lineage_id, *, mutation=None
    ) -> ReportShareRevocationResponse:
        project_id = project_identity(project_id)
        request, selected = revoke_intent(lineage_id, mutation)
        return self._transport.request(
            operations.REPORT_SHARES_REVOKE_INACTIVE,
            ReportShareRevocationResponse,
            path=operations.REPORT_SHARES_REVOKE_INACTIVE.path_for(project_id=project_id),
            body=request,
            options=selected,
            validate=lambda result: check_revocation(result, project_id, request),
        )


class AsyncReports:
    def __init__(self, transport):
        self._transport = transport

    async def publish(self, project_id, version, *, mutation=None) -> ReportPublicationResponse:
        project_id = project_identity(project_id)
        request, selected = publish_intent(version, mutation)
        return await self._transport.request(
            operations.REPORT_VERSION_PUBLISH,
            ReportPublicationResponse,
            path=operations.REPORT_VERSION_PUBLISH.path_for(project_id=project_id),
            body=request,
            options=selected,
            validate=lambda result: check_publication(result, project_id, request),
        )

    async def revoke_inactive(
        self, project_id, lineage_id, *, mutation=None
    ) -> ReportShareRevocationResponse:
        project_id = project_identity(project_id)
        request, selected = revoke_intent(lineage_id, mutation)
        return await self._transport.request(
            operations.REPORT_SHARES_REVOKE_INACTIVE,
            ReportShareRevocationResponse,
            path=operations.REPORT_SHARES_REVOKE_INACTIVE.path_for(project_id=project_id),
            body=request,
            options=selected,
            validate=lambda result: check_revocation(result, project_id, request),
        )
