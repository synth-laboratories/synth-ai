"""Publish report versions and manage their scoped shares as the calling user.

SYN-5157. The backend admits the producer and forwards the caller's own API
key to Forge as the acting credential, so evidence custody and sharing never
exceed that user's grants. Callers without a raw API key are refused.
"""

from datetime import datetime
from uuid import uuid4

from pydantic import TypeAdapter

from synth_ai.contracts import forge as operations
from synth_ai.contracts.forge import (
    REPORT_SHARE_GET,
    REPORT_SHARE_PUT,
    REPORT_SHARE_REVOKE,
    REPORT_SHARES_LIST,
)
from synth_ai.core.parameters import identifier, verified
from synth_ai.options import MutationOptions

from .contracts.report_manifests import ReportVersion, version_digest
from .records import project_identity
from .types import (
    ReportPublicationResponse,
    ReportPublishIntent,
    ReportShareIntent,
    ReportShareListResponse,
    ReportShareResponse,
    ReportShareRevocationResponse,
    ReportShareRevokeIntent,
    ReportShareRevokeOneIntent,
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
        revocation.scope.project_id == project_id and revocation.lineage_id == request.lineage_id,
        "Forge returned a different share revocation report",
    )


def share_intent(audience, invited, expires_at, mutation):
    """Named principals only: ``organization`` or ``invited`` members.

    There is no public/unlisted option (owner decision 2026-10-28). ``expires_at``
    must be timezone-aware; ``None`` means the share lasts until revoked.
    """
    if expires_at is not None and (
        not isinstance(expires_at, datetime) or expires_at.tzinfo is None
    ):
        raise ValueError("expires_at must be a timezone-aware datetime")
    key, selected = _selected(mutation)
    request = ReportShareIntent(
        operation_id=key,
        audience=audience,
        invited=tuple(invited),
        expires_at=expires_at,
    )
    return request, selected


def revoke_share_intent(mutation):
    key, selected = _selected(mutation)
    return ReportShareRevokeOneIntent(operation_id=key), selected


def _digest(value):
    if not isinstance(value, str) or len(value) != 64 or set(value) - set("0123456789abcdef"):
        raise ValueError("version digest must be 64 lowercase hex characters")
    return value


def check_share(result, project_id, digest):
    share = result.share
    return verified(
        result,
        share.scope.project_id == project_id and share.version_sha256 == digest,
        "Forge returned a different report share",
    )


def check_share_list(result, project_id, after, limit):
    digests = [share.version_sha256 for share in result.shares]
    return verified(
        result,
        len(digests) <= limit
        and digests == sorted(set(digests))
        and (after is None or all(digest > after for digest in digests))
        and all(share.scope.project_id == project_id for share in result.shares),
        "Forge returned an out-of-contract report share page",
    )


def _share_calls(project_id, digest, audience, invited, expires_at, mutation):
    project_id, digest = project_identity(project_id), _digest(digest)
    request, selected = share_intent(audience, invited, expires_at, mutation)
    return {
        "operation": REPORT_SHARE_PUT,
        "model": ReportShareResponse,
        "path": REPORT_SHARE_PUT.path_for(project_id=project_id, digest=digest),
        "body": request,
        "options": selected,
        "validate": lambda result: check_share(result, project_id, digest),
    }


def _get_call(project_id, digest, request):
    project_id, digest = project_identity(project_id), _digest(digest)
    return {
        "operation": REPORT_SHARE_GET,
        "model": ReportShareResponse,
        "path": REPORT_SHARE_GET.path_for(project_id=project_id, digest=digest),
        "options": request,
        "validate": lambda result: check_share(result, project_id, digest),
    }


def _list_call(project_id, after, limit, request):
    project_id = project_identity(project_id)
    if after is not None:
        after = _digest(after)
    if type(limit) is not int or not 1 <= limit <= 1000:
        raise ValueError("limit must be an integer from 1 to 1000")
    query = {"limit": limit} if after is None else {"after": after, "limit": limit}
    return {
        "operation": REPORT_SHARES_LIST,
        "model": ReportShareListResponse,
        "path": REPORT_SHARES_LIST.path_for(project_id=project_id),
        "query": query,
        "options": request,
        "validate": lambda result: check_share_list(result, project_id, after, limit),
    }


def _revoke_call(project_id, digest, mutation):
    project_id, digest = project_identity(project_id), _digest(digest)
    request, selected = revoke_share_intent(mutation)
    return {
        "operation": REPORT_SHARE_REVOKE,
        "model": ReportShareResponse,
        "path": REPORT_SHARE_REVOKE.path_for(project_id=project_id, digest=digest),
        "body": request,
        "options": selected,
        "validate": lambda result: check_share(result, project_id, digest),
    }


def _send(transport, call):
    call = dict(call)
    return transport.request(call.pop("operation"), call.pop("model"), **call)


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

    def share(
        self,
        project_id,
        digest,
        *,
        audience,
        invited=(),
        expires_at=None,
        mutation=None,
    ) -> ReportShareResponse:
        """Share one exact version with the organization or invited members."""
        return _send(
            self._transport,
            _share_calls(project_id, digest, audience, invited, expires_at, mutation),
        )

    def get_share(self, project_id, digest, *, request=None) -> ReportShareResponse:
        return _send(self._transport, _get_call(project_id, digest, request))

    def list_shares(
        self, project_id, *, after=None, limit=100, request=None
    ) -> ReportShareListResponse:
        return _send(self._transport, _list_call(project_id, after, limit, request))

    def revoke_share(self, project_id, digest, *, mutation=None) -> ReportShareResponse:
        return _send(self._transport, _revoke_call(project_id, digest, mutation))


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

    async def share(
        self,
        project_id,
        digest,
        *,
        audience,
        invited=(),
        expires_at=None,
        mutation=None,
    ) -> ReportShareResponse:
        """Share one exact version with the organization or invited members."""
        return await _send(
            self._transport,
            _share_calls(project_id, digest, audience, invited, expires_at, mutation),
        )

    async def get_share(self, project_id, digest, *, request=None) -> ReportShareResponse:
        return await _send(self._transport, _get_call(project_id, digest, request))

    async def list_shares(
        self, project_id, *, after=None, limit=100, request=None
    ) -> ReportShareListResponse:
        return await _send(self._transport, _list_call(project_id, after, limit, request))

    async def revoke_share(self, project_id, digest, *, mutation=None) -> ReportShareResponse:
        return await _send(self._transport, _revoke_call(project_id, digest, mutation))
