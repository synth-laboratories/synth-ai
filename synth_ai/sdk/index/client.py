"""Unreleased Index transport adapters; no retry loop or local search fallback.

See sibling docs/drafts/synth-index-api-design-2026-09-12.md.
Backend owns authorization and usage receipts. The injected transport owns its
lifetime; these adapters do not discover credentials or create extra clients.
"""

from collections.abc import Sequence
from hashlib import sha256
from uuid import uuid4

from synth_ai.core.http.async_transport import AsyncHttpTransport
from synth_ai.core.http.transport import HttpTransport

from .artifacts import ArtifactPublicationResponse
from .contracts import ContributionReference
from .contributions import ContributionDraft, ContributionUploadPrepared, ContributionUploadSpec
from .search import (
    ContentsResult,
    ContentsSpec,
    SearchContent,
    SearchFilters,
    SearchResult,
    SearchScope,
    SearchSpec,
)
from .submission import ContributionSubmission, ContributionSubmitSpec


def _search_spec(
    spec: SearchSpec | None,
    query: str | None,
    scope: SearchScope | None,
    filters: SearchFilters | None,
    max_results: int | None,
) -> SearchSpec:
    if spec is not None:
        if any(value is not None for value in (query, scope, filters, max_results)):
            raise ValueError("Pass either SearchSpec or search keyword arguments, not both")
        return spec
    if query is None:
        raise ValueError("Index search requires a query or SearchSpec")
    return SearchSpec(
        query=query,
        scope=scope if scope is not None else SearchScope(),
        filters=filters if filters is not None else SearchFilters(),
        content=SearchContent(max_results=5 if max_results is None else max_results),
    )


def _contents_spec(
    spec: ContentsSpec | None,
    references: Sequence[ContributionReference] | None,
    search_id: str | None,
    max_bytes: int | None,
) -> ContentsSpec:
    if spec is not None:
        if any(value is not None for value in (references, search_id, max_bytes)):
            raise ValueError("Pass either ContentsSpec or contents keyword arguments, not both")
        return spec
    if references is None:
        raise ValueError("Index contents requires references or ContentsSpec")
    return ContentsSpec(
        references=tuple(references),
        search_id=search_id,
        max_bytes=65_536 if max_bytes is None else max_bytes,
    )


def _search_result(payload: object, spec: SearchSpec) -> SearchResult:
    result = SearchResult.model_validate(payload)
    if result.usage.billing_scope != spec.scope.visibility:
        raise ValueError("Index response billing scope does not match the request")
    if len(result.results) > spec.content.max_results or any(
        len(hit.highlights) > spec.content.max_excerpts_per_result for hit in result.results
    ):
        raise ValueError("Index response exceeds requested result or excerpt bounds")
    return result


def _contents_result(payload: object, spec: ContentsSpec) -> ContentsResult:
    result = ContentsResult.model_validate(payload)
    expected = {(item.contribution_id, item.revision_id) for item in spec.references}
    actual = {(item.reference.contribution_id, item.reference.revision_id) for item in result.items}
    if actual != expected:
        raise ValueError("Index contents response does not match requested revisions")
    if sum(len((item.text or "").encode("utf-8")) for item in result.items) > spec.max_bytes:
        raise ValueError("Index contents response exceeds requested byte bound")
    return result


def _search_headers(idempotency_key: str | None) -> dict[str, str]:
    key = str(uuid4()) if idempotency_key is None else idempotency_key
    if not 1 <= len(key) <= 128 or not all(
        character.isascii() and (character.isalnum() or character in "_.-") for character in key
    ):
        raise ValueError("Index idempotency key must be 1–128 safe ASCII characters")
    return {"Idempotency-Key": key}


def _upload_path(draft: ContributionDraft, spec: ContributionUploadSpec) -> str:
    reference = draft.reference
    if (spec.package.contribution_id, spec.package.revision_id) != (
        reference.contribution_id,
        reference.revision_id,
    ):
        raise ValueError("Upload package must match the server-issued draft")
    return f"/api/v1/index/contributions/{reference.contribution_id}/revisions/{reference.revision_id}/upload"


def _upload_result(
    payload: object, draft: ContributionDraft, spec: ContributionUploadSpec
) -> ContributionUploadPrepared:
    result = ContributionUploadPrepared.model_validate(payload)
    if (
        result.transfer.publication_id != str(spec.publication_id)
        or result.transfer.collection_id != str(draft.collection_id)
        or result.transfer.revision != 1
    ):
        raise ValueError("Upload transfer does not match the requested draft publication")
    if type(spec.package).model_validate_json(result.descriptor_json) != spec.package:
        raise ValueError("Upload descriptor differs from the submitted package")
    expected = {
        asset.object.logical_path: asset.object.digest_sha256 for asset in spec.package.assets
    }
    expected["contribution.json"] = sha256(result.descriptor_json.encode("utf-8")).hexdigest()
    actual = {
        target.logical_path: target.digest_sha256 for target in result.transfer.upload_targets
    }
    # Artifact prepare omits already-present objects on resumable uploads.
    # Finalize, not an exact target count, proves the complete object set exists.
    if len(actual) != len(result.transfer.upload_targets) or any(
        expected.get(path) != digest for path, digest in actual.items()
    ):
        raise ValueError("Upload targets do not match declared contribution objects")
    return result


def _finalize_path(draft: ContributionDraft, prepared: ContributionUploadPrepared) -> str:
    if (
        prepared.transfer.collection_id != str(draft.collection_id)
        or prepared.transfer.revision != 1
    ):
        raise ValueError("Finalization transfer does not match draft collection")
    reference = draft.reference
    return f"/api/v1/index/contributions/{reference.contribution_id}/revisions/{reference.revision_id}/finalize"


def _finalize_result(
    payload: object, prepared: ContributionUploadPrepared
) -> ArtifactPublicationResponse:
    result = ArtifactPublicationResponse.model_validate(payload)
    transfer = prepared.transfer
    if (
        result.publication_id != transfer.publication_id
        or result.collection_id != transfer.collection_id
        or result.revision != transfer.revision
        or result.manifest_digest != transfer.manifest_digest
        or result.status != "committed"
    ):
        raise ValueError("Finalization response does not match committed prepared publication")
    return result


class ContentsAPI:
    def __init__(self, transport: HttpTransport) -> None:
        self._transport = transport

    def retrieve(
        self,
        spec: ContentsSpec | None = None,
        *,
        references: Sequence[ContributionReference] | None = None,
        search_id: str | None = None,
        max_bytes: int | None = None,
    ) -> ContentsResult:
        """Retrieve exact revision contents under current backend authorization."""
        spec = _contents_spec(spec, references, search_id, max_bytes)
        payload = self._transport.request_json(
            "POST",
            "/api/v1/index/contents",
            json_body=spec.model_dump(mode="json"),
            operation_id="index.contents.retrieve",
        )
        return _contents_result(payload, spec)


class ContributionsAPI:
    def __init__(self, transport: HttpTransport) -> None:
        self._transport = transport

    def finalize(
        self, draft: ContributionDraft, prepared: ContributionUploadPrepared
    ) -> ArtifactPublicationResponse:
        """Verify uploaded bytes through the backend; does not submit or publish."""
        payload = self._transport.request_json(
            "POST",
            _finalize_path(draft, prepared),
            json_body={"publication_id": prepared.transfer.publication_id},
            operation_id="index.contributions.upload.finalize",
        )
        return _finalize_result(payload, prepared)

    def submit(
        self, reference: ContributionReference, spec: ContributionSubmitSpec
    ) -> ContributionSubmission:
        """Submit finalized bytes for review; does not approve or publish research."""
        payload = self._transport.request_json(
            "POST",
            f"/api/v1/index/contributions/{reference.contribution_id}/revisions/{reference.revision_id}/submit",
            json_body=spec.model_dump(mode="json"),
            operation_id="index.contributions.submit",
        )
        result = ContributionSubmission.model_validate(payload)
        if result.reference != reference:
            raise ValueError("Submission response does not match requested revision")
        return result

    def prepare_upload(
        self, draft: ContributionDraft, spec: ContributionUploadSpec
    ) -> ContributionUploadPrepared:
        """Prepare transfer instructions only; reuse the publication ID for retries.

        No local files are read and no signed URL is contacted. Finalization,
        submission, and public release remain separate deliberate operations.
        """
        payload = self._transport.request_json(
            "POST",
            _upload_path(draft, spec),
            json_body=spec.model_dump(mode="json"),
            operation_id="index.contributions.upload.prepare",
        )
        return _upload_result(payload, draft, spec)

    def create(self, *, idempotency_key: str) -> ContributionDraft:
        """Create a private draft. Persist and reuse the key after uncertain failures."""
        if not isinstance(idempotency_key, str):
            raise ValueError("Draft creation requires an explicit idempotency key")
        payload = self._transport.request_json(
            "POST",
            "/api/v1/index/contributions",
            json_body={},
            headers=_search_headers(idempotency_key),
            operation_id="index.contributions.create",
        )
        return ContributionDraft.model_validate(payload)


class AsyncContributionsAPI:
    def __init__(self, transport: AsyncHttpTransport) -> None:
        self._transport = transport

    async def finalize(
        self, draft: ContributionDraft, prepared: ContributionUploadPrepared
    ) -> ArtifactPublicationResponse:
        """Finalize the prepared publication without changing its release audience."""
        payload = await self._transport.request_json(
            "POST",
            _finalize_path(draft, prepared),
            json_body={"publication_id": prepared.transfer.publication_id},
            operation_id="index.contributions.upload.finalize",
        )
        return _finalize_result(payload, prepared)

    async def submit(
        self, reference: ContributionReference, spec: ContributionSubmitSpec
    ) -> ContributionSubmission:
        """Submit finalized bytes; retry the same publication after uncertain failures."""
        payload = await self._transport.request_json(
            "POST",
            f"/api/v1/index/contributions/{reference.contribution_id}/revisions/{reference.revision_id}/submit",
            json_body=spec.model_dump(mode="json"),
            operation_id="index.contributions.submit",
        )
        result = ContributionSubmission.model_validate(payload)
        if result.reference != reference:
            raise ValueError("Submission response does not match requested revision")
        return result

    async def prepare_upload(
        self, draft: ContributionDraft, spec: ContributionUploadSpec
    ) -> ContributionUploadPrepared:
        """Prepare only; no file reads, byte transfer, finalization, or publication."""
        payload = await self._transport.request_json(
            "POST",
            _upload_path(draft, spec),
            json_body=spec.model_dump(mode="json"),
            operation_id="index.contributions.upload.prepare",
        )
        return _upload_result(payload, draft, spec)

    async def create(self, *, idempotency_key: str) -> ContributionDraft:
        """Create a private draft without submitting, publishing, or awarding credits."""
        if not isinstance(idempotency_key, str):
            raise ValueError("Draft creation requires an explicit idempotency key")
        payload = await self._transport.request_json(
            "POST",
            "/api/v1/index/contributions",
            json_body={},
            headers=_search_headers(idempotency_key),
            operation_id="index.contributions.create",
        )
        return ContributionDraft.model_validate(payload)


class IndexAPI:
    def __init__(self, transport: HttpTransport) -> None:
        self._transport = transport
        self.contents = ContentsAPI(transport)
        self.contributions = ContributionsAPI(transport)

    def search(
        self,
        spec: SearchSpec | None = None,
        *,
        query: str | None = None,
        scope: SearchScope | None = None,
        filters: SearchFilters | None = None,
        max_results: int | None = None,
        idempotency_key: str | None = None,
    ) -> SearchResult:
        """Execute one fast search; reuse an explicit key for application-level retries."""
        spec = _search_spec(spec, query, scope, filters, max_results)
        payload = self._transport.request_json(
            "POST",
            "/api/v1/index/search",
            json_body=spec.model_dump(mode="json"),
            headers=_search_headers(idempotency_key),
            operation_id="index.search",
        )
        return _search_result(payload, spec)


class AsyncContentsAPI:
    def __init__(self, transport: AsyncHttpTransport) -> None:
        self._transport = transport

    async def retrieve(
        self,
        spec: ContentsSpec | None = None,
        *,
        references: Sequence[ContributionReference] | None = None,
        search_id: str | None = None,
        max_bytes: int | None = None,
    ) -> ContentsResult:
        """Retrieve exact revision contents without blocking the caller's event loop."""
        spec = _contents_spec(spec, references, search_id, max_bytes)
        payload = await self._transport.request_json(
            "POST",
            "/api/v1/index/contents",
            json_body=spec.model_dump(mode="json"),
            operation_id="index.contents.retrieve",
        )
        return _contents_result(payload, spec)


class AsyncIndexAPI:
    def __init__(self, transport: AsyncHttpTransport) -> None:
        self._transport = transport
        self.contents = AsyncContentsAPI(transport)
        self.contributions = AsyncContributionsAPI(transport)

    async def search(
        self,
        spec: SearchSpec | None = None,
        *,
        query: str | None = None,
        scope: SearchScope | None = None,
        filters: SearchFilters | None = None,
        max_results: int | None = None,
        idempotency_key: str | None = None,
    ) -> SearchResult:
        """Execute one fast search; failures propagate, never become empty results."""
        spec = _search_spec(spec, query, scope, filters, max_results)
        payload = await self._transport.request_json(
            "POST",
            "/api/v1/index/search",
            json_body=spec.model_dump(mode="json"),
            headers=_search_headers(idempotency_key),
            operation_id="index.search",
        )
        return _search_result(payload, spec)
