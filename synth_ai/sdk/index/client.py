"""Unreleased Index transport adapters; no retry loop or local search fallback.

See sibling docs/drafts/synth-index-api-design-2026-09-12.md.
Backend owns authorization and usage receipts. The injected transport owns its
lifetime; these adapters do not discover credentials or create extra clients.
Every operation is declared once in ``OPERATIONS`` (method, path template) and
executed identically by the sync and async clients; parity tests read it.
"""

import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from hashlib import sha256
from typing import Any
from uuid import uuid4

from synth_ai.core.http.async_transport import AsyncHttpTransport
from synth_ai.core.http.transport import HttpTransport

from .artifacts import ArtifactPublicationResponse
from .catalog import CollectionList, IndexCapabilities, IndexUsageSummary, RewardsSummary, TagList
from .contracts import ContributionReference
from .contributions import ContributionDraft, ContributionUploadPrepared, ContributionUploadSpec
from .lifecycle import (
    Assessment,
    AssessmentCreateSpec,
    AssessmentList,
    Contribution,
    ContributionRevision,
    ContributionRevisionList,
    Publication,
    PublishSpec,
    RevisionCreateSpec,
    WithdrawSpec,
)
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
from .transfer import upload_bytes, upload_bytes_sync

_PREFIX = "/api/v1/index"
_REVISION = "/contributions/{contribution_id}/revisions/{revision_id}"

# operation_id -> (HTTP method, path template). Single SDK source of truth.
OPERATIONS: Mapping[str, tuple[str, str]] = {
    "index.capabilities": ("GET", f"{_PREFIX}/capabilities"),
    "index.search": ("POST", f"{_PREFIX}/search"),
    "index.contents.retrieve": ("POST", f"{_PREFIX}/contents"),
    "index.contributions.create": ("POST", f"{_PREFIX}/contributions"),
    "index.contributions.retrieve": ("GET", f"{_PREFIX}/contributions/{{contribution_id}}"),
    "index.contributions.revisions.create": (
        "POST",
        f"{_PREFIX}/contributions/{{contribution_id}}/revisions",
    ),
    "index.contributions.revisions.retrieve": ("GET", f"{_PREFIX}{_REVISION}"),
    "index.contributions.upload.prepare": ("POST", f"{_PREFIX}{_REVISION}/upload"),
    "index.contributions.upload.finalize": ("POST", f"{_PREFIX}{_REVISION}/finalize"),
    "index.contributions.submit": ("POST", f"{_PREFIX}{_REVISION}/submit"),
    "index.contributions.assessments.list": ("GET", f"{_PREFIX}{_REVISION}/assessments"),
    "index.contributions.assessments.create": ("POST", f"{_PREFIX}{_REVISION}/review"),
    "index.contributions.publish": ("POST", f"{_PREFIX}/contributions/{{contribution_id}}/publish"),
    "index.contributions.withdraw": (
        "POST",
        f"{_PREFIX}/contributions/{{contribution_id}}/withdraw",
    ),
    "index.tags.list": ("GET", f"{_PREFIX}/tags"),
    "index.collections.list": ("GET", f"{_PREFIX}/collections"),
    "index.me.contributions": ("GET", f"{_PREFIX}/me/contributions"),
    "index.me.usage": ("GET", f"{_PREFIX}/me/usage"),
    "index.me.rewards": ("GET", f"{_PREFIX}/me/rewards"),
}

_IDENTIFIER = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9_.-]{0,127}$")


@dataclass(frozen=True, slots=True)
class _Call:
    operation_id: str
    parse: Callable[[object], Any]
    path_parameters: Mapping[str, str] | None = None
    json_body: dict[str, Any] | None = None
    headers: dict[str, str] | None = None

    def request(self) -> tuple[str, str, dict[str, Any]]:
        method, template = OPERATIONS[self.operation_id]
        for name, value in (self.path_parameters or {}).items():
            if not _IDENTIFIER.fullmatch(value):
                raise ValueError(f"{name} must be an Index identifier")
        kwargs: dict[str, Any] = {"operation_id": self.operation_id}
        if self.json_body is not None:
            kwargs["json_body"] = self.json_body
        if self.headers is not None:
            kwargs["headers"] = self.headers
        return method, template.format(**(self.path_parameters or {})), kwargs


def _run(transport: HttpTransport, call: _Call) -> Any:
    method, path, kwargs = call.request()
    return call.parse(transport.request_json(method, path, **kwargs))


async def _arun(transport: AsyncHttpTransport, call: _Call) -> Any:
    method, path, kwargs = call.request()
    return call.parse(await transport.request_json(method, path, **kwargs))


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


def _explicit_key(idempotency_key: str, action: str) -> dict[str, str]:
    if not isinstance(idempotency_key, str):
        raise ValueError(f"{action} requires an explicit idempotency key")
    return _search_headers(idempotency_key)


def _revision_parameters(reference: ContributionReference) -> dict[str, str]:
    return {"contribution_id": reference.contribution_id, "revision_id": reference.revision_id}


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


def _matching(expected: ContributionReference, message: str) -> Callable[[object], Any]:
    def parse(payload: object) -> ContributionSubmission:
        result = ContributionSubmission.model_validate(payload)
        if result.reference != expected:
            raise ValueError(message)
        return result

    return parse


# ---- call builders shared by sync and async clients -------------------------


def _capabilities_call() -> _Call:
    return _Call("index.capabilities", IndexCapabilities.model_validate)


def _search_call(spec: SearchSpec, idempotency_key: str | None) -> _Call:
    return _Call(
        "index.search",
        lambda payload: _search_result(payload, spec),
        json_body=spec.model_dump(mode="json"),
        headers=_search_headers(idempotency_key),
    )


def _contents_call(spec: ContentsSpec) -> _Call:
    return _Call(
        "index.contents.retrieve",
        lambda payload: _contents_result(payload, spec),
        json_body=spec.model_dump(mode="json"),
    )


def _create_call(idempotency_key: str) -> _Call:
    return _Call(
        "index.contributions.create",
        ContributionDraft.model_validate,
        json_body={},
        headers=_explicit_key(idempotency_key, "Draft creation"),
    )


def _retrieve_call(contribution_id: str) -> _Call:
    return _Call(
        "index.contributions.retrieve",
        Contribution.model_validate,
        path_parameters={"contribution_id": contribution_id},
    )


def _prepare_call(draft: ContributionDraft, spec: ContributionUploadSpec) -> _Call:
    reference = draft.reference
    if (spec.package.contribution_id, spec.package.revision_id) != (
        reference.contribution_id,
        reference.revision_id,
    ):
        raise ValueError("Upload package must match the server-issued draft")
    return _Call(
        "index.contributions.upload.prepare",
        lambda payload: _upload_result(payload, draft, spec),
        path_parameters=_revision_parameters(reference),
        json_body=spec.model_dump(mode="json"),
    )


def _finalize_call(draft: ContributionDraft, prepared: ContributionUploadPrepared) -> _Call:
    if (
        prepared.transfer.collection_id != str(draft.collection_id)
        or prepared.transfer.revision != 1
    ):
        raise ValueError("Finalization transfer does not match draft collection")
    return _Call(
        "index.contributions.upload.finalize",
        lambda payload: _finalize_result(payload, prepared),
        path_parameters=_revision_parameters(draft.reference),
        json_body={"publication_id": prepared.transfer.publication_id},
    )


def _submit_call(reference: ContributionReference, spec: ContributionSubmitSpec) -> _Call:
    return _Call(
        "index.contributions.submit",
        _matching(reference, "Submission response does not match requested revision"),
        path_parameters=_revision_parameters(reference),
        json_body=spec.model_dump(mode="json"),
    )


def _publication_call(
    operation_id: str, contribution_id: str, spec: PublishSpec | WithdrawSpec
) -> _Call:
    def parse(payload: object) -> Publication:
        result = Publication.model_validate(payload)
        if result.contribution_id != contribution_id:
            raise ValueError("Publication response does not match requested Contribution")
        return result

    return _Call(
        operation_id,
        parse,
        path_parameters={"contribution_id": contribution_id},
        json_body=spec.model_dump(mode="json"),
    )


def _revision_create_call(
    contribution_id: str, spec: RevisionCreateSpec, idempotency_key: str
) -> _Call:
    def parse(payload: object) -> ContributionDraft:
        draft = ContributionDraft.model_validate(payload)
        if draft.reference.contribution_id != contribution_id:
            raise ValueError("Revision draft does not match requested Contribution")
        return draft

    return _Call(
        "index.contributions.revisions.create",
        parse,
        path_parameters={"contribution_id": contribution_id},
        json_body=spec.model_dump(mode="json"),
        headers=_explicit_key(idempotency_key, "Revision creation"),
    )


def _revision_retrieve_call(reference: ContributionReference) -> _Call:
    def parse(payload: object) -> ContributionRevision:
        revision = ContributionRevision.model_validate(payload)
        if revision.reference != reference:
            raise ValueError("Revision response does not match requested revision")
        return revision

    return _Call(
        "index.contributions.revisions.retrieve",
        parse,
        path_parameters=_revision_parameters(reference),
    )


def _assessments_list_call(reference: ContributionReference) -> _Call:
    return _Call(
        "index.contributions.assessments.list",
        AssessmentList.model_validate,
        path_parameters=_revision_parameters(reference),
    )


def _assessment_create_call(reference: ContributionReference, spec: AssessmentCreateSpec) -> _Call:
    def parse(payload: object) -> Assessment:
        assessment = Assessment.model_validate(payload)
        if assessment.reference != reference or assessment.manifest_digest != spec.manifest_digest:
            raise ValueError("Assessment does not bind the requested sealed revision")
        return assessment

    return _Call(
        "index.contributions.assessments.create",
        parse,
        path_parameters=_revision_parameters(reference),
        json_body=spec.model_dump(mode="json"),
    )


# ---- sync client -------------------------------------------------------------


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
        return _run(
            self._transport, _contents_call(_contents_spec(spec, references, search_id, max_bytes))
        )


class RevisionsAPI:
    def __init__(self, transport: HttpTransport) -> None:
        self._transport = transport

    def create(
        self, contribution_id: str, spec: RevisionCreateSpec, *, idempotency_key: str
    ) -> ContributionDraft:
        """Open a private revision draft after changes were requested or for an update."""
        return _run(self._transport, _revision_create_call(contribution_id, spec, idempotency_key))

    def retrieve(self, reference: ContributionReference) -> ContributionRevision:
        """Read one exact revision's status and assessments under current access."""
        return _run(self._transport, _revision_retrieve_call(reference))


class AssessmentsAPI:
    def __init__(self, transport: HttpTransport) -> None:
        self._transport = transport

    def list(self, reference: ContributionReference) -> AssessmentList:
        return _run(self._transport, _assessments_list_call(reference))

    def create(self, reference: ContributionReference, spec: AssessmentCreateSpec) -> Assessment:
        """Record a reviewer decision; requires reviewer authority, never self-approval."""
        return _run(self._transport, _assessment_create_call(reference, spec))


class ContributionsAPI:
    def __init__(self, transport: HttpTransport) -> None:
        self._transport = transport
        self.revisions = RevisionsAPI(transport)
        self.assessments = AssessmentsAPI(transport)

    def create(self, *, idempotency_key: str) -> ContributionDraft:
        """Create a private draft. Persist and reuse the key after uncertain failures."""
        return _run(self._transport, _create_call(idempotency_key))

    def retrieve(self, contribution_id: str) -> Contribution:
        """Read a Contribution's current published revision and lifecycle status."""
        return _run(self._transport, _retrieve_call(contribution_id))

    def prepare_upload(
        self, draft: ContributionDraft, spec: ContributionUploadSpec
    ) -> ContributionUploadPrepared:
        """Prepare transfer instructions only; reuse the publication ID for retries.

        No local files are read and no signed URL is contacted. Finalization,
        submission, and public release remain separate deliberate operations.
        """
        return _run(self._transport, _prepare_call(draft, spec))

    def upload(self, prepared: ContributionUploadPrepared, content: Mapping[str, bytes]) -> None:
        """Transfer exactly the declared asset bytes to prepared storage targets."""
        upload_bytes_sync(prepared, content)

    def finalize(
        self, draft: ContributionDraft, prepared: ContributionUploadPrepared
    ) -> ArtifactPublicationResponse:
        """Verify uploaded bytes through the backend; does not submit or publish."""
        return _run(self._transport, _finalize_call(draft, prepared))

    def submit(
        self, reference: ContributionReference, spec: ContributionSubmitSpec
    ) -> ContributionSubmission:
        """Submit finalized bytes for review; does not approve or publish research."""
        return _run(self._transport, _submit_call(reference, spec))

    def publish(self, contribution_id: str, spec: PublishSpec) -> Publication:
        """Publish an approved revision; requires explicit publication authority."""
        return _run(
            self._transport,
            _publication_call("index.contributions.publish", contribution_id, spec),
        )

    def withdraw(self, contribution_id: str, spec: WithdrawSpec) -> Publication:
        """Withdraw from search and new reads; prior downloads cannot be recalled."""
        return _run(
            self._transport,
            _publication_call("index.contributions.withdraw", contribution_id, spec),
        )


class TagsAPI:
    def __init__(self, transport: HttpTransport) -> None:
        self._transport = transport

    def list(self) -> TagList:
        return _run(self._transport, _Call("index.tags.list", TagList.model_validate))


class CollectionsAPI:
    def __init__(self, transport: HttpTransport) -> None:
        self._transport = transport

    def list(self) -> CollectionList:
        return _run(self._transport, _Call("index.collections.list", CollectionList.model_validate))


class AccountAPI:
    """Authenticated caller's own Contributions, Index usage and awarded credits."""

    def __init__(self, transport: HttpTransport) -> None:
        self._transport = transport

    def contributions(self) -> ContributionRevisionList:
        return _run(
            self._transport,
            _Call("index.me.contributions", ContributionRevisionList.model_validate),
        )

    def usage(self) -> IndexUsageSummary:
        return _run(self._transport, _Call("index.me.usage", IndexUsageSummary.model_validate))

    def rewards(self) -> RewardsSummary:
        return _run(self._transport, _Call("index.me.rewards", RewardsSummary.model_validate))


class IndexAPI:
    def __init__(self, transport: HttpTransport) -> None:
        self._transport = transport
        self.contents = ContentsAPI(transport)
        self.contributions = ContributionsAPI(transport)
        self.tags = TagsAPI(transport)
        self.collections = CollectionsAPI(transport)
        self.account = AccountAPI(transport)

    def capabilities(self) -> IndexCapabilities:
        """Discover supported modes, scopes, limits and lifecycle operations."""
        return _run(self._transport, _capabilities_call())

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
        return _run(self._transport, _search_call(spec, idempotency_key))


# ---- async client ------------------------------------------------------------


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
        return await _arun(
            self._transport, _contents_call(_contents_spec(spec, references, search_id, max_bytes))
        )


class AsyncRevisionsAPI:
    def __init__(self, transport: AsyncHttpTransport) -> None:
        self._transport = transport

    async def create(
        self, contribution_id: str, spec: RevisionCreateSpec, *, idempotency_key: str
    ) -> ContributionDraft:
        return await _arun(
            self._transport, _revision_create_call(contribution_id, spec, idempotency_key)
        )

    async def retrieve(self, reference: ContributionReference) -> ContributionRevision:
        return await _arun(self._transport, _revision_retrieve_call(reference))


class AsyncAssessmentsAPI:
    def __init__(self, transport: AsyncHttpTransport) -> None:
        self._transport = transport

    async def list(self, reference: ContributionReference) -> AssessmentList:
        return await _arun(self._transport, _assessments_list_call(reference))

    async def create(
        self, reference: ContributionReference, spec: AssessmentCreateSpec
    ) -> Assessment:
        return await _arun(self._transport, _assessment_create_call(reference, spec))


class AsyncContributionsAPI:
    def __init__(self, transport: AsyncHttpTransport) -> None:
        self._transport = transport
        self.revisions = AsyncRevisionsAPI(transport)
        self.assessments = AsyncAssessmentsAPI(transport)

    async def create(self, *, idempotency_key: str) -> ContributionDraft:
        """Create a private draft without submitting, publishing, or awarding credits."""
        return await _arun(self._transport, _create_call(idempotency_key))

    async def retrieve(self, contribution_id: str) -> Contribution:
        return await _arun(self._transport, _retrieve_call(contribution_id))

    async def prepare_upload(
        self, draft: ContributionDraft, spec: ContributionUploadSpec
    ) -> ContributionUploadPrepared:
        """Prepare only; no file reads, byte transfer, finalization, or publication."""
        return await _arun(self._transport, _prepare_call(draft, spec))

    async def upload(
        self, prepared: ContributionUploadPrepared, content: Mapping[str, bytes]
    ) -> None:
        await upload_bytes(prepared, content)

    async def finalize(
        self, draft: ContributionDraft, prepared: ContributionUploadPrepared
    ) -> ArtifactPublicationResponse:
        """Finalize the prepared publication without changing its release audience."""
        return await _arun(self._transport, _finalize_call(draft, prepared))

    async def submit(
        self, reference: ContributionReference, spec: ContributionSubmitSpec
    ) -> ContributionSubmission:
        """Submit finalized bytes; retry the same publication after uncertain failures."""
        return await _arun(self._transport, _submit_call(reference, spec))

    async def publish(self, contribution_id: str, spec: PublishSpec) -> Publication:
        return await _arun(
            self._transport,
            _publication_call("index.contributions.publish", contribution_id, spec),
        )

    async def withdraw(self, contribution_id: str, spec: WithdrawSpec) -> Publication:
        return await _arun(
            self._transport,
            _publication_call("index.contributions.withdraw", contribution_id, spec),
        )


class AsyncTagsAPI:
    def __init__(self, transport: AsyncHttpTransport) -> None:
        self._transport = transport

    async def list(self) -> TagList:
        return await _arun(self._transport, _Call("index.tags.list", TagList.model_validate))


class AsyncCollectionsAPI:
    def __init__(self, transport: AsyncHttpTransport) -> None:
        self._transport = transport

    async def list(self) -> CollectionList:
        return await _arun(
            self._transport, _Call("index.collections.list", CollectionList.model_validate)
        )


class AsyncAccountAPI:
    def __init__(self, transport: AsyncHttpTransport) -> None:
        self._transport = transport

    async def contributions(self) -> ContributionRevisionList:
        return await _arun(
            self._transport,
            _Call("index.me.contributions", ContributionRevisionList.model_validate),
        )

    async def usage(self) -> IndexUsageSummary:
        return await _arun(
            self._transport, _Call("index.me.usage", IndexUsageSummary.model_validate)
        )

    async def rewards(self) -> RewardsSummary:
        return await _arun(
            self._transport, _Call("index.me.rewards", RewardsSummary.model_validate)
        )


class AsyncIndexAPI:
    def __init__(self, transport: AsyncHttpTransport) -> None:
        self._transport = transport
        self.contents = AsyncContentsAPI(transport)
        self.contributions = AsyncContributionsAPI(transport)
        self.tags = AsyncTagsAPI(transport)
        self.collections = AsyncCollectionsAPI(transport)
        self.account = AsyncAccountAPI(transport)

    async def capabilities(self) -> IndexCapabilities:
        return await _arun(self._transport, _capabilities_call())

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
        return await _arun(self._transport, _search_call(spec, idempotency_key))
