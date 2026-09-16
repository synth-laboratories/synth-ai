"""Index transport adapters; no local search fallback.

See sibling docs/drafts/synth-index-api-design-2026-09-12.md.
Backend owns authorization and usage receipts. The injected transport owns its
lifetime; these adapters do not discover credentials or create extra clients.
Every operation is declared once in ``OPERATIONS`` (method, path template), and
one resource tree serves both clients: the sync client runs calls on the sync
transport, the async client returns awaitables from the async transport.
"""

import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from hashlib import sha256
from typing import Any
from uuid import uuid4

from pydantic import BaseModel

from synth_ai.core.http.async_transport import AsyncHttpTransport
from synth_ai.core.http.transport import HttpTransport

from .artifacts import ArtifactPublicationFinalize, ArtifactPublicationResponse
from .catalog import (
    Capabilities,
    CollectionGrant,
    CollectionGrantRevoked,
    CollectionGrants,
    CollectionGrantSpec,
    Collections,
    ContestEntry,
    ContestEntrySpec,
    ContestReviewSpec,
    ContestScoreSpec,
    ContestSpec,
    ContestStatusSpec,
    ContestView,
    IndexUsageSummary,
    Leaderboard,
    MyRewards,
    ProfilePinsSpec,
    ProfileSpec,
    ProfileView,
    PromoCreditSummary,
    RewardAward,
    RewardAwardSpec,
    RewardReverseSpec,
    TagRegistry,
)
from .contracts import ContributionReference
from .contributions import (
    ContributionDraft,
    ContributionUploadPrepared,
    ContributionUploadSpec,
    CreateContributionSpec,
    ResearchDraftSpec,
    ResearchLookupView,
)
from .lifecycle import (
    Assessment,
    Assessments,
    ContributionView,
    MeView,
    MyContributions,
    PublicationSpec,
    PublicationStatus,
    ReviewList,
    ReviewSpec,
    RevisionCreateSpec,
    RevisionView,
    WithdrawalSpec,
)
from .search import (
    ContentsResult,
    ContentsSpec,
    PublicSearchResult,
    SearchContent,
    SearchFilters,
    SearchResult,
    SearchScope,
    SearchSpec,
)
from .submission import ContributionSubmission, ContributionSubmitSpec, RevisionStatus
from .transfer import upload_bytes, upload_bytes_sync

_P = "/api/v1/index"
_C = f"{_P}/contributions/{{contribution_id}}"
_R = f"{_C}/revisions/{{revision_id}}"
_K = f"{_P}/contests/{{contest_id}}"

# operation_id -> (HTTP method, path template). Mirrors backend operation IDs.
OPERATIONS: Mapping[str, tuple[str, str]] = {
    "index.capabilities": ("GET", f"{_P}/capabilities"),
    "index.search": ("POST", f"{_P}/search"),
    "index.contents.retrieve": ("POST", f"{_P}/contents"),
    "index.contributions.create": ("POST", f"{_P}/contributions"),
    "index.contributions.research.create": ("POST", f"{_P}/contributions/research"),
    "index.contributions.research.lookup": ("POST", f"{_P}/contributions/research/lookup"),
    "index.contributions.retrieve": ("GET", _C),
    "index.contributions.publication.create": ("POST", f"{_C}/publication"),
    "index.contributions.withdrawal.create": ("POST", f"{_C}/withdrawal"),
    "index.contributions.revisions.create": ("POST", f"{_C}/revisions"),
    "index.contributions.revisions.retrieve": ("GET", _R),
    "index.contributions.assessments.list": ("GET", f"{_R}/assessments"),
    "index.contributions.assets.retrieve": ("GET", f"{_R}/assets/{{asset_id}}"),
    "index.contributions.upload.prepare": ("POST", f"{_R}/upload"),
    "index.contributions.upload.finalize": ("POST", f"{_R}/finalize"),
    "index.contributions.submit": ("POST", f"{_R}/submit"),
    "index.contributions.reviews.create": ("POST", f"{_R}/reviews"),
    "index.reviews.list": ("GET", f"{_P}/reviews"),
    "index.tags.list": ("GET", f"{_P}/tags"),
    "index.collections.list": ("GET", f"{_P}/collections"),
    "index.collections.grants.list": ("GET", f"{_P}/collections/{{collection_id}}/grants"),
    "index.collections.grants.create": ("POST", f"{_P}/collections/{{collection_id}}/grants"),
    "index.collections.grants.revoke": (
        "DELETE",
        f"{_P}/collections/{{collection_id}}/grants/{{grant_id}}",
    ),
    "index.me.retrieve": ("GET", f"{_P}/me"),
    "index.me.contributions.list": ("GET", f"{_P}/me/contributions"),
    "index.me.usage": ("GET", f"{_P}/me/usage"),
    "index.me.promo_credit": ("GET", f"{_P}/me/promo-credit"),
    "index.me.rewards.list": ("GET", f"{_P}/me/rewards"),
    "index.me.profile.update": ("PUT", f"{_P}/me/profile"),
    "index.me.profile.pins.update": ("PUT", f"{_P}/me/profile/pins"),
    "index.profiles.retrieve": ("GET", f"{_P}/profiles/{{principal_id}}"),
    "index.rewards.award": ("POST", f"{_P}/rewards/awards"),
    "index.rewards.reverse": ("POST", f"{_P}/rewards/awards/{{award_id}}/reverse"),
    "index.contests.create": ("POST", f"{_P}/contests"),
    "index.contests.retrieve": ("GET", _K),
    "index.contests.status.update": ("POST", f"{_K}/status"),
    "index.contests.leaderboard": ("GET", f"{_K}/leaderboard"),
    "index.contests.entries.create": ("POST", f"{_K}/entries"),
    "index.contests.entries.score": ("POST", f"{_K}/entries/{{entry_id}}/score"),
    "index.contests.entries.review": ("POST", f"{_K}/entries/{{entry_id}}/review"),
}

# The complete credential-free customer surface: the eight operations a reader
# can call with no account at all. Every one of them is also reachable on the
# authenticated client, which sees private work in addition.
PUBLIC_OPERATIONS: Mapping[str, tuple[str, str]] = {
    "index.public.search": ("POST", f"{_P}/public/search"),
    "index.public.contents.retrieve": ("POST", f"{_P}/public/contents"),
    "index.public.capabilities": ("GET", f"{_P}/public/capabilities"),
    "index.public.tags.list": ("GET", f"{_P}/public/tags"),
    "index.public.contributions.retrieve": (
        "GET",
        f"{_P}/public/contributions/{{contribution_id}}",
    ),
    "index.public.contributions.revisions.retrieve": (
        "GET",
        f"{_P}/public/contributions/{{contribution_id}}/revisions/{{revision_id}}",
    ),
    "index.public.contributions.assets.retrieve": (
        "GET",
        f"{_P}/public/contributions/{{contribution_id}}/revisions/{{revision_id}}"
        f"/assets/{{asset_id}}",
    ),
    "index.public.profiles.retrieve": ("GET", f"{_P}/public/profiles/{{principal_id}}"),
}

_IDENTIFIER = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9_.-]{0,127}$")


@dataclass(frozen=True, slots=True)
class _Call:
    """One declared request: the typed body it sends and the model it parses.

    ``body`` is the request model itself, not a pre-serialized dict, so the model
    each operation sends is inspectable next to the one it parses; the parity
    gate compares both against the backend contract. ``keep_none`` preserves
    explicit ``null`` fields for bodies whose wire form always carries them.
    """

    operation_id: str
    parse: Callable[[Any], Any]
    path_parameters: Mapping[str, str] | None = None
    body: BaseModel | None = None
    keep_none: bool = False
    headers: dict[str, str] | None = None
    params: dict[str, Any] | None = None
    raw: bool = False

    @property
    def request_model(self) -> type[BaseModel] | None:
        return None if self.body is None else type(self.body)

    @property
    def response_model(self) -> type[BaseModel] | None:
        """The model ``parse`` validates, or ``None`` for raw asset bytes."""
        if self.raw:
            return None
        model = getattr(self.parse, "model", None) or getattr(self.parse, "__self__", None)
        if not (isinstance(model, type) and issubclass(model, BaseModel)):
            raise TypeError(f"{self.operation_id} parses no declared response model")
        return model

    def request(self) -> tuple[str, str, dict[str, Any]]:
        operation = OPERATIONS.get(self.operation_id) or PUBLIC_OPERATIONS.get(self.operation_id)
        if operation is None:
            raise ValueError(f"Unknown Index operation: {self.operation_id}")
        method, template = operation
        for name, value in (self.path_parameters or {}).items():
            if not isinstance(value, str) or not _IDENTIFIER.fullmatch(value):
                raise ValueError(f"{name} must be an Index identifier")
        kwargs: dict[str, Any] = {"operation_id": self.operation_id}
        if self.params:
            kwargs["params"] = self.params
        if not self.raw:
            if self.body is not None:
                kwargs["json_body"] = self.body.model_dump(
                    mode="json", exclude_none=not self.keep_none
                )
            if self.headers is not None:
                kwargs["headers"] = self.headers
        return method, template.format(**(self.path_parameters or {})), kwargs


def _sync_runner(transport: HttpTransport) -> Callable[[_Call], Any]:
    def run(call: _Call) -> Any:
        method, path, kwargs = call.request()
        send = transport.request_bytes if call.raw else transport.request_json
        return call.parse(send(method, path, **kwargs))

    return run


def _async_runner(transport: AsyncHttpTransport) -> Callable[[_Call], Any]:
    async def run(call: _Call) -> Any:
        method, path, kwargs = call.request()
        send = transport.request_bytes if call.raw else transport.request_json
        return call.parse(await send(method, path, **kwargs))

    return run


def _key(idempotency_key: str | None, *, required: str | None = None) -> dict[str, str]:
    if idempotency_key is None and required is not None:
        raise ValueError(f"{required} requires an explicit idempotency key")
    key = str(uuid4()) if idempotency_key is None else idempotency_key
    if (
        not isinstance(key, str)
        or not 1 <= len(key) <= 128
        or not all(
            character.isascii() and (character.isalnum() or character in "_.-") for character in key
        )
    ):
        raise ValueError("Index idempotency key must be 1–128 safe ASCII characters")
    return {"Idempotency-Key": key}


def _parser(model: type[BaseModel], parse: Callable[[object], Any]) -> Callable[[object], Any]:
    """Name the model a response-checking parser validates."""
    parse.model = model  # type: ignore[attr-defined]
    return parse


def _revision(reference: ContributionReference) -> dict[str, str]:
    return {"contribution_id": reference.contribution_id, "revision_id": reference.revision_id}


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
    _search_delivery_bounds(result, spec)
    return result


def _public_search_result(payload: object, spec: SearchSpec) -> PublicSearchResult:
    result = PublicSearchResult.model_validate(payload)
    _search_delivery_bounds(result, spec)
    return result


def _search_delivery_bounds(result: SearchResult | PublicSearchResult, spec: SearchSpec) -> None:
    """Both delivery modes honor the caller's context bounds, not only global caps."""
    if len(result.results) > spec.content.max_results or any(
        len(hit.highlights) > spec.content.max_excerpts_per_result for hit in result.results
    ):
        raise ValueError("Index response exceeds requested result or excerpt bounds")


def _contents_result(payload: object, spec: ContentsSpec) -> ContentsResult:
    result = ContentsResult.model_validate(payload)
    expected = {(item.contribution_id, item.revision_id) for item in spec.references}
    actual = {(item.reference.contribution_id, item.reference.revision_id) for item in result.items}
    if actual != expected:
        raise ValueError("Index contents response does not match requested revisions")
    if sum(len((item.text or "").encode("utf-8")) for item in result.items) > spec.max_bytes:
        raise ValueError("Index contents response exceeds requested byte bound")
    return result


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


def _bound(model: Any, check: Callable[[Any], bool], message: str) -> Callable[[object], Any]:
    def parse(payload: object) -> Any:
        result = model.model_validate(payload)
        if not check(result):
            raise ValueError(message)
        return result

    return _parser(model, parse)


class _Resource:
    def __init__(self, run: Callable[[_Call], Any], asynchronous: bool) -> None:
        self._run = run
        self._asynchronous = asynchronous


class ContentsAPI(_Resource):
    def retrieve(
        self,
        spec: ContentsSpec | None = None,
        *,
        references: Sequence[ContributionReference] | None = None,
        search_id: str | None = None,
        max_bytes: int | None = None,
    ) -> Any:
        """Retrieve exact revision contents under current backend authorization."""
        spec = _contents_spec(spec, references, search_id, max_bytes)
        return self._run(
            _Call(
                "index.contents.retrieve",
                _parser(ContentsResult, lambda payload: _contents_result(payload, spec)),
                body=spec, keep_none=True,
            )
        )


class RevisionsAPI(_Resource):
    def create(
        self, contribution_id: str, spec: RevisionCreateSpec, *, idempotency_key: str
    ) -> Any:
        """Open a private revision draft (owner only); replaying the key returns it."""
        return self._run(
            _Call(
                "index.contributions.revisions.create",
                _bound(
                    ContributionDraft,
                    lambda draft: draft.reference.contribution_id == contribution_id,
                    "Revision draft does not match requested Contribution",
                ),
                path_parameters={"contribution_id": contribution_id},
                body=spec,
                headers=_key(idempotency_key, required="Revision creation"),
            )
        )

    def retrieve(self, reference: ContributionReference) -> Any:
        """Exact revision status, sealed package, assessments and citation."""
        return self._run(
            _Call(
                "index.contributions.revisions.retrieve",
                _bound(
                    RevisionView,
                    lambda view: view.reference == reference,
                    "Revision response does not match requested revision",
                ),
                path_parameters=_revision(reference),
            )
        )


class AssessmentsAPI(_Resource):
    def list(self, reference: ContributionReference) -> Any:
        return self._run(
            _Call(
                "index.contributions.assessments.list",
                Assessments.model_validate,
                path_parameters=_revision(reference),
            )
        )


class ReviewsAPI(_Resource):
    """Reviewer queue and decisions; needs a review grant, never self-review."""

    def list(self, *, status: RevisionStatus | str | None = None, cursor: str | None = None) -> Any:
        params: dict[str, Any] = {}
        if status is not None:
            params["status"] = RevisionStatus(status).value
        if cursor is not None:
            if not _IDENTIFIER.fullmatch(cursor):
                raise ValueError("cursor must be an Index identifier")
            params["cursor"] = cursor
        return self._run(_Call("index.reviews.list", ReviewList.model_validate, params=params))

    def create(
        self,
        reference: ContributionReference,
        spec: ReviewSpec,
        *,
        idempotency_key: str | None = None,
    ) -> Any:
        """Record a decision on the exact sealed revision; retries return the original."""
        return self._run(
            _Call(
                "index.contributions.reviews.create",
                _bound(
                    Assessment,
                    lambda item: (
                        item.reference == reference
                        and (
                            spec.manifest_digest is None
                            or item.manifest_digest == spec.manifest_digest
                        )
                    ),
                    "Assessment does not bind the requested sealed revision",
                ),
                path_parameters=_revision(reference),
                body=spec,
                # The route requires the header. A generated key still makes one
                # call retry-safe inside the transport; pass your own to make an
                # application-level retry return the original decision.
                headers=_key(idempotency_key),
            )
        )


class AssetsAPI(_Resource):
    def retrieve(self, reference: ContributionReference, asset_id: str) -> Any:
        """Download one declared asset's bytes under current authorization."""
        return self._run(
            _Call(
                "index.contributions.assets.retrieve",
                bytes,
                path_parameters={**_revision(reference), "asset_id": asset_id},
                raw=True,
            )
        )


class ContributionsAPI(_Resource):
    def __init__(self, run: Callable[[_Call], Any], asynchronous: bool) -> None:
        super().__init__(run, asynchronous)
        self.revisions = RevisionsAPI(run, asynchronous)
        self.assessments = AssessmentsAPI(run, asynchronous)
        self.reviews = ReviewsAPI(run, asynchronous)
        self.assets = AssetsAPI(run, asynchronous)

    def create(self, *, idempotency_key: str) -> Any:
        """Create a private draft. Persist and reuse the key after uncertain failures."""
        return self._run(
            _Call(
                "index.contributions.create",
                ContributionDraft.model_validate,
                body=CreateContributionSpec(),
                headers=_key(idempotency_key, required="Draft creation"),
            )
        )

    def create_research(self, spec: ResearchDraftSpec, *, idempotency_key: str) -> Any:
        """Allocate a private SYNTH-origin draft for a vetted export.

        The backend requires an active research-import grant and checks source
        paths. Reuse the same key and spec after an uncertain response.
        """
        return self._run(
            _Call(
                "index.contributions.research.create",
                ContributionDraft.model_validate,
                body=spec, keep_none=True,
                headers=_key(idempotency_key, required="Research draft creation"),
            )
        )

    def lookup_research(self, spec: ResearchDraftSpec, *, idempotency_key: str) -> Any:
        """Read what one research allocation key produced, without allocating.

        This is the recovery read for an allocation whose response was lost, and
        for saved intake state from an older client: the server answers from the
        caller's own receipt for ``idempotency_key`` and the exact ``spec``.
        A 404 ``research_receipt_absent`` means nothing was allocated under that
        key for this account; a 409 means the key was used for different input.
        """
        return self._run(
            _Call(
                "index.contributions.research.lookup",
                ResearchLookupView.model_validate,
                body=spec,
                keep_none=True,
                headers=_key(idempotency_key, required="Research allocation lookup"),
            )
        )

    def retrieve(self, contribution_id: str) -> Any:
        """Contribution, its current revision and revision history (404 when hidden)."""
        return self._run(
            _Call(
                "index.contributions.retrieve",
                _bound(
                    ContributionView,
                    lambda view: view.contribution_id == contribution_id,
                    "Contribution response does not match the request",
                ),
                path_parameters={"contribution_id": contribution_id},
            )
        )

    def prepare_upload(self, draft: ContributionDraft, spec: ContributionUploadSpec) -> Any:
        """Prepare transfer instructions only; reuse the publication ID for retries."""
        reference = draft.reference
        if (spec.package.contribution_id, spec.package.revision_id) != (
            reference.contribution_id,
            reference.revision_id,
        ):
            raise ValueError("Upload package must match the server-issued draft")
        return self._run(
            _Call(
                "index.contributions.upload.prepare",
                _parser(
                    ContributionUploadPrepared,
                    lambda payload: _upload_result(payload, draft, spec),
                ),
                path_parameters=_revision(reference),
                body=spec, keep_none=True,
            )
        )

    def upload(self, prepared: ContributionUploadPrepared, content: Mapping[str, bytes]) -> Any:
        """Transfer exactly the declared asset bytes to prepared storage targets."""
        if self._asynchronous:
            return upload_bytes(prepared, content)
        return upload_bytes_sync(prepared, content)

    def finalize(self, draft: ContributionDraft, prepared: ContributionUploadPrepared) -> Any:
        """Verify uploaded bytes through the backend; does not submit or publish."""
        if (
            prepared.transfer.collection_id != str(draft.collection_id)
            or prepared.transfer.revision != 1
        ):
            raise ValueError("Finalization transfer does not match draft collection")
        return self._run(
            _Call(
                "index.contributions.upload.finalize",
                _parser(
                    ArtifactPublicationResponse,
                    lambda payload: _finalize_result(payload, prepared),
                ),
                path_parameters=_revision(draft.reference),
                body=ArtifactPublicationFinalize(publication_id=prepared.transfer.publication_id),
                keep_none=True,
            )
        )

    def submit(self, reference: ContributionReference, spec: ContributionSubmitSpec) -> Any:
        """Submit finalized bytes for review; does not approve or publish research."""
        return self._run(
            _Call(
                "index.contributions.submit",
                _bound(
                    ContributionSubmission,
                    lambda item: item.reference == reference,
                    "Submission response does not match requested revision",
                ),
                path_parameters=_revision(reference),
                body=spec, keep_none=True,
            )
        )

    def _publication(
        self, operation_id: str, contribution_id: str, spec: Any, idempotency_key: str | None
    ) -> Any:
        return self._run(
            _Call(
                operation_id,
                _bound(
                    PublicationStatus,
                    lambda item: item.contribution_id == contribution_id,
                    "Publication response does not match requested Contribution",
                ),
                path_parameters={"contribution_id": contribution_id},
                body=spec,
                headers=_key(idempotency_key),
            )
        )

    def publish(
        self, contribution_id: str, spec: PublicationSpec, *, idempotency_key: str | None = None
    ) -> Any:
        """Publish an independently approved revision; requires a publisher grant."""
        return self._publication(
            "index.contributions.publication.create", contribution_id, spec, idempotency_key
        )

    def withdraw(
        self, contribution_id: str, spec: WithdrawalSpec, *, idempotency_key: str | None = None
    ) -> Any:
        """Withdraw from search and new reads; prior downloads cannot be recalled."""
        return self._publication(
            "index.contributions.withdrawal.create", contribution_id, spec, idempotency_key
        )


class TagsAPI(_Resource):
    def list(self) -> Any:
        return self._run(_Call("index.tags.list", TagRegistry.model_validate))


class CollectionGrantsAPI(_Resource):
    """Owner-only explicit shares of a Contribution's released storage."""

    def list(self, collection_id: str) -> Any:
        return self._run(
            _Call(
                "index.collections.grants.list",
                CollectionGrants.model_validate,
                path_parameters={"collection_id": collection_id},
            )
        )

    def create(self, collection_id: str, spec: CollectionGrantSpec) -> Any:
        return self._run(
            _Call(
                "index.collections.grants.create",
                CollectionGrant.model_validate,
                path_parameters={"collection_id": collection_id},
                body=spec,
            )
        )

    def revoke(self, collection_id: str, grant_id: str) -> Any:
        return self._run(
            _Call(
                "index.collections.grants.revoke",
                CollectionGrantRevoked.model_validate,
                path_parameters={"collection_id": collection_id, "grant_id": grant_id},
            )
        )


class CollectionsAPI(_Resource):
    def __init__(self, run: Callable[[_Call], Any], asynchronous: bool) -> None:
        super().__init__(run, asynchronous)
        self.grants = CollectionGrantsAPI(run, asynchronous)

    def list(self) -> Any:
        return self._run(_Call("index.collections.list", Collections.model_validate))


class AccountAPI(_Resource):
    """Authenticated caller's identity, work, usage, awarded credits and profile."""

    def retrieve(self) -> Any:
        return self._run(_Call("index.me.retrieve", MeView.model_validate))

    def contributions(self) -> Any:
        return self._run(_Call("index.me.contributions.list", MyContributions.model_validate))

    def promo_credit(self) -> Any:
        """Read the current private-search promo balance for this organization.

        Returns a summary whose ``credit`` is ``None`` when the organization is
        not enrolled. An exhausted balance is a balance, not an error: the
        refusal only happens when a private search is actually attempted.
        """
        return self._run(
            _Call("index.me.promo_credit", PromoCreditSummary.model_validate)
        )

    def usage(self) -> Any:
        return self._run(_Call("index.me.usage", IndexUsageSummary.model_validate))

    def rewards(self) -> Any:
        return self._run(_Call("index.me.rewards.list", MyRewards.model_validate))

    def update_profile(self, spec: ProfileSpec) -> Any:
        return self._run(
            _Call("index.me.profile.update", ProfileView.model_validate, body=spec)
        )

    def update_pins(self, spec: ProfilePinsSpec) -> Any:
        return self._run(
            _Call("index.me.profile.pins.update", ProfileView.model_validate, body=spec)
        )


class ProfilesAPI(_Resource):
    def retrieve(self, principal_id: str) -> Any:
        return self._run(
            _Call(
                "index.profiles.retrieve",
                ProfileView.model_validate,
                path_parameters={"principal_id": principal_id},
            )
        )


class RewardsAPI(_Resource):
    """Award program operations; require an award grant. Credits, never cash."""

    def award(self, spec: RewardAwardSpec, *, idempotency_key: str) -> Any:
        return self._run(
            _Call(
                "index.rewards.award",
                RewardAward.model_validate,
                body=spec,
                headers=_key(idempotency_key, required="Reward award"),
            )
        )

    def reverse(self, award_id: str, spec: RewardReverseSpec) -> Any:
        return self._run(
            _Call(
                "index.rewards.reverse",
                RewardAward.model_validate,
                path_parameters={"award_id": award_id},
                body=spec,
            )
        )


class ContestEntriesAPI(_Resource):
    def create(self, contest_id: str, spec: ContestEntrySpec) -> Any:
        return self._run(
            _Call(
                "index.contests.entries.create",
                ContestEntry.model_validate,
                path_parameters={"contest_id": contest_id},
                body=spec,
            )
        )

    def score(self, contest_id: str, entry_id: str, spec: ContestScoreSpec) -> Any:
        return self._run(
            _Call(
                "index.contests.entries.score",
                ContestEntry.model_validate,
                path_parameters={"contest_id": contest_id, "entry_id": entry_id},
                body=spec,
            )
        )

    def review(self, contest_id: str, entry_id: str, spec: ContestReviewSpec) -> Any:
        return self._run(
            _Call(
                "index.contests.entries.review",
                ContestEntry.model_validate,
                path_parameters={"contest_id": contest_id, "entry_id": entry_id},
                body=spec,
            )
        )


class ContestsAPI(_Resource):
    def __init__(self, run: Callable[[_Call], Any], asynchronous: bool) -> None:
        super().__init__(run, asynchronous)
        self.entries = ContestEntriesAPI(run, asynchronous)

    def create(self, spec: ContestSpec) -> Any:
        return self._run(
            _Call("index.contests.create", ContestView.model_validate, body=spec)
        )

    def retrieve(self, contest_id: str) -> Any:
        return self._run(
            _Call(
                "index.contests.retrieve",
                ContestView.model_validate,
                path_parameters={"contest_id": contest_id},
            )
        )

    def update_status(self, contest_id: str, spec: ContestStatusSpec) -> Any:
        return self._run(
            _Call(
                "index.contests.status.update",
                ContestView.model_validate,
                path_parameters={"contest_id": contest_id},
                body=spec,
            )
        )

    def leaderboard(self, contest_id: str) -> Any:
        return self._run(
            _Call(
                "index.contests.leaderboard",
                Leaderboard.model_validate,
                path_parameters={"contest_id": contest_id},
            )
        )


class _IndexRoot(_Resource):
    def __init__(self, run: Callable[[_Call], Any], asynchronous: bool) -> None:
        super().__init__(run, asynchronous)
        self.contents = ContentsAPI(run, asynchronous)
        self.contributions = ContributionsAPI(run, asynchronous)
        self.reviews = self.contributions.reviews
        self.tags = TagsAPI(run, asynchronous)
        self.collections = CollectionsAPI(run, asynchronous)
        self.account = AccountAPI(run, asynchronous)
        self.profiles = ProfilesAPI(run, asynchronous)
        self.rewards = RewardsAPI(run, asynchronous)
        self.contests = ContestsAPI(run, asynchronous)

    def capabilities(self) -> Any:
        """Discover supported modes, scopes, limits and the caller's capabilities."""
        return self._run(_Call("index.capabilities", Capabilities.model_validate))

    def search(
        self,
        spec: SearchSpec | None = None,
        *,
        query: str | None = None,
        scope: SearchScope | None = None,
        filters: SearchFilters | None = None,
        max_results: int | None = None,
        idempotency_key: str | None = None,
    ) -> Any:
        """Execute one fast search; reuse an explicit key for application-level retries.

        Failures propagate as typed errors; they never become empty results.
        """
        spec = _search_spec(spec, query, scope, filters, max_results)
        return self._run(
            _Call(
                "index.search",
                _parser(SearchResult, lambda payload: _search_result(payload, spec)),
                body=spec, keep_none=True,
                headers=_key(idempotency_key),
            )
        )


class IndexAPI(_IndexRoot):
    """Blocking client over ``HttpTransport``."""

    def __init__(self, transport: HttpTransport) -> None:
        self._transport = transport
        super().__init__(_sync_runner(transport), asynchronous=False)


class AsyncIndexAPI(_IndexRoot):
    """Async client over ``AsyncHttpTransport``; every call returns an awaitable."""

    def __init__(self, transport: AsyncHttpTransport) -> None:
        self._transport = transport
        super().__init__(_async_runner(transport), asynchronous=True)


class PublicContentsAPI(_Resource):
    def retrieve(
        self,
        spec: ContentsSpec | None = None,
        *,
        references: Sequence[ContributionReference] | None = None,
        max_bytes: int | None = None,
    ) -> Any:
        """Retrieve public exact-revision contents without account authority."""
        spec = _contents_spec(spec, references, None, max_bytes)
        if spec.search_id is not None:
            raise ValueError("Anonymous public contents cannot use a search receipt")
        return self._run(
            _Call(
                "index.public.contents.retrieve",
                _parser(ContentsResult, lambda payload: _contents_result(payload, spec)),
                body=spec, keep_none=True,
            )
        )


class PublicAssetsAPI(_Resource):
    def retrieve(self, reference: ContributionReference, asset_id: str) -> Any:
        """Download one declared asset of a published revision, with no account."""
        return self._run(
            _Call(
                "index.public.contributions.assets.retrieve",
                bytes,
                path_parameters={**_revision(reference), "asset_id": asset_id},
                raw=True,
            )
        )


class PublicProfilesAPI(_Resource):
    def retrieve(self, principal_id: str) -> Any:
        """Read a contributor profile as an anonymous reader sees it."""
        return self._run(
            _Call(
                "index.public.profiles.retrieve",
                ProfileView.model_validate,
                path_parameters={"principal_id": principal_id},
            )
        )


class PublicTagsAPI(_Resource):
    def list(self) -> Any:
        """Read the public tag registry used by search filters."""
        return self._run(_Call("index.public.tags.list", TagRegistry.model_validate))


class PublicRevisionsAPI(_Resource):
    def retrieve(self, reference: ContributionReference) -> Any:
        return self._run(
            _Call(
                "index.public.contributions.revisions.retrieve",
                _bound(
                    RevisionView,
                    lambda view: view.reference == reference,
                    "Revision response does not match requested revision",
                ),
                path_parameters=_revision(reference),
            )
        )


class PublicContributionsAPI(_Resource):
    def __init__(self, run: Callable[[_Call], Any], asynchronous: bool) -> None:
        super().__init__(run, asynchronous)
        self.revisions = PublicRevisionsAPI(run, asynchronous)
        self.assets = PublicAssetsAPI(run, asynchronous)

    def retrieve(self, contribution_id: str) -> Any:
        return self._run(
            _Call(
                "index.public.contributions.retrieve",
                _bound(
                    ContributionView,
                    lambda view: view.contribution_id == contribution_id,
                    "Contribution response does not match the request",
                ),
                path_parameters={"contribution_id": contribution_id},
            )
        )


class _PublicIndexRoot(_Resource):
    def __init__(self, run: Callable[[_Call], Any], asynchronous: bool) -> None:
        super().__init__(run, asynchronous)
        self.contents = PublicContentsAPI(run, asynchronous)
        self.contributions = PublicContributionsAPI(run, asynchronous)
        self.tags = PublicTagsAPI(run, asynchronous)
        self.profiles = PublicProfilesAPI(run, asynchronous)

    def capabilities(self) -> Any:
        """Discover the public surface: public visibility only, no write features."""
        return self._run(
            _Call("index.public.capabilities", Capabilities.model_validate)
        )

    def search(
        self,
        spec: SearchSpec | None = None,
        *,
        query: str | None = None,
        scope: SearchScope | None = None,
        filters: SearchFilters | None = None,
        max_results: int | None = None,
        idempotency_key: str | None = None,
    ) -> PublicSearchResult:
        """Search reviewed public Contributions with no account or usage receipt."""
        del idempotency_key
        spec = _search_spec(spec, query, scope, filters, max_results)
        if spec.scope.visibility != "public" or spec.scope.collection_ids:
            raise ValueError("Anonymous Index search is public-only")
        return self._run(
            _Call(
                "index.public.search",
                _parser(PublicSearchResult, lambda payload: _public_search_result(payload, spec)),
                body=spec, keep_none=True,
            )
        )


class PublicIndexAPI(_PublicIndexRoot):
    """Blocking, read-only API over an injected credential-free transport."""

    def __init__(self, transport: HttpTransport) -> None:
        self._transport = transport
        super().__init__(_sync_runner(transport), asynchronous=False)


class AsyncPublicIndexAPI(_PublicIndexRoot):
    """Async, read-only API over an injected credential-free transport."""

    def __init__(self, transport: AsyncHttpTransport) -> None:
        self._transport = transport
        super().__init__(_async_runner(transport), asynchronous=True)
