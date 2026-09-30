"""Authenticated and public Index clients over backend-owned Search state.

The service, not this package, owns authorization, funding, and usage receipts.
Sync and async clients share the same resource tree; neither silently falls
back to local search or creates another transport. Retain a durable Search ID
after an uncertain response and reconcile it before starting new work.
"""

import asyncio
import re
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from hashlib import sha256
from typing import Any
from uuid import UUID, uuid4

from synth_ai.core.errors import SynthError
from synth_ai.core.http.async_transport import AsyncHttpTransport
from synth_ai.core.http.transport import HttpTransport

from .artifacts import (
    ArtifactPublicationPrepare,
    ArtifactPublicationPrepareResponse,
    ArtifactPublicationResponse,
)
from .catalog import (
    AccessFundingAccount,
    BillingPolicyUpdate,
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
from .classification import ClassificationDecisionView, ClassificationSpec, ClassificationView
from .contracts import ContributionReference
from .contributions import (
    ContributionDraft,
    ContributionUploadPrepared,
    ContributionUploadSpec,
    ResearchDraftSpec,
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
from .package import ContributionPackage
from .public_search import PublicSearchOperations
from .qa import (
    FENCED_ACTIONS,
    AcceptAssignmentSpec,
    AdjudicationSpec,
    AppealSpec,
    AssignmentSpec,
    AssignmentView,
    CaseAction,
    CaseEvents,
    CaseEventSpec,
    CaseEventView,
    CaseView,
    CreateCaseSpec,
    EscalationSpec,
    EventVisibility,
    FencedCaseRequest,
    InternalNoteSpec,
)
from .qa_checks import CheckAttemptView, CheckReport, RecordCheckSpec
from .qa_preflight import PreflightResult, RunPreflightSpec
from .qa_reviews import RecordReviewSpec, ReviewFact, ReviewReport
from .research import (
    ReleaseConsentSpec,
    ReleaseConsentView,
    ReleaseDisclosure,
    ReleaseResearchView,
    ReproductionAttestationSpec,
    ReproductionReceipt,
    ResearchArchiveAllocation,
    ResearchArchiveAllocationSpec,
    ResearchArchiveView,
    ResearchBindingSpec,
    ResearchRevocationSpec,
)
from .retry import (
    DEFAULT_INDEX_RETRY_POLICY,
    IndexRetryPolicy,
    is_transient_search_failure,
    search_id_from_error,
)
from .search import (
    ContentsResult,
    ContentsSpec,
    Search,
    SearchBillingConstraints,
    SearchCancellation,
    SearchContent,
    SearchEventPage,
    SearchExecutionCancelledError,
    SearchExecutionFailedError,
    SearchExecutionLimits,
    SearchFilters,
    SearchMode,
    SearchResult,
    SearchScope,
    SearchSpec,
    SearchState,
    SearchWaitTimeoutError,
)
from .submission import ContributionSubmission, ContributionSubmitSpec, RevisionStatus
from .transfer import upload_bytes, upload_bytes_sync
from .usage_accounting import SearchUsageReceipt, SearchUsageSummary

_P = "/api/v1/index"
_C = f"{_P}/contributions/{{contribution_id}}"
_R = f"{_C}/revisions/{{revision_id}}"
_K = f"{_P}/contests/{{contest_id}}"

# operation_id -> (HTTP method, path template). Mirrors backend operation IDs.
OPERATIONS: Mapping[str, tuple[str, str]] = {
    "index.capabilities": ("GET", f"{_P}/capabilities"),
    "index.search": ("POST", f"{_P}/search"),
    "index.searches.create": ("POST", f"{_P}/searches"),
    "index.searches.get": ("GET", f"{_P}/searches/{{search_id}}"),
    "index.searches.result": ("GET", f"{_P}/searches/{{search_id}}/result"),
    "index.searches.events": ("GET", f"{_P}/searches/{{search_id}}/events"),
    "index.searches.cancel": ("POST", f"{_P}/searches/{{search_id}}/cancel"),
    "index.contents.retrieve": ("POST", f"{_P}/contents"),
    "index.contributions.create": ("POST", f"{_P}/contributions"),
    "index.contributions.research.create": ("POST", f"{_P}/contributions/research"),
    "index.qa.cases.create": ("POST", _P + "/qa/cases"),
    "index.qa.cases.get": ("GET", _P + "/qa/cases/{case_id}"),
    "index.qa.events.list": ("GET", _P + "/qa/cases/{case_id}/events"),
    "index.qa.events.create": ("POST", _P + "/qa/cases/{case_id}/events"),
    "index.qa.appeals.create": ("POST", _P + "/qa/cases/{case_id}/appeals"),
    "index.qa.escalations.create": ("POST", _P + "/qa/cases/{case_id}/escalations"),
    "index.qa.adjudications.create": ("POST", _P + "/qa/cases/{case_id}/adjudications"),
    "index.qa.notes.create": ("POST", _P + "/qa/cases/{case_id}/internal-notes"),
    "index.qa.assignments.create": ("POST", _P + "/qa/cases/{case_id}/assignments"),
    "index.qa.assignments.list": ("GET", _P + "/qa/assignments"),
    "index.qa.assignments.accept": ("POST", _P + "/qa/assignments/{assignment_id}/accept"),
    "index.qa.assignments.revoke": ("POST", _P + "/qa/assignments/{assignment_id}/revoke"),
    "index.qa.package.retrieve": ("GET", _P + "/qa/cases/{case_id}/package"),
    "index.qa.assets.retrieve": ("GET", _P + "/qa/cases/{case_id}/assets/{asset_id}"),
    "index.qa.checks.list": ("GET", _P + "/qa/cases/{case_id}/checks"),
    "index.qa.checks.record": ("POST", _P + "/qa/cases/{case_id}/checks"),
    "index.qa.checks.preflight": ("POST", _P + "/qa/cases/{case_id}/checks/preflight"),
    "index.qa.checks.secret_scan": ("POST", _P + "/qa/cases/{case_id}/checks/secret-scan"),
    "index.qa.reviews.list": ("GET", _P + "/qa/cases/{case_id}/reviews"),
    "index.qa.reviews.record": ("POST", _P + "/qa/cases/{case_id}/reviews"),
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
    "index.classification.get": ("GET", f"{_R}/classification"),
    "index.classification.create": ("POST", f"{_R}/classification-decisions"),
    "index.research.release.get": ("GET", f"{_R}/release-research"),
    "index.research.archive.get": ("GET", f"{_R}/research-archive"),
    "index.research.archive.grants.list": ("GET", f"{_R}/research-archive/grants"),
    "index.research.archive.grants.create": ("POST", f"{_R}/research-archive/grants"),
    "index.research.archive.grants.revoke": (
        "DELETE",
        f"{_R}/research-archive/grants/{{grant_id}}",
    ),
    "index.research.archives.create": ("POST", f"{_C}/research-archives"),
    "index.research.archives.upload.prepare": (
        "POST",
        f"{_C}/research-archives/{{snapshot_id}}/upload",
    ),
    "index.research.archives.upload.finalize": (
        "POST",
        f"{_C}/research-archives/{{snapshot_id}}/finalize",
    ),
    "index.research.binding.create": ("POST", f"{_R}/research-binding"),
    "index.research.consent.create": ("POST", f"{_R}/release-consent"),
    "index.research.reproduction.create": ("POST", f"{_R}/reproduction-attestations"),
    "index.research.disclosure.revoke": ("POST", f"{_R}/disclosure-revocation"),
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
    "index.me.search_usage_receipt": ("GET", f"{_P}/me/usage/searches/{{search_id}}"),
    "index.me.operation_usage_summary": ("GET", f"{_P}/me/usage/operations"),
    "index.me.operation_usage_export": ("GET", f"{_P}/me/usage/operations/export"),
    "index.me.promo_credit": ("GET", f"{_P}/me/promo-credit"),
    "index.me.access_funding": ("GET", f"{_P}/me/access-funding"),
    "index.me.access_funding.update": ("PUT", f"{_P}/me/access-funding/{{mode}}"),
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

# Anonymous callers may browse published research and run the free public
# search (Index Search v0.2). Private-scope search still uses index.search.
PUBLIC_OPERATIONS: Mapping[str, tuple[str, str]] = {
    "index.public.classification.get": (
        "GET",
        f"{_P}/public/contributions/{{contribution_id}}/revisions/{{revision_id}}/classification",
    ),
    "index.public.research.release.get": (
        "GET",
        f"{_P}/public/contributions/{{contribution_id}}/revisions/{{revision_id}}/release-research",
    ),
    "index.public.search": ("POST", f"{_P}/public/search"),
    "index.public.searches.get": ("GET", f"{_P}/public/searches/{{search_id}}"),
    "index.public.searches.result": ("GET", f"{_P}/public/searches/{{search_id}}/result"),
    "index.public.searches.cancel": ("POST", f"{_P}/public/searches/{{search_id}}/cancel"),
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
    operation_id: str
    parse: Callable[[Any], Any]
    path_parameters: Mapping[str, str] | None = None
    json_body: dict[str, Any] | None = None
    headers: dict[str, str] | None = None
    params: dict[str, Any] | None = None
    raw: bool = False
    # Per-request transport timeout; None keeps the client's default.
    timeout_seconds: float | None = None

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
        if self.timeout_seconds is not None:
            kwargs["timeout_seconds"] = self.timeout_seconds
        if not self.raw:
            if self.json_body is not None:
                kwargs["json_body"] = self.json_body
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


def _body(spec: Any) -> dict[str, Any]:
    return spec.model_dump(mode="json", exclude_none=True)


def _revision(reference: ContributionReference) -> dict[str, str]:
    return {"contribution_id": reference.contribution_id, "revision_id": reference.revision_id}


def _search_spec(
    spec: SearchSpec | None,
    query: str | None,
    scope: SearchScope | None,
    filters: SearchFilters | None,
    max_results: int | None,
    mode: SearchMode | str | None = None,
    limits: SearchExecutionLimits | None = None,
    billing: SearchBillingConstraints | None = None,
) -> SearchSpec:
    if spec is not None:
        if any(
            value is not None
            for value in (query, scope, filters, max_results, mode, limits, billing)
        ):
            raise ValueError("Pass either SearchSpec or search keyword arguments, not both")
        return spec
    if query is None:
        raise ValueError("Index search requires a query or SearchSpec")
    return SearchSpec(
        query=query,
        mode=SearchMode.FAST if mode is None else SearchMode(mode),
        scope=scope if scope is not None else SearchScope(),
        filters=filters if filters is not None else SearchFilters(),
        content=SearchContent(max_results=5 if max_results is None else max_results),
        limits=limits,
        billing=billing if billing is not None else SearchBillingConstraints(),
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
    if (
        result.requested_mode != spec.mode
        or result.effective_mode != spec.mode
        or result.usage.billing_scope != spec.scope.visibility
    ):
        raise ValueError("Index response billing scope does not match the request")
    _search_delivery_bounds(result, spec)
    return result


def _search_delivery_bounds(result: SearchResult, spec: SearchSpec) -> None:
    """Citations stay within the caller's requested result bound."""
    if len(result.citations) > spec.content.max_results:
        raise ValueError("Index response exceeds requested citation bound")


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

    return parse


class _Resource:
    def __init__(self, run: Callable[[_Call], Any], asynchronous: bool) -> None:
        self._run = run
        self._asynchronous = asynchronous


#: Upper bound for one lifecycle status read while waiting (prod 2026-09-27
#: P12: one poll response never arrived and the client sat on the 120 s
#: transport timeout, then reported the Search as failed). A slower read is
#: abandoned and retried; the overall wait deadline is unchanged.
STATUS_POLL_TIMEOUT_SECONDS = 20.0


def _poll_timeout(deadline: float) -> float:
    return max(1.0, min(STATUS_POLL_TIMEOUT_SECONDS, deadline - time.monotonic()))


class SearchHandle:
    """Blocking durable Search handle; local timeout never mutates server state."""

    def __init__(
        self,
        searches: "SearchesAPI",
        snapshot: Search,
        *,
        retry: IndexRetryPolicy = DEFAULT_INDEX_RETRY_POLICY,
    ) -> None:
        self._searches = searches
        self.snapshot = snapshot
        self._retry = retry

    @property
    def search_id(self) -> str:
        """Stable server Search ID to retain for reconnects and receipts."""
        return self.snapshot.search_id

    def refresh(self, *, timeout_seconds: float | None = None) -> Search:
        """Fetch the latest durable lifecycle state without creating another Search."""
        self.snapshot = self._searches.get(self.search_id, timeout_seconds=timeout_seconds)
        return self.snapshot

    def events(self, *, after: int = 0, limit: int = 200) -> SearchEventPage:
        """Read lifecycle events after a sequence cursor; does not wait for completion."""
        return self._searches.events(self.search_id, after=after, limit=limit)

    def cancel(self) -> SearchCancellation:
        """Request cancellation of this Search; completion may race the request."""
        return self._searches.cancel(self.search_id)

    def result(self) -> SearchResult:
        """Fetch the delivered result and validate it against the original request."""
        return self._searches.result(self.search_id, self.snapshot.spec)

    def wait(self, *, timeout_seconds: float = 120.0, poll_seconds: float = 0.25) -> SearchResult:
        """Poll until terminal, without cancelling or resubmitting on local timeout.

        Raises ``SearchWaitTimeoutError`` with the Search ID when the local
        deadline expires. Keep that ID and reconnect to the same Search.
        """
        if timeout_seconds <= 0 or not 0.01 <= poll_seconds <= 5:
            raise ValueError(
                "wait timeout must be positive and poll interval within 0.01..5 seconds"
            )
        deadline = time.monotonic() + timeout_seconds
        while self.snapshot.state in {SearchState.QUEUED, SearchState.RUNNING}:
            if time.monotonic() >= deadline:
                raise SearchWaitTimeoutError(self.search_id)
            time.sleep(min(poll_seconds, max(0, deadline - time.monotonic())))
            try:
                self.refresh(timeout_seconds=_poll_timeout(deadline))
            except SynthError as error:
                # The Search keeps running server-side; a transient read
                # failure is polled through until the local deadline.
                if not is_transient_search_failure(error):
                    raise
        if self.snapshot.state == SearchState.FAILED:
            if self.snapshot.failure is None:
                raise ValueError("failed Search omitted its typed failure")
            raise SearchExecutionFailedError(self.search_id, self.snapshot.failure)
        if self.snapshot.state == SearchState.CANCELLED:
            raise SearchExecutionCancelledError(self.search_id)
        return self._searches._retrying_sync(lambda: self.result(), self._retry)


class AsyncSearchHandle:
    """Async durable Search handle; local timeout never mutates server state."""

    def __init__(
        self,
        searches: "SearchesAPI",
        snapshot: Search,
        *,
        retry: IndexRetryPolicy = DEFAULT_INDEX_RETRY_POLICY,
    ) -> None:
        self._searches = searches
        self.snapshot = snapshot
        self._retry = retry

    @property
    def search_id(self) -> str:
        """Stable server Search ID to retain for reconnects and receipts."""
        return self.snapshot.search_id

    async def refresh(self, *, timeout_seconds: float | None = None) -> Search:
        """Fetch the latest durable lifecycle state without creating another Search."""
        self.snapshot = await self._searches.get(self.search_id, timeout_seconds=timeout_seconds)
        return self.snapshot

    async def events(self, *, after: int = 0, limit: int = 200) -> SearchEventPage:
        """Read lifecycle events after a sequence cursor; does not wait for completion."""
        return await self._searches.events(self.search_id, after=after, limit=limit)

    async def cancel(self) -> SearchCancellation:
        """Request cancellation of this Search; completion may race the request."""
        return await self._searches.cancel(self.search_id)

    async def result(self) -> SearchResult:
        """Fetch the delivered result and validate it against the original request."""
        return await self._searches.result(self.search_id, self.snapshot.spec)

    async def wait(
        self, *, timeout_seconds: float = 120.0, poll_seconds: float = 0.25
    ) -> SearchResult:
        """Poll until terminal without cancelling or resubmitting on local timeout.

        Raises ``SearchWaitTimeoutError`` with the Search ID when the local
        deadline expires. Keep that ID and reconnect to the same Search.
        """
        if timeout_seconds <= 0 or not 0.01 <= poll_seconds <= 5:
            raise ValueError(
                "wait timeout must be positive and poll interval within 0.01..5 seconds"
            )
        deadline = time.monotonic() + timeout_seconds
        while self.snapshot.state in {SearchState.QUEUED, SearchState.RUNNING}:
            if time.monotonic() >= deadline:
                raise SearchWaitTimeoutError(self.search_id)
            await asyncio.sleep(min(poll_seconds, max(0, deadline - time.monotonic())))
            try:
                await self.refresh(timeout_seconds=_poll_timeout(deadline))
            except SynthError as error:
                if not is_transient_search_failure(error):
                    raise
        if self.snapshot.state == SearchState.FAILED:
            if self.snapshot.failure is None:
                raise ValueError("failed Search omitted its typed failure")
            raise SearchExecutionFailedError(self.search_id, self.snapshot.failure)
        if self.snapshot.state == SearchState.CANCELLED:
            raise SearchExecutionCancelledError(self.search_id)
        return await self._searches._retrying_async(self.result, self._retry)


class SearchesAPI(_Resource):
    """Durable Search lifecycle over the backend-owned execution ledger."""

    def create(
        self,
        spec: SearchSpec,
        *,
        idempotency_key: str | None = None,
        retry: IndexRetryPolicy = DEFAULT_INDEX_RETRY_POLICY,
    ) -> Any:
        """Create one durable Search and return its handle.

        Always sends an ``Idempotency-Key`` (a fresh UUID when none is given).
        A transient failure (502/503/504, timeout, network) is retried with the
        same request and key, bounded by ``retry``; the backend returns the
        Search it already admitted instead of creating another. A failure that
        names the admitted Search (``X-Synth-Search-Id`` or ``search_id``) is
        resolved by reading that Search. Keep the key if you retry yourself.
        """
        call = _Call(
            "index.searches.create",
            Search.model_validate,
            json_body=_body(spec),
            headers=_key(idempotency_key),
        )
        if self._asynchronous:

            async def asynchronous_handle() -> AsyncSearchHandle:
                snapshot = await self._create_async(call, retry)
                return AsyncSearchHandle(self, snapshot, retry=retry)

            return asynchronous_handle()
        return SearchHandle(self, self._create_sync(call, retry), retry=retry)

    def _create_sync(self, call: _Call, policy: IndexRetryPolicy) -> Search:
        attempt, waited = 0, 0.0
        while True:
            try:
                return self._run(call)
            except SynthError as error:
                admitted = search_id_from_error(error)
                if admitted is not None and is_transient_search_failure(error):
                    return self._reconnect_sync(admitted, error, policy)
                delay = policy.next_delay(error, attempt_index=attempt, waited_seconds=waited)
                if delay is None:
                    raise
                time.sleep(delay)
                waited += delay
                attempt += 1

    async def _create_async(self, call: _Call, policy: IndexRetryPolicy) -> Search:
        attempt, waited = 0, 0.0
        while True:
            try:
                return await self._run(call)
            except SynthError as error:
                admitted = search_id_from_error(error)
                if admitted is not None and is_transient_search_failure(error):
                    return await self._reconnect_async(admitted, error, policy)
                delay = policy.next_delay(error, attempt_index=attempt, waited_seconds=waited)
                if delay is None:
                    raise
                await asyncio.sleep(delay)
                waited += delay
                attempt += 1

    def _reconnect_sync(
        self, search_id: str, cause: SynthError, policy: IndexRetryPolicy
    ) -> Search:
        try:
            return self._retrying_sync(lambda: self.get(search_id), policy)
        except SynthError as error:
            # Re-raise the create failure: it names the admitted Search.
            raise cause from error

    async def _reconnect_async(
        self, search_id: str, cause: SynthError, policy: IndexRetryPolicy
    ) -> Search:
        try:
            return await self._retrying_async(lambda: self.get(search_id), policy)
        except SynthError as error:
            raise cause from error

    @staticmethod
    def _retrying_sync(operation: Callable[[], Any], policy: IndexRetryPolicy) -> Any:
        """Run an idempotent read, retrying transient failures within ``policy``."""
        attempt, waited = 0, 0.0
        while True:
            try:
                return operation()
            except SynthError as error:
                delay = policy.next_delay(error, attempt_index=attempt, waited_seconds=waited)
                if delay is None:
                    raise
                time.sleep(delay)
                waited += delay
                attempt += 1

    @staticmethod
    async def _retrying_async(operation: Callable[[], Any], policy: IndexRetryPolicy) -> Any:
        attempt, waited = 0, 0.0
        while True:
            try:
                return await operation()
            except SynthError as error:
                delay = policy.next_delay(error, attempt_index=attempt, waited_seconds=waited)
                if delay is None:
                    raise
                await asyncio.sleep(delay)
                waited += delay
                attempt += 1

    def get(self, search_id: str, *, timeout_seconds: float | None = None) -> Any:
        """Retrieve a Search's latest state by its stable server ID.

        ``timeout_seconds`` bounds this one read (default: the client timeout).
        """
        return self._run(
            _Call(
                "index.searches.get",
                _bound(Search, lambda item: item.search_id == search_id, "Search ID mismatch"),
                path_parameters={"search_id": search_id},
                timeout_seconds=timeout_seconds,
            )
        )

    def result(self, search_id: str, spec: SearchSpec) -> Any:
        """Retrieve a delivered result, bound to its Search ID and request spec."""

        def parse(payload: object) -> SearchResult:
            result = _search_result(payload, spec)
            if result.search_id != search_id:
                raise ValueError("Search result ID mismatch")
            return result

        return self._run(
            _Call(
                "index.searches.result",
                parse,
                path_parameters={"search_id": search_id},
            )
        )

    def events(self, search_id: str, *, after: int = 0, limit: int = 200) -> Any:
        """Read up to ``limit`` lifecycle events after the sequence cursor."""
        if after < 0 or not 1 <= limit <= 200:
            raise ValueError("Search event cursor or limit is out of bounds")
        return self._run(
            _Call(
                "index.searches.events",
                _bound(
                    SearchEventPage,
                    lambda page: page.search_id == search_id,
                    "Search event page ID mismatch",
                ),
                path_parameters={"search_id": search_id},
                params={"after": after, "limit": limit},
            )
        )

    def cancel(self, search_id: str) -> Any:
        """Request cancellation of an existing durable Search."""
        return self._run(
            _Call(
                "index.searches.cancel",
                _bound(
                    SearchCancellation,
                    lambda item: item.search_id == search_id,
                    "Search cancellation ID mismatch",
                ),
                path_parameters={"search_id": search_id},
            )
        )


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
                lambda payload: _contents_result(payload, spec),
                json_body=spec.model_dump(mode="json"),
            )
        )


def _release_research(
    payload: object, reference: ContributionReference
) -> ReleaseResearchView | None:
    if payload is None:
        return None
    return _bound(
        ReleaseResearchView,
        lambda view: view.disclosure.reference == reference,
        "Release research response does not match requested revision",
    )(payload)


class PublicReleaseResearchAPI(_Resource):
    """Read public-safe disclosure and reproduction without archive/session identities.

    Examples:
        result = public.contents.release_research.retrieve(reference)
    """
    def retrieve(self, reference: ContributionReference) -> Any:
        """Read safe released-output proof; historical unbound releases return None.

        See sibling backend/notes/specifications/synth-index/research-archive-release.md.

        Args:
            reference: Exact Contribution and revision addressed by this operation.

        Returns:
            ReleaseResearchView | Awaitable[ReleaseResearchView]: Validated result bound to the requested exact identities and intent.

        Raises:
            ValueError: Requested constraints or returned identity/content bindings are invalid.

        Examples:
            result = public.contents.release_research.retrieve(reference)
        """
        return self._run(
            _Call(
                "index.public.research.release.get",
                lambda payload: _release_research(payload, reference),
                path_parameters=_revision(reference),
            )
        )


class PublicClassificationAPI(_Resource):
    """Read public-safe effective tags under current public revision authority.

    Examples:
        result = public.contents.classifications.retrieve(reference)
    """
    def retrieve(self, reference: ContributionReference) -> Any:
        """Read safe effective metadata; see tag-classification.md.

        Args:
            reference: Exact Contribution and revision addressed by this operation.

        Returns:
            ClassificationView | Awaitable[ClassificationView]: Validated result bound to the requested exact identities and intent.

        Raises:
            ValueError: Requested constraints or returned identity/content bindings are invalid.

        Examples:
            result = public.contents.classifications.retrieve(reference)
        """
        return self._run(
            _Call(
                "index.public.classification.get",
                _bound(
                    ClassificationView,
                    lambda view: view.reference == reference,
                    "Classification response does not match requested revision",
                ),
                path_parameters=_revision(reference),
            )
        )


class ClassificationsAPI(_Resource):
    """Review effective tags and record independent reviewer classifications.

    Examples:
        result = index.contributions.classifications.retrieve(reference)
    """
    def retrieve(self, reference: ContributionReference) -> Any:
        """Read metadata under the current revision ACL; see tag-classification.md.

        Args:
            reference: Exact Contribution and revision addressed by this operation.

        Returns:
            ClassificationView | Awaitable[ClassificationView]: Validated result bound to the requested exact identities and intent.

        Raises:
            ValueError: Requested constraints or returned identity/content bindings are invalid.

        Examples:
            result = index.contributions.classifications.retrieve(reference)
        """
        return self._run(
            _Call(
                "index.classification.get",
                _bound(
                    ClassificationView,
                    lambda view: view.reference == reference,
                    "Classification response does not match requested revision",
                ),
                path_parameters=_revision(reference),
            )
        )

    def create(
        self, reference: ContributionReference, spec: ClassificationSpec, *, idempotency_key: str
    ) -> Any:
        """Classify exact sealed bytes with a current independent reviewer grant.

        See sibling backend/notes/specifications/synth-index/tag-classification.md.
        Retry the identical identity and intent after an uncertain response.

        Args:
            reference: Exact Contribution and revision addressed by this operation.
            spec: Typed exact-input request required by this operation; does not infer publication consent.
            idempotency_key: Persisted request key reused for an identical uncertain retry.

        Returns:
            ClassificationDecisionView | Awaitable[ClassificationDecisionView]: Validated result bound to the requested exact identities and intent.

        Raises:
            ValueError: Requested constraints or returned identity/content bindings are invalid.

        Examples:
            result = index.contributions.classifications.create(reference, spec, idempotency_key=idempotency_key)
        """
        return self._run(
            _Call(
                "index.classification.create",
                _bound(
                    ClassificationDecisionView,
                    lambda view: (
                        view.reference == reference
                        and view.manifest_digest == spec.manifest_digest
                        and view.registry_version == spec.registry_version
                        and view.generation == spec.expected_generation + 1
                        and view.accepted_tag_ids == spec.accepted_tag_ids
                    ),
                    "Classification decision does not match requested intent",
                ),
                path_parameters=_revision(reference),
                json_body=_body(spec),
                headers=_key(idempotency_key, required="Classification decision"),
            )
        )


class ResearchAPI(_Resource):
    """Explicit authenticated research operations, distinct from ordinary Search.

    See sibling backend/notes/specifications/synth-index/research-archive-release.md.
    Retries bind the same exact content; no method infers consent or publishes.

    Examples:
        result = index.contributions.research.release(reference)
    """

    def release(self, reference: ContributionReference) -> Any:
        """Read the public-safe release disclosure under current revision access.

        Args:
            reference: Exact Contribution and revision addressed by this operation.

        Returns:
            ReleaseResearchView | Awaitable[ReleaseResearchView]: Validated result bound to the requested exact identities and intent.

        Raises:
            ValueError: Requested constraints or returned identity/content bindings are invalid.

        Examples:
            result = index.contributions.research.release(reference)
        """
        return self._run(
            _Call(
                "index.research.release.get",
                lambda payload: _release_research(payload, reference),
                path_parameters=_revision(reference),
            )
        )

    def archive(self, reference: ContributionReference) -> Any:
        """Read private frozen inputs under a current explicit archive grant.

        Args:
            reference: Exact Contribution and revision addressed by this operation.

        Returns:
            ResearchArchiveView | Awaitable[ResearchArchiveView]: Validated result bound to the requested exact identities and intent.

        Raises:
            ValueError: Requested constraints or returned identity/content bindings are invalid.

        Examples:
            result = index.contributions.research.archive(reference)
        """
        return self._run(
            _Call(
                "index.research.archive.get",
                _bound(
                    ResearchArchiveView,
                    lambda view: view.binding.disclosure.reference == reference,
                    "Private research response does not match requested revision",
                ),
                path_parameters=_revision(reference),
            )
        )

    def archive_grants(self, reference: ContributionReference) -> Any:
        """List this owner's named readers of one frozen archive revision.

        Args:
            reference: Exact Contribution and revision addressed by this operation.

        Returns:
            CollectionGrants | Awaitable[CollectionGrants]: Validated result bound to the requested exact identities and intent.

        Raises:
            ValueError: Requested constraints or returned identity/content bindings are invalid.

        Examples:
            result = index.contributions.research.archive_grants(reference)
        """
        return self._run(
            _Call(
                "index.research.archive.grants.list",
                CollectionGrants.model_validate,
                path_parameters=_revision(reference),
            )
        )

    def grant_archive(self, reference: ContributionReference, spec: CollectionGrantSpec) -> Any:
        """Grant one named user manifest and object access to this revision.

        Args:
            reference: Exact Contribution and revision addressed by this operation.
            spec: Typed exact-input request required by this operation; does not infer publication consent.

        Returns:
            CollectionGrant | Awaitable[CollectionGrant]: Validated result bound to the requested exact identities and intent.

        Raises:
            ValueError: Requested constraints or returned identity/content bindings are invalid.

        Examples:
            result = index.contributions.research.grant_archive(reference, spec)
        """
        return self._run(
            _Call(
                "index.research.archive.grants.create",
                _bound(
                    CollectionGrant,
                    lambda grant: (
                        grant.subject_kind == spec.subject_kind
                        and grant.subject_id == spec.subject_id
                        and set(grant.operations) == {"read_manifest", "read_object"}
                    ),
                    "Archive grant does not match requested reader and operations",
                ),
                path_parameters=_revision(reference),
                json_body=_body(spec),
            )
        )

    def revoke_archive_grant(self, reference: ContributionReference, grant_id: str) -> Any:
        """Revoke both archive read operations for a named reader.

        Args:
            reference: Exact Contribution and revision addressed by this operation.
            grant_id: Identifier of the exact archive read grant being revoked.

        Returns:
            CollectionGrantRevoked | Awaitable[CollectionGrantRevoked]: Validated result bound to the requested exact identities and intent.

        Raises:
            ValueError: Requested constraints or returned identity/content bindings are invalid.

        Examples:
            result = index.contributions.research.revoke_archive_grant(reference, grant_id)
        """
        return self._run(
            _Call(
                "index.research.archive.grants.revoke",
                _bound(
                    CollectionGrantRevoked,
                    lambda receipt: receipt.grant_id == grant_id and receipt.revoked,
                    "Archive revocation does not match requested grant",
                ),
                path_parameters={**_revision(reference), "grant_id": grant_id},
            )
        )

    def allocate_archive(self, contribution_id: str, spec: ResearchArchiveAllocationSpec) -> Any:
        """Allocate a private frozen-input archive for the requested snapshot.

        Args:
            contribution_id: Contribution whose owner requests private archive allocation or upload.
            spec: Typed exact-input request required by this operation; does not infer publication consent.

        Returns:
            ResearchArchiveAllocation | Awaitable[ResearchArchiveAllocation]: Validated result bound to the requested exact identities and intent.

        Raises:
            ValueError: Requested constraints or returned identity/content bindings are invalid.

        Examples:
            result = index.contributions.research.allocate_archive(contribution_id, spec)
        """
        return self._run(
            _Call(
                "index.research.archives.create",
                _bound(
                    ResearchArchiveAllocation,
                    lambda collection: (
                        collection.scope.owner_namespace == "contribution_research_archives"
                        and collection.scope.owner_resource_id == spec.snapshot_id
                        and collection.scope.visibility == "private"
                    ),
                    "Archive allocation must return the requested private snapshot scope",
                ),
                path_parameters={"contribution_id": contribution_id},
                json_body=_body(spec),
            )
        )

    def prepare_archive_upload(
        self, contribution_id: str, snapshot_id: str, spec: ArtifactPublicationPrepare
    ) -> Any:
        """Prepare only the allocated private snapshot; never retain signed URLs.

        Args:
            contribution_id: Contribution whose owner requests private archive allocation or upload.
            snapshot_id: Frozen snapshot identity already allocated to this private archive.
            spec: Typed exact-input request required by this operation; does not infer publication consent.

        Returns:
            ArtifactPublicationPrepareResponse | Awaitable[ArtifactPublicationPrepareResponse]: Validated result bound to the requested exact identities and intent.

        Raises:
            ValueError: Requested constraints or returned identity/content bindings are invalid.

        Examples:
            result = index.contributions.research.prepare_archive_upload(contribution_id, snapshot_id, spec)
        """
        if spec.revision != 1 or spec.manifest_schema_version != "synth.research.snapshot.v1":
            raise ValueError("Archive upload requires snapshot v1 at revision 1")
        expected = {obj.logical_path: obj.digest_sha256 for obj in spec.objects}
        return self._run(
            _Call(
                "index.research.archives.upload.prepare",
                _bound(
                    ArtifactPublicationPrepareResponse,
                    lambda result: (
                        result.publication_id == spec.publication_id
                        and result.collection_id == spec.collection_id
                        and result.revision == 1
                        and len({target.logical_path for target in result.upload_targets})
                        == len(result.upload_targets)
                        and all(
                            expected.get(target.logical_path) == target.digest_sha256
                            for target in result.upload_targets
                        )
                    ),
                    "Archive transfer differs from requested snapshot objects",
                ),
                path_parameters={"contribution_id": contribution_id, "snapshot_id": snapshot_id},
                json_body=_body(spec),
            )
        )

    def finalize_archive_upload(
        self, contribution_id: str, snapshot_id: str, publication_id: str, *, collection_id: str
    ) -> Any:
        """Verify and commit exact private snapshot bytes; this does not publish a release.

        Args:
            contribution_id: Contribution whose owner requests private archive allocation or upload.
            snapshot_id: Frozen snapshot identity already allocated to this private archive.
            publication_id: Exact prepared artifact publication identity being finalized.
            collection_id: Expected private archive collection UUID used to verify finalization.

        Returns:
            ArtifactPublicationResponse | Awaitable[ArtifactPublicationResponse]: Validated result bound to the requested exact identities and intent.

        Raises:
            ValueError: Requested constraints or returned identity/content bindings are invalid.

        Examples:
            result = index.contributions.research.finalize_archive_upload(contribution_id, snapshot_id, publication_id, collection_id=collection_id)
        """
        return self._run(
            _Call(
                "index.research.archives.upload.finalize",
                _bound(
                    ArtifactPublicationResponse,
                    lambda result: (
                        result.publication_id == publication_id
                        and result.collection_id == collection_id
                        and result.revision == 1
                        and result.status == "committed"
                    ),
                    "Archive finalization differs from requested snapshot",
                ),
                path_parameters={"contribution_id": contribution_id, "snapshot_id": snapshot_id},
                json_body={"publication_id": publication_id},
            )
        )

    def bind(self, reference: ContributionReference, spec: ResearchBindingSpec) -> Any:
        """Bind exact frozen research inputs to the approved release disclosure.

        Args:
            reference: Exact Contribution and revision addressed by this operation.
            spec: Typed exact-input request required by this operation; does not infer publication consent.

        Returns:
            ReleaseDisclosure | Awaitable[ReleaseDisclosure]: Validated result bound to the requested exact identities and intent.

        Raises:
            ValueError: Requested constraints or returned identity/content bindings are invalid.

        Examples:
            result = index.contributions.research.bind(reference, spec)
        """
        if spec.binding.disclosure.reference != reference:
            raise ValueError("Research binding must match the requested revision")
        return self._run(
            _Call(
                "index.research.binding.create",
                _bound(
                    ReleaseDisclosure,
                    lambda disclosure: disclosure == spec.binding.disclosure,
                    "Research binding response differs from exact submitted disclosure",
                ),
                path_parameters=_revision(reference),
                json_body=_body(spec),
            )
        )

    def consent(self, reference: ContributionReference, spec: ReleaseConsentSpec) -> Any:
        """Explicit author consent for the exact manifest, disclosure and audience.

        Args:
            reference: Exact Contribution and revision addressed by this operation.
            spec: Typed exact-input request required by this operation; does not infer publication consent.

        Returns:
            ReleaseConsentView | Awaitable[ReleaseConsentView]: Validated result bound to the requested exact identities and intent.

        Raises:
            ValueError: Requested constraints or returned identity/content bindings are invalid.

        Examples:
            result = index.contributions.research.consent(reference, spec)
        """
        return self._run(
            _Call(
                "index.research.consent.create",
                _bound(
                    ReleaseConsentView,
                    lambda view: (
                        view.revision_id == reference.revision_id
                        and view.manifest_digest == spec.manifest_digest
                        and view.disclosure_digest == spec.disclosure_digest
                        and view.audience == spec.audience
                    ),
                    "Consent response differs from exact requested content",
                ),
                path_parameters=_revision(reference),
                json_body=_body(spec),
            )
        )

    def attest(self, reference: ContributionReference, spec: ReproductionAttestationSpec) -> Any:
        """Record a reproduction observation against the exact derivation binding.

        Args:
            reference: Exact Contribution and revision addressed by this operation.
            spec: Typed exact-input request required by this operation; does not infer publication consent.

        Returns:
            ReproductionReceipt | Awaitable[ReproductionReceipt]: Validated result bound to the requested exact identities and intent.

        Raises:
            ValueError: Requested constraints or returned identity/content bindings are invalid.

        Examples:
            result = index.contributions.research.attest(reference, spec)
        """
        return self._run(
            _Call(
                "index.research.reproduction.create",
                _bound(
                    ReproductionReceipt,
                    lambda receipt: receipt == spec.receipt,
                    "Attestation response differs from exact submitted receipt",
                ),
                path_parameters=_revision(reference),
                json_body=_body(spec),
            )
        )

    def revoke(self, reference: ContributionReference, spec: ResearchRevocationSpec) -> Any:
        """Revoke the exact approved disclosure; historical downloads cannot be recalled.

        Args:
            reference: Exact Contribution and revision addressed by this operation.
            spec: Typed exact-input request required by this operation; does not infer publication consent.

        Returns:
            dict | Awaitable[dict]: Validated result bound to the requested exact identities and intent.

        Raises:
            ValueError: Requested constraints or returned identity/content bindings are invalid.

        Examples:
            result = index.contributions.research.revoke(reference, spec)
        """
        def parse(payload):
            if (
                payload != {"revision_id": reference.revision_id, "revoked": True}
                or payload.get("revoked") is not True
            ):
                raise ValueError("Revocation response does not match requested revision")
            return payload

        return self._run(
            _Call(
                "index.research.disclosure.revoke",
                parse,
                path_parameters=_revision(reference),
                json_body=_body(spec),
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
                json_body=_body(spec),
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
                json_body=_body(spec),
                headers=None if idempotency_key is None else _key(idempotency_key),
            )
        )


class AssetsAPI(_Resource):
    def retrieve(self, reference: ContributionReference, asset_id: str) -> Any:
        """Download one declared asset's bytes under current authorization.

        Not available at the public launch: the service answers 403
        ``index_asset_download_unavailable`` (not retried). Search results and
        Contribution reads include quoted passages from the files instead.
        """
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
        self.research = ResearchAPI(run, asynchronous)
        self.classifications = ClassificationsAPI(run, asynchronous)
        self.assessments = AssessmentsAPI(run, asynchronous)
        self.reviews = ReviewsAPI(run, asynchronous)
        self.assets = AssetsAPI(run, asynchronous)

    def create(self, *, idempotency_key: str) -> Any:
        """Create a private draft. Persist and reuse the key after uncertain failures."""
        return self._run(
            _Call(
                "index.contributions.create",
                ContributionDraft.model_validate,
                json_body={},
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
                json_body=spec.model_dump(mode="json"),
                headers=_key(idempotency_key, required="Research draft creation"),
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
                lambda payload: _upload_result(payload, draft, spec),
                path_parameters=_revision(reference),
                json_body=spec.model_dump(mode="json"),
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
            # This is the artifact-publication revision inside the draft's own
            # collection (backend pins it to 1). Repaired child Contribution
            # revisions get a fresh collection, so they also transfer at 1.
            or prepared.transfer.revision != 1
        ):
            raise ValueError("Finalization transfer does not match draft collection")
        return self._run(
            _Call(
                "index.contributions.upload.finalize",
                lambda payload: _finalize_result(payload, prepared),
                path_parameters=_revision(draft.reference),
                json_body={"publication_id": prepared.transfer.publication_id},
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
                json_body=spec.model_dump(mode="json"),
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
                json_body=_body(spec),
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
                json_body=_body(spec),
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
        return self._run(_Call("index.me.promo_credit", PromoCreditSummary.model_validate))

    def usage(self) -> Any:
        return self._run(_Call("index.me.usage", IndexUsageSummary.model_validate))

    def access_funding(self) -> Any:
        """Read effective Fast/Deep access, consent, allowances, and wallet holds."""
        return self._run(_Call("index.me.access_funding", AccessFundingAccount.model_validate))

    def update_billing_policy(self, mode: SearchMode | str, policy: BillingPolicyUpdate) -> Any:
        """Replace this org's versioned wallet consent and cap for one mode."""
        selected = SearchMode(mode)
        return self._run(
            _Call(
                "index.me.access_funding.update",
                lambda payload: payload,
                path_parameters={"mode": selected.value},
                json_body=policy.model_dump(mode="json"),
            )
        )

    def search_usage(self, search_id: str) -> Any:
        """Return the physical-consumption and charge receipt for one search."""
        return self._run(
            _Call(
                "index.me.search_usage_receipt",
                SearchUsageReceipt.model_validate,
                path_parameters={"search_id": search_id},
            )
        )

    def operation_usage(
        self,
        period_start: datetime,
        period_end: datetime,
        *,
        limit: int = 100,
        offset: int = 0,
    ) -> Any:
        """Aggregate search attempts, cost, and charges for an aware time range."""
        if period_start.tzinfo is None or period_end.tzinfo is None:
            raise ValueError("Index usage periods must be timezone-aware")
        if not 1 <= limit <= 500 or offset < 0:
            raise ValueError("Index usage page must use limit 1..500 and offset >= 0")
        return self._run(
            _Call(
                "index.me.operation_usage_summary",
                SearchUsageSummary.model_validate,
                params={
                    "period_start": period_start.isoformat(),
                    "period_end": period_end.isoformat(),
                    "include_consumption": "true",
                    "limit": limit,
                    "offset": offset,
                },
            )
        )

    def export_operation_usage(self, period_start: datetime, period_end: datetime) -> Any:
        """Download the tenant-scoped operation aggregate as CSV bytes."""
        if period_start.tzinfo is None or period_end.tzinfo is None:
            raise ValueError("Index usage periods must be timezone-aware")
        return self._run(
            _Call(
                "index.me.operation_usage_export",
                lambda payload: payload,
                params={
                    "period_start": period_start.isoformat(),
                    "period_end": period_end.isoformat(),
                    "include_consumption": "true",
                },
                raw=True,
            )
        )

    def rewards(self) -> Any:
        return self._run(_Call("index.me.rewards.list", MyRewards.model_validate))

    def update_profile(self, spec: ProfileSpec) -> Any:
        return self._run(
            _Call("index.me.profile.update", ProfileView.model_validate, json_body=_body(spec))
        )

    def update_pins(self, spec: ProfilePinsSpec) -> Any:
        return self._run(
            _Call("index.me.profile.pins.update", ProfileView.model_validate, json_body=_body(spec))
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
                json_body=_body(spec),
                headers=_key(idempotency_key, required="Reward award"),
            )
        )

    def reverse(self, award_id: str, spec: RewardReverseSpec) -> Any:
        return self._run(
            _Call(
                "index.rewards.reverse",
                RewardAward.model_validate,
                path_parameters={"award_id": award_id},
                json_body=_body(spec),
            )
        )


class ContestEntriesAPI(_Resource):
    def create(self, contest_id: str, spec: ContestEntrySpec) -> Any:
        return self._run(
            _Call(
                "index.contests.entries.create",
                ContestEntry.model_validate,
                path_parameters={"contest_id": contest_id},
                json_body=_body(spec),
            )
        )

    def score(self, contest_id: str, entry_id: str, spec: ContestScoreSpec) -> Any:
        return self._run(
            _Call(
                "index.contests.entries.score",
                ContestEntry.model_validate,
                path_parameters={"contest_id": contest_id, "entry_id": entry_id},
                json_body=_body(spec),
            )
        )

    def review(self, contest_id: str, entry_id: str, spec: ContestReviewSpec) -> Any:
        return self._run(
            _Call(
                "index.contests.entries.review",
                ContestEntry.model_validate,
                path_parameters={"contest_id": contest_id, "entry_id": entry_id},
                json_body=_body(spec),
            )
        )


class ContestsAPI(_Resource):
    def __init__(self, run: Callable[[_Call], Any], asynchronous: bool) -> None:
        super().__init__(run, asynchronous)
        self.entries = ContestEntriesAPI(run, asynchronous)

    def create(self, spec: ContestSpec) -> Any:
        return self._run(
            _Call("index.contests.create", ContestView.model_validate, json_body=_body(spec))
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
                json_body=_body(spec),
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


class QaAPI(_Resource):
    """Private revision-bound QA. See sibling backend contribution-qa-cases spec.

    Backend rollout is disabled until qualified. No local fallback, automatic
    scientific approval, public publication or reward is implied by these calls.
    Every method is awaitable on AsyncIndexAPI and blocking on IndexAPI.
    """

    def _case(self, case_id):
        return {"case_id": str(UUID(str(case_id)))}

    def _cursor(self, after):
        if type(after) is not int or after < 0:
            raise ValueError("after must be a nonnegative sequence")
        return {"after": after}

    def create_case(self, spec: CreateCaseSpec):
        """Open review for the exact sealed Contribution revision.

        Every call is blocking on IndexAPI and awaitable on AsyncIndexAPI.
        Backend permissions and rollout remain authoritative.

        Args:
            spec: Typed request naming the exact inputs required by this operation.

        Returns:
            CaseView | Awaitable[CaseView]: CaseView bound to the submitted revision, manifest and rubric.

        Raises:
            ValueError: Request identifiers, bounds or returned binding are invalid.
        Examples:
            result = index.qa.create_case(spec)
        """
        return self._run(
            _Call(
                "index.qa.cases.create",
                _bound(
                    CaseView,
                    lambda view: view.reference == spec.reference
                    and view.manifest_digest == spec.manifest_digest
                    and view.rubric_version == spec.rubric_version,
                    "QA case differs from sealed request",
                ),
                json_body=spec.model_dump(mode="json"),
            )
        )

    def case(self, case_id):
        """Read the current authorized revision-bound QA case.

        Every call is blocking on IndexAPI and awaitable on AsyncIndexAPI.
        Backend permissions and rollout remain authoritative.

        Args:
            case_id (str | UUID): UUID of the authorized revision-bound QA case.

        Returns:
            CaseView | Awaitable[CaseView]: CaseView matching the requested case identifier.

        Raises:
            ValueError: Request identifiers, bounds or returned binding are invalid.
        Examples:
            result = index.qa.case(case_id)
        """
        path = self._case(case_id)
        return self._run(
            _Call(
                "index.qa.cases.get",
                _bound(
                    CaseView,
                    lambda view: str(view.case_id) == path["case_id"],
                    "QA case differs from request",
                ),
                path_parameters=path,
            )
        )

    def events(self, case_id, *, after=0):
        """Read the next authorized QA conversation page.

        Every call is blocking on IndexAPI and awaitable on AsyncIndexAPI.
        Backend permissions and rollout remain authoritative.

        Args:
            case_id (str | UUID): UUID of the authorized revision-bound QA case.
            after (int): Nonnegative case-sequence cursor; zero starts the first page.

        Returns:
            CaseEvents | Awaitable[CaseEvents]: CaseEvents containing ordered events and a continuation cursor.

        Raises:
            ValueError: Request identifiers, bounds or returned binding are invalid.
        Examples:
            result = index.qa.events(case_id, after=0)
        """
        return self._run(
            _Call(
                "index.qa.events.list",
                CaseEvents.model_validate,
                path_parameters=self._case(case_id),
                params=self._cursor(after),
            )
        )

    def append_event(self, case_id, spec: CaseEventSpec, *, idempotency_key: str):
        """Append a shared conversation action; use dedicated routes for fenced decisions.

        Every call is blocking on IndexAPI and awaitable on AsyncIndexAPI.
        Backend permissions and rollout remain authoritative.

        Args:
            case_id (str | UUID): UUID of the authorized revision-bound QA case.
            spec: Typed request naming the exact inputs required by this operation.
            idempotency_key: Persisted request key reused for an uncertain retry of the same operation.

        Returns:
            CaseEventView | Awaitable[CaseEventView]: CaseEventView matching the next expected sequence, action and message.

        Raises:
            ValueError: Request identifiers, bounds or returned binding are invalid.
        Examples:
            result = index.qa.append_event(case_id, spec, idempotency_key=idempotency_key)
        """
        if spec.action in FENCED_ACTIONS:
            raise ValueError(
                f"{spec.action.value} needs the manifest/rubric-fenced route; use "
                "qa.appeal, qa.escalate or qa.adjudicate"
            )
        return self._run(
            _Call(
                "index.qa.events.create",
                _bound(
                    CaseEventView,
                    lambda event: event.sequence == spec.expected_version + 1
                    and event.action == spec.action
                    and event.message == spec.message,
                    "QA event differs from request",
                ),
                path_parameters=self._case(case_id),
                json_body=spec.model_dump(mode="json"),
                headers=_key(idempotency_key, required="QA event"),
            )
        )

    def _act(self, action: CaseAction, case_id, expected_version: int, message: str, key: str):
        return self.append_event(
            case_id,
            CaseEventSpec(expected_version=expected_version, action=action, message=message),
            idempotency_key=key,
        )

    # Plain conversation actions use POST /qa/cases/{id}/events; the backend state
    # machine enforces per-role legality and refuses appeal/escalate/adjudicate there.
    def send_message(self, case_id, expected_version: int, message: str, *, idempotency_key: str):
        """Send a shared message at the expected QA case version.

        Every call is blocking on IndexAPI and awaitable on AsyncIndexAPI.
        Backend permissions and rollout remain authoritative.

        Args:
            case_id (str | UUID): UUID of the authorized revision-bound QA case.
            expected_version: Current case version expected by this write; stale versions are refused.
            message: Bounded message text for the shared conversation action.
            idempotency_key: Persisted request key reused for an uncertain retry of the same operation.

        Returns:
            CaseEventView | Awaitable[CaseEventView]: Recorded shared message event.

        Raises:
            ValueError: Request identifiers, bounds or returned binding are invalid.
        Examples:
            result = index.qa.send_message(case_id, expected_version, message, idempotency_key=idempotency_key)
        """
        return self._act(CaseAction.MESSAGE, case_id, expected_version, message, idempotency_key)

    def request_changes(
        self, case_id, expected_version: int, message: str, *, idempotency_key: str
    ):
        """Request contributor changes against the current QA version.

        Every call is blocking on IndexAPI and awaitable on AsyncIndexAPI.
        Backend permissions and rollout remain authoritative.

        Args:
            case_id (str | UUID): UUID of the authorized revision-bound QA case.
            expected_version: Current case version expected by this write; stale versions are refused.
            message: Bounded message text for the shared conversation action.
            idempotency_key: Persisted request key reused for an uncertain retry of the same operation.

        Returns:
            CaseEventView | Awaitable[CaseEventView]: Recorded change-request event; does not authorize a repaired revision.

        Raises:
            ValueError: Request identifiers, bounds or returned binding are invalid.
        Examples:
            result = index.qa.request_changes(case_id, expected_version, message, idempotency_key=idempotency_key)
        """
        return self._act(
            CaseAction.REQUEST_CHANGES, case_id, expected_version, message, idempotency_key
        )

    def respond(self, case_id, expected_version: int, message: str, *, idempotency_key: str):
        """Respond to findings at the expected QA case version.

        Every call is blocking on IndexAPI and awaitable on AsyncIndexAPI.
        Backend permissions and rollout remain authoritative.

        Args:
            case_id (str | UUID): UUID of the authorized revision-bound QA case.
            expected_version: Current case version expected by this write; stale versions are refused.
            message: Bounded message text for the shared conversation action.
            idempotency_key: Persisted request key reused for an uncertain retry of the same operation.

        Returns:
            CaseEventView | Awaitable[CaseEventView]: Recorded contributor response event.

        Raises:
            ValueError: Request identifiers, bounds or returned binding are invalid.
        Examples:
            result = index.qa.respond(case_id, expected_version, message, idempotency_key=idempotency_key)
        """
        return self._act(CaseAction.RESPOND, case_id, expected_version, message, idempotency_key)

    # Appeal, escalation, adjudication and internal notes are dedicated POST routes.
    # Each names the exact case version, sealed manifest digest and rubric version, so
    # a decision written against other bytes or another rubric is refused by the backend.
    def _fenced(
        self,
        operation: str,
        case_id,
        spec: FencedCaseRequest,
        action: CaseAction,
        visibility: EventVisibility,
        idempotency_key: str,
    ):
        return self._run(
            _Call(
                operation,
                _bound(
                    CaseEventView,
                    lambda event: event.sequence == spec.expected_version + 1
                    and event.action == action
                    and event.message == spec.message
                    and event.visibility == visibility,
                    "QA event differs from request",
                ),
                path_parameters=self._case(case_id),
                json_body=spec.model_dump(mode="json"),
                headers=_key(idempotency_key, required="QA event"),
            )
        )

    def escalate(self, case_id, spec: EscalationSpec, *, idempotency_key: str):
        """Escalate the exact manifest and rubric to a coordinator.

        Every call is blocking on IndexAPI and awaitable on AsyncIndexAPI.
        Backend permissions and rollout remain authoritative.

        Args:
            case_id (str | UUID): UUID of the authorized revision-bound QA case.
            spec: Typed request naming the exact inputs required by this operation.
            idempotency_key: Persisted request key reused for an uncertain retry of the same operation.

        Returns:
            CaseEventView | Awaitable[CaseEventView]: Shared escalation event; does not grant publication authority.

        Raises:
            ValueError: Request identifiers, bounds or returned binding are invalid.
        Examples:
            result = index.qa.escalate(case_id, spec, idempotency_key=idempotency_key)
        """
        return self._fenced(
            "index.qa.escalations.create",
            case_id,
            spec,
            CaseAction.ESCALATE,
            EventVisibility.SHARED,
            idempotency_key,
        )

    def appeal(self, case_id, spec: AppealSpec, *, idempotency_key: str):
        """Appeal the current decision against exact reviewed inputs.

        Every call is blocking on IndexAPI and awaitable on AsyncIndexAPI.
        Backend permissions and rollout remain authoritative.

        Args:
            case_id (str | UUID): UUID of the authorized revision-bound QA case.
            spec: Typed request naming the exact inputs required by this operation.
            idempotency_key: Persisted request key reused for an uncertain retry of the same operation.

        Returns:
            CaseEventView | Awaitable[CaseEventView]: Shared appeal event; publication remains separate.

        Raises:
            ValueError: Request identifiers, bounds or returned binding are invalid.
        Examples:
            result = index.qa.appeal(case_id, spec, idempotency_key=idempotency_key)
        """
        return self._fenced(
            "index.qa.appeals.create",
            case_id,
            spec,
            CaseAction.APPEAL,
            EventVisibility.SHARED,
            idempotency_key,
        )

    def adjudicate(self, case_id, spec: AdjudicationSpec, *, idempotency_key: str):
        """Reopen independent review through an authorized coordinator.

        Every call is blocking on IndexAPI and awaitable on AsyncIndexAPI.
        Backend permissions and rollout remain authoritative.

        Args:
            case_id (str | UUID): UUID of the authorized revision-bound QA case.
            spec: Typed request naming the exact inputs required by this operation.
            idempotency_key: Persisted request key reused for an uncertain retry of the same operation.

        Returns:
            CaseEventView | Awaitable[CaseEventView]: Shared adjudication event, never direct publication approval.

        Raises:
            ValueError: Request identifiers, bounds or returned binding are invalid.
        Examples:
            result = index.qa.adjudicate(case_id, spec, idempotency_key=idempotency_key)
        """
        return self._fenced(
            "index.qa.adjudications.create",
            case_id,
            spec,
            CaseAction.ADJUDICATE,
            EventVisibility.SHARED,
            idempotency_key,
        )

    def add_internal_note(self, case_id, spec: InternalNoteSpec, *, idempotency_key: str):
        """Record a reviewer/coordinator note excluded from contributor disclosure.

        Every call is blocking on IndexAPI and awaitable on AsyncIndexAPI.
        Backend permissions and rollout remain authoritative.

        Args:
            case_id (str | UUID): UUID of the authorized revision-bound QA case.
            spec: Typed request naming the exact inputs required by this operation.
            idempotency_key: Persisted request key reused for an uncertain retry of the same operation.

        Returns:
            CaseEventView | Awaitable[CaseEventView]: Internal message event whose visibility is checked before return.

        Raises:
            ValueError: Request identifiers, bounds or returned binding are invalid.
        Examples:
            result = index.qa.add_internal_note(case_id, spec, idempotency_key=idempotency_key)
        """
        return self._fenced(
            "index.qa.notes.create",
            case_id,
            spec,
            CaseAction.MESSAGE,
            EventVisibility.INTERNAL,
            idempotency_key,
        )

    def invite_reviewer(self, case_id, spec: AssignmentSpec):
        """Invite an independent reviewer for this exact QA case.

        Every call is blocking on IndexAPI and awaitable on AsyncIndexAPI.
        Backend permissions and rollout remain authoritative.

        Args:
            case_id (str | UUID): UUID of the authorized revision-bound QA case.
            spec: Typed request naming the exact inputs required by this operation.

        Returns:
            AssignmentView | Awaitable[AssignmentView]: AssignmentView matching the requested case, reviewer and organization.

        Raises:
            ValueError: Request identifiers, bounds or returned binding are invalid.
        Examples:
            result = index.qa.invite_reviewer(case_id, spec)
        """
        path = self._case(case_id)
        return self._run(
            _Call(
                "index.qa.assignments.create",
                _bound(
                    AssignmentView,
                    lambda invitation: str(invitation.case_id) == path["case_id"]
                    and invitation.reviewer_user_id == spec.reviewer_user_id
                    and invitation.reviewer_org_id == spec.reviewer_org_id,
                    "QA invitation differs from request",
                ),
                path_parameters=path,
                json_body=spec.model_dump(mode="json"),
            )
        )

    def assignments(self):
        """List the caller’s authorized QA reviewer assignments.

        Every call is blocking on IndexAPI and awaitable on AsyncIndexAPI.
        Backend permissions and rollout remain authoritative.

        Returns:
            tuple[AssignmentView, ...] | Awaitable[tuple[AssignmentView, ...]]: Tuple of at most 100 validated AssignmentView records.

        Raises:
            ValueError: Request identifiers, bounds or returned binding are invalid.
        Examples:
            result = index.qa.assignments()
        """
        def parse(payload):
            if not isinstance(payload, list) or len(payload) > 100:
                raise ValueError("QA assignment list bound invalid")
            return tuple(AssignmentView.model_validate(item) for item in payload)

        return self._run(_Call("index.qa.assignments.list", parse))

    def _assignment(self, action, assignment_id, spec=None):
        identifier = str(UUID(str(assignment_id)))
        return self._run(
            _Call(
                "index.qa.assignments." + action,
                _bound(
                    AssignmentView,
                    lambda assignment: str(assignment.assignment_id) == identifier,
                    "QA assignment differs from request",
                ),
                path_parameters={"assignment_id": identifier},
                json_body=spec.model_dump(mode="json") if spec else None,
            )
        )

    def accept_assignment(self, assignment_id, spec: AcceptAssignmentSpec):
        """Accept a reviewer assignment with conflict and provenance declarations.

        Every call is blocking on IndexAPI and awaitable on AsyncIndexAPI.
        Backend permissions and rollout remain authoritative.

        Args:
            assignment_id (str | UUID): UUID of the reviewer assignment to accept or revoke.
            spec: Typed request naming the exact inputs required by this operation.

        Returns:
            AssignmentView | Awaitable[AssignmentView]: AssignmentView matching the requested assignment identifier.

        Raises:
            ValueError: Request identifiers, bounds or returned binding are invalid.
        Examples:
            result = index.qa.accept_assignment(assignment_id, spec)
        """
        return self._assignment("accept", assignment_id, spec)

    def revoke_assignment(self, assignment_id):
        """Revoke an authorized reviewer assignment.

        Every call is blocking on IndexAPI and awaitable on AsyncIndexAPI.
        Backend permissions and rollout remain authoritative.

        Args:
            assignment_id (str | UUID): UUID of the reviewer assignment to accept or revoke.

        Returns:
            AssignmentView | Awaitable[AssignmentView]: Current AssignmentView matching the requested assignment.

        Raises:
            ValueError: Request identifiers, bounds or returned binding are invalid.
        Examples:
            result = index.qa.revoke_assignment(assignment_id)
        """
        return self._assignment("revoke", assignment_id)

    def package(self, case: CaseView):
        """Read the exact sealed package under current QA assignment authority.

        Every call is blocking on IndexAPI and awaitable on AsyncIndexAPI.
        Backend permissions and rollout remain authoritative.

        Args:
            case: Current CaseView identifying the exact revision whose package is requested.

        Returns:
            ContributionPackage | Awaitable[ContributionPackage]: ContributionPackage whose Contribution and revision match the supplied case.

        Raises:
            ValueError: Request identifiers, bounds or returned binding are invalid.
        Examples:
            result = index.qa.package(case)
        """
        return self._run(
            _Call(
                "index.qa.package.retrieve",
                _bound(
                    ContributionPackage,
                    lambda package: (package.contribution_id, package.revision_id)
                    == (case.reference.contribution_id, case.reference.revision_id),
                    "QA package differs from case revision",
                ),
                path_parameters=self._case(case.case_id),
            )
        )

    def asset(self, case_id, asset_id: str, *, digest_sha256: str, size_bytes: int):
        """Read exact assigned bytes using digest and size from the sealed package.

        Every call is blocking on IndexAPI and awaitable on AsyncIndexAPI.
        Backend permissions and rollout remain authoritative.

        Args:
            case_id (str | UUID): UUID of the authorized revision-bound QA case.
            asset_id: Identifier of a declared asset in the sealed package.
            digest_sha256: Expected lowercase SHA-256 of the sealed asset bytes.
            size_bytes: Exact expected byte count, from zero through 4 MiB inclusive.

        Returns:
            bytes | Awaitable[bytes]: Raw bytes verified against the requested SHA-256 and byte count.

        Raises:
            ValueError: Request identifiers, bounds or returned binding are invalid.
        Examples:
            result = index.qa.asset(case_id, asset_id, digest_sha256=digest_sha256, size_bytes=size_bytes)
        """
        if (
            not re.fullmatch(r"[0-9a-f]{64}", digest_sha256)
            or type(size_bytes) is not int
            or not 0 <= size_bytes <= 4 * 1024 * 1024
        ):
            raise ValueError("QA asset needs exact bounded size and SHA256")

        def parse(body):
            if (
                not isinstance(body, bytes)
                or len(body) != size_bytes
                or sha256(body).hexdigest() != digest_sha256
            ):
                raise ValueError("QA asset differs from sealed declaration")
            return body

        return self._run(
            _Call(
                "index.qa.assets.retrieve",
                parse,
                path_parameters={**self._case(case_id), "asset_id": asset_id},
                raw=True,
            )
        )

    def checks(self, case_id, *, after=0):
        """Read recorded producer check attempts for one QA case.

        Every call is blocking on IndexAPI and awaitable on AsyncIndexAPI.
        Backend permissions and rollout remain authoritative.

        Args:
            case_id (str | UUID): UUID of the authorized revision-bound QA case.
            after (int): Nonnegative case-sequence cursor; zero starts the first page.

        Returns:
            CheckReport | Awaitable[CheckReport]: CheckReport bound to the case with attempts, findings and continuation cursor.

        Raises:
            ValueError: Request identifiers, bounds or returned binding are invalid.
        Examples:
            result = index.qa.checks(case_id, after=0)
        """
        path = self._case(case_id)
        return self._run(
            _Call(
                "index.qa.checks.list",
                _bound(
                    CheckReport,
                    lambda report: all(
                        str(attempt.case_id) == path["case_id"] for attempt in report.attempts
                    ),
                    "QA checks differ from case",
                ),
                path_parameters=path,
                params=self._cursor(after),
            )
        )

    def record_check(self, case_id, spec: RecordCheckSpec, *, idempotency_key: str):
        """Record a fenced producer check receipt and its actionable findings.

        Every call is blocking on IndexAPI and awaitable on AsyncIndexAPI.
        Backend permissions and rollout remain authoritative.

        Args:
            case_id (str | UUID): UUID of the authorized revision-bound QA case.
            spec: Typed request naming the exact inputs required by this operation.
            idempotency_key: Persisted request key reused for an uncertain retry of the same operation.

        Returns:
            CheckAttemptView | Awaitable[CheckAttemptView]: CheckAttemptView matching the attempt, case, manifest and rubric.

        Raises:
            ValueError: Request identifiers, bounds or returned binding are invalid.
        Examples:
            result = index.qa.record_check(case_id, spec, idempotency_key=idempotency_key)
        """
        path = self._case(case_id)
        return self._run(
            _Call(
                "index.qa.checks.record",
                _bound(
                    CheckAttemptView,
                    lambda attempt: str(attempt.case_id) == path["case_id"]
                    and attempt.attempt_id == spec.attempt_id
                    and attempt.manifest_digest == spec.manifest_digest
                    and attempt.rubric_version == spec.rubric_version,
                    "QA check differs from request",
                ),
                path_parameters=path,
                json_body=spec.model_dump(mode="json"),
                headers=_key(idempotency_key, required="QA check"),
            )
        )

    def preflight(self, case_id, spec: RunPreflightSpec):
        """Request the backend’s configured bounded QA preflight checks.

        Every call is blocking on IndexAPI and awaitable on AsyncIndexAPI.
        Backend permissions and rollout remain authoritative.

        Args:
            case_id (str | UUID): UUID of the authorized revision-bound QA case.
            spec: Typed request naming the exact inputs required by this operation.

        Returns:
            PreflightResult | Awaitable[PreflightResult]: PreflightResult with attempts matching this case and producer run.

        Raises:
            ValueError: Request identifiers, bounds or returned binding are invalid.
        Examples:
            result = index.qa.preflight(case_id, spec)
        """
        path = self._case(case_id)
        return self._run(
            _Call(
                "index.qa.checks.preflight",
                _bound(
                    PreflightResult,
                    lambda batch: all(
                        str(attempt.case_id) == path["case_id"] and attempt.run_id == spec.run_id
                        for attempt in batch.attempts
                    ),
                    "QA preflight differs from request",
                ),
                path_parameters=path,
                json_body=spec.model_dump(mode="json"),
            )
        )

    def secret_scan(self, case_id, spec: RunPreflightSpec):
        """Request the configured privacy secret scan for the current sealed package.

        Every call is blocking on IndexAPI and awaitable on AsyncIndexAPI.
        Backend permissions and rollout remain authoritative.

        Args:
            case_id (str | UUID): UUID of the authorized revision-bound QA case.
            spec: Typed request naming the exact inputs required by this operation.

        Returns:
            CheckAttemptView | Awaitable[CheckAttemptView]: CheckAttemptView for privacy.secret_scan matching this case and run.

        Raises:
            ValueError: Request identifiers, bounds or returned binding are invalid.
        Examples:
            result = index.qa.secret_scan(case_id, spec)
        """
        path = self._case(case_id)
        return self._run(
            _Call(
                "index.qa.checks.secret_scan",
                _bound(
                    CheckAttemptView,
                    lambda attempt: str(attempt.case_id) == path["case_id"]
                    and attempt.run_id == spec.run_id
                    and attempt.gate == "privacy.secret_scan",
                    "QA secret scan differs from request",
                ),
                path_parameters=path,
                json_body=spec.model_dump(mode="json"),
            )
        )

    def reviews(self, case_id, *, after=0):
        """Read the next independent review page for this QA case.

        Every call is blocking on IndexAPI and awaitable on AsyncIndexAPI.
        Backend permissions and rollout remain authoritative.

        Args:
            case_id (str | UUID): UUID of the authorized revision-bound QA case.
            after (int): Nonnegative case-sequence cursor; zero starts the first page.

        Returns:
            ReviewReport | Awaitable[ReviewReport]: ReviewReport with strictly ordered case sequences and a validated continuation.

        Raises:
            ValueError: Request identifiers, bounds or returned binding are invalid.
        Examples:
            result = index.qa.reviews(case_id, after=0)
        """
        path = self._case(case_id)

        def parse(payload):
            report = ReviewReport.model_validate(payload)
            sequences = [fact.case_sequence for fact in report.reviews]
            if any(
                str(fact.case_id) != path["case_id"] or fact.case_sequence <= after
                for fact in report.reviews
            ) or sequences != sorted(set(sequences)):
                raise ValueError("QA review report differs from requested page")
            if report.next_after is not None and (
                len(report.reviews) != 3 or report.next_after != sequences[-1]
            ):
                raise ValueError("QA review continuation invalid")
            return report

        return self._run(
            _Call("index.qa.reviews.list", parse, path_parameters=path, params=self._cursor(after))
        )

    def record_review(self, case_id, spec: RecordReviewSpec, *, idempotency_key: str):
        """Record an independent assignment’s rubric judgments at exact sealed inputs.

        Every call is blocking on IndexAPI and awaitable on AsyncIndexAPI.
        Backend permissions and rollout remain authoritative.

        Args:
            case_id (str | UUID): UUID of the authorized revision-bound QA case.
            spec: Typed request naming the exact inputs required by this operation.
            idempotency_key: Persisted request key reused for an uncertain retry of the same operation.

        Returns:
            ReviewFact | Awaitable[ReviewFact]: ReviewFact matching the supplied case and review; not publication authority.

        Raises:
            ValueError: Request identifiers, bounds or returned binding are invalid.
        Examples:
            result = index.qa.record_review(case_id, spec, idempotency_key=idempotency_key)
        """
        path = self._case(case_id)
        headers = _key(idempotency_key, required="QA content review")
        if len(idempotency_key) > 119:
            raise ValueError("QA review key exceeds 119 characters")
        return self._run(
            _Call(
                "index.qa.reviews.record",
                _bound(
                    ReviewFact,
                    lambda fact: str(fact.case_id) == path["case_id"] and fact.review == spec,
                    "QA review differs from sealed request",
                ),
                path_parameters=path,
                json_body=spec.model_dump(mode="json"),
                headers=headers,
            )
        )


class _IndexRoot(_Resource):
    def __init__(self, run: Callable[[_Call], Any], asynchronous: bool) -> None:
        super().__init__(run, asynchronous)
        self.contents = ContentsAPI(run, asynchronous)
        self.searches = SearchesAPI(run, asynchronous)
        self.contributions = ContributionsAPI(run, asynchronous)
        self.qa = QaAPI(run, asynchronous)
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
        mode: SearchMode | str | None = None,
        limits: SearchExecutionLimits | None = None,
        billing: SearchBillingConstraints | None = None,
        idempotency_key: str | None = None,
    ) -> Any:
        """Execute fast immediately or create-and-wait for one durable deep Search.

        Reuse the key after an uncertain response. A local deep wait timeout
        raises ``SearchWaitTimeoutError`` with its Search ID and does not cancel or
        resubmit work.
        """
        spec = _search_spec(spec, query, scope, filters, max_results, mode, limits, billing)
        if spec.mode == SearchMode.DEEP:
            key = _key(idempotency_key)["Idempotency-Key"]
            handle = self.searches.create(spec, idempotency_key=key)
            if self._asynchronous:

                async def await_deep() -> SearchResult:
                    asynchronous_handle = await handle
                    return await asynchronous_handle.wait(
                        timeout_seconds=(spec.limits or SearchExecutionLimits()).deadline_seconds
                        + 5
                    )

                return await_deep()
            return handle.wait(
                timeout_seconds=(spec.limits or SearchExecutionLimits()).deadline_seconds + 5
            )
        return self._run(
            _Call(
                "index.search",
                lambda payload: _search_result(payload, spec),
                json_body=spec.model_dump(mode="json"),
                headers=_key(idempotency_key),
            )
        )


class IndexAPI(_IndexRoot, PublicSearchOperations):
    """Blocking client over ``HttpTransport``.

    ``search()`` is the authenticated, funded path (private scope, wallet).
    ``public_search()`` is the free Index Search v0.2 public route, which is
    anonymous-only: the backend refuses this client's API key with
    ``PublicSearchAuthenticatedError`` (409). Use ``search()`` (paid) instead.
    """

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
                lambda payload: _contents_result(payload, spec),
                json_body=spec.model_dump(mode="json"),
            )
        )


class PublicAssetsAPI(_Resource):
    def retrieve(self, reference: ContributionReference, asset_id: str) -> Any:
        """Download one declared asset of a published revision, with no account.

        Not available at the public launch: the service answers 403
        ``index_asset_download_unavailable`` (not retried).
        """
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
        self.release_research = PublicReleaseResearchAPI(run, asynchronous)
        self.classifications = PublicClassificationAPI(run, asynchronous)
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
        return self._run(_Call("index.public.capabilities", Capabilities.model_validate))


class PublicIndexAPI(_PublicIndexRoot, PublicSearchOperations):
    """Blocking anonymous API: browse plus the free public search (v0.2)."""

    def __init__(self, transport: HttpTransport) -> None:
        self._transport = transport
        super().__init__(_sync_runner(transport), asynchronous=False)


class AsyncPublicIndexAPI(_PublicIndexRoot):
    """Async, browse-only API over an injected credential-free transport."""

    def __init__(self, transport: AsyncHttpTransport) -> None:
        self._transport = transport
        super().__init__(_async_runner(transport), asynchronous=True)
