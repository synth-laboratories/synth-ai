"""Stable synchronous and asynchronous Factory operations.

# See: specifications/sdk/core_research_migration.md
"""

from __future__ import annotations

from synth_ai.core.contracts.json_value import JsonObject, JsonValue
from synth_ai.core.http.async_transport import AsyncHttpTransport
from synth_ai.core.http.request import HttpRequest
from synth_ai.core.http.transport import HttpTransport
from synth_ai.core.research.contracts._wire import array_value
from synth_ai.core.research.contracts.common import EffortId, FactoryCandidateId, FactoryId
from synth_ai.core.research.contracts.factories import (
    Effort,
    EffortPatch,
    EffortSpec,
    Factory,
    FactoryCandidate,
    FactoryCandidateGradingRequest,
    FactoryCandidateGradingStatus,
    FactoryChampionDecision,
    FactoryChampionEvent,
    FactoryChampionRollbackRequest,
    FactoryChampionSelectRequest,
    FactoryPatch,
    FactorySpec,
    FactoryTransition,
    FactoryTransitionResult,
)
from synth_ai.core.research.contracts.factory_lenses import (
    FactoryBestResults,
    FactoryEvaluationLens,
    FactoryLensSpec,
    FactoryPreferenceEvent,
    FactoryPreferenceRequest,
    FactoryResultEvaluation,
    FactoryResultEvaluationRequest,
)
from synth_ai.core.research.operations import research_operation
from synth_ai.core.research.traces import (
    AsyncFactoryTraceStoreAPI,
    FactoryTraceStoreAPI,
)


def _request(
    operation_id: str,
    path: str,
    *,
    query: JsonObject | None = None,
    body: JsonObject | None = None,
) -> HttpRequest:
    return HttpRequest(
        research_operation(operation_id),
        path,
        query=query or {},
        body=body,
    )


def _factories(value: JsonValue) -> tuple[Factory, ...]:
    return tuple(
        Factory.from_wire(item) for item in array_value(value, operation_id="list_factories")
    )


def _efforts(value: JsonValue) -> tuple[Effort, ...]:
    return tuple(
        Effort.from_wire(item) for item in array_value(value, operation_id="list_factory_efforts")
    )


def _candidates(value: JsonValue) -> tuple[FactoryCandidate, ...]:
    return tuple(
        FactoryCandidate.from_wire(item)
        for item in array_value(value, operation_id="list_factory_candidates")
    )


def _lenses(value: JsonValue) -> tuple[FactoryEvaluationLens, ...]:
    return tuple(
        FactoryEvaluationLens.from_wire(item)
        for item in array_value(value, operation_id="list_factory_evaluation_lenses")
    )


def _champion_events(value: JsonValue) -> tuple[FactoryChampionEvent, ...]:
    return tuple(
        FactoryChampionEvent.from_wire(item)
        for item in array_value(value, operation_id="list_factory_champion_events")
    )


def _grading_request(
    request: FactoryCandidateGradingRequest | JsonObject,
) -> FactoryCandidateGradingRequest:
    if isinstance(request, FactoryCandidateGradingRequest):
        return request
    grading = request.get("grading")
    if not isinstance(grading, dict):
        raise ValueError("factory candidate grading request requires a grading object")
    return FactoryCandidateGradingRequest(grading=dict(grading))


def _champion_select_request(
    request: FactoryChampionSelectRequest | JsonObject,
) -> FactoryChampionSelectRequest:
    if isinstance(request, FactoryChampionSelectRequest):
        return request
    baseline_score = request.get("baseline_score")
    if isinstance(baseline_score, bool) or not isinstance(baseline_score, (int, float)):
        raise ValueError("factory champion selection requires numeric baseline_score")
    effort_id = request.get("effort_id")
    if effort_id is not None and not isinstance(effort_id, str):
        raise ValueError("factory champion selection effort_id must be a string")
    return FactoryChampionSelectRequest(
        baseline_score=float(baseline_score),
        effort_id=EffortId(effort_id) if effort_id is not None else None,
    )


def _champion_rollback_request(
    request: FactoryChampionRollbackRequest | JsonObject,
) -> FactoryChampionRollbackRequest:
    if isinstance(request, FactoryChampionRollbackRequest):
        return request
    candidate_id = request.get("to_candidate_id")
    reason = request.get("reason")
    effort_id = request.get("effort_id")
    if not isinstance(candidate_id, str):
        raise ValueError("factory champion rollback requires to_candidate_id")
    if not isinstance(reason, str):
        raise ValueError("factory champion rollback requires reason")
    if effort_id is not None and not isinstance(effort_id, str):
        raise ValueError("factory champion rollback effort_id must be a string")
    return FactoryChampionRollbackRequest(
        to_candidate_id=FactoryCandidateId(candidate_id),
        reason=reason,
        effort_id=EffortId(effort_id) if effort_id is not None else None,
    )


class FactoryEffortsAPI:
    """Stable Effort lifecycle nested beneath the Factory namespace."""

    def __init__(self, transport: HttpTransport) -> None:
        self._transport = transport

    def create(self, request: EffortSpec) -> Effort:
        """Create an Effort from a typed Effort specification.

        Args:
            request: Effort specification to serialize into the create request body.

        Returns:
            The created Effort.
        """
        value = self._transport.execute(
            _request("create_effort", "/smr/efforts", body=request.to_wire())
        )
        return Effort.from_wire(value)

    def list(self, factory_id: FactoryId) -> tuple[Effort, ...]:
        """List Efforts for a Factory.

        Args:
            factory_id: Factory whose Efforts to list.

        Returns:
            The Efforts returned by the backend.
        """
        value = self._transport.execute(
            _request(
                "list_factory_efforts",
                f"/smr/factories/{factory_id}/efforts",
            )
        )
        return _efforts(value)

    def retrieve(self, effort_id: EffortId) -> Effort:
        """Retrieve an Effort.

        Args:
            effort_id: Effort to retrieve.

        Returns:
            The requested Effort.
        """
        value = self._transport.execute(_request("retrieve_effort", f"/smr/efforts/{effort_id}"))
        return Effort.from_wire(value)

    def update(self, effort_id: EffortId, request: EffortPatch) -> Effort:
        """Update mutable Effort fields from a typed Effort patch.

        Args:
            effort_id: Effort to update.
            request: Effort patch to serialize into the update request body.

        Returns:
            The updated Effort.

        Raises:
            ValueError: If the Effort patch does not change any fields.
        """
        value = self._transport.execute(
            _request(
                "update_effort",
                f"/smr/efforts/{effort_id}",
                body=request.to_wire(),
            )
        )
        return Effort.from_wire(value)


class FactoryCandidatesAPI:
    """Immutable candidate discovery and benchmark-owned grading intake."""

    def __init__(self, transport: HttpTransport) -> None:
        self._transport = transport

    def list(
        self,
        factory_id: FactoryId,
        *,
        grading_status: FactoryCandidateGradingStatus | str | None = None,
        effort_id: EffortId | None = None,
        limit: int = 200,
    ) -> tuple[FactoryCandidate, ...]:
        """List candidates for a Factory.

        Args:
            factory_id: Factory whose candidates to list.
            grading_status: Optional grading status filter.
            effort_id: Optional Effort filter.
            limit: Maximum number of candidates to request.

        Returns:
            The candidates returned by the backend.
        """
        query: JsonObject = {"limit": limit}
        if grading_status is not None:
            query["grading_status"] = (
                grading_status.value
                if isinstance(grading_status, FactoryCandidateGradingStatus)
                else grading_status
            )
        if effort_id is not None:
            query["effort_id"] = effort_id
        value = self._transport.execute(
            _request(
                "list_factory_candidates",
                f"/smr/factories/{factory_id}/candidates",
                query=query,
            )
        )
        return _candidates(value)

    def record_grading(
        self,
        factory_id: FactoryId,
        candidate_id: FactoryCandidateId,
        request: FactoryCandidateGradingRequest | JsonObject,
    ) -> FactoryCandidate:
        """Record benchmark-owned grading for a Factory candidate.

        Args:
            factory_id: Factory that owns the candidate.
            candidate_id: Candidate to grade.
            request: Grading request or mapping to serialize into the grading request body.

        Returns:
            The updated candidate.

        Raises:
            ValueError: If a mapping request lacks a grading object or contains invalid grading
                details.
        """
        value = self._transport.execute(
            _request(
                "record_factory_candidate_grading",
                f"/smr/factories/{factory_id}/candidates/{candidate_id}/grading",
                body=_grading_request(request).to_wire(),
            )
        )
        return FactoryCandidate.from_wire(value)


class FactoryChampionsAPI:
    """Deterministic champion selection and append-only decision history."""

    def __init__(self, transport: HttpTransport) -> None:
        self._transport = transport

    def select(
        self,
        factory_id: FactoryId,
        request: FactoryChampionSelectRequest | JsonObject,
    ) -> FactoryChampionDecision:
        """Select a champion for a Factory.

        Args:
            factory_id: Factory whose champion to select.
            request: Champion selection request or mapping to serialize into the request body.

        Returns:
            The champion decision returned by the backend.

        Raises:
            ValueError: If a mapping request lacks a numeric baseline score or has an invalid
                Effort id.
        """
        value = self._transport.execute(
            _request(
                "select_factory_champion",
                f"/smr/factories/{factory_id}/champion/select",
                body=_champion_select_request(request).to_wire(),
            )
        )
        return FactoryChampionDecision.from_wire(value)

    def rollback(
        self,
        factory_id: FactoryId,
        request: FactoryChampionRollbackRequest | JsonObject,
    ) -> FactoryChampionDecision:
        """Rollback a Factory champion to a candidate.

        Args:
            factory_id: Factory whose champion to rollback.
            request: Champion rollback request or mapping to serialize into the request body.

        Returns:
            The champion decision returned by the backend.

        Raises:
            ValueError: If a mapping request lacks rollback fields or contains invalid field types.
        """
        value = self._transport.execute(
            _request(
                "rollback_factory_champion",
                f"/smr/factories/{factory_id}/champion/rollback",
                body=_champion_rollback_request(request).to_wire(),
            )
        )
        return FactoryChampionDecision.from_wire(value)

    def list_events(
        self,
        factory_id: FactoryId,
        *,
        limit: int = 100,
    ) -> tuple[FactoryChampionEvent, ...]:
        """List champion events for a Factory.

        Args:
            factory_id: Factory whose champion events to list.
            limit: Maximum number of champion events to request.

        Returns:
            The champion events returned by the backend.
        """
        value = self._transport.execute(
            _request(
                "list_factory_champion_events",
                f"/smr/factories/{factory_id}/champion/events",
                query={"limit": limit},
            )
        )
        return _champion_events(value)


class FactoryLensesAPI:
    """Optional optimization lens, derived best-so-far, and human preference.

    Every method here requires the Factory to be on the champion-free Result
    authority; the backend returns a typed 409 otherwise rather than storing
    configuration that could never take effect.

    A Factory that optimizes nothing never touches this namespace. That is the
    ordinary case: ``results.best_so_far`` answers ``optimizes=False`` and the
    Factory works exactly as well.
    """

    def __init__(self, transport: HttpTransport) -> None:
        self._transport = transport

    def define(self, factory_id: FactoryId, request: FactoryLensSpec) -> FactoryEvaluationLens:
        """Declare how this Factory compares its Results.

        Re-declaring an existing ``lens_key`` appends a new immutable version
        rather than editing the old one, so a best-so-far answer computed under
        an earlier version stays reproducible.

        Args:
            factory_id: Factory to declare the lens on.
            request: Lens specification.

        Returns:
            The stored lens version.
        """
        value = self._transport.execute(
            _request(
                "define_factory_evaluation_lens",
                f"/smr/factories/{factory_id}/lenses",
                body=request.to_wire(),
            )
        )
        return FactoryEvaluationLens.from_wire(value)

    def list(
        self,
        factory_id: FactoryId,
        *,
        include_superseded: bool = False,
        limit: int = 100,
    ) -> tuple[FactoryEvaluationLens, ...]:
        """List lens versions, newest per key.

        Args:
            factory_id: Factory whose lenses to list.
            include_superseded: Include older versions. Needed to audit a
                best-so-far answer produced under a definition since replaced.
            limit: Maximum number of rows to request.

        Returns:
            The lens versions returned by the backend.
        """
        value = self._transport.execute(
            _request(
                "list_factory_evaluation_lenses",
                f"/smr/factories/{factory_id}/lenses",
                query={"include_superseded": include_superseded, "limit": limit},
            )
        )
        return _lenses(value)

    def best_so_far(self, factory_id: FactoryId) -> FactoryBestResults:
        """Derived best-so-far for every lens this Factory declares.

        Args:
            factory_id: Factory to query.

        Returns:
            Per-lens outcomes. ``optimizes is False`` means the Factory
            hillclimbs nothing, which is a valid steady state and must render
            differently from "no results yet".
        """
        value = self._transport.execute(
            _request(
                "retrieve_factory_best_results",
                f"/smr/factories/{factory_id}/results/best-so-far",
            )
        )
        return FactoryBestResults.from_wire(value)

    def record_evaluation(
        self,
        factory_id: FactoryId,
        result_id: str,
        request: FactoryResultEvaluationRequest,
    ) -> FactoryResultEvaluation:
        """Post one externally owned verdict for a Result under a lens.

        Idempotent under ``attempt_key``: a retrying evaluator converges on one
        row. A correction uses a *new* attempt key, so the record of what was
        believed when survives.

        Args:
            factory_id: Factory that owns the Result.
            result_id: Result envelope id, or the WorkProduct id it pins.
            request: The verdict to store.

        Returns:
            The stored evaluation.
        """
        value = self._transport.execute(
            _request(
                "record_factory_result_evaluation",
                f"/smr/factories/{factory_id}/results/{result_id}/evaluations",
                body=request.to_wire(),
            )
        )
        return FactoryResultEvaluation.from_wire(value)

    def prefer(
        self, factory_id: FactoryId, request: FactoryPreferenceRequest
    ) -> FactoryPreferenceEvent:
        """Record a human preference beside the derived best.

        Preference does not overwrite the derived best. A reviewer choosing one
        Result while a lens computes another is real information; collapsing
        them into one pointer destroys it.

        Args:
            factory_id: Factory to record the preference on.
            request: The preference event.

        Returns:
            The stored event.
        """
        value = self._transport.execute(
            _request(
                "record_factory_result_preference",
                f"/smr/factories/{factory_id}/results/prefer",
                body=request.to_wire(),
            )
        )
        return FactoryPreferenceEvent.from_wire(value)


class FactoriesAPI:
    """Stable Factory lifecycle, Efforts, candidates, and champion decisions."""

    def __init__(self, transport: HttpTransport) -> None:
        self._transport = transport
        self.efforts = FactoryEffortsAPI(transport)
        self.candidates = FactoryCandidatesAPI(transport)
        self.champions = FactoryChampionsAPI(transport)
        self.lenses = FactoryLensesAPI(transport)

    def trace_store(self, factory_id: FactoryId) -> FactoryTraceStoreAPI:
        """Open the Factory's managed Trace V5 store (no network call)."""
        return FactoryTraceStoreAPI(self._transport, str(factory_id))

    def create(self, request: FactorySpec) -> Factory:
        """Create a Factory from a typed Factory specification.

        Args:
            request: Factory specification to serialize into the create request body.

        Returns:
            The created Factory.
        """
        value = self._transport.execute(
            _request("create_factory", "/smr/factories", body=request.to_wire())
        )
        return Factory.from_wire(value)

    def list(self, *, include_archived: bool = False) -> tuple[Factory, ...]:
        """List Factories visible to the authenticated organization.

        Args:
            include_archived: Whether to include archived Factories in the result.

        Returns:
            The Factories returned by the backend.
        """
        value = self._transport.execute(
            _request(
                "list_factories",
                "/smr/factories",
                query={"include_archived": include_archived},
            )
        )
        return _factories(value)

    def retrieve(self, factory_id: FactoryId) -> Factory:
        """Retrieve a Factory.

        Args:
            factory_id: Factory to retrieve.

        Returns:
            The requested Factory.
        """
        value = self._transport.execute(
            _request("retrieve_factory", f"/smr/factories/{factory_id}")
        )
        return Factory.from_wire(value)

    def update(self, factory_id: FactoryId, request: FactoryPatch) -> Factory:
        """Update mutable Factory fields from a typed Factory patch.

        Args:
            factory_id: Factory to update.
            request: Factory patch to serialize into the update request body.

        Returns:
            The updated Factory.
        """
        value = self._transport.execute(
            _request(
                "update_factory",
                f"/smr/factories/{factory_id}",
                body=request.to_wire(),
            )
        )
        return Factory.from_wire(value)

    def start(
        self,
        factory_id: FactoryId,
        request: FactoryTransition | None = None,
    ) -> FactoryTransitionResult:
        """Apply the start FactoryLifecycle transition to a Factory.

        Args:
            factory_id: Factory to start.
            request: Optional Factory transition metadata and preview flag.

        Returns:
            The Factory transition result returned by the backend.
        """
        return self._transition(factory_id, "start", request)

    def pause(
        self,
        factory_id: FactoryId,
        request: FactoryTransition | None = None,
    ) -> FactoryTransitionResult:
        """Apply the pause FactoryLifecycle transition to a Factory.

        Args:
            factory_id: Factory to pause.
            request: Optional Factory transition metadata and preview flag.

        Returns:
            The Factory transition result returned by the backend.
        """
        return self._transition(factory_id, "pause", request)

    def resume(
        self,
        factory_id: FactoryId,
        request: FactoryTransition | None = None,
    ) -> FactoryTransitionResult:
        """Apply the resume FactoryLifecycle transition to a Factory.

        Args:
            factory_id: Factory to resume.
            request: Optional Factory transition metadata and preview flag.

        Returns:
            The Factory transition result returned by the backend.
        """
        return self._transition(factory_id, "resume", request)

    def archive(
        self,
        factory_id: FactoryId,
        request: FactoryTransition | None = None,
    ) -> FactoryTransitionResult:
        """Apply the archive FactoryLifecycle transition to a Factory.

        Args:
            factory_id: Factory to archive.
            request: Optional Factory transition metadata and preview flag.

        Returns:
            The Factory transition result returned by the backend.
        """
        return self._transition(factory_id, "archive", request)

    def _transition(
        self,
        factory_id: FactoryId,
        command: str,
        request: FactoryTransition | None,
    ) -> FactoryTransitionResult:
        value = self._transport.execute(
            _request(
                f"{command}_factory",
                f"/smr/factories/{factory_id}/{command}",
                body=(request or FactoryTransition()).to_wire(),
            )
        )
        return FactoryTransitionResult.from_wire(value)


class AsyncFactoryEffortsAPI:
    """Native asynchronous peer of :class:`FactoryEffortsAPI`."""

    def __init__(self, transport: AsyncHttpTransport) -> None:
        self._transport = transport

    async def create(self, request: EffortSpec) -> Effort:
        """Create an Effort from a typed Effort specification.

        Args:
            request: Effort specification to serialize into the create request body.

        Returns:
            The created Effort.
        """
        value = await self._transport.execute(
            _request("create_effort", "/smr/efforts", body=request.to_wire())
        )
        return Effort.from_wire(value)

    async def list(self, factory_id: FactoryId) -> tuple[Effort, ...]:
        """List Efforts for a Factory.

        Args:
            factory_id: Factory whose Efforts to list.

        Returns:
            The Efforts returned by the backend.
        """
        value = await self._transport.execute(
            _request(
                "list_factory_efforts",
                f"/smr/factories/{factory_id}/efforts",
            )
        )
        return _efforts(value)

    async def retrieve(self, effort_id: EffortId) -> Effort:
        """Retrieve an Effort.

        Args:
            effort_id: Effort to retrieve.

        Returns:
            The requested Effort.
        """
        value = await self._transport.execute(
            _request("retrieve_effort", f"/smr/efforts/{effort_id}")
        )
        return Effort.from_wire(value)

    async def update(self, effort_id: EffortId, request: EffortPatch) -> Effort:
        """Update mutable Effort fields from a typed Effort patch.

        Args:
            effort_id: Effort to update.
            request: Effort patch to serialize into the update request body.

        Returns:
            The updated Effort.

        Raises:
            ValueError: If the Effort patch does not change any fields.
        """
        value = await self._transport.execute(
            _request(
                "update_effort",
                f"/smr/efforts/{effort_id}",
                body=request.to_wire(),
            )
        )
        return Effort.from_wire(value)


class AsyncFactoryCandidatesAPI:
    """Native asynchronous peer of :class:`FactoryCandidatesAPI`."""

    def __init__(self, transport: AsyncHttpTransport) -> None:
        self._transport = transport

    async def list(
        self,
        factory_id: FactoryId,
        *,
        grading_status: FactoryCandidateGradingStatus | str | None = None,
        effort_id: EffortId | None = None,
        limit: int = 200,
    ) -> tuple[FactoryCandidate, ...]:
        """List candidates for a Factory.

        Args:
            factory_id: Factory whose candidates to list.
            grading_status: Optional grading status filter.
            effort_id: Optional Effort filter.
            limit: Maximum number of candidates to request.

        Returns:
            The candidates returned by the backend.
        """
        query: JsonObject = {"limit": limit}
        if grading_status is not None:
            query["grading_status"] = (
                grading_status.value
                if isinstance(grading_status, FactoryCandidateGradingStatus)
                else grading_status
            )
        if effort_id is not None:
            query["effort_id"] = effort_id
        value = await self._transport.execute(
            _request(
                "list_factory_candidates",
                f"/smr/factories/{factory_id}/candidates",
                query=query,
            )
        )
        return _candidates(value)

    async def record_grading(
        self,
        factory_id: FactoryId,
        candidate_id: FactoryCandidateId,
        request: FactoryCandidateGradingRequest | JsonObject,
    ) -> FactoryCandidate:
        """Record benchmark-owned grading for a Factory candidate.

        Args:
            factory_id: Factory that owns the candidate.
            candidate_id: Candidate to grade.
            request: Grading request or mapping to serialize into the grading request body.

        Returns:
            The updated candidate.

        Raises:
            ValueError: If a mapping request lacks a grading object or contains invalid grading
                details.
        """
        value = await self._transport.execute(
            _request(
                "record_factory_candidate_grading",
                f"/smr/factories/{factory_id}/candidates/{candidate_id}/grading",
                body=_grading_request(request).to_wire(),
            )
        )
        return FactoryCandidate.from_wire(value)


class AsyncFactoryChampionsAPI:
    """Native asynchronous peer of :class:`FactoryChampionsAPI`."""

    def __init__(self, transport: AsyncHttpTransport) -> None:
        self._transport = transport

    async def select(
        self,
        factory_id: FactoryId,
        request: FactoryChampionSelectRequest | JsonObject,
    ) -> FactoryChampionDecision:
        """Select a champion for a Factory.

        Args:
            factory_id: Factory whose champion to select.
            request: Champion selection request or mapping to serialize into the request body.

        Returns:
            The champion decision returned by the backend.

        Raises:
            ValueError: If a mapping request lacks a numeric baseline score or has an invalid
                Effort id.
        """
        value = await self._transport.execute(
            _request(
                "select_factory_champion",
                f"/smr/factories/{factory_id}/champion/select",
                body=_champion_select_request(request).to_wire(),
            )
        )
        return FactoryChampionDecision.from_wire(value)

    async def rollback(
        self,
        factory_id: FactoryId,
        request: FactoryChampionRollbackRequest | JsonObject,
    ) -> FactoryChampionDecision:
        """Rollback a Factory champion to a candidate.

        Args:
            factory_id: Factory whose champion to rollback.
            request: Champion rollback request or mapping to serialize into the request body.

        Returns:
            The champion decision returned by the backend.

        Raises:
            ValueError: If a mapping request lacks rollback fields or contains invalid field types.
        """
        value = await self._transport.execute(
            _request(
                "rollback_factory_champion",
                f"/smr/factories/{factory_id}/champion/rollback",
                body=_champion_rollback_request(request).to_wire(),
            )
        )
        return FactoryChampionDecision.from_wire(value)

    async def list_events(
        self,
        factory_id: FactoryId,
        *,
        limit: int = 100,
    ) -> tuple[FactoryChampionEvent, ...]:
        """List champion events for a Factory.

        Args:
            factory_id: Factory whose champion events to list.
            limit: Maximum number of champion events to request.

        Returns:
            The champion events returned by the backend.
        """
        value = await self._transport.execute(
            _request(
                "list_factory_champion_events",
                f"/smr/factories/{factory_id}/champion/events",
                query={"limit": limit},
            )
        )
        return _champion_events(value)


class AsyncFactoryLensesAPI:
    """Optional optimization lens, derived best-so-far, and human preference.

    Every method here requires the Factory to be on the champion-free Result
    authority; the backend returns a typed 409 otherwise rather than storing
    configuration that could never take effect.

    A Factory that optimizes nothing never touches this namespace. That is the
    ordinary case: ``results.best_so_far`` answers ``optimizes=False`` and the
    Factory works exactly as well.
    """

    def __init__(self, transport: AsyncHttpTransport) -> None:
        self._transport = transport

    async def define(
        self, factory_id: FactoryId, request: FactoryLensSpec
    ) -> FactoryEvaluationLens:
        """Declare how this Factory compares its Results.

        Re-declaring an existing ``lens_key`` appends a new immutable version
        rather than editing the old one, so a best-so-far answer computed under
        an earlier version stays reproducible.

        Args:
            factory_id: Factory to declare the lens on.
            request: Lens specification.

        Returns:
            The stored lens version.
        """
        value = await self._transport.execute(
            _request(
                "define_factory_evaluation_lens",
                f"/smr/factories/{factory_id}/lenses",
                body=request.to_wire(),
            )
        )
        return FactoryEvaluationLens.from_wire(value)

    async def list(
        self,
        factory_id: FactoryId,
        *,
        include_superseded: bool = False,
        limit: int = 100,
    ) -> tuple[FactoryEvaluationLens, ...]:
        """List lens versions, newest per key.

        Args:
            factory_id: Factory whose lenses to list.
            include_superseded: Include older versions. Needed to audit a
                best-so-far answer produced under a definition since replaced.
            limit: Maximum number of rows to request.

        Returns:
            The lens versions returned by the backend.
        """
        value = await self._transport.execute(
            _request(
                "list_factory_evaluation_lenses",
                f"/smr/factories/{factory_id}/lenses",
                query={"include_superseded": include_superseded, "limit": limit},
            )
        )
        return _lenses(value)

    async def best_so_far(self, factory_id: FactoryId) -> FactoryBestResults:
        """Derived best-so-far for every lens this Factory declares.

        Args:
            factory_id: Factory to query.

        Returns:
            Per-lens outcomes. ``optimizes is False`` means the Factory
            hillclimbs nothing, which is a valid steady state and must render
            differently from "no results yet".
        """
        value = await self._transport.execute(
            _request(
                "retrieve_factory_best_results",
                f"/smr/factories/{factory_id}/results/best-so-far",
            )
        )
        return FactoryBestResults.from_wire(value)

    async def record_evaluation(
        self,
        factory_id: FactoryId,
        result_id: str,
        request: FactoryResultEvaluationRequest,
    ) -> FactoryResultEvaluation:
        """Post one externally owned verdict for a Result under a lens.

        Idempotent under ``attempt_key``: a retrying evaluator converges on one
        row. A correction uses a *new* attempt key, so the record of what was
        believed when survives.

        Args:
            factory_id: Factory that owns the Result.
            result_id: Result envelope id, or the WorkProduct id it pins.
            request: The verdict to store.

        Returns:
            The stored evaluation.
        """
        value = await self._transport.execute(
            _request(
                "record_factory_result_evaluation",
                f"/smr/factories/{factory_id}/results/{result_id}/evaluations",
                body=request.to_wire(),
            )
        )
        return FactoryResultEvaluation.from_wire(value)

    async def prefer(
        self, factory_id: FactoryId, request: FactoryPreferenceRequest
    ) -> FactoryPreferenceEvent:
        """Record a human preference beside the derived best.

        Preference does not overwrite the derived best. A reviewer choosing one
        Result while a lens computes another is real information; collapsing
        them into one pointer destroys it.

        Args:
            factory_id: Factory to record the preference on.
            request: The preference event.

        Returns:
            The stored event.
        """
        value = await self._transport.execute(
            _request(
                "record_factory_result_preference",
                f"/smr/factories/{factory_id}/results/prefer",
                body=request.to_wire(),
            )
        )
        return FactoryPreferenceEvent.from_wire(value)


class AsyncFactoriesAPI:
    """Native asynchronous Factory lifecycle with sync surface parity."""

    def __init__(self, transport: AsyncHttpTransport) -> None:
        self._transport = transport
        self.efforts = AsyncFactoryEffortsAPI(transport)
        self.candidates = AsyncFactoryCandidatesAPI(transport)
        self.champions = AsyncFactoryChampionsAPI(transport)
        self.lenses = AsyncFactoryLensesAPI(transport)

    def trace_store(self, factory_id: FactoryId) -> AsyncFactoryTraceStoreAPI:
        """Open the Factory's async managed Trace V5 store (no network call)."""
        return AsyncFactoryTraceStoreAPI(self._transport, str(factory_id))

    async def create(self, request: FactorySpec) -> Factory:
        """Create a Factory from a typed Factory specification.

        Args:
            request: Factory specification to serialize into the create request body.

        Returns:
            The created Factory.
        """
        value = await self._transport.execute(
            _request("create_factory", "/smr/factories", body=request.to_wire())
        )
        return Factory.from_wire(value)

    async def list(self, *, include_archived: bool = False) -> tuple[Factory, ...]:
        """List Factories visible to the authenticated organization.

        Args:
            include_archived: Whether to include archived Factories in the result.

        Returns:
            The Factories returned by the backend.
        """
        value = await self._transport.execute(
            _request(
                "list_factories",
                "/smr/factories",
                query={"include_archived": include_archived},
            )
        )
        return _factories(value)

    async def retrieve(self, factory_id: FactoryId) -> Factory:
        """Retrieve a Factory.

        Args:
            factory_id: Factory to retrieve.

        Returns:
            The requested Factory.
        """
        value = await self._transport.execute(
            _request("retrieve_factory", f"/smr/factories/{factory_id}")
        )
        return Factory.from_wire(value)

    async def update(self, factory_id: FactoryId, request: FactoryPatch) -> Factory:
        """Update mutable Factory fields from a typed Factory patch.

        Args:
            factory_id: Factory to update.
            request: Factory patch to serialize into the update request body.

        Returns:
            The updated Factory.
        """
        value = await self._transport.execute(
            _request(
                "update_factory",
                f"/smr/factories/{factory_id}",
                body=request.to_wire(),
            )
        )
        return Factory.from_wire(value)

    async def start(
        self,
        factory_id: FactoryId,
        request: FactoryTransition | None = None,
    ) -> FactoryTransitionResult:
        """Apply the start FactoryLifecycle transition to a Factory.

        Args:
            factory_id: Factory to start.
            request: Optional Factory transition metadata and preview flag.

        Returns:
            The Factory transition result returned by the backend.
        """
        return await self._transition(factory_id, "start", request)

    async def pause(
        self,
        factory_id: FactoryId,
        request: FactoryTransition | None = None,
    ) -> FactoryTransitionResult:
        """Apply the pause FactoryLifecycle transition to a Factory.

        Args:
            factory_id: Factory to pause.
            request: Optional Factory transition metadata and preview flag.

        Returns:
            The Factory transition result returned by the backend.
        """
        return await self._transition(factory_id, "pause", request)

    async def resume(
        self,
        factory_id: FactoryId,
        request: FactoryTransition | None = None,
    ) -> FactoryTransitionResult:
        """Apply the resume FactoryLifecycle transition to a Factory.

        Args:
            factory_id: Factory to resume.
            request: Optional Factory transition metadata and preview flag.

        Returns:
            The Factory transition result returned by the backend.
        """
        return await self._transition(factory_id, "resume", request)

    async def archive(
        self,
        factory_id: FactoryId,
        request: FactoryTransition | None = None,
    ) -> FactoryTransitionResult:
        """Apply the archive FactoryLifecycle transition to a Factory.

        Args:
            factory_id: Factory to archive.
            request: Optional Factory transition metadata and preview flag.

        Returns:
            The Factory transition result returned by the backend.
        """
        return await self._transition(factory_id, "archive", request)

    async def _transition(
        self,
        factory_id: FactoryId,
        command: str,
        request: FactoryTransition | None,
    ) -> FactoryTransitionResult:
        value = await self._transport.execute(
            _request(
                f"{command}_factory",
                f"/smr/factories/{factory_id}/{command}",
                body=(request or FactoryTransition()).to_wire(),
            )
        )
        return FactoryTransitionResult.from_wire(value)


ResearchFactoriesAPI = FactoriesAPI
AsyncResearchFactoriesAPI = AsyncFactoriesAPI


__all__ = [
    "AsyncFactoriesAPI",
    "AsyncFactoryCandidatesAPI",
    "AsyncFactoryChampionsAPI",
    "AsyncFactoryEffortsAPI",
    "AsyncFactoryLensesAPI",
    "AsyncResearchFactoriesAPI",
    "FactoriesAPI",
    "FactoryCandidatesAPI",
    "FactoryChampionsAPI",
    "FactoryEffortsAPI",
    "FactoryLensesAPI",
    "ResearchFactoriesAPI",
]
