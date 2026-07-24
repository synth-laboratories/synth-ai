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
from synth_ai.core.research.operations import research_operation


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
        value = self._transport.execute(
            _request("create_effort", "/smr/efforts", body=request.to_wire())
        )
        return Effort.from_wire(value)

    def list(self, factory_id: FactoryId) -> tuple[Effort, ...]:
        value = self._transport.execute(
            _request(
                "list_factory_efforts",
                f"/smr/factories/{factory_id}/efforts",
            )
        )
        return _efforts(value)

    def retrieve(self, effort_id: EffortId) -> Effort:
        value = self._transport.execute(_request("retrieve_effort", f"/smr/efforts/{effort_id}"))
        return Effort.from_wire(value)

    def update(self, effort_id: EffortId, request: EffortPatch) -> Effort:
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
        value = self._transport.execute(
            _request(
                "list_factory_champion_events",
                f"/smr/factories/{factory_id}/champion/events",
                query={"limit": limit},
            )
        )
        return _champion_events(value)


class FactoriesAPI:
    """Stable Factory lifecycle, Efforts, candidates, and champion decisions."""

    def __init__(self, transport: HttpTransport) -> None:
        self._transport = transport
        self.efforts = FactoryEffortsAPI(transport)
        self.candidates = FactoryCandidatesAPI(transport)
        self.champions = FactoryChampionsAPI(transport)

    def create(self, request: FactorySpec) -> Factory:
        value = self._transport.execute(
            _request("create_factory", "/smr/factories", body=request.to_wire())
        )
        return Factory.from_wire(value)

    def list(self, *, include_archived: bool = False) -> tuple[Factory, ...]:
        value = self._transport.execute(
            _request(
                "list_factories",
                "/smr/factories",
                query={"include_archived": include_archived},
            )
        )
        return _factories(value)

    def retrieve(self, factory_id: FactoryId) -> Factory:
        value = self._transport.execute(
            _request("retrieve_factory", f"/smr/factories/{factory_id}")
        )
        return Factory.from_wire(value)

    def update(self, factory_id: FactoryId, request: FactoryPatch) -> Factory:
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
        return self._transition(factory_id, "start", request)

    def pause(
        self,
        factory_id: FactoryId,
        request: FactoryTransition | None = None,
    ) -> FactoryTransitionResult:
        return self._transition(factory_id, "pause", request)

    def resume(
        self,
        factory_id: FactoryId,
        request: FactoryTransition | None = None,
    ) -> FactoryTransitionResult:
        return self._transition(factory_id, "resume", request)

    def archive(
        self,
        factory_id: FactoryId,
        request: FactoryTransition | None = None,
    ) -> FactoryTransitionResult:
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
        value = await self._transport.execute(
            _request("create_effort", "/smr/efforts", body=request.to_wire())
        )
        return Effort.from_wire(value)

    async def list(self, factory_id: FactoryId) -> tuple[Effort, ...]:
        value = await self._transport.execute(
            _request(
                "list_factory_efforts",
                f"/smr/factories/{factory_id}/efforts",
            )
        )
        return _efforts(value)

    async def retrieve(self, effort_id: EffortId) -> Effort:
        value = await self._transport.execute(
            _request("retrieve_effort", f"/smr/efforts/{effort_id}")
        )
        return Effort.from_wire(value)

    async def update(self, effort_id: EffortId, request: EffortPatch) -> Effort:
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
        value = await self._transport.execute(
            _request(
                "list_factory_champion_events",
                f"/smr/factories/{factory_id}/champion/events",
                query={"limit": limit},
            )
        )
        return _champion_events(value)


class AsyncFactoriesAPI:
    """Native asynchronous Factory lifecycle with sync surface parity."""

    def __init__(self, transport: AsyncHttpTransport) -> None:
        self._transport = transport
        self.efforts = AsyncFactoryEffortsAPI(transport)
        self.candidates = AsyncFactoryCandidatesAPI(transport)
        self.champions = AsyncFactoryChampionsAPI(transport)

    async def create(self, request: FactorySpec) -> Factory:
        value = await self._transport.execute(
            _request("create_factory", "/smr/factories", body=request.to_wire())
        )
        return Factory.from_wire(value)

    async def list(self, *, include_archived: bool = False) -> tuple[Factory, ...]:
        value = await self._transport.execute(
            _request(
                "list_factories",
                "/smr/factories",
                query={"include_archived": include_archived},
            )
        )
        return _factories(value)

    async def retrieve(self, factory_id: FactoryId) -> Factory:
        value = await self._transport.execute(
            _request("retrieve_factory", f"/smr/factories/{factory_id}")
        )
        return Factory.from_wire(value)

    async def update(self, factory_id: FactoryId, request: FactoryPatch) -> Factory:
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
        return await self._transition(factory_id, "start", request)

    async def pause(
        self,
        factory_id: FactoryId,
        request: FactoryTransition | None = None,
    ) -> FactoryTransitionResult:
        return await self._transition(factory_id, "pause", request)

    async def resume(
        self,
        factory_id: FactoryId,
        request: FactoryTransition | None = None,
    ) -> FactoryTransitionResult:
        return await self._transition(factory_id, "resume", request)

    async def archive(
        self,
        factory_id: FactoryId,
        request: FactoryTransition | None = None,
    ) -> FactoryTransitionResult:
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
    "AsyncResearchFactoriesAPI",
    "FactoriesAPI",
    "FactoryCandidatesAPI",
    "FactoryChampionsAPI",
    "FactoryEffortsAPI",
    "ResearchFactoriesAPI",
]
