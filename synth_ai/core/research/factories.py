"""Stable synchronous and asynchronous Factory operations.

# See: specifications/sdk/core_research_migration.md
"""

from __future__ import annotations

from synth_ai.core.contracts.json_value import JsonObject, JsonValue
from synth_ai.core.http.async_transport import AsyncHttpTransport
from synth_ai.core.http.request import HttpRequest
from synth_ai.core.http.transport import HttpTransport
from synth_ai.core.research.contracts._wire import array_value
from synth_ai.core.research.contracts.common import EffortId, FactoryId
from synth_ai.core.research.contracts.factories import (
    Effort,
    EffortPatch,
    EffortSpec,
    Factory,
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
        Factory.from_wire(item)
        for item in array_value(value, operation_id="list_factories")
    )


def _efforts(value: JsonValue) -> tuple[Effort, ...]:
    return tuple(
        Effort.from_wire(item)
        for item in array_value(value, operation_id="list_factory_efforts")
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
        value = self._transport.execute(
            _request("retrieve_effort", f"/smr/efforts/{effort_id}")
        )
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


class FactoriesAPI:
    """Small stable Factory lifecycle: create, inspect, transition, and Efforts."""

    def __init__(self, transport: HttpTransport) -> None:
        self._transport = transport
        self.efforts = FactoryEffortsAPI(transport)

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


class AsyncFactoriesAPI:
    """Native asynchronous Factory lifecycle with sync surface parity."""

    def __init__(self, transport: AsyncHttpTransport) -> None:
        self._transport = transport
        self.efforts = AsyncFactoryEffortsAPI(transport)

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
    "AsyncFactoryEffortsAPI",
    "AsyncResearchFactoriesAPI",
    "FactoriesAPI",
    "FactoryEffortsAPI",
    "ResearchFactoriesAPI",
]
