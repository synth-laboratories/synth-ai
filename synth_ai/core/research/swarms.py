"""Synchronous and native asynchronous swarm operations."""

from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncIterator, Iterator

from synth_ai.core.contracts.json_value import JsonObject, JsonValue
from synth_ai.core.http.async_transport import AsyncHttpTransport
from synth_ai.core.http.request import HttpRequest
from synth_ai.core.http.transport import HttpTransport
from synth_ai.core.research.contracts._wire import array_value
from synth_ai.core.research.contracts.common import ProjectId, SwarmId
from synth_ai.core.research.contracts.swarms import (
    BranchResult,
    BranchSpec,
    ResolvedSwarmConfiguration,
    Swarm,
    SwarmPreflight,
    SwarmSpec,
)
from synth_ai.core.research.contracts.usage import SwarmUsage
from synth_ai.core.research.events import SwarmEvent, decode_swarm_event
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


def _swarms(value: JsonValue, *, operation_id: str) -> tuple[Swarm, ...]:
    return tuple(
        Swarm.from_wire(item)
        for item in array_value(value, operation_id=operation_id)
    )


def _wait_arguments(timeout_seconds: float, poll_interval_seconds: float) -> None:
    if timeout_seconds <= 0:
        raise ValueError("timeout_seconds must be positive")
    if poll_interval_seconds <= 0:
        raise ValueError("poll_interval_seconds must be positive")


class SwarmHandle:
    def __init__(self, api: SwarmsAPI, swarm: Swarm) -> None:
        self._api = api
        self.swarm_id = swarm.swarm_id
        self.project_id = swarm.project_id
        self.initial = swarm

    def retrieve(self) -> Swarm:
        return self._api.retrieve(self.swarm_id)

    def configuration(self) -> ResolvedSwarmConfiguration:
        """Return the immutable resolved configuration bound to this swarm."""
        return self._api.configuration(self.swarm_id)

    def usage(self) -> SwarmUsage:
        """Return typed cost, token, actor, and freshness evidence."""
        return self._api.usage(self.swarm_id)

    def wait(
        self,
        *,
        timeout_seconds: float = 3600.0,
        poll_interval_seconds: float = 2.0,
    ) -> Swarm:
        return self._api.wait(
            self.swarm_id,
            timeout_seconds=timeout_seconds,
            poll_interval_seconds=poll_interval_seconds,
        )

    def pause(self) -> Swarm:
        return self._api.pause(self.swarm_id)

    def resume(self) -> Swarm:
        return self._api.resume(self.swarm_id)

    def cancel(self) -> Swarm:
        return self._api.cancel(self.swarm_id)

    def events(
        self,
        *,
        last_event_id: str | None = None,
    ) -> Iterator[SwarmEvent]:
        yield from self._api.events(self.swarm_id, last_event_id=last_event_id)


class SwarmsAPI:
    def __init__(self, transport: HttpTransport) -> None:
        self._transport = transport

    def preflight(
        self,
        request: SwarmSpec,
        *,
        project_id: ProjectId | None = None,
    ) -> SwarmPreflight:
        if project_id is None:
            operation_id = "preflight_one_off_run"
            path = "/smr/runs:one-off/launch-preflight"
        else:
            operation_id = "preflight_project_run"
            path = f"/smr/projects/{project_id}/launch-preflight"
        value = self._transport.execute(
            _request(operation_id, path, body=request.to_wire())
        )
        return SwarmPreflight.from_wire(value)

    def create(
        self,
        request: SwarmSpec,
        *,
        project_id: ProjectId | None = None,
    ) -> SwarmHandle:
        if project_id is None:
            operation_id = "trigger_one_off_run"
            path = "/smr/runs:one-off"
        else:
            operation_id = "trigger_project_run"
            path = f"/smr/projects/{project_id}/trigger"
        value = self._transport.execute(
            _request(operation_id, path, body=request.to_wire())
        )
        return SwarmHandle(self, Swarm.from_wire(value))

    def list(
        self,
        project_id: ProjectId,
        *,
        limit: int = 100,
        cursor: str | None = None,
    ) -> tuple[Swarm, ...]:
        query: JsonObject = {"limit": limit}
        if cursor is not None:
            query["cursor"] = cursor
        value = self._transport.execute(
            _request(
                "list_project_runs",
                f"/smr/projects/{project_id}/runs",
                query=query,
            )
        )
        return _swarms(value, operation_id="list_project_runs")

    def retrieve(self, swarm_id: SwarmId) -> Swarm:
        value = self._transport.execute(
            _request("retrieve_run", f"/smr/runs/{swarm_id}")
        )
        return Swarm.from_wire(value)

    def configuration(self, swarm_id: SwarmId) -> ResolvedSwarmConfiguration:
        """Return the versioned, redacted configuration snapshot for a swarm."""
        value = self._transport.execute(
            _request(
                "retrieve_swarm_configuration",
                f"/smr/runs/{swarm_id}/configuration",
            )
        )
        return ResolvedSwarmConfiguration.from_wire(value)

    def usage(self, swarm_id: SwarmId) -> SwarmUsage:
        """Return typed cost, token, actor, and freshness evidence."""
        value = self._transport.execute(
            _request(
                "retrieve_swarm_usage",
                f"/smr/runs/{swarm_id}/usage-summary",
            )
        )
        return SwarmUsage.from_wire(value)

    def wait(
        self,
        swarm_id: SwarmId,
        *,
        timeout_seconds: float = 3600.0,
        poll_interval_seconds: float = 2.0,
    ) -> Swarm:
        _wait_arguments(timeout_seconds, poll_interval_seconds)
        deadline = time.monotonic() + timeout_seconds
        while True:
            swarm = self.retrieve(swarm_id)
            if swarm.state.is_terminal:
                return swarm
            if time.monotonic() >= deadline:
                raise TimeoutError(f"swarm {swarm_id} did not reach a terminal state")
            time.sleep(poll_interval_seconds)

    def pause(self, swarm_id: SwarmId) -> Swarm:
        value = self._transport.execute(
            _request("pause_run", f"/smr/runs/{swarm_id}/pause")
        )
        return Swarm.from_wire(value)

    def resume(self, swarm_id: SwarmId) -> Swarm:
        value = self._transport.execute(
            _request("resume_run", f"/smr/runs/{swarm_id}/resume")
        )
        return Swarm.from_wire(value)

    def cancel(self, swarm_id: SwarmId) -> Swarm:
        self._transport.execute(_request("stop_run", f"/smr/runs/{swarm_id}/stop"))
        return self.retrieve(swarm_id)

    def branch(
        self,
        swarm_id: SwarmId,
        request: BranchSpec,
    ) -> BranchResult:
        value = self._transport.execute(
            _request(
                "branch_run",
                f"/smr/runs/{swarm_id}/branches",
                body=request.to_wire(),
            )
        )
        return BranchResult.from_wire(value)

    def events(
        self,
        swarm_id: SwarmId,
        *,
        last_event_id: str | None = None,
    ) -> Iterator[SwarmEvent]:
        for event in self._transport.stream_sse(
            f"/smr/runs/{swarm_id}/runtime/stream",
            last_event_id=last_event_id,
            timeout_seconds=None,
            operation_id="stream_run_events",
        ):
            yield decode_swarm_event(event)


class AsyncSwarmHandle:
    def __init__(self, api: AsyncSwarmsAPI, swarm: Swarm) -> None:
        self._api = api
        self.swarm_id = swarm.swarm_id
        self.project_id = swarm.project_id
        self.initial = swarm

    async def retrieve(self) -> Swarm:
        return await self._api.retrieve(self.swarm_id)

    async def configuration(self) -> ResolvedSwarmConfiguration:
        """Return the immutable resolved configuration bound to this swarm."""
        return await self._api.configuration(self.swarm_id)

    async def usage(self) -> SwarmUsage:
        """Return typed cost, token, actor, and freshness evidence."""
        return await self._api.usage(self.swarm_id)

    async def wait(
        self,
        *,
        timeout_seconds: float = 3600.0,
        poll_interval_seconds: float = 2.0,
    ) -> Swarm:
        return await self._api.wait(
            self.swarm_id,
            timeout_seconds=timeout_seconds,
            poll_interval_seconds=poll_interval_seconds,
        )

    async def pause(self) -> Swarm:
        return await self._api.pause(self.swarm_id)

    async def resume(self) -> Swarm:
        return await self._api.resume(self.swarm_id)

    async def cancel(self) -> Swarm:
        return await self._api.cancel(self.swarm_id)

    async def events(
        self,
        *,
        last_event_id: str | None = None,
    ) -> AsyncIterator[SwarmEvent]:
        async for event in self._api.events(
            self.swarm_id,
            last_event_id=last_event_id,
        ):
            yield event


class AsyncSwarmsAPI:
    def __init__(self, transport: AsyncHttpTransport) -> None:
        self._transport = transport

    async def preflight(
        self,
        request: SwarmSpec,
        *,
        project_id: ProjectId | None = None,
    ) -> SwarmPreflight:
        if project_id is None:
            operation_id = "preflight_one_off_run"
            path = "/smr/runs:one-off/launch-preflight"
        else:
            operation_id = "preflight_project_run"
            path = f"/smr/projects/{project_id}/launch-preflight"
        value = await self._transport.execute(
            _request(operation_id, path, body=request.to_wire())
        )
        return SwarmPreflight.from_wire(value)

    async def create(
        self,
        request: SwarmSpec,
        *,
        project_id: ProjectId | None = None,
    ) -> AsyncSwarmHandle:
        if project_id is None:
            operation_id = "trigger_one_off_run"
            path = "/smr/runs:one-off"
        else:
            operation_id = "trigger_project_run"
            path = f"/smr/projects/{project_id}/trigger"
        value = await self._transport.execute(
            _request(operation_id, path, body=request.to_wire())
        )
        return AsyncSwarmHandle(self, Swarm.from_wire(value))

    async def list(
        self,
        project_id: ProjectId,
        *,
        limit: int = 100,
        cursor: str | None = None,
    ) -> tuple[Swarm, ...]:
        query: JsonObject = {"limit": limit}
        if cursor is not None:
            query["cursor"] = cursor
        value = await self._transport.execute(
            _request("list_project_runs", f"/smr/projects/{project_id}/runs", query=query)
        )
        return _swarms(value, operation_id="list_project_runs")

    async def retrieve(self, swarm_id: SwarmId) -> Swarm:
        value = await self._transport.execute(
            _request("retrieve_run", f"/smr/runs/{swarm_id}")
        )
        return Swarm.from_wire(value)

    async def configuration(self, swarm_id: SwarmId) -> ResolvedSwarmConfiguration:
        """Return the versioned, redacted configuration snapshot for a swarm."""
        value = await self._transport.execute(
            _request(
                "retrieve_swarm_configuration",
                f"/smr/runs/{swarm_id}/configuration",
            )
        )
        return ResolvedSwarmConfiguration.from_wire(value)

    async def usage(self, swarm_id: SwarmId) -> SwarmUsage:
        """Return typed cost, token, actor, and freshness evidence."""
        value = await self._transport.execute(
            _request(
                "retrieve_swarm_usage",
                f"/smr/runs/{swarm_id}/usage-summary",
            )
        )
        return SwarmUsage.from_wire(value)

    async def wait(
        self,
        swarm_id: SwarmId,
        *,
        timeout_seconds: float = 3600.0,
        poll_interval_seconds: float = 2.0,
    ) -> Swarm:
        _wait_arguments(timeout_seconds, poll_interval_seconds)
        deadline = time.monotonic() + timeout_seconds
        while True:
            swarm = await self.retrieve(swarm_id)
            if swarm.state.is_terminal:
                return swarm
            if time.monotonic() >= deadline:
                raise TimeoutError(f"swarm {swarm_id} did not reach a terminal state")
            await asyncio.sleep(poll_interval_seconds)

    async def pause(self, swarm_id: SwarmId) -> Swarm:
        value = await self._transport.execute(
            _request("pause_run", f"/smr/runs/{swarm_id}/pause")
        )
        return Swarm.from_wire(value)

    async def resume(self, swarm_id: SwarmId) -> Swarm:
        value = await self._transport.execute(
            _request("resume_run", f"/smr/runs/{swarm_id}/resume")
        )
        return Swarm.from_wire(value)

    async def cancel(self, swarm_id: SwarmId) -> Swarm:
        await self._transport.execute(_request("stop_run", f"/smr/runs/{swarm_id}/stop"))
        return await self.retrieve(swarm_id)

    async def branch(
        self,
        swarm_id: SwarmId,
        request: BranchSpec,
    ) -> BranchResult:
        value = await self._transport.execute(
            _request("branch_run", f"/smr/runs/{swarm_id}/branches", body=request.to_wire())
        )
        return BranchResult.from_wire(value)

    async def events(
        self,
        swarm_id: SwarmId,
        *,
        last_event_id: str | None = None,
    ) -> AsyncIterator[SwarmEvent]:
        async for event in self._transport.stream_sse(
            f"/smr/runs/{swarm_id}/runtime/stream",
            last_event_id=last_event_id,
            timeout_seconds=None,
            operation_id="stream_run_events",
        ):
            yield decode_swarm_event(event)


ResearchSwarmHandle = SwarmHandle
ResearchSwarmsAPI = SwarmsAPI
AsyncResearchSwarmHandle = AsyncSwarmHandle
AsyncResearchSwarmsAPI = AsyncSwarmsAPI


__all__ = [
    "AsyncSwarmHandle",
    "AsyncSwarmsAPI",
    "SwarmHandle",
    "SwarmsAPI",
    "AsyncResearchSwarmHandle",
    "AsyncResearchSwarmsAPI",
    "ResearchSwarmHandle",
    "ResearchSwarmsAPI",
]
