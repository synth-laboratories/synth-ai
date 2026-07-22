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
from synth_ai.core.research.contracts.activity import ActivityWindow, SwarmActivity
from synth_ai.core.research.contracts.common import (
    ArtifactId,
    ParticipantSessionId,
    ProjectId,
    SwarmId,
    WorkProductId,
)
from synth_ai.core.research.contracts.evidence import (
    ContentDisposition,
    SwarmEvidence,
)
from synth_ai.core.research.contracts.status import SwarmStatus
from synth_ai.core.research.contracts.swarms import (
    BranchResult,
    BranchSpec,
    ResolvedSwarmConfiguration,
    Swarm,
    SwarmPreflight,
    SwarmSpec,
)
from synth_ai.core.research.contracts.transcript import (
    SwarmTranscriptPage,
    TranscriptView,
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
    return tuple(Swarm.from_wire(item) for item in array_value(value, operation_id=operation_id))


def _wait_arguments(timeout_seconds: float, poll_interval_seconds: float) -> None:
    if timeout_seconds <= 0:
        raise ValueError("timeout_seconds must be positive")
    if poll_interval_seconds <= 0:
        raise ValueError("poll_interval_seconds must be positive")


def _transcript_limit(limit: int) -> int:
    if type(limit) is not int or not 1 <= limit <= 500:
        raise ValueError("limit must be an integer from 1 through 500")
    return limit


def _stream_timeout(timeout_seconds: float | None) -> float | None:
    if timeout_seconds is not None and timeout_seconds <= 0:
        raise ValueError("timeout_seconds must be positive when provided")
    return timeout_seconds


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

    def evidence(self) -> SwarmEvidence:
        """Return durable artifact and WorkProduct evidence."""
        return self._api.evidence(self.swarm_id)

    def activity(
        self,
        window: ActivityWindow = ActivityWindow(),
    ) -> SwarmActivity:
        """Return one bounded actor, task, message, event, and output snapshot."""
        return self._api.activity(self.swarm_id, window)

    def status(self) -> SwarmStatus:
        """Return the cheap authoritative status projection."""
        return self._api.status(self.swarm_id)

    def workspace_archive(self, *, timeout_seconds: float | None = None) -> bytes:
        """Download the run-owned workspace archive bytes."""
        return self._api.workspace_archive(
            self.swarm_id,
            timeout_seconds=timeout_seconds,
        )

    def transcript(
        self,
        *,
        participant_session_id: ParticipantSessionId | None = None,
        cursor: str | None = None,
        limit: int = 200,
        view: TranscriptView = TranscriptView.OPERATOR,
    ) -> SwarmTranscriptPage:
        """Return one page from the live or terminal transcript authority."""
        return self._api.transcript(
            self.swarm_id,
            participant_session_id=participant_session_id,
            cursor=cursor,
            limit=limit,
            view=view,
        )

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
        transcript_cursor: str | None = None,
        view: TranscriptView = TranscriptView.OPERATOR,
        last_event_id: str | None = None,
        timeout_seconds: float | None = None,
    ) -> Iterator[SwarmEvent]:
        yield from self._api.events(
            self.swarm_id,
            transcript_cursor=transcript_cursor,
            view=view,
            last_event_id=last_event_id,
            timeout_seconds=timeout_seconds,
        )


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
        value = self._transport.execute(_request(operation_id, path, body=request.to_wire()))
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
        value = self._transport.execute(_request(operation_id, path, body=request.to_wire()))
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
        value = self._transport.execute(_request("retrieve_run", f"/smr/runs/{swarm_id}"))
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

    def evidence(self, swarm_id: SwarmId) -> SwarmEvidence:
        """Return durable artifact and WorkProduct evidence."""
        value = self._transport.execute(
            _request(
                "retrieve_swarm_evidence",
                f"/smr/runs/{swarm_id}/evidence",
            )
        )
        return SwarmEvidence.from_wire(value)

    def activity(
        self,
        swarm_id: SwarmId,
        window: ActivityWindow = ActivityWindow(),
    ) -> SwarmActivity:
        """Return one bounded actor, task, message, event, and output snapshot."""
        value = self._transport.execute(
            _request(
                "retrieve_swarm_activity",
                f"/smr/runs/{swarm_id}/activity",
                query=window.to_query(),
            )
        )
        return SwarmActivity.from_wire(value)

    def status(self, swarm_id: SwarmId) -> SwarmStatus:
        """Return the cheap authoritative status projection for a swarm."""
        value = self._transport.execute(
            _request(
                "retrieve_swarm_status",
                f"/smr/runs/{swarm_id}/status",
            )
        )
        return SwarmStatus.from_wire(value)

    def workspace_archive(
        self,
        swarm_id: SwarmId,
        *,
        timeout_seconds: float | None = None,
    ) -> bytes:
        """Download the run-owned workspace archive bytes."""
        return self._transport.request_bytes(
            "GET",
            f"/smr/runs/{swarm_id}/workspace/archive",
            timeout_seconds=timeout_seconds,
            operation_id="retrieve_swarm_workspace_archive",
        )

    def transcript(
        self,
        swarm_id: SwarmId,
        *,
        participant_session_id: ParticipantSessionId | None = None,
        cursor: str | None = None,
        limit: int = 200,
        view: TranscriptView = TranscriptView.OPERATOR,
    ) -> SwarmTranscriptPage:
        """Return one page with explicit replay authority and cursor semantics."""
        query: JsonObject = {
            "limit": _transcript_limit(limit),
            "view": TranscriptView(view).value,
        }
        if participant_session_id is not None:
            query["participant_session_id"] = participant_session_id
        if cursor is not None:
            query["cursor"] = cursor
        value = self._transport.execute(
            _request(
                "list_run_transcript",
                f"/smr/runs/{swarm_id}/runtime/transcript",
                query=query,
            )
        )
        return SwarmTranscriptPage.from_wire(value)

    def artifact_content(
        self,
        artifact_id: ArtifactId,
        *,
        disposition: ContentDisposition = ContentDisposition.INLINE,
        timeout_seconds: float | None = None,
    ) -> bytes:
        """Read durable bytes for an artifact advertised by swarm evidence."""
        operation = research_operation("retrieve_swarm_artifact_content")
        return self._transport.request_bytes(
            operation.method.value,
            f"/smr/artifacts/{artifact_id}/content",
            params={"disposition": disposition.value},
            timeout_seconds=timeout_seconds,
            operation_id=str(operation.operation_id),
        )

    def work_product_content(
        self,
        work_product_id: WorkProductId,
        *,
        disposition: ContentDisposition = ContentDisposition.INLINE,
        timeout_seconds: float | None = None,
    ) -> bytes:
        """Read durable bytes for a WorkProduct advertised by swarm evidence."""
        operation = research_operation("retrieve_swarm_work_product_content")
        return self._transport.request_bytes(
            operation.method.value,
            f"/smr/work-products/{work_product_id}/content",
            params={"disposition": disposition.value},
            timeout_seconds=timeout_seconds,
            operation_id=str(operation.operation_id),
        )

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
        value = self._transport.execute(_request("pause_run", f"/smr/runs/{swarm_id}/pause"))
        return Swarm.from_wire(value)

    def resume(self, swarm_id: SwarmId) -> Swarm:
        value = self._transport.execute(_request("resume_run", f"/smr/runs/{swarm_id}/resume"))
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
        transcript_cursor: str | None = None,
        view: TranscriptView = TranscriptView.OPERATOR,
        last_event_id: str | None = None,
        timeout_seconds: float | None = None,
    ) -> Iterator[SwarmEvent]:
        query: JsonObject = {"view": TranscriptView(view).value}
        if transcript_cursor is not None:
            query["transcript_cursor"] = transcript_cursor
        for event in self._transport.stream_sse(
            f"/smr/runs/{swarm_id}/runtime/stream",
            params=query,
            last_event_id=last_event_id,
            timeout_seconds=_stream_timeout(timeout_seconds),
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

    async def evidence(self) -> SwarmEvidence:
        """Return durable artifact and WorkProduct evidence."""
        return await self._api.evidence(self.swarm_id)

    async def activity(
        self,
        window: ActivityWindow = ActivityWindow(),
    ) -> SwarmActivity:
        """Return one bounded actor, task, message, event, and output snapshot."""
        return await self._api.activity(self.swarm_id, window)

    async def status(self) -> SwarmStatus:
        """Return the cheap authoritative status projection."""
        return await self._api.status(self.swarm_id)

    async def workspace_archive(
        self,
        *,
        timeout_seconds: float | None = None,
    ) -> bytes:
        """Download the run-owned workspace archive bytes."""
        return await self._api.workspace_archive(
            self.swarm_id,
            timeout_seconds=timeout_seconds,
        )

    async def transcript(
        self,
        *,
        participant_session_id: ParticipantSessionId | None = None,
        cursor: str | None = None,
        limit: int = 200,
        view: TranscriptView = TranscriptView.OPERATOR,
    ) -> SwarmTranscriptPage:
        """Return one page from the live or terminal transcript authority."""
        return await self._api.transcript(
            self.swarm_id,
            participant_session_id=participant_session_id,
            cursor=cursor,
            limit=limit,
            view=view,
        )

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
        transcript_cursor: str | None = None,
        view: TranscriptView = TranscriptView.OPERATOR,
        last_event_id: str | None = None,
        timeout_seconds: float | None = None,
    ) -> AsyncIterator[SwarmEvent]:
        async for event in self._api.events(
            self.swarm_id,
            transcript_cursor=transcript_cursor,
            view=view,
            last_event_id=last_event_id,
            timeout_seconds=timeout_seconds,
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
        value = await self._transport.execute(_request(operation_id, path, body=request.to_wire()))
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
        value = await self._transport.execute(_request(operation_id, path, body=request.to_wire()))
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
        value = await self._transport.execute(_request("retrieve_run", f"/smr/runs/{swarm_id}"))
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

    async def evidence(self, swarm_id: SwarmId) -> SwarmEvidence:
        """Return durable artifact and WorkProduct evidence."""
        value = await self._transport.execute(
            _request(
                "retrieve_swarm_evidence",
                f"/smr/runs/{swarm_id}/evidence",
            )
        )
        return SwarmEvidence.from_wire(value)

    async def activity(
        self,
        swarm_id: SwarmId,
        window: ActivityWindow = ActivityWindow(),
    ) -> SwarmActivity:
        """Return one bounded actor, task, message, event, and output snapshot."""
        value = await self._transport.execute(
            _request(
                "retrieve_swarm_activity",
                f"/smr/runs/{swarm_id}/activity",
                query=window.to_query(),
            )
        )
        return SwarmActivity.from_wire(value)

    async def status(self, swarm_id: SwarmId) -> SwarmStatus:
        """Return the cheap authoritative status projection for a swarm."""
        value = await self._transport.execute(
            _request(
                "retrieve_swarm_status",
                f"/smr/runs/{swarm_id}/status",
            )
        )
        return SwarmStatus.from_wire(value)

    async def workspace_archive(
        self,
        swarm_id: SwarmId,
        *,
        timeout_seconds: float | None = None,
    ) -> bytes:
        """Download the run-owned workspace archive bytes."""
        return await self._transport.request_bytes(
            "GET",
            f"/smr/runs/{swarm_id}/workspace/archive",
            timeout_seconds=timeout_seconds,
            operation_id="retrieve_swarm_workspace_archive",
        )

    async def transcript(
        self,
        swarm_id: SwarmId,
        *,
        participant_session_id: ParticipantSessionId | None = None,
        cursor: str | None = None,
        limit: int = 200,
        view: TranscriptView = TranscriptView.OPERATOR,
    ) -> SwarmTranscriptPage:
        """Return one page with explicit replay authority and cursor semantics."""
        query: JsonObject = {
            "limit": _transcript_limit(limit),
            "view": TranscriptView(view).value,
        }
        if participant_session_id is not None:
            query["participant_session_id"] = participant_session_id
        if cursor is not None:
            query["cursor"] = cursor
        value = await self._transport.execute(
            _request(
                "list_run_transcript",
                f"/smr/runs/{swarm_id}/runtime/transcript",
                query=query,
            )
        )
        return SwarmTranscriptPage.from_wire(value)

    async def artifact_content(
        self,
        artifact_id: ArtifactId,
        *,
        disposition: ContentDisposition = ContentDisposition.INLINE,
        timeout_seconds: float | None = None,
    ) -> bytes:
        """Read durable bytes for an artifact advertised by swarm evidence."""
        operation = research_operation("retrieve_swarm_artifact_content")
        return await self._transport.request_bytes(
            operation.method.value,
            f"/smr/artifacts/{artifact_id}/content",
            params={"disposition": disposition.value},
            timeout_seconds=timeout_seconds,
            operation_id=str(operation.operation_id),
        )

    async def work_product_content(
        self,
        work_product_id: WorkProductId,
        *,
        disposition: ContentDisposition = ContentDisposition.INLINE,
        timeout_seconds: float | None = None,
    ) -> bytes:
        """Read durable bytes for a WorkProduct advertised by swarm evidence."""
        operation = research_operation("retrieve_swarm_work_product_content")
        return await self._transport.request_bytes(
            operation.method.value,
            f"/smr/work-products/{work_product_id}/content",
            params={"disposition": disposition.value},
            timeout_seconds=timeout_seconds,
            operation_id=str(operation.operation_id),
        )

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
        value = await self._transport.execute(_request("pause_run", f"/smr/runs/{swarm_id}/pause"))
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
        transcript_cursor: str | None = None,
        view: TranscriptView = TranscriptView.OPERATOR,
        last_event_id: str | None = None,
        timeout_seconds: float | None = None,
    ) -> AsyncIterator[SwarmEvent]:
        query: JsonObject = {"view": TranscriptView(view).value}
        if transcript_cursor is not None:
            query["transcript_cursor"] = transcript_cursor
        async for event in self._transport.stream_sse(
            f"/smr/runs/{swarm_id}/runtime/stream",
            params=query,
            last_event_id=last_event_id,
            timeout_seconds=_stream_timeout(timeout_seconds),
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
