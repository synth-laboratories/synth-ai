"""Durable run reads and explicit control. See backend SDK surface.md."""

from __future__ import annotations

import asyncio
import time

from synth_ai.contracts import swarms as operations
from synth_ai.core.http.async_client import AsyncHttpTransport
from synth_ai.core.http.client import HttpTransport
from synth_ai.core.observation import AsyncEventStream, EventStream
from synth_ai.core.parameters import identifier, page_parameters, verified
from synth_ai.errors import RequestTimeoutError
from synth_ai.options import MutationOptions, ObservationOptions, ReadOptions
from synth_ai.types.base import EventCursor, Page, PageCursor
from synth_ai.types.receipts import ControlReceipt
from synth_ai.types.usage import UsageOperation, UsageSummary

from .outputs import AsyncOutputs, Outputs
from .types.programs import ProgramRunView
from .types.runs import ResolvedConfiguration, RunSnapshot, SwarmEvent, SwarmResult, TaskOutcome


class SwarmRun:
    def events(
        self, *, observation: ObservationOptions, cursor: EventCursor | None = None
    ) -> EventStream[SwarmEvent]:
        """Create a lazy bounded observer; close never cancels remote work."""
        return EventStream(self.event_page, cursor, observation, owner=self._transport)

    def __init__(self, transport: HttpTransport, run_id: str) -> None:
        self._transport = transport
        self.run_id = identifier(run_id)
        self.outputs = Outputs(transport, self.run_id)

    def status(self, *, request: ReadOptions | None = None) -> RunSnapshot:
        """Read current authorized status; no implicit control mutation."""
        return self._transport.request(
            operations.GET,
            RunSnapshot,
            path=f"/swarms/runs/{self.run_id}",
            options=request,
            validate=lambda result: verified(
                result, result.run_id == self.run_id, "Run response identity mismatch"
            ),
        )

    def program(self, *, request: ReadOptions | None = None) -> ProgramRunView:
        """Inspect the admitted program and its projection without changing it."""
        return self._transport.request(
            operations.PROGRAM,
            ProgramRunView,
            path=f"/swarms/runs/{self.run_id}/program",
            options=request,
            validate=lambda result: verified(
                result, result.run_id == self.run_id, "Program run identity mismatch"
            ),
        )

    def configuration(self, *, request: ReadOptions | None = None) -> ResolvedConfiguration:
        """Read current authorized configuration; no implicit control mutation."""
        return self._transport.request(
            operations.CONFIGURATION,
            ResolvedConfiguration,
            path=f"/swarms/runs/{self.run_id}/configuration",
            options=request,
            validate=lambda result: verified(
                result, result.run_id == self.run_id, "Run response identity mismatch"
            ),
        )

    def result(self, *, request: ReadOptions | None = None) -> SwarmResult:
        """Read current authorized result; no implicit control mutation."""
        return self._transport.request(
            operations.RESULT,
            SwarmResult,
            path=f"/swarms/runs/{self.run_id}/result",
            options=request,
            validate=lambda result: verified(
                result, result.run_id == self.run_id, "Run response identity mismatch"
            ),
        )

    def usage(self, *, request: ReadOptions | None = None) -> UsageSummary:
        """Read current authorized usage; no implicit control mutation."""
        return self._transport.request(
            operations.USAGE,
            UsageSummary,
            path=f"/swarms/runs/{self.run_id}/usage",
            options=request,
        )

    def event_page(
        self, *, limit: int, cursor: EventCursor | None = None, request: ReadOptions | None = None
    ) -> Page[SwarmEvent, EventCursor]:
        """Read a bounded page; cursors never transfer authority."""
        return self._transport.request(
            operations.EVENTS,
            Page[SwarmEvent, EventCursor],
            path=f"/swarms/runs/{self.run_id}/events",
            query=page_parameters(limit, cursor),
            options=request,
            validate=lambda result: verified(
                result,
                len(result.items) <= limit
                and all(item.run_id == self.run_id for item in result.items),
                "Run page bound or identity mismatch",
            ),
        )

    def tasks(
        self, *, limit: int, cursor: PageCursor | None = None, request: ReadOptions | None = None
    ) -> Page[TaskOutcome]:
        """Read a bounded page; cursors never transfer authority."""
        return self._transport.request(
            operations.TASKS,
            Page[TaskOutcome],
            path=f"/swarms/runs/{self.run_id}/tasks",
            query=page_parameters(limit, cursor),
            options=request,
            validate=lambda result: verified(
                result,
                len(result.items) <= limit
                and all(item.run_id == self.run_id for item in result.items),
                "Run page bound or identity mismatch",
            ),
        )

    def usage_operations(
        self, *, limit: int, cursor: PageCursor | None = None, request: ReadOptions | None = None
    ) -> Page[UsageOperation]:
        """Read a bounded page; cursors never transfer authority."""
        return self._transport.request(
            operations.USAGE_ITEMS,
            Page[UsageOperation],
            path=f"/swarms/runs/{self.run_id}/usage/operations",
            query=page_parameters(limit, cursor),
            options=request,
            validate=lambda result: verified(
                result, len(result.items) <= limit, "Run page bound or identity mismatch"
            ),
        )

    def cancel(self, *, mutation: MutationOptions | None = None) -> ControlReceipt:
        """Request cancellation; acceptance does not imply settlement or cleanup."""
        return self._transport.request(
            operations.CANCEL,
            ControlReceipt,
            path=f"/swarms/runs/{self.run_id}/cancel",
            options=mutation,
            validate=lambda result: verified(
                result, result.resource_id == self.run_id, "Cancel identity mismatch"
            ),
        )

    def control(self, operation_id: str, *, request: ReadOptions | None = None) -> ControlReceipt:
        """Read a control receipt without issuing another cancellation."""
        operation_id = identifier(operation_id)
        return self._transport.request(
            operations.CONTROL,
            ControlReceipt,
            path=f"/swarms/runs/{self.run_id}/controls/{operation_id}",
            options=request,
            validate=lambda result: verified(
                result,
                result.operation_id == operation_id and result.resource_id == self.run_id,
                "Control identity mismatch",
            ),
        )

    def wait(self, *, observation: ObservationOptions) -> SwarmResult:
        """Wait for canonical output within a local deadline; never cancel remote work."""
        deadline = time.monotonic() + observation.timeout_seconds
        polls = min(10000, int(observation.timeout_seconds / observation.poll_interval_seconds) + 2)
        for _ in range(polls):
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise RequestTimeoutError("Swarm observation expired; remote work continues")
            snapshot = self.status(
                request=ReadOptions(
                    timeout_seconds=min(remaining, observation.idle_timeout_seconds),
                    read_retry_count=observation.retry_count_max,
                )
            )
            if snapshot.result_available:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise RequestTimeoutError("Swarm observation expired; remote work continues")
                return self.result(
                    request=ReadOptions(
                        timeout_seconds=remaining, read_retry_count=observation.retry_count_max
                    )
                )
            time.sleep(min(observation.poll_interval_seconds, max(0, deadline - time.monotonic())))
        raise RequestTimeoutError("Swarm observation expired; remote work continues")


class AsyncSwarmRun:
    def events(
        self, *, observation: ObservationOptions, cursor: EventCursor | None = None
    ) -> AsyncEventStream[SwarmEvent]:
        """Create a lazy native-async observer without starting a background task."""
        return AsyncEventStream(self.event_page, cursor, observation, owner=self._transport)

    def __init__(self, transport: AsyncHttpTransport, run_id: str) -> None:
        self._transport = transport
        self.run_id = identifier(run_id)
        self.outputs = AsyncOutputs(transport, self.run_id)

    async def status(self, *, request: ReadOptions | None = None) -> RunSnapshot:
        """Read current authorized status; no implicit control mutation."""
        return await self._transport.request(
            operations.GET,
            RunSnapshot,
            path=f"/swarms/runs/{self.run_id}",
            options=request,
            validate=lambda result: verified(
                result, result.run_id == self.run_id, "Run response identity mismatch"
            ),
        )

    async def program(self, *, request: ReadOptions | None = None) -> ProgramRunView:
        """Inspect the admitted program and its projection without changing it."""
        return await self._transport.request(
            operations.PROGRAM,
            ProgramRunView,
            path=f"/swarms/runs/{self.run_id}/program",
            options=request,
            validate=lambda result: verified(
                result, result.run_id == self.run_id, "Program run identity mismatch"
            ),
        )

    async def configuration(self, *, request: ReadOptions | None = None) -> ResolvedConfiguration:
        """Read current authorized configuration; no implicit control mutation."""
        return await self._transport.request(
            operations.CONFIGURATION,
            ResolvedConfiguration,
            path=f"/swarms/runs/{self.run_id}/configuration",
            options=request,
            validate=lambda result: verified(
                result, result.run_id == self.run_id, "Run response identity mismatch"
            ),
        )

    async def result(self, *, request: ReadOptions | None = None) -> SwarmResult:
        """Read current authorized result; no implicit control mutation."""
        return await self._transport.request(
            operations.RESULT,
            SwarmResult,
            path=f"/swarms/runs/{self.run_id}/result",
            options=request,
            validate=lambda result: verified(
                result, result.run_id == self.run_id, "Run response identity mismatch"
            ),
        )

    async def usage(self, *, request: ReadOptions | None = None) -> UsageSummary:
        """Read current authorized usage; no implicit control mutation."""
        return await self._transport.request(
            operations.USAGE,
            UsageSummary,
            path=f"/swarms/runs/{self.run_id}/usage",
            options=request,
        )

    async def event_page(
        self, *, limit: int, cursor: EventCursor | None = None, request: ReadOptions | None = None
    ) -> Page[SwarmEvent, EventCursor]:
        """Read a bounded page; cursors never transfer authority."""
        return await self._transport.request(
            operations.EVENTS,
            Page[SwarmEvent, EventCursor],
            path=f"/swarms/runs/{self.run_id}/events",
            query=page_parameters(limit, cursor),
            options=request,
            validate=lambda result: verified(
                result,
                len(result.items) <= limit
                and all(item.run_id == self.run_id for item in result.items),
                "Run page bound or identity mismatch",
            ),
        )

    async def tasks(
        self, *, limit: int, cursor: PageCursor | None = None, request: ReadOptions | None = None
    ) -> Page[TaskOutcome]:
        """Read a bounded page; cursors never transfer authority."""
        return await self._transport.request(
            operations.TASKS,
            Page[TaskOutcome],
            path=f"/swarms/runs/{self.run_id}/tasks",
            query=page_parameters(limit, cursor),
            options=request,
            validate=lambda result: verified(
                result,
                len(result.items) <= limit
                and all(item.run_id == self.run_id for item in result.items),
                "Run page bound or identity mismatch",
            ),
        )

    async def usage_operations(
        self, *, limit: int, cursor: PageCursor | None = None, request: ReadOptions | None = None
    ) -> Page[UsageOperation]:
        """Read a bounded page; cursors never transfer authority."""
        return await self._transport.request(
            operations.USAGE_ITEMS,
            Page[UsageOperation],
            path=f"/swarms/runs/{self.run_id}/usage/operations",
            query=page_parameters(limit, cursor),
            options=request,
            validate=lambda result: verified(
                result, len(result.items) <= limit, "Run page bound or identity mismatch"
            ),
        )

    async def cancel(self, *, mutation: MutationOptions | None = None) -> ControlReceipt:
        """Request cancellation; acceptance does not imply settlement or cleanup."""
        return await self._transport.request(
            operations.CANCEL,
            ControlReceipt,
            path=f"/swarms/runs/{self.run_id}/cancel",
            options=mutation,
            validate=lambda result: verified(
                result, result.resource_id == self.run_id, "Cancel identity mismatch"
            ),
        )

    async def control(
        self, operation_id: str, *, request: ReadOptions | None = None
    ) -> ControlReceipt:
        """Read a control receipt without issuing another cancellation."""
        operation_id = identifier(operation_id)
        return await self._transport.request(
            operations.CONTROL,
            ControlReceipt,
            path=f"/swarms/runs/{self.run_id}/controls/{operation_id}",
            options=request,
            validate=lambda result: verified(
                result,
                result.operation_id == operation_id and result.resource_id == self.run_id,
                "Control identity mismatch",
            ),
        )

    async def wait(self, *, observation: ObservationOptions) -> SwarmResult:
        """Wait for canonical output within a local deadline; never cancel remote work."""
        deadline = time.monotonic() + observation.timeout_seconds
        polls = min(10000, int(observation.timeout_seconds / observation.poll_interval_seconds) + 2)
        for _ in range(polls):
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise RequestTimeoutError("Swarm observation expired; remote work continues")
            snapshot = await self.status(
                request=ReadOptions(
                    timeout_seconds=min(remaining, observation.idle_timeout_seconds),
                    read_retry_count=observation.retry_count_max,
                )
            )
            if snapshot.result_available:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise RequestTimeoutError("Swarm observation expired; remote work continues")
                return await self.result(
                    request=ReadOptions(
                        timeout_seconds=remaining, read_retry_count=observation.retry_count_max
                    )
                )
            await asyncio.sleep(
                min(observation.poll_interval_seconds, max(0, deadline - time.monotonic()))
            )
        raise RequestTimeoutError("Swarm observation expired; remote work continues")
