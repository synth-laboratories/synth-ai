"""Durable run reads and explicit control. See backend SDK surface.md."""

from __future__ import annotations

import asyncio
import time
import threading
from typing import get_args

from synth_ai.contracts import swarms as operations
from synth_ai.contracts.operations import Operation
from synth_ai.core.http.async_client import AsyncHttpTransport
from synth_ai.core.http.client import HttpTransport
from synth_ai.core.observation import AsyncEventStream, EventStream
from synth_ai.core.parameters import identifier, page_parameters, verified
from synth_ai.errors import (
    ContractError,
    OutcomeUncertainError,
    RequestTimeoutError,
    ResultNotReadyError,
    SynthError,
)
from synth_ai.options import MutationOptions, ObservationOptions, ReadOptions
from synth_ai.sdk.core.links import AsyncLinks, Links
from synth_ai.types.base import EventCursor, Page, PageCursor
from synth_ai.types.receipts import ControlCommand, ControlReceipt, MutationResolution
from synth_ai.types.refs import Link, Ref, Relation
from synth_ai.types.usage import UsageOperation, UsageSummary

from .outputs import AsyncOutputs, Outputs
from .types.programs import ProgramRunView
from .types.runs import (
    KnownRunState,
    ResolvedConfiguration,
    RunSnapshot,
    RunState,
    SwarmEvent,
    SwarmResult,
    TaskOutcome,
)


def _control_receipt(value: ControlReceipt, run_id: str, command: ControlCommand) -> ControlReceipt:
    return verified(
        value,
        value.resource_id == run_id and value.command in (None, command),
        "Control identity or command mismatch",
    )


def _recovery_read(deadline: float, error: OutcomeUncertainError) -> ReadOptions:
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise error
    return ReadOptions(timeout_seconds=remaining, read_retry_count=0)


def _target_states(value: RunState | tuple[RunState, ...]) -> frozenset[RunState]:
    values = (value,) if isinstance(value, str) else value
    if (
        not isinstance(values, tuple)
        or not values
        or any(item not in get_args(KnownRunState) for item in values)
    ):
        raise ValueError("At least one supported run state is required")
    return frozenset(values)


class RunIdentity:
    run_id: str
    _reference: Ref | None = None
    _project_id: str | None = None

    @property
    def ref(self) -> Ref:
        """Exact frozen launch identity, retained from an authoritative snapshot."""
        if self._reference is None:
            raise ContractError(
                "Read run status from a backend providing exact run references",
                code="run_ref_unavailable",
                stage="run_reference",
            )
        return self._reference

    def _observe_snapshot(self, snapshot: RunSnapshot) -> RunSnapshot:
        with self._identity_lock:
            return self._observe_locked(snapshot)

    def _observe_locked(self, snapshot: RunSnapshot) -> RunSnapshot:
        verified(
            snapshot,
            snapshot.run_id == self.run_id
            and (self._project_id is None or snapshot.project_id == self._project_id)
            and (self._reference is None or snapshot.ref in (None, self._reference)),
            "Run response changed its immutable launch identity",
        )
        self._project_id = snapshot.project_id
        if snapshot.ref is not None:
            self._reference = snapshot.ref
        return snapshot

    def _link_project(self) -> str:
        if self._project_id is None or self._reference is None:
            raise ContractError(
                "Read run status before listing provenance", code="run_ref_unavailable"
            )
        return self._project_id


class SwarmRun(RunIdentity):
    def events(
        self, *, observation: ObservationOptions, cursor: EventCursor | None = None
    ) -> EventStream[SwarmEvent]:
        """Create a lazy bounded observer; close never cancels remote work."""
        return EventStream(self.event_page, cursor, observation, owner=self._transport)

    def __init__(
        self, transport: HttpTransport, run_id: str, *, snapshot: RunSnapshot | None = None
    ) -> None:
        self._transport = transport
        self.run_id = identifier(run_id)
        self._identity_lock = threading.Lock()
        from .labels import RunLabels

        self.labels = RunLabels(self)
        self.outputs = Outputs(transport, self.run_id)
        if snapshot is not None:
            self._observe_snapshot(snapshot)

    def objectives(self, *, request=None):
        return self.status(request=request).objectives

    def download(self, destination, *, options=None):
        from .bundle import download

        return download(self, destination, options)

    def produced(
        self,
        *,
        limit: int = 200,
        cursor: PageCursor | None = None,
        request: ReadOptions | None = None,
    ) -> Page[Link]:
        return Links(self._transport).list(
            self._link_project(),
            subject=self.ref,
            relation=Relation.PRODUCED,
            limit=limit,
            cursor=cursor,
            request=request,
        )

    def consumed(
        self,
        *,
        limit: int = 200,
        cursor: PageCursor | None = None,
        request: ReadOptions | None = None,
    ) -> Page[Link]:
        return Links(self._transport).list(
            self._link_project(),
            subject=self.ref,
            relation=Relation.CONSUMED,
            limit=limit,
            cursor=cursor,
            request=request,
        )

    def status(self, *, request: ReadOptions | None = None) -> RunSnapshot:
        """Read current authorized status; no implicit control mutation."""
        return self._transport.request(
            operations.GET,
            RunSnapshot,
            path=f"/swarms/runs/{self.run_id}",
            options=request,
            validate=self._observe_snapshot,
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
        """Request cancel; receipt progress is independent of accounting and cleanup."""
        return self._send_control(operations.CANCEL, "cancel", mutation)

    def pause(self, *, mutation: MutationOptions | None = None) -> ControlReceipt:
        """Request pause; receipt progress is independent of accounting and cleanup."""
        return self._send_control(operations.PAUSE, "pause", mutation)

    def resume(self, *, mutation: MutationOptions | None = None) -> ControlReceipt:
        """Request resume; receipt progress is independent of accounting and cleanup."""
        return self._send_control(operations.RESUME, "resume", mutation)

    def _send_control(
        self, operation: Operation, command: ControlCommand, mutation: MutationOptions | None
    ) -> ControlReceipt:
        # One absolute local deadline includes the POST and at most two reads.
        timeout = mutation.timeout_seconds if mutation is not None else None
        deadline = time.monotonic() + (timeout or self._transport.request_timeout_seconds)
        try:
            return self._transport.request(
                operation,
                ControlReceipt,
                path=f"/swarms/runs/{self.run_id}/{command}",
                options=mutation,
                validate=lambda value: _control_receipt(value, self.run_id, command),
            )
        except OutcomeUncertainError as error:
            identity = error.identity
            if identity is None:
                raise
            if (
                identity.operation != operation.id
                or identity.resource_path != f"/api/v1/swarms/runs/{self.run_id}/{command}"
            ):
                raise ContractError(
                    "Uncertain control returned a different identity", identity=identity
                ) from error
            try:
                resolution = self._transport.request(
                    operations.RESOLVE,
                    MutationResolution,
                    query={
                        "key": identity.key,
                        "operation": identity.operation,
                        "resource_path": identity.resource_path,
                        "payload_sha256": identity.payload_sha256,
                    },
                    options=_recovery_read(deadline, error),
                    validate=lambda value: verified(
                        value, value.identity == identity, "Resolver identity mismatch"
                    ),
                )
                if (
                    resolution.state not in {"accepted", "pending", "not_accepted"}
                    or resolution.metadata.operation_id is None
                ):
                    raise error
                if resolution.resource is not None and resolution.resource.kind != "control":
                    raise ContractError(
                        "Resolver returned a different resource kind", identity=identity
                    )
                # The resolver metadata names public custody; resource_id may
                # instead name the underlying owner input. Never repeat POST.
                result = self.control(
                    resolution.metadata.operation_id, request=_recovery_read(deadline, error)
                )
                return _control_receipt(result, self.run_id, command)
            except SynthError as recovery_error:
                if recovery_error is error:
                    raise
                raise error from recovery_error

    def control(self, operation_id: str, *, request: ReadOptions | None = None) -> ControlReceipt:
        """Read a control receipt without issuing another control."""
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

    def wait_for_state(
        self, state: RunState | tuple[RunState, ...], *, observation: ObservationOptions
    ) -> RunSnapshot:
        """Observe target states within the deadline and maximum_events status reads."""
        targets = _target_states(state)
        deadline = time.monotonic() + observation.timeout_seconds
        for _ in range(observation.maximum_events):
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            snapshot = self.status(
                request=ReadOptions(
                    timeout_seconds=min(remaining, observation.idle_timeout_seconds),
                    read_retry_count=observation.retry_count_max,
                )
            )
            if time.monotonic() >= deadline:
                break
            if snapshot.state in targets:
                return snapshot
            if snapshot.execution_closed:
                raise ResultNotReadyError(
                    f"Run {self.run_id} ended in state {snapshot.state} before the requested state",
                    code="run_state_unreachable",
                    request_id=snapshot.metadata.request_id,
                    retryable=False,
                )
            time.sleep(min(observation.poll_interval_seconds, max(0, deadline - time.monotonic())))
        raise RequestTimeoutError(
            "Swarm state observation expired; remote work continues", code="run_observation_expired"
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
            if snapshot.state == "paused":
                raise ResultNotReadyError(
                    f"Run {self.run_id} is paused; its original deadline continues",
                    code="run_paused",
                    request_id=snapshot.metadata.request_id,
                    retryable=False,
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


class AsyncSwarmRun(RunIdentity):
    def events(
        self, *, observation: ObservationOptions, cursor: EventCursor | None = None
    ) -> AsyncEventStream[SwarmEvent]:
        """Create a lazy native-async observer without starting a background task."""
        return AsyncEventStream(self.event_page, cursor, observation, owner=self._transport)

    def __init__(
        self, transport: AsyncHttpTransport, run_id: str, *, snapshot: RunSnapshot | None = None
    ) -> None:
        self._transport = transport
        self.run_id = identifier(run_id)
        self._identity_lock = threading.Lock()
        from .labels import AsyncRunLabels

        self.labels = AsyncRunLabels(self)
        self.outputs = AsyncOutputs(transport, self.run_id)
        if snapshot is not None:
            self._observe_snapshot(snapshot)

    async def objectives(self, *, request=None):
        return (await self.status(request=request)).objectives

    async def download(self, destination, *, options=None):
        from .bundle import download_async

        return await download_async(self, destination, options)

    async def produced(
        self,
        *,
        limit: int = 200,
        cursor: PageCursor | None = None,
        request: ReadOptions | None = None,
    ) -> Page[Link]:
        return await AsyncLinks(self._transport).list(
            self._link_project(),
            subject=self.ref,
            relation=Relation.PRODUCED,
            limit=limit,
            cursor=cursor,
            request=request,
        )

    async def consumed(
        self,
        *,
        limit: int = 200,
        cursor: PageCursor | None = None,
        request: ReadOptions | None = None,
    ) -> Page[Link]:
        return await AsyncLinks(self._transport).list(
            self._link_project(),
            subject=self.ref,
            relation=Relation.CONSUMED,
            limit=limit,
            cursor=cursor,
            request=request,
        )

    async def status(self, *, request: ReadOptions | None = None) -> RunSnapshot:
        """Read current authorized status; no implicit control mutation."""
        return await self._transport.request(
            operations.GET,
            RunSnapshot,
            path=f"/swarms/runs/{self.run_id}",
            options=request,
            validate=self._observe_snapshot,
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
        """Request cancel; receipt progress is independent of accounting and cleanup."""
        return await self._send_control(operations.CANCEL, "cancel", mutation)

    async def pause(self, *, mutation: MutationOptions | None = None) -> ControlReceipt:
        """Request pause; receipt progress is independent of accounting and cleanup."""
        return await self._send_control(operations.PAUSE, "pause", mutation)

    async def resume(self, *, mutation: MutationOptions | None = None) -> ControlReceipt:
        """Request resume; receipt progress is independent of accounting and cleanup."""
        return await self._send_control(operations.RESUME, "resume", mutation)

    async def _send_control(
        self, operation: Operation, command: ControlCommand, mutation: MutationOptions | None
    ) -> ControlReceipt:
        # One absolute local deadline includes the POST and at most two reads.
        timeout = mutation.timeout_seconds if mutation is not None else None
        deadline = time.monotonic() + (timeout or self._transport.request_timeout_seconds)
        try:
            return await self._transport.request(
                operation,
                ControlReceipt,
                path=f"/swarms/runs/{self.run_id}/{command}",
                options=mutation,
                validate=lambda value: _control_receipt(value, self.run_id, command),
            )
        except OutcomeUncertainError as error:
            identity = error.identity
            if identity is None:
                raise
            if (
                identity.operation != operation.id
                or identity.resource_path != f"/api/v1/swarms/runs/{self.run_id}/{command}"
            ):
                raise ContractError(
                    "Uncertain control returned a different identity", identity=identity
                ) from error
            try:
                resolution = await self._transport.request(
                    operations.RESOLVE,
                    MutationResolution,
                    query={
                        "key": identity.key,
                        "operation": identity.operation,
                        "resource_path": identity.resource_path,
                        "payload_sha256": identity.payload_sha256,
                    },
                    options=_recovery_read(deadline, error),
                    validate=lambda value: verified(
                        value, value.identity == identity, "Resolver identity mismatch"
                    ),
                )
                if (
                    resolution.state not in {"accepted", "pending", "not_accepted"}
                    or resolution.metadata.operation_id is None
                ):
                    raise error
                if resolution.resource is not None and resolution.resource.kind != "control":
                    raise ContractError(
                        "Resolver returned a different resource kind", identity=identity
                    )
                # The resolver metadata names public custody; resource_id may
                # instead name the underlying owner input. Never repeat POST.
                result = await self.control(
                    resolution.metadata.operation_id, request=_recovery_read(deadline, error)
                )
                return _control_receipt(result, self.run_id, command)
            except SynthError as recovery_error:
                if recovery_error is error:
                    raise
                raise error from recovery_error

    async def control(
        self, operation_id: str, *, request: ReadOptions | None = None
    ) -> ControlReceipt:
        """Read a control receipt without issuing another control."""
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

    async def wait_for_state(
        self, state: RunState | tuple[RunState, ...], *, observation: ObservationOptions
    ) -> RunSnapshot:
        """Observe target states within the deadline and maximum_events status reads."""
        targets = _target_states(state)
        deadline = time.monotonic() + observation.timeout_seconds
        for _ in range(observation.maximum_events):
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            snapshot = await self.status(
                request=ReadOptions(
                    timeout_seconds=min(remaining, observation.idle_timeout_seconds),
                    read_retry_count=observation.retry_count_max,
                )
            )
            if time.monotonic() >= deadline:
                break
            if snapshot.state in targets:
                return snapshot
            if snapshot.execution_closed:
                raise ResultNotReadyError(
                    f"Run {self.run_id} ended in state {snapshot.state} before the requested state",
                    code="run_state_unreachable",
                    request_id=snapshot.metadata.request_id,
                    retryable=False,
                )
            await asyncio.sleep(
                min(observation.poll_interval_seconds, max(0, deadline - time.monotonic()))
            )
        raise RequestTimeoutError(
            "Swarm state observation expired; remote work continues", code="run_observation_expired"
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
            if snapshot.state == "paused":
                raise ResultNotReadyError(
                    f"Run {self.run_id} is paused; its original deadline continues",
                    code="run_paused",
                    request_id=snapshot.metadata.request_id,
                    retryable=False,
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
