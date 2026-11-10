"""Observe milestones using durable-feed wakeups and authoritative run snapshots.

The feed opens before each initial read and stays open through Core's last run
milestone (cleanup_complete), so nothing polls after execution closes. Each
wake is confirmed with one getRun; periodic reconciliation covers lost wakes. Each wait owns and closes its
single pending feed read, and cancellation never mutates the server-side run.
"""

from __future__ import annotations

import asyncio
import math
from collections.abc import AsyncIterator
from contextlib import AbstractAsyncContextManager
from typing import Protocol

import httpx

from ._event_stream import RunMilestone
from ._generated.models import RunEvent, RunSnapshot
from ._protocol import CompletionMilestone
from ._retry import RetryPolicy
from ._stream import StreamCheckpoint
from .errors import (
    ResponseContractError,
    ResultUnavailableError,
    RunFailedError,
    RunHistoryUnavailableError,
    TransportFailureError,
    WaitTimeoutError,
    local_problem,
)

# Repair for a lost wake only: Core's stream carries every milestone, so an
# idle open feed costs one getRun per interval, never one per poll_interval.
_RECONCILE_SECONDS = 30.0


class ObservationSource(Protocol):
    retry_policy: RetryPolicy

    def subscribe_events(
        self, run_id: str, state: StreamCheckpoint
    ) -> AbstractAsyncContextManager[AsyncIterator[RunEvent | RunMilestone]]: ...

    async def request(
        self, operation_id: str, model: type[RunSnapshot], *, path: dict[str, str]
    ) -> RunSnapshot: ...


class RunObserver:
    def __init__(self, source: ObservationSource) -> None:
        self._source = source

    async def wait(
        self,
        run_id: str,
        *,
        timeout: float,
        poll_interval: float,
        until: CompletionMilestone,
        raise_on_failure: bool,
    ) -> RunSnapshot:
        validate_wait(timeout, poll_interval, until)
        checkpoint = StreamCheckpoint(run_id=run_id)
        snapshot: RunSnapshot | None = None

        async def read() -> RunSnapshot:
            nonlocal snapshot
            snapshot = await self._source.request("getRun", RunSnapshot, path={"run_id": run_id})
            return snapshot

        def reached(current: RunSnapshot) -> bool:
            if current.reached(until):
                if raise_on_failure and current.state == "failed":
                    raise RunFailedError("Run " + run_id + " failed", snapshot=current)
                return True
            if until == CompletionMilestone.RESULT_AVAILABLE and current.finished:
                raise ResultUnavailableError(
                    "Run has finished without an available result", snapshot=current
                )
            if current.metadata.get("imported_history") is not None:
                raise RunHistoryUnavailableError(
                    "This historical run cannot prove the requested milestone", snapshot=current
                )
            return False

        async def next_event(
            events: AsyncIterator[RunEvent | RunMilestone],
        ) -> RunEvent | RunMilestone:
            return await anext(events)

        try:
            async with asyncio.timeout(timeout):
                for attempt in range(self._source.retry_policy.attempts):
                    try:
                        async with self._source.subscribe_events(run_id, checkpoint) as events:
                            if reached(await read()):
                                assert snapshot is not None
                                return snapshot
                            pending = asyncio.create_task(next_event(events))
                            final = False
                            try:
                                while True:
                                    done, _ = await asyncio.wait(
                                        {pending}, timeout=max(_RECONCILE_SECONDS, poll_interval)
                                    )
                                    ended = False
                                    if done:
                                        try:
                                            item = pending.result()
                                        except StopAsyncIteration:
                                            ended = True
                                        else:
                                            final = final or (
                                                isinstance(item, RunMilestone) and item.final
                                            )
                                    if reached(await read()):
                                        assert snapshot is not None
                                        return snapshot
                                    if ended:
                                        break
                                    if done:
                                        pending = asyncio.create_task(next_event(events))
                            finally:
                                pending.cancel()
                                await asyncio.gather(pending, return_exceptions=True)
                            if final:
                                # Core sent its last milestone and closed the feed,
                                # yet the run does not show the requested one.
                                error = ResponseContractError(
                                    local_problem(
                                        "event_stream_ended_before_milestone", "listRunEvents"
                                    )
                                )
                                error.checkpoint = checkpoint
                                raise error
                    except httpx.TransportError:
                        pass
                    # A feed that ends before its final milestone (lifetime cap,
                    # disconnect) resumes from the checkpoint.
                    if attempt + 1 < self._source.retry_policy.attempts:
                        delay = self._source.retry_policy.delay(None, attempt)
                        await asyncio.sleep(max(poll_interval, delay or 0))
                error = TransportFailureError(
                    local_problem("event_stream_interrupted", "listRunEvents")
                )
                error.checkpoint = checkpoint
                raise error
        except TimeoutError as cause:
            raise WaitTimeoutError(
                "Timed out waiting for " + run_id, snapshot=snapshot, run_id=run_id
            ) from cause


def validate_wait(timeout: float, poll_interval: float, until: str) -> None:
    if (
        not math.isfinite(timeout)
        or not math.isfinite(poll_interval)
        or timeout <= 0
        or poll_interval <= 0
    ):
        raise ValueError("timeout and poll_interval must be finite and positive")
    if until not in {"finished", "execution_closed", "result_available"}:
        raise ValueError("unknown completion milestone")
