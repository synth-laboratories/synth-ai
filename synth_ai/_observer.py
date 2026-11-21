"""Observe milestones using durable-feed wakeups and authoritative run snapshots.

The feed opens before each initial read and stays open through cleanup. Each
milestone wakes an authoritative read; periodic reconciliation covers lost wakes.
Each wait owns and closes its single pending feed read, and cancellation never
mutates the server-side run. See docs/protocols.md for recovery semantics.
"""

from __future__ import annotations

import asyncio
import math
from collections.abc import AsyncIterator
from contextlib import AbstractAsyncContextManager
from typing import Protocol

import httpx

from ._generated.models import RunEvent, RunMilestone, RunSnapshot
from ._protocol import CompletionMilestone
from ._retry import RetryPolicy, stream_refusal_retryable
from ._stream import StreamCheckpoint
from .errors import (
    ResponseContractError,
    ResultUnavailableError,
    RunFailedError,
    RunHistoryUnavailableError,
    RunPausedError,
    SwarmsAPIError,
    WaitTimeoutError,
    local_problem,
    stream_interrupted,
)

# Reconcile lost wakes on an open feed; milestones drive normal progress.
_RECONCILE_SECONDS = 30.0
# Gateway statuses a Core deploy produces on reads. Reads are safe to repeat, so a
# wait backs off and retries them until its own timeout; mutations never do.
_TRANSIENT_READ_STATUSES = frozenset({502, 503, 504})
_TRANSIENT_BACKOFF_MAX_SECONDS = 5.0


def transient_read_failure(error: SwarmsAPIError) -> bool:
    """A gateway failure on a read that a wait outlasts, unless the server says otherwise."""
    return (
        not isinstance(error, ResponseContractError)
        and error.status in _TRANSIENT_READ_STATUSES
        and error.problem.service != "synth-ai"
        and error.problem.retryable is not False
    )


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
        reading = False
        # Per channel, the latest gateway failure not yet followed by a success there
        # (a snapshot read, or a delivered event); raised instead of a bare timeout when
        # the outage outlasts the caller's deadline. Reads succeeding does not clear a
        # feed that keeps failing, and the reverse.
        outages: dict[str, SwarmsAPIError] = {}
        outage_counts = {"read": 0, "feed": 0}

        async def outlast(channel: str, error: SwarmsAPIError) -> None:
            outages[channel] = error
            delay = self._source.retry_policy.delay(None, min(outage_counts[channel], 6))
            outage_counts[channel] += 1
            await asyncio.sleep(max(poll_interval, min(delay or 0, _TRANSIENT_BACKOFF_MAX_SECONDS)))

        def recovered(channel: str) -> None:
            outages.pop(channel, None)
            outage_counts[channel] = 0

        async def read() -> RunSnapshot:
            # Marks snapshot failures so they are never mistaken for feed refusals.
            nonlocal snapshot, reading
            reading = True
            while True:
                try:
                    snapshot = await self._source.request(
                        "getRun", RunSnapshot, path={"run_id": run_id}
                    )
                    break
                except SwarmsAPIError as error:
                    if not transient_read_failure(error):
                        raise
                    await outlast("read", error)
            recovered("read")
            reading = False
            return snapshot

        def reached(current: RunSnapshot) -> bool:
            if current.reached(until):
                if raise_on_failure and current.state == "failed":
                    raise RunFailedError("Run " + run_id + " failed", snapshot=current)
                return True
            if current.state == "paused":
                raise RunPausedError("Run " + run_id + " is paused", snapshot=current)
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
                attempts = self._source.retry_policy.attempts
                cause: Exception | None = None
                attempt = 0
                while attempt < attempts:
                    reading = False
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
                                            recovered("feed")
                                            final = final or (
                                                isinstance(item, RunMilestone) and item.final
                                            )
                                        except StopAsyncIteration:
                                            ended = True
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
                                error = ResponseContractError(
                                    local_problem(
                                        "event_stream_ended_before_milestone", "listRunEvents"
                                    )
                                )
                                error.checkpoint = checkpoint
                                raise error
                    except SwarmsAPIError as error:
                        # A deploy's gateway failures on the feed reconnect from the
                        # checkpoint without spending an attempt; the timeout bounds them.
                        if not reading and transient_read_failure(error):
                            await outlast("feed", error)
                            continue
                        # A feed refusal the server marks retryable reconnects from the
                        # checkpoint; the next attempt rereads before any milestone.
                        if not reading and stream_refusal_retryable(error.problem):
                            cause = error
                        else:
                            raise
                    except httpx.TransportError as error:
                        cause = error
                    if attempt + 1 < attempts:
                        delay = self._source.retry_policy.delay(None, attempt)
                        await asyncio.sleep(max(poll_interval, delay or 0))
                    attempt += 1
                raise stream_interrupted(attempts, checkpoint, snapshot) from cause
        except TimeoutError as timed_out:
            outage = outages.get("read") or outages.get("feed")
            if outage is not None:
                raise outage from timed_out
            raise WaitTimeoutError(
                "Timed out waiting for " + run_id, snapshot=snapshot, run_id=run_id
            ) from timed_out


def validate_wait(timeout: float, poll_interval: float, until: str) -> None:
    if (
        not math.isfinite(timeout)
        or not math.isfinite(poll_interval)
        or timeout <= 0
        or poll_interval <= 0
    ):
        raise ValueError("timeout and poll_interval must be finite and positive")
    if until not in {"finished", "execution_closed", "result_available", "paused"}:
        raise ValueError("unknown completion milestone")
