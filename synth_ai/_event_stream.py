"""Decode one durable feed connection; the transport owns reconnects and HTTP lifetime.

Checkpoints describe receipt, not durable application processing. Updating the
checkpoint before yield makes cancellation at generator suspension explicit.
"""

from __future__ import annotations

import asyncio
import math
from contextlib import aclosing
from typing import TYPE_CHECKING, cast

from ._generated.models import (
    ManagerMessageState,
    PostedMessageKind,
    Problem,
    RunEvent,
    RunMessage,
    RunMessageEventPayload,
    RunMessagePage,
    RunMilestone,
)
from ._limits import CHECKPOINT_MAX_EVENTS, CHECKPOINT_MAX_ID_LENGTH
from ._pagination import paginate
from ._stream import StreamCheckpoint, bounded_lines, frames
from .errors import (
    ConflictError,
    MessageReplyTimeoutError,
    MessageReplyUnavailableError,
    MessagingUnavailableError,
    ResponseContractError,
    ResultUnavailableError,
    local_problem,
    raise_problem,
)

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator, Awaitable, Callable
    from typing import NoReturn

    import httpx

    from ._generated.models import RunSnapshot
    from ._transport import Transport

MESSAGE_EVENT_KINDS = frozenset(
    {
        "run.message.accepted",
        "run.message.delivered",
        "run.message.disposed",
        "run.message.expired",
        "run.message.posted",
    }
)


def require_available_messages(page: RunMessagePage) -> RunMessagePage:
    """Fail with MessagingUnavailableError when Core marks message history unavailable."""
    if page.availability.kind == "unavailable":
        raise MessagingUnavailableError(
            local_problem("messaging_unavailable", "listRunMessages", status=503)
        )
    return page


def extract_message_event_id(event: RunEvent) -> str:
    """Validate and return the message_id carried by a run.message.* event."""
    if event.payload is None:
        raise ResponseContractError(
            local_problem("run_message_event_invalid", "listRunEvents", status=502)
        )
    return cast("RunMessageEventPayload", event.payload).message_id


def is_correlated_reply(message: RunMessage, message_id: str) -> bool:
    """Return True when `message` is a non-customer posted reply to `message_id`."""
    return (
        message.reply_to_message_id == message_id
        and message.kind == PostedMessageKind.REPLY
        and message.state.kind == ManagerMessageState.POSTED
        and message.author != "customer"
    )


async def iter_available_messages(
    fetch_page: Callable[[str | None], Awaitable[RunMessagePage]],
    *,
    cursor: str | None = None,
) -> AsyncGenerator[RunMessage, None]:
    """Yield message pages while failing closed if Core reports messaging unavailable."""

    async def _page(position: str | None) -> RunMessagePage:
        return require_available_messages(await fetch_page(position))

    async for item in paginate(_page, cursor=cursor):
        yield item


async def follow_run_messages(
    transport: Transport,
    run_id: str,
    iter_history: Callable[[], AsyncGenerator[RunMessage, None]],
    fetch_message: Callable[[str], Awaitable[RunMessage]],
) -> AsyncGenerator[RunMessage, None]:
    """Yield historical messages followed by live message events from one feed."""
    checkpoint = StreamCheckpoint(run_id=run_id)
    async with transport.subscribe_events(run_id, checkpoint) as events:
        async for item in iter_history():
            yield item
        async for event in events:
            if isinstance(event, RunMilestone):
                if event.final:
                    return
                continue
            if event.kind not in MESSAGE_EVENT_KINDS:
                continue
            yield await fetch_message(extract_message_event_id(event))


async def wait_for_correlated_reply(
    stream: AsyncGenerator[RunMessage, None],
    run_id: str,
    message_id: str,
    timeout: float,
) -> RunMessage:
    """Wait for a non-customer posted reply correlated to `message_id`."""
    if not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("timeout must be finite and positive")
    try:
        async with asyncio.timeout(timeout):
            async with aclosing(stream) as messages:
                async for message in messages:
                    if is_correlated_reply(message, message_id):
                        return message
    except TimeoutError:
        raise MessageReplyTimeoutError(run_id=run_id, message_id=message_id) from None
    raise MessageReplyUnavailableError(run_id=run_id, message_id=message_id)


async def reraise_result_unavailable(
    run_id: str,
    refusal: ConflictError,
    fetch_snapshot: Callable[[str], Awaitable[RunSnapshot]],
) -> NoReturn:
    """Convert `result_unavailable` 409 conflicts into `ResultUnavailableError` with snapshot."""
    if refusal.code != "result_unavailable":
        raise refusal
    snapshot = await fetch_snapshot(run_id)
    raise ResultUnavailableError(
        "Run " + run_id + " has not finished; wait() for it first", snapshot=snapshot
    ) from refusal


async def decode_events(
    response: httpx.Response, run_id: str, state: StreamCheckpoint
) -> AsyncGenerator[RunEvent | RunMilestone, None]:
    validate_stream_response(response)
    async for frame in frames(bounded_lines(response.aiter_bytes())):
        if frame.event == "error":
            raise_problem(
                Problem.model_validate_json(frame.data),
                request_id=response.headers.get("x-request-id"),
            )
        if frame.event == "milestone":
            milestone = RunMilestone.model_validate_json(frame.data)
            if milestone.run_id != run_id:
                raise ValueError("milestone belongs to another run")
            if frame.id is None or not frame.id or len(frame.id) > CHECKPOINT_MAX_ID_LENGTH:
                raise ValueError("milestone without a resumable cursor")
            duplicate = frame.id == state.last_event_id
            state.last_event_id = frame.id
            state.pending_event_ids.clear()
            if not duplicate:
                yield milestone
            continue
        if frame.event not in {"message", "event"}:
            raise ValueError("unknown event frame type")
        item = RunEvent.model_validate_json(frame.data) if frame.data else None
        if item is not None and item.run_id != run_id:
            raise ValueError("event belongs to another run")
        # Replayed committed frames and partial expanded sequences are both
        # suppressed. Cursor values are opaque: never order them numerically.
        if frame.id is not None and len(frame.id) > CHECKPOINT_MAX_ID_LENGTH:
            raise ValueError("event cursor exceeds checkpoint bound")
        duplicate = frame.id is not None and frame.id == state.last_event_id
        if item is not None:
            duplicate = duplicate or item.event_id in state.pending_event_ids
            if len(item.event_id) > CHECKPOINT_MAX_ID_LENGTH:
                raise ValueError("event identity exceeds checkpoint bound")
            if (
                item.event_id not in state.pending_event_ids
                and len(state.pending_event_ids) == CHECKPOINT_MAX_EVENTS
            ):
                raise ValueError("expanded event sequence exceeds checkpoint bound")
            state.pending_event_ids.add(item.event_id)
        if frame.id is not None:
            state.last_event_id = frame.id or None
            state.pending_event_ids.clear()
        if item is not None and not duplicate:
            yield item


def validate_stream_response(response: httpx.Response) -> None:
    if response.headers.get("content-type", "").split(";")[0].strip() != "text/event-stream":
        raise ValueError("expected event stream")
    if response.headers.get("content-encoding", "identity").lower() != "identity":
        raise ValueError("compressed event stream refused")
