"""Decode one durable feed connection; the transport owns reconnects and HTTP lifetime.

Checkpoints describe receipt, not durable application processing. Updating the
checkpoint before yield makes cancellation at generator suspension explicit.
"""

from __future__ import annotations

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
from ._stream import StreamCheckpoint, bounded_lines, frames
from .errors import (
    MessagingUnavailableError,
    ResponseContractError,
    local_problem,
    raise_problem,
)

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator

    import httpx

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
