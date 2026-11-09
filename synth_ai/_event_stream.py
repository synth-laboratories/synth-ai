"""Decode one durable feed connection; the transport owns reconnects and HTTP lifetime.

Checkpoints describe receipt, not durable application processing. Updating the
checkpoint before yield makes cancellation at generator suspension explicit.
"""

from collections.abc import AsyncGenerator

import httpx

from ._generated.models import Problem, RunEvent
from ._limits import CHECKPOINT_MAX_EVENTS, CHECKPOINT_MAX_ID_LENGTH
from ._stream import StreamCheckpoint, bounded_lines, frames
from .errors import raise_problem


async def decode_events(
    response: httpx.Response, run_id: str, state: StreamCheckpoint
) -> AsyncGenerator[RunEvent, None]:
    validate_stream_response(response)
    async for frame in frames(bounded_lines(response.aiter_bytes())):
        if frame.event == "error":
            raise_problem(
                Problem.model_validate_json(frame.data),
                request_id=response.headers.get("x-request-id"),
            )
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
