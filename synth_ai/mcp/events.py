"""MCP Events (SEP-3415 / io.modelcontextprotocol/events) for stdio MCP (SYN-6058).

Supports `events/list`, `events/poll`, and `events/stream` (`deliveryModes: ["poll", "push"]`)
backed by `AsyncSynthClient`. Webhook subscriptions (`events/subscribe` / `events/unsubscribe`)
are intentionally deferred to SYN-6061.
"""

from __future__ import annotations

from collections.abc import Mapping
from contextlib import aclosing
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from synth_ai._generated.models import RunMilestone
from synth_ai._stream import StreamCheckpoint
from synth_ai.errors import (
    NotFoundError,
    SwarmsAPIError,
    TransportFailureError,
)

from ._events_catalog import (
    CREDENTIAL_REVOKED,
    DEFAULT_POLL_LIMIT,
    DEFAULT_STREAM_TIMEOUT_S,
    EVENT_CATALOG,
    EVENT_SWARMS_RUN_MILESTONE,
    EXTENSION_CAPABILITIES,
    EXTENSION_METHODS,
    EXTENSION_URI,
    INTERNAL_ERROR,
    INVALID_PARAMS,
    KNOWN_EVENT_NAMES,
    MAX_POLL_LIMIT,
    MAX_POLL_SCAN_PAGES,
    MAX_STREAM_TIMEOUT_S,
    MILESTONE_LEVELS,
    MILESTONE_ORDER,
    POLL_INTERVAL_IDLE_MS,
    POLL_INTERVAL_MORE_MS,
    POLL_INTERVAL_SCAN_MS,
    RESOURCE_NOT_FOUND,
    UPSTREAM_PAGE_LIMIT,
    EventCursor,
    EventError,
    catalog_for_run_event,
    decode_cursor,
    encode_cursor,
    envelope_from_milestone,
    envelope_from_run_event,
    list_events_payload,
    matches_catalog,
    parse_input,
    parse_int_param,
    snapshot_milestone_level,
)

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from pydantic import JsonValue

    from synth_ai.aclient import AsyncSynthClient

__all__ = [
    "EVENT_CATALOG",
    "EXTENSION_CAPABILITIES",
    "EXTENSION_METHODS",
    "EXTENSION_URI",
    "EventCursor",
    "EventError",
    "decode_cursor",
    "encode_cursor",
    "list_events_payload",
    "poll_events",
    "stream_events",
]


def _translate_api_error(error: SwarmsAPIError, run_id: str) -> EventError:
    if error.problem.status == 401:
        return EventError(
            CREDENTIAL_REVOKED,
            "The credential is no longer authorized.",
            data={"code": "credential_revoked"},
        )
    if isinstance(error, NotFoundError) or error.problem.status in {403, 404}:
        return EventError(RESOURCE_NOT_FOUND, "Run not found.", data={"run_id": run_id})
    if error.problem.code == "invalid_cursor":
        return EventError(
            INVALID_PARAMS,
            "Invalid params: cursor rejected by run event store.",
            data={"code": error.problem.code},
        )
    return EventError(
        INTERNAL_ERROR,
        error.problem.detail or "Run events unavailable.",
        data={"code": error.problem.code},
    )


async def poll_events(
    client: AsyncSynthClient,
    params: Mapping[str, JsonValue] | None,
) -> dict[str, JsonValue]:
    if not isinstance(params, Mapping):
        raise EventError(INVALID_PARAMS, "Invalid params: expected object.")
    raw_event = params.get("event")
    if not isinstance(raw_event, str) or raw_event.strip() not in KNOWN_EVENT_NAMES:
        raise EventError(
            INVALID_PARAMS,
            "Invalid params: unknown event type.",
            data={"event": raw_event if isinstance(raw_event, str) else ""},
        )
    event_name = raw_event.strip()
    filt = parse_input((event_name,), params.get("input"))
    limit = parse_int_param(
        params.get("limit"),
        field="limit",
        default=DEFAULT_POLL_LIMIT,
        minimum=1,
        maximum=MAX_POLL_LIMIT,
    )
    raw_cursor = params.get("cursor")
    if raw_cursor is not None and not isinstance(raw_cursor, str):
        raise EventError(INVALID_PARAMS, "Invalid params: cursor must be a string.")
    cursor = decode_cursor(raw_cursor or "", expected_event=event_name, expected_run_id=filt.run_id)
    now_iso = datetime.now(UTC).isoformat().replace("+00:00", "Z")

    if event_name == EVENT_SWARMS_RUN_MILESTONE:
        try:
            snap = await client.swarms.get(filt.run_id)
        except SwarmsAPIError as error:
            raise _translate_api_error(error, filt.run_id) from error
        reached = snapshot_milestone_level(snap)
        ts = snap.closed_at or snap.started_at or now_iso
        matched_ms: list[JsonValue] = []
        next_level = cursor.milestone_level
        has_more = False
        for level in range(cursor.milestone_level + 1, reached + 1):
            ms_name = MILESTONE_ORDER[level - 1]
            if filt.milestone is not None and filt.milestone != ms_name:
                next_level = level
                continue
            if len(matched_ms) == limit:
                has_more = True
                break
            milestone = RunMilestone.model_validate(
                {
                    "milestone": ms_name,
                    "run_id": filt.run_id,
                    "state": str(snap.state),
                    "execution_closed": level >= 1,
                    "accounting_final": level >= 2,
                    "cleanup_complete": level >= 3,
                    "result_available": bool(snap.result_available),
                    "failure": snap.failure,
                    "final": level == len(MILESTONE_ORDER),
                }
            )
            matched_ms.append(envelope_from_milestone(milestone, ts))
            next_level = level
        return {
            "events": matched_ms,
            "cursor": encode_cursor(
                EventCursor(
                    event=EVENT_SWARMS_RUN_MILESTONE,
                    run_id=filt.run_id,
                    milestone_level=next_level,
                )
            ),
            "hasMore": has_more,
            "truncated": False,
            "nextPollMs": POLL_INTERVAL_MORE_MS if has_more else POLL_INTERVAL_IDLE_MS,
        }

    matched: list[JsonValue] = []
    seen_ids: set[str] = set()
    cur_upstream = cursor.upstream
    cur_skip = cursor.skip
    truncated = False
    has_more = False

    for page_num in range(MAX_POLL_SCAN_PAGES):
        try:
            page = await client.swarms.events(
                filt.run_id,
                cursor=cur_upstream or None,
                limit=UPSTREAM_PAGE_LIMIT,
            )
        except SwarmsAPIError as error:
            if error.problem.status == 410 and cur_upstream:
                truncated = True
                cur_upstream = ""
                cur_skip = 0
                continue
            raise _translate_api_error(error, filt.run_id) from error

        if getattr(page, "truncated", False) is True:
            truncated = True
        if cur_skip > len(page.items):
            truncated = True
            cur_skip = 0

        reached_limit = False
        next_skip = len(page.items)
        for idx in range(cur_skip, len(page.items)):
            item = page.items[idx]
            if not matches_catalog(event_name, item.kind, filt):
                continue
            if item.event_id in seen_ids:
                continue
            if len(matched) == limit:
                reached_limit = True
                next_skip = idx
                break
            seen_ids.add(item.event_id)
            matched.append(envelope_from_run_event(event_name, item, now_iso))

        if reached_limit:
            has_more = True
            cur_skip = next_skip
            break

        cur_skip = 0
        if page.has_more and page.next_cursor is not None and page.next_cursor != "":
            cur_upstream = page.next_cursor
            if len(matched) == limit or page_num == MAX_POLL_SCAN_PAGES - 1:
                has_more = True
                break
            continue
        if page.resume_cursor is not None and page.resume_cursor != "":
            cur_upstream = page.resume_cursor
        has_more = False
        break

    next_poll_ms = (
        POLL_INTERVAL_IDLE_MS
        if not has_more
        else (POLL_INTERVAL_SCAN_MS if not matched else POLL_INTERVAL_MORE_MS)
    )
    return {
        "events": matched,
        "cursor": encode_cursor(
            EventCursor(
                event=event_name,
                run_id=filt.run_id,
                upstream=cur_upstream,
                skip=cur_skip,
            )
        ),
        "hasMore": has_more,
        "truncated": truncated,
        "nextPollMs": next_poll_ms,
    }


def _parse_stream_events(params: Mapping[str, JsonValue]) -> tuple[str, ...]:
    raw_single = params.get("event")
    raw_multi = params.get("events")
    if raw_single is not None and raw_multi is not None:
        raise EventError(
            INVALID_PARAMS, "Invalid params: specify either event or events, not both."
        )
    if raw_multi is not None:
        if not isinstance(raw_multi, list) or not raw_multi:
            raise EventError(
                INVALID_PARAMS, "Invalid params: events must be a non-empty array of event names."
            )
        ordered: list[str] = []
        seen: set[str] = set()
        for item in raw_multi:
            if not isinstance(item, str) or item.strip() not in KNOWN_EVENT_NAMES:
                raise EventError(
                    INVALID_PARAMS,
                    "Invalid params: unknown event type.",
                    data={"event": item if isinstance(item, str) else ""},
                )
            cleaned = item.strip()
            if cleaned not in seen:
                seen.add(cleaned)
                ordered.append(cleaned)
        return tuple(ordered)
    if not isinstance(raw_single, str) or raw_single.strip() not in KNOWN_EVENT_NAMES:
        raise EventError(
            INVALID_PARAMS,
            "Invalid params: unknown event type.",
            data={"event": raw_single if isinstance(raw_single, str) else ""},
        )
    return (raw_single.strip(),)


async def stream_events(
    client: AsyncSynthClient,
    params: Mapping[str, JsonValue] | None,
    emit_notification: Callable[[dict[str, JsonValue]], Awaitable[None]],
) -> dict[str, JsonValue]:
    if not isinstance(params, Mapping):
        raise EventError(INVALID_PARAMS, "Invalid params: expected object.")
    requested = _parse_stream_events(params)
    filt = parse_input(requested, params.get("input"))
    timeout_seconds = parse_int_param(
        params.get("timeoutSeconds"),
        field="timeoutSeconds",
        default=DEFAULT_STREAM_TIMEOUT_S,
        minimum=1,
        maximum=MAX_STREAM_TIMEOUT_S,
    )
    raw_cursor = params.get("cursor")
    if raw_cursor is not None and not isinstance(raw_cursor, str):
        raise EventError(INVALID_PARAMS, "Invalid params: cursor must be a string.")
    cursor_event = requested[0] if len(requested) == 1 else ""
    parsed_cursor = decode_cursor(
        raw_cursor or "", expected_event=cursor_event, expected_run_id=filt.run_id
    )
    want_milestones = EVENT_SWARMS_RUN_MILESTONE in requested
    checkpoint = StreamCheckpoint(
        run_id=filt.run_id,
        last_event_id=parsed_cursor.stream_resume_id or None,
    )
    delivered = 0
    status = "completed"
    now_iso = datetime.now(UTC).isoformat().replace("+00:00", "Z")

    try:
        async with aclosing(
            client.swarms.watch(
                filt.run_id,
                checkpoint=checkpoint,
                timeout=float(timeout_seconds),
            )
        ) as stream:
            async for item in stream:
                if isinstance(item, RunMilestone):
                    ms_name = str(item.milestone)
                    if want_milestones and (filt.milestone is None or filt.milestone == ms_name):
                        env = envelope_from_milestone(item, now_iso)
                        env["cursor"] = encode_cursor(
                            EventCursor(
                                event=cursor_event,
                                run_id=filt.run_id,
                                milestone_level=MILESTONE_LEVELS.get(ms_name, 0),
                                stream_resume_id=checkpoint.last_event_id or "",
                            )
                        )
                        await emit_notification(env)
                        delivered += 1
                    continue

                matched_catalog = catalog_for_run_event(requested, item.kind, filt)
                if matched_catalog is None:
                    continue
                env = envelope_from_run_event(matched_catalog, item, now_iso)
                env["cursor"] = encode_cursor(
                    EventCursor(
                        event=cursor_event,
                        run_id=filt.run_id,
                        stream_resume_id=checkpoint.last_event_id or "",
                    )
                )
                await emit_notification(env)
                delivered += 1
    except TransportFailureError as error:
        if error.problem.code == "event_stream_timeout":
            status = "timeout"
        else:
            raise EventError(
                INTERNAL_ERROR,
                error.problem.detail or "Run event stream failed.",
                data={"code": error.problem.code},
            ) from error
    except SwarmsAPIError as error:
        raise _translate_api_error(error, filt.run_id) from error

    return {
        "status": status,
        "delivered": delivered,
        "cursor": encode_cursor(
            EventCursor(
                event=cursor_event,
                run_id=filt.run_id,
                stream_resume_id=checkpoint.last_event_id or "",
            )
        ),
        "run_id": filt.run_id,
    }
