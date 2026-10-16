"""Valid SSE behavior paired with protocol laws; no live services.

See https://html.spec.whatwg.org/multipage/server-sent-events.html#parsing-an-event-stream
"""

import asyncio

import pytest
from streaming_probe import collect_events


@pytest.mark.parametrize("mode", ["sync", "async"])
@pytest.mark.parametrize("line_ending", ["\n", "\r", "\r\n"])
def test_valid_line_endings_multiline_data_and_cursor_persist(mode, line_ending):
    lines = [
        ": keepalive",
        "event: progress",
        "id: stable",
        "retry: 1200",
        "data: first",
        "data: second",
        "",
        "data: third",
        "",
        "",
    ]
    events = asyncio.run(
        collect_events(mode, line_ending.join(lines).encode(), "text/event-stream; charset=utf-8")
    )
    assert len(events) == 2
    assert events[0].event == "progress"
    assert events[0].data == "first\nsecond"
    assert events[0].retry_milliseconds == 1200
    assert events[0].event_id == "stable"
    assert events[1].event == "message"
    assert events[1].event_id == "stable"
    assert events[1].data == "third"


@pytest.mark.parametrize("mode", ["sync", "async"])
def test_empty_data_event_and_explicit_cursor_reset(mode):
    events = asyncio.run(collect_events(mode, b"id: prior\ndata: first\n\nid:\ndata:\n\n"))
    assert len(events) == 2
    assert events[0].event_id == "prior"
    assert events[1].event_id == ""
    assert events[1].data == ""


@pytest.mark.parametrize("mode", ["sync", "async"])
def test_comment_only_and_unknown_field_blocks_do_not_emit(mode):
    assert asyncio.run(collect_events(mode, b": keepalive\nignored: value\n\n")) == []
