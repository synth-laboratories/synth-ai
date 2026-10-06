"""SSE protocol acceptance: EX-08–EX-13.

See https://html.spec.whatwg.org/multipage/server-sent-events.html#parsing-an-event-stream
and SDK testing/specifications/sdk/core_research_migration.md.
"""

import asyncio

import pytest
from streaming_probe import collect_events
from synth_ai.core.errors import ContractMismatchError


@pytest.mark.parametrize("mode", ["sync", "async"])
@pytest.mark.parametrize("value", ["nonsense", "-1", "+12", "1_000", "١٢"])
def test_invalid_retry_field_is_ignored__EX08(mode, value):
    body = f"retry: {value}\ndata: retained\n\n".encode()
    try:
        events = asyncio.run(collect_events(mode, body))
    except ValueError as failure:
        pytest.fail(f"EX-08: invalid retry field terminates valid event stream: {failure}")
    assert [event.data for event in events] == ["retained"], "EX-08: valid data lost"
    assert events[0].retry_milliseconds is None, "EX-08: non-ASCII-digit retry accepted"


@pytest.mark.parametrize("mode", ["sync", "async"])
def test_nul_event_id_preserves_previous_id__EX09(mode):
    body = b"id: retained\ndata: first\n\nid: invalid\x00id\ndata: second\n\n"
    events = asyncio.run(collect_events(mode, body))
    assert len(events) == 2
    assert events[0].event_id == "retained"
    assert events[1].event_id == "retained", "EX-09: NUL-containing id overwrites replay cursor"


@pytest.mark.parametrize("mode", ["sync", "async"])
def test_empty_event_type_uses_message__EX10(mode):
    events = asyncio.run(collect_events(mode, b"event:\ndata: retained\n\n"))
    assert len(events) == 1
    assert events[0].event == "message", "EX-10: empty event type loses default message routing"


@pytest.mark.parametrize("mode", ["sync", "async"])
@pytest.mark.parametrize("body", [b"data: unfinished", b"data: unfinished\n"])
def test_eof_does_not_dispatch_unterminated_event__EX11(mode, body):
    events = asyncio.run(collect_events(mode, body))
    assert events == [], "EX-11: disconnect dispatches event lacking terminating blank line"


@pytest.mark.parametrize("mode", ["sync", "async"])
def test_leading_bom_does_not_drop_first_event__EX12(mode):
    events = asyncio.run(collect_events(mode, "\ufeffdata: retained\n\n".encode()))
    assert len(events) == 1, "EX-12: leading UTF-8 BOM drops first event"
    assert events[0].data == "retained", "EX-12: first event content changed"


@pytest.mark.parametrize("mode", ["sync", "async"])
@pytest.mark.parametrize("content_type", [None, "application/json", "text/html"])
def test_wrong_stream_media_type_is_typed_failure__EX13(mode, content_type):
    try:
        asyncio.run(collect_events(mode, b"data: retained\n\n", content_type))
    except ContractMismatchError as refusal:
        assert refusal.operation == "offline_events", (
            "EX-13: failed stream loses operation identity"
        )
        assert refusal.retryable is False, "EX-13: wrong protocol is not transient"
        return
    pytest.fail("EX-13: non-event-stream response accepted as SSE data")
