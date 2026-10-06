"""Server-sent event contracts and strict decoder."""

from __future__ import annotations

import json
from collections.abc import AsyncIterable, AsyncIterator, Iterable, Iterator
from dataclasses import dataclass

from synth_ai.core.contracts.json_value import JsonValue


@dataclass(frozen=True, slots=True)
class SseEvent:
    event: str
    data: str
    event_id: str | None = None
    retry_milliseconds: int | None = None

    def json_data(self) -> JsonValue:
        return json.loads(self.data)


class SseDecoder:
    """Incremental SSE decoder shared by sync and async transports."""

    def __init__(self) -> None:
        self._first_line = True
        self._event_type = "message"
        self._event_id: str | None = None
        self._retry_milliseconds: int | None = None
        self._data_lines: list[str] = []

    def feed_line(self, raw_line: str) -> SseEvent | None:
        line = raw_line.rstrip("\r")
        if self._first_line:
            line = line.removeprefix("\ufeff")
            self._first_line = False
        if not line:
            return self.finish_event()
        if line.startswith(":"):
            return None
        field, separator, value = line.partition(":")
        if separator and value.startswith(" "):
            value = value[1:]
        if field == "event":
            self._event_type = value
        elif field == "data":
            self._data_lines.append(value)
        elif field == "id" and "\x00" not in value:
            self._event_id = value
        elif field == "retry" and value and value.isascii() and value.isdigit():
            self._retry_milliseconds = int(value)
        return None

    def finish_event(self) -> SseEvent | None:
        if not self._data_lines:
            self._event_type = "message"
            self._retry_milliseconds = None
            return None
        event = SseEvent(
            self._event_type or "message",
            "\n".join(self._data_lines),
            self._event_id,
            self._retry_milliseconds,
        )
        self._event_type = "message"
        self._retry_milliseconds = None
        self._data_lines = []
        return event


def iter_sse_events(lines: Iterable[str]) -> Iterator[SseEvent]:
    decoder = SseDecoder()
    for line in lines:
        event = decoder.feed_line(line)
        if event is not None:
            yield event


async def iter_sse_events_async(lines: AsyncIterable[str]) -> AsyncIterator[SseEvent]:
    decoder = SseDecoder()
    async for line in lines:
        event = decoder.feed_line(line)
        if event is not None:
            yield event


__all__ = ["SseDecoder", "SseEvent", "iter_sse_events", "iter_sse_events_async"]
