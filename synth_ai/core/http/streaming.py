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
        self._event_type = "message"
        self._event_id: str | None = None
        self._retry_milliseconds: int | None = None
        self._data_lines: list[str] = []

    def feed_line(self, raw_line: str) -> SseEvent | None:
        line = raw_line.rstrip("\r")
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
        elif field == "id":
            self._event_id = value
        elif field == "retry":
            try:
                self._retry_milliseconds = int(value)
            except ValueError as exc:
                raise ValueError(f"invalid SSE retry value {value!r}") from exc
        return None

    def finish_event(self) -> SseEvent | None:
        if not self._data_lines:
            self._event_type = "message"
            self._retry_milliseconds = None
            return None
        event = SseEvent(
            self._event_type,
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
    event = decoder.finish_event()
    if event is not None:
        yield event


async def iter_sse_events_async(lines: AsyncIterable[str]) -> AsyncIterator[SseEvent]:
    decoder = SseDecoder()
    async for line in lines:
        event = decoder.feed_line(line)
        if event is not None:
            yield event
    event = decoder.finish_event()
    if event is not None:
        yield event


__all__ = ["SseDecoder", "SseEvent", "iter_sse_events", "iter_sse_events_async"]
