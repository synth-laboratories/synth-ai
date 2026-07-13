"""Small streaming and preview helpers."""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class BinaryPayloadPreview:
    media_type: str
    size_bytes: int
    preview_text: str


def preview_binary_payload(
    payload: bytes,
    *,
    media_type: str = "application/octet-stream",
    preview_bytes: int = 128,
) -> BinaryPayloadPreview:
    preview = payload[:preview_bytes].decode("utf-8", errors="replace")
    return BinaryPayloadPreview(
        media_type=media_type,
        size_bytes=len(payload),
        preview_text=preview,
    )


@dataclass(frozen=True, slots=True)
class SseEvent:
    event: str
    data: str
    event_id: str | None = None
    retry: int | None = None

    def json_data(self) -> Any:
        return json.loads(self.data)


def iter_sse_events(lines: Iterable[str]) -> Iterator[SseEvent]:
    event_type = "message"
    event_id: str | None = None
    retry: int | None = None
    data_lines: list[str] = []

    def flush() -> SseEvent | None:
        nonlocal event_type, event_id, retry, data_lines
        if not data_lines:
            event_type = "message"
            retry = None
            return None
        event = SseEvent(
            event=event_type or "message",
            data="\n".join(data_lines),
            event_id=event_id,
            retry=retry,
        )
        event_type = "message"
        retry = None
        data_lines = []
        return event

    for raw_line in lines:
        line = raw_line.rstrip("\r")
        if line == "":
            event = flush()
            if event is not None:
                yield event
            continue
        if line.startswith(":"):
            continue
        field, separator, value = line.partition(":")
        if separator and value.startswith(" "):
            value = value[1:]
        if field == "event":
            event_type = value
        elif field == "data":
            data_lines.append(value)
        elif field == "id":
            event_id = value
        elif field == "retry":
            try:
                retry = int(value)
            except ValueError:
                retry = None
    event = flush()
    if event is not None:
        yield event


__all__ = [
    "BinaryPayloadPreview",
    "SseEvent",
    "iter_sse_events",
    "preview_binary_payload",
]
