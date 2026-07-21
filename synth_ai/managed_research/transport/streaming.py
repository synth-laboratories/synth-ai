"""Small streaming and preview helpers."""

from __future__ import annotations

from dataclasses import dataclass

from synth_ai.core.http.streaming import SseEvent, iter_sse_events


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


__all__ = [
    "BinaryPayloadPreview",
    "SseEvent",
    "iter_sse_events",
    "preview_binary_payload",
]
