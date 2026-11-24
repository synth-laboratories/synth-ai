"""The McpOutputContent representation of output bytes (contract 1.31.0)."""

from __future__ import annotations

import base64
import contextlib
import hashlib

from ._generated.models import McpOutputContent

__all__ = ["inline_content", "textual_media"]

TEXTUAL_MEDIA = frozenset(
    {
        "application/json",
        "application/x-ndjson",
        "application/jsonl",
        "application/xml",
        "application/yaml",
        "application/x-yaml",
        "application/toml",
        "application/javascript",
        "application/x-sh",
        "application/sql",
    }
)


def textual_media(content_type: str) -> bool:
    """Media types whose bytes are meant to be read as text (Core's textualMedia)."""
    media = content_type.split(";")[0].strip().lower()
    return media.startswith("text/") or media.endswith(("+json", "+xml")) or media in TEXTUAL_MEDIA


def inline_content(content: bytes, media_type: str) -> McpOutputContent:
    """Text only for a textual media type with valid UTF-8; base64 otherwise."""
    encoding, text = "base64", base64.b64encode(content).decode("ascii")
    if textual_media(media_type):
        with contextlib.suppress(UnicodeDecodeError):
            encoding, text = "utf8", content.decode("utf-8", errors="strict")
    return McpOutputContent.model_validate(
        {
            "media_type": media_type,
            "byte_size": len(content),
            "sha256": hashlib.sha256(content).hexdigest(),
            "encoding": encoding,
            "content": text,
        }
    )
