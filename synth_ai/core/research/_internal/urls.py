"""Backend URL helpers."""

import os
from urllib.parse import urlparse, urlunparse

BACKEND_URL_BASE = (os.getenv("SYNTH_BACKEND_URL") or "https://api.usesynth.ai").strip()


def _strip_terminal_segment(path: str, segment: str) -> str:
    trimmed = path.rstrip("/")
    if trimmed.endswith(segment):
        return trimmed[: -len(segment)].rstrip("/")
    return trimmed


def normalize_backend_base(url: str) -> str:
    parsed = urlparse(str(url or "").strip())
    path = _strip_terminal_segment(parsed.path, "/v1")
    path = _strip_terminal_segment(path, "/api")
    normalized = parsed._replace(path=path.rstrip("/"), query="", fragment="")
    return urlunparse(normalized)


__all__ = ["BACKEND_URL_BASE", "normalize_backend_base"]
