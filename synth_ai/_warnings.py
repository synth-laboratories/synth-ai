"""Deduplicated server deprecation, sunset, and client-upgrade warnings."""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import httpx

__all__ = ["observe_version_warnings", "reset_warning_cache"]

_MAX_SEEN_KEYS = 256
_seen_warnings: set[tuple[str, ...]] = set()


def reset_warning_cache() -> None:
    """Clear deduplication state (used by tests)."""
    _seen_warnings.clear()


def _emit_once(key: tuple[str, ...], message: str) -> None:
    if key in _seen_warnings:
        return
    if len(_seen_warnings) >= _MAX_SEEN_KEYS:
        _seen_warnings.clear()
    _seen_warnings.add(key)
    warnings.warn(message, DeprecationWarning, stacklevel=4)


def observe_version_warnings(response: httpx.Response, operation_id: str) -> None:
    """Surface RFC 9745/8594 deprecation/sunset and client grace-window advisories.

    Warnings are emitted via Python's warnings module (never stdout), so CLI
    --json output and structured SDK callers remain unaffected.
    """
    deprecation = (response.headers.get("Deprecation") or "").strip()
    sunset = (response.headers.get("Sunset") or "").strip()
    link = (response.headers.get("Link") or "").strip()
    if deprecation or sunset:
        parts = [f"Synth API operation {operation_id!r} is deprecated"]
        if deprecation:
            parts.append(f"since {deprecation}")
        if sunset:
            parts.append(f"(sunset: {sunset})")
        if link:
            parts.append(f"see {link}")
        _emit_once(("op", operation_id, deprecation, sunset, link), " ".join(parts))

    recommended = (response.headers.get("X-Synth-Client-Upgrade-Recommended") or "").strip()
    client_dep = (response.headers.get("X-Synth-Client-Deprecation") or "").strip()
    if recommended or client_dep:
        detail = (
            client_dep
            if client_dep
            else f"synth-ai upgrade recommended to {recommended} or newer: pip install -U synth-ai"
        )
        _emit_once(("client", recommended, client_dep), detail)
