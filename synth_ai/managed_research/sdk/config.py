"""Client configuration helpers for the Managed Research SDK."""

from __future__ import annotations

import os

from synth_ai.managed_research.auth import BACKEND_URL_BASE, get_api_key, normalize_backend_base

DEFAULT_TIMEOUT_SECONDS = 30.0
DEFAULT_WORKSPACE_ARCHIVE_DOWNLOAD_TIMEOUT_SECONDS = 600.0
DEFAULT_MISC_PROJECT_ALIAS = "00000000-0000-0000-0000-000000000000"
OPENAI_TRANSPORT_MODE_BACKEND_BFF = "backend_bff"
OPENAI_TRANSPORT_MODE_DIRECT_HP = "direct_hp"
OPENAI_TRANSPORT_MODE_AUTO = "auto"
OPENAI_VALID_TRANSPORT_MODES = {
    OPENAI_TRANSPORT_MODE_BACKEND_BFF,
    OPENAI_TRANSPORT_MODE_DIRECT_HP,
    OPENAI_TRANSPORT_MODE_AUTO,
}


def resolve_backend_base(backend_base: str | None) -> str:
    candidate = str(backend_base or os.getenv("SYNTH_BACKEND_URL") or BACKEND_URL_BASE).strip()
    if not candidate:
        candidate = "https://api.usesynth.ai"
    return normalize_backend_base(candidate).rstrip("/")


def resolve_api_key(api_key: str | None) -> str:
    if api_key and api_key.strip():
        return api_key.strip()
    resolved = get_api_key("SYNTH_API_KEY", required=True)
    if not resolved:
        raise ValueError("api_key is required (provide api_key or set SYNTH_API_KEY)")
    return resolved


def auth_headers(api_key: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}


def optional_str(value: str | None) -> str | None:
    normalized = str(value or "").strip()
    return normalized or None


def resolve_openai_transport_mode(value: str | None) -> str:
    normalized = str(value or OPENAI_TRANSPORT_MODE_AUTO).strip().lower()
    if normalized not in OPENAI_VALID_TRANSPORT_MODES:
        allowed = ", ".join(sorted(OPENAI_VALID_TRANSPORT_MODES))
        raise ValueError(f"openai_transport_mode must be one of: {allowed}")
    return normalized


__all__ = [
    "DEFAULT_MISC_PROJECT_ALIAS",
    "DEFAULT_TIMEOUT_SECONDS",
    "DEFAULT_WORKSPACE_ARCHIVE_DOWNLOAD_TIMEOUT_SECONDS",
    "OPENAI_TRANSPORT_MODE_AUTO",
    "OPENAI_TRANSPORT_MODE_BACKEND_BFF",
    "OPENAI_TRANSPORT_MODE_DIRECT_HP",
    "OPENAI_VALID_TRANSPORT_MODES",
    "auth_headers",
    "optional_str",
    "resolve_api_key",
    "resolve_backend_base",
    "resolve_openai_transport_mode",
]
