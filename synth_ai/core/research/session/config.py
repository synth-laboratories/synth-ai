"""Client configuration helpers for the Managed Research SDK."""

from __future__ import annotations

import os

from synth_ai.core.auth.credentials import resolve_api_credential
from synth_ai.core.utils.urls import (
    REQUIRE_EXPLICIT_BACKEND_ENV,
    default_backend_base,
    normalize_backend_base,
)

DEFAULT_TIMEOUT_SECONDS = 30.0
DEFAULT_WORKSPACE_ARCHIVE_DOWNLOAD_TIMEOUT_SECONDS = 600.0
DEFAULT_MISC_PROJECT_ALIAS = "00000000-0000-0000-0000-000000000000"
OPENAI_TRANSPORT_MODE_BACKEND_BFF = "backend_bff"
OPENAI_TRANSPORT_MODE_DIRECT_HP = "direct_hp"
OPENAI_TRANSPORT_MODE_AUTO = "auto"
OPENAI_VALID_TRANSPORT_MODES = {
    OPENAI_TRANSPORT_MODE_DIRECT_HP,
}


def resolve_backend_base(backend_base: str | None) -> str:
    """Resolve the backend base URL from, in order: explicit argument,
    SYNTH_BACKEND_URL, then the package default (prod unless configured
    otherwise via SYNTH_BACKEND_URL_OVERRIDE / environment detection).

    The package default is deliberate for customers. It is a hazard for
    internal callers, who should set SYNTH_REQUIRE_EXPLICIT_BACKEND=1 so that
    reaching this fallback raises instead of silently targeting production.
    """
    explicit = str(backend_base or "").strip()
    if explicit:
        return normalize_backend_base(explicit).rstrip("/")

    from_env = str(os.getenv("SYNTH_BACKEND_URL") or "").strip()
    if from_env:
        return normalize_backend_base(from_env).rstrip("/")

    # Single default path, and the only one: `default_backend_base` raises under
    # SYNTH_REQUIRE_EXPLICIT_BACKEND rather than silently targeting production.
    return normalize_backend_base(default_backend_base()).rstrip("/")


def resolve_api_key(api_key: str | None) -> str:
    return resolve_api_credential(api_key).value


def auth_headers(api_key: str) -> dict[str, str]:
    return resolve_api_credential(api_key).authorization_headers()


def optional_str(value: str | None) -> str | None:
    normalized = str(value or "").strip()
    return normalized or None


def resolve_openai_transport_mode(value: str | None) -> str:
    normalized = str(value or "").strip().lower()
    if normalized in {
        OPENAI_TRANSPORT_MODE_AUTO,
        OPENAI_TRANSPORT_MODE_BACKEND_BFF,
    }:
        raise ValueError(
            f"openai_transport_mode={normalized!r} was retired with the backend "
            "managed-agents proxy; direct_hp requires an explicit Horizons Private "
            "base URL and credential"
        )
    if normalized not in OPENAI_VALID_TRANSPORT_MODES:
        raise ValueError(
            "openai_transport_mode must be explicitly set to direct_hp with an explicit "
            "Horizons Private base URL and credential"
        )
    return normalized


__all__ = [
    "DEFAULT_MISC_PROJECT_ALIAS",
    "DEFAULT_TIMEOUT_SECONDS",
    "DEFAULT_WORKSPACE_ARCHIVE_DOWNLOAD_TIMEOUT_SECONDS",
    "REQUIRE_EXPLICIT_BACKEND_ENV",
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
