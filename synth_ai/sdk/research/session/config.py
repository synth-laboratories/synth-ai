"""Client configuration helpers for the Managed Research SDK."""

from __future__ import annotations

import os

from synth_ai.core.auth.credentials import resolve_api_credential
from synth_ai.core.utils.urls import resolve_synth_backend_url

DEFAULT_TIMEOUT_SECONDS = 30.0
DEFAULT_WORKSPACE_ARCHIVE_DOWNLOAD_TIMEOUT_SECONDS = 600.0
DEFAULT_MISC_PROJECT_ALIAS = "00000000-0000-0000-0000-000000000000"

REQUIRE_EXPLICIT_BACKEND_ENV = "SYNTH_REQUIRE_EXPLICIT_BACKEND"


def _require_explicit_backend() -> bool:
    """Opt-in strictness for internal tooling.

    Customer code wants the prod default: pip install, call, reach production.
    Internal tooling (evals, dock, operator scripts) wants the opposite — an
    unnamed backend should be an error, not a silent prod call. Such callers
    set SYNTH_REQUIRE_EXPLICIT_BACKEND=1 and pass backend_base explicitly.
    """
    return str(os.getenv(REQUIRE_EXPLICIT_BACKEND_ENV) or "").strip().lower() in {
        "1",
        "true",
        "yes",
    }


def resolve_backend_base(backend_base: str | None) -> str:
    """Use the shared construction-time backend configuration authority."""
    return resolve_synth_backend_url(backend_base)


def resolve_api_key(api_key: str | None) -> str:
    return resolve_api_credential(api_key).value


def auth_headers(api_key: str) -> dict[str, str]:
    return resolve_api_credential(api_key).authorization_headers()


def optional_str(value: str | None) -> str | None:
    normalized = str(value or "").strip()
    return normalized or None


__all__ = [
    "DEFAULT_MISC_PROJECT_ALIAS",
    "DEFAULT_TIMEOUT_SECONDS",
    "DEFAULT_WORKSPACE_ARCHIVE_DOWNLOAD_TIMEOUT_SECONDS",
    "auth_headers",
    "optional_str",
    "resolve_api_key",
    "resolve_backend_base",
]
