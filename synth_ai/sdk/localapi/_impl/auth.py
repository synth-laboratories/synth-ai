"""Authentication helpers shared by Task Apps."""

from __future__ import annotations

import os
from collections.abc import Iterable
from contextlib import suppress
from typing import Any

import synth_ai_py

from .errors import http_exception

_API_KEY_ENV = "ENVIRONMENT_API_KEY"
_API_KEY_HEADER = "x-api-key"
_API_KEYS_HEADER = "x-api-keys"
_AUTH_HEADER = "authorization"


def normalize_environment_api_key() -> str | None:
    """Ensure `ENVIRONMENT_API_KEY` is populated from dev fallbacks.

    Returns the resolved key (if any) so callers can branch on configuration.
    """

    fn = getattr(synth_ai_py, "localapi_normalize_environment_api_key", None)
    if callable(fn):
        return fn()
    # Fallback: promote DEV_ENVIRONMENT_API_KEY to ENVIRONMENT_API_KEY if needed.
    key = os.environ.get(_API_KEY_ENV, "").strip()
    if key:
        return key
    dev_key = os.environ.get("DEV_ENVIRONMENT_API_KEY", "").strip()
    if dev_key:
        os.environ[_API_KEY_ENV] = dev_key
        return dev_key
    return None


def allowed_environment_api_keys() -> set[str]:
    """Return the set of valid environment API keys for this Task App.

    Includes:
    - The primary ENVIRONMENT_API_KEY (normalized from dev fallbacks if needed)
    - Any comma-separated aliases from ENVIRONMENT_API_KEY_ALIASES
    """

    fn = getattr(synth_ai_py, "localapi_allowed_environment_api_keys", None)
    if callable(fn):
        return set(fn())
    keys: set[str] = set()
    primary = os.environ.get(_API_KEY_ENV, "").strip()
    if primary:
        keys.add(primary)
    aliases_raw = os.environ.get("ENVIRONMENT_API_KEY_ALIASES", "")
    if aliases_raw:
        for part in aliases_raw.split(","):
            candidate = part.strip()
            if candidate:
                keys.add(candidate)
    return keys


def _header_values(request: Any, header: str) -> Iterable[str]:
    header_lower = header.lower()
    if request is None:
        return []
    headers = getattr(request, "headers", None)
    if headers:
        raw = headers.get(header) or headers.get(header_lower)
        if raw is not None:
            return [raw]
    if isinstance(request, dict):
        raw = request.get(header) or request.get(header_lower)
        if raw is not None:
            return [raw]
    # Support passing explicit header dict via keyword arg on FastAPI route handlers
    for attr in ("headers", "state"):
        maybe = getattr(request, attr, None)
        if isinstance(maybe, dict):
            raw = maybe.get(header) or maybe.get(header_lower)
            if raw is not None:
                return [raw]
    return []


def _split_csv(values: Iterable[str]) -> list[str]:
    seen: list[str] = []
    for v in values:
        if not isinstance(v, str):
            continue
        for part in v.split(","):
            trimmed = part.strip()
            if trimmed:
                seen.append(trimmed)
    return seen


def _extract_candidates(request: Any) -> list[str]:
    single = list(_header_values(request, _API_KEY_HEADER))
    multi = list(_header_values(request, _API_KEYS_HEADER))
    auths = list(_header_values(request, _AUTH_HEADER))
    bearer: list[str] = []
    for a in auths:
        if isinstance(a, str) and a.lower().startswith("bearer "):
            bearer.append(a.split(" ", 1)[1].strip())
    return _split_csv(single + multi + bearer)


def _raw_header_values(request: Any) -> list[str]:
    return (
        list(_header_values(request, _API_KEY_HEADER))
        + list(_header_values(request, _API_KEYS_HEADER))
        + list(_header_values(request, _AUTH_HEADER))
    )


def is_api_key_header_authorized(request: Any) -> bool:
    """Return True if any header-provided key matches any allowed environment key."""

    header_values = _raw_header_values(request)
    fn = getattr(synth_ai_py, "localapi_is_api_key_header_authorized", None)
    if callable(fn):
        return fn(header_values)
    allowed = allowed_environment_api_keys()
    return any(h in allowed for h in header_values if isinstance(h, str))


def require_api_key_dependency(request: Any) -> None:
    """FastAPI dependency enforcing Task App authentication headers."""

    allowed = allowed_environment_api_keys()
    if not allowed:
        raise http_exception(
            503, "missing_environment_api_key", "ENVIRONMENT_API_KEY is not configured"
        )

    header_values = _raw_header_values(request)
    fn = getattr(synth_ai_py, "localapi_is_api_key_header_authorized", None)
    authorized = (
        fn(header_values)
        if callable(fn)
        else any(h in allowed for h in header_values if isinstance(h, str))
    )
    if not authorized:
        candidates = _extract_candidates(request)
        with suppress(Exception):
            print(
                {
                    "task_auth_failed": True,
                    "allowed_first15": [k[:15] for k in allowed],
                    "allowed_count": len(allowed),
                    "got_first15": [c[:15] for c in candidates],
                    "got_lens": [len(c) for c in candidates],
                    "have_x_api_key": bool(_header_values(request, _API_KEY_HEADER)),
                    "have_x_api_keys": bool(_header_values(request, _API_KEYS_HEADER)),
                    "have_authorization": bool(_header_values(request, _AUTH_HEADER)),
                },
                flush=True,
            )
        # Use 400 to make failures unmistakable during preflight
        raise http_exception(
            400,
            "unauthorised",
            "API key missing or invalid",
            extra={
                "allowed_first15": [k[:15] for k in allowed],
                "allowed_count": len(allowed),
                "got_first15": [c[:15] for c in candidates],
                "got_lens": [len(c) for c in candidates],
            },
        )
