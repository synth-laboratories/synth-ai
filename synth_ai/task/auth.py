"""Authentication helpers shared by Task Apps."""

from __future__ import annotations

import os
from collections.abc import Iterable
from contextlib import suppress
from typing import Any

from .errors import http_exception

_API_KEY_ENV = "ENVIRONMENT_API_KEY"
_DEV_API_KEY_ENVS = ("dev_environment_api_key", "DEV_ENVIRONMENT_API_KEY")
_API_KEY_HEADER = "x-api-key"
_API_KEYS_HEADER = "x-api-keys"
_AUTH_HEADER = "authorization"
_API_KEY_ALIASES_ENV = (
    "ENVIRONMENT_API_KEY_ALIASES"  # comma-separated list of additional valid keys
)


def _mask(value: str, *, prefix: int = 4) -> str:
    if not value:
        return "<empty>"
    visible = value[:prefix]
    return f"{visible}{'…' if len(value) > prefix else ''}"


def normalize_environment_api_key() -> str | None:
    """Ensure `ENVIRONMENT_API_KEY` is populated from dev fallbacks.

    Returns the resolved key (if any) so callers can branch on configuration.
    """

    key = os.getenv(_API_KEY_ENV)
    if key:
        return key
    for env in _DEV_API_KEY_ENVS:
        candidate = os.getenv(env)
        if candidate:
            os.environ[_API_KEY_ENV] = candidate
            print(
                f"[task:auth] {_API_KEY_ENV} set from {env} (prefix={_mask(candidate)})",
                flush=True,
            )
            return candidate
    return None


def allowed_environment_api_keys() -> set[str]:
    """Return the set of valid environment API keys for this Task App.

    Includes:
    - The primary ENVIRONMENT_API_KEY (normalized from dev fallbacks if needed)
    - Any comma-separated aliases from ENVIRONMENT_API_KEY_ALIASES
    """
    keys: set[str] = set()
    primary = normalize_environment_api_key()
    if primary:
        keys.add(primary)
    aliases = (os.getenv(_API_KEY_ALIASES_ENV) or "").strip()
    if aliases:
        for part in aliases.split(","):
            trimmed = part.strip()
            if trimmed:
                keys.add(trimmed)
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


def is_api_key_header_authorized(request: Any) -> bool:
    """Return True if any header-provided key matches any allowed environment key."""

    allowed = allowed_environment_api_keys()
    if not allowed:
        return False
    single = list(_header_values(request, _API_KEY_HEADER))
    multi = list(_header_values(request, _API_KEYS_HEADER))
    auths = list(_header_values(request, _AUTH_HEADER))
    bearer: list[str] = []
    for a in auths:
        if isinstance(a, str) and a.lower().startswith("bearer "):
            bearer.append(a.split(" ", 1)[1].strip())
    candidates = _split_csv(single + multi + bearer)
    return any(candidate in allowed for candidate in candidates)


def require_api_key_dependency(request: Any) -> None:
    """FastAPI dependency enforcing Task App authentication headers."""

    allowed = allowed_environment_api_keys()
    if not allowed:
        raise http_exception(
            503, "missing_environment_api_key", "ENVIRONMENT_API_KEY is not configured"
        )
    # Build candidate list for verbose diagnostics
    single = list(_header_values(request, _API_KEY_HEADER))
    multi = list(_header_values(request, _API_KEYS_HEADER))
    auths = list(_header_values(request, _AUTH_HEADER))
    bearer: list[str] = []
    for a in auths:
        if isinstance(a, str) and a.lower().startswith("bearer "):
            bearer.append(a.split(" ", 1)[1].strip())
    candidates = _split_csv(single + multi + bearer)
    if not any(candidate in allowed for candidate in candidates):
        with suppress(Exception):
            print(
                {
                    "task_auth_failed": True,
                    "allowed_first15": [k[:15] for k in allowed],
                    "allowed_count": len(allowed),
                    "got_first15": [c[:15] for c in candidates],
                    "got_lens": [len(c) for c in candidates],
                    "have_x_api_key": bool(single),
                    "have_x_api_keys": bool(multi),
                    "have_authorization": bool(auths),
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
