from __future__ import annotations

"""Authentication helpers shared by Task Apps."""

import os
from typing import Iterable, Optional, Any

from .errors import http_exception

_API_KEY_ENV = "ENVIRONMENT_API_KEY"
_DEV_API_KEY_ENVS = ("dev_environment_api_key", "DEV_ENVIRONMENT_API_KEY")
_API_KEY_HEADER = "x-api-key"
_API_KEYS_HEADER = "x-api-keys"


def _mask(value: str, *, prefix: int = 4) -> str:
    if not value:
        return "<empty>"
    visible = value[:prefix]
    return f"{visible}{'…' if len(value) > prefix else ''}"


def normalize_environment_api_key() -> Optional[str]:
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
    """Return True if `request` carries an authorised API key header."""

    expected = normalize_environment_api_key()
    if not expected:
        return False
    single = list(_header_values(request, _API_KEY_HEADER))
    multi = list(_header_values(request, _API_KEYS_HEADER))
    candidates = _split_csv(single + multi)
    return any(candidate == expected for candidate in candidates)


def require_api_key_dependency(request: Any) -> None:
    """FastAPI dependency enforcing Task App authentication headers."""

    expected = normalize_environment_api_key()
    if not expected:
        raise http_exception(503, "missing_environment_api_key", "ENVIRONMENT_API_KEY is not configured")
    if not is_api_key_header_authorized(request):
        raise http_exception(401, "unauthorised", "API key missing or invalid")
