"""Authentication helpers shared by Containers."""

from __future__ import annotations

import ipaddress
import logging
import os
from collections.abc import Iterable
from threading import Lock
from typing import Any

try:
    import synth_ai_py
except Exception:  # pragma: no cover - optional in minimal runtime images
    synth_ai_py = None  # type: ignore[assignment]

from .errors import http_exception

_LOGGER = logging.getLogger(__name__)

_AUTH_HEADER = "authorization"
_CONTAINER_AUTH_HEADER = "x-synth-container-authorization"
_RELAY_MARKER_HEADER = "x-synth-relay"
_CONTAINER_AUTH_MODE_ENV = "SYNTH_CONTAINER_AUTH_MODE"
_CONTAINER_AUTH_MODE_REQUIRED = "required"
_CONTAINER_AUTH_MODE_OPTIONAL_LOCAL = "optional_local"
_CONTAINER_AUTH_MODE_DISABLED = "disabled"
_CONTAINER_AUTH_MODES = {
    _CONTAINER_AUTH_MODE_REQUIRED,
    _CONTAINER_AUTH_MODE_OPTIONAL_LOCAL,
    _CONTAINER_AUTH_MODE_DISABLED,
}
_ROUTE_SCOPE_MAP: dict[str, str] = {
    "/rollout": "rollout",
    "/task_info": "task_info",
    "/info": "task_info",
}
_AUTH_SCHEME_PASETO = "paseto"
_AUTH_SCHEME_NONE = "none"
_AUTH_ADOPTION_COUNTS: dict[str, int] = {
    _AUTH_SCHEME_PASETO: 0,
    _AUTH_SCHEME_NONE: 0,
}
_AUTH_ADOPTION_LOCK = Lock()
_AUTH_ADOPTION_LOG_EVERY = 100


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


def _request_path(request: Any) -> str:
    url = getattr(request, "url", None)
    path = getattr(url, "path", None)
    if isinstance(path, str) and path.strip():
        normalized = path.strip()
        if normalized != "/" and normalized.endswith("/"):
            normalized = normalized.rstrip("/")
        return normalized
    return ""


def _required_scope_for_request(request: Any) -> str | None:
    path = _request_path(request)
    return _ROUTE_SCOPE_MAP.get(path)


def _container_auth_mode() -> str:
    raw = os.environ.get(_CONTAINER_AUTH_MODE_ENV, "").strip().lower()
    if raw in _CONTAINER_AUTH_MODES:
        return raw
    return _CONTAINER_AUTH_MODE_REQUIRED


def _request_host(request: Any) -> str:
    client = getattr(request, "client", None)
    host = getattr(client, "host", None)
    if isinstance(host, str) and host.strip():
        return host.strip()
    forwarded_for = next(iter(_header_values(request, "x-forwarded-for")), None)
    if isinstance(forwarded_for, str) and forwarded_for.strip():
        return forwarded_for.split(",", 1)[0].strip()
    return ""


def _is_loopback_host(host: str) -> bool:
    candidate = host.strip()
    if not candidate:
        return False
    if candidate.lower() == "localhost":
        return True
    try:
        return ipaddress.ip_address(candidate).is_loopback
    except ValueError:
        return False


def _is_relay_marked(request: Any) -> bool:
    marker = next(iter(_header_values(request, _RELAY_MARKER_HEADER)), None)
    if not isinstance(marker, str):
        return False
    lowered = marker.strip().lower()
    return lowered not in ("", "0", "false", "no", "off")


def _optional_local_unauth_allowed(request: Any) -> bool:
    if _container_auth_mode() != _CONTAINER_AUTH_MODE_OPTIONAL_LOCAL:
        return False
    path = _request_path(request)
    if path not in {"/rollout", "/task_info", "/info"}:
        return False
    if _is_relay_marked(request):
        return False
    return _is_loopback_host(_request_host(request))


def _auth_adoption_snapshot_unlocked() -> dict[str, Any]:
    total = sum(_AUTH_ADOPTION_COUNTS.values())
    paseto = _AUTH_ADOPTION_COUNTS[_AUTH_SCHEME_PASETO]
    none = _AUTH_ADOPTION_COUNTS[_AUTH_SCHEME_NONE]
    if total <= 0:
        return {
            "total": 0,
            "paseto": 0,
            "none": 0,
            "paseto_pct": 0.0,
            "none_pct": 0.0,
        }
    return {
        "total": total,
        "paseto": paseto,
        "none": none,
        "paseto_pct": round((paseto / total) * 100.0, 2),
        "none_pct": round((none / total) * 100.0, 2),
    }


def container_auth_adoption_metrics() -> dict[str, Any]:
    """Return auth scheme adoption counts and percentages for this process."""

    with _AUTH_ADOPTION_LOCK:
        return _auth_adoption_snapshot_unlocked()


def _reset_container_auth_adoption_metrics_for_tests() -> None:
    with _AUTH_ADOPTION_LOCK:
        _AUTH_ADOPTION_COUNTS[_AUTH_SCHEME_PASETO] = 0
        _AUTH_ADOPTION_COUNTS[_AUTH_SCHEME_NONE] = 0


def _record_auth_scheme(auth_scheme: str, *, path: str, mode: str) -> None:
    if auth_scheme not in _AUTH_ADOPTION_COUNTS:
        return

    should_log_adoption = False
    snapshot: dict[str, Any] = {}
    with _AUTH_ADOPTION_LOCK:
        _AUTH_ADOPTION_COUNTS[auth_scheme] += 1
        total = sum(_AUTH_ADOPTION_COUNTS.values())
        should_log_adoption = total == 1 or total % _AUTH_ADOPTION_LOG_EVERY == 0
        if should_log_adoption:
            snapshot = _auth_adoption_snapshot_unlocked()

    if should_log_adoption:
        _LOGGER.info(
            "container_auth_adoption total=%s paseto=%s none=%s paseto_pct=%.2f none_pct=%.2f",
            snapshot["total"],
            snapshot["paseto"],
            snapshot["none"],
            snapshot["paseto_pct"],
            snapshot["none_pct"],
        )


def require_api_key_dependency(request: Any) -> None:
    """FastAPI dependency enforcing Container authentication headers."""

    mode = _container_auth_mode()
    path = _request_path(request)
    if path in {"/health", "/health/"}:
        _record_auth_scheme(_AUTH_SCHEME_NONE, path=path, mode=mode)
        return
    if mode == _CONTAINER_AUTH_MODE_DISABLED:
        _record_auth_scheme(_AUTH_SCHEME_NONE, path=path, mode=mode)
        return

    host = _request_host(request)
    relay_marked = _is_relay_marked(request)
    container_auth_header = next(iter(_header_values(request, _CONTAINER_AUTH_HEADER)), None)
    if isinstance(container_auth_header, str) and container_auth_header.strip():
        verify_fn = (
            getattr(synth_ai_py, "container_verify_paseto_header", None) if synth_ai_py else None
        )
        if callable(verify_fn):
            required_scope = _required_scope_for_request(request)
            try:
                verify_fn(container_auth_header, required_scope=required_scope)
                _record_auth_scheme(_AUTH_SCHEME_PASETO, path=path, mode=mode)
                return
            except Exception as exc:
                raise http_exception(
                    401,
                    "auth_invalid",
                    "Container token invalid or expired",
                    extra={
                        "auth_scheme": "paseto",
                        "required_scope": required_scope,
                        "deny_reason": str(exc),
                    },
                ) from exc
        have_authorization = bool(_header_values(request, _AUTH_HEADER))
        if relay_marked and _is_loopback_host(host) and have_authorization:
            # In some local/in-process runtimes the optional Rust bindings are
            # unavailable. Relay requests are already authenticated upstream by
            # worker token, so allow loopback relay traffic when both auth
            # headers are present.
            _LOGGER.warning(
                "container_auth_verifier_unavailable path=%s host=%s mode=%s relay_marked=true; accepting loopback relay request",
                path,
                host,
                mode,
            )
            _record_auth_scheme(_AUTH_SCHEME_PASETO, path=path, mode=mode)
            return

    if _optional_local_unauth_allowed(request):
        _record_auth_scheme(_AUTH_SCHEME_NONE, path=path, mode=mode)
        return

    have_authorization = bool(_header_values(request, _AUTH_HEADER))
    have_container_auth = bool(_header_values(request, _CONTAINER_AUTH_HEADER))
    _record_auth_scheme(_AUTH_SCHEME_NONE, path=path, mode=mode)
    raise http_exception(
        401,
        "auth_missing",
        "Container token is required",
        extra={
            "auth_scheme": "none",
            "required_header": _CONTAINER_AUTH_HEADER,
            "have_authorization": have_authorization,
            "have_container_auth": have_container_auth,
            "auth_mode": mode,
            "relay_marked": relay_marked,
            "is_loopback": _is_loopback_host(host),
        },
    )
