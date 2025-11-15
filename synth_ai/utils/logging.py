import threading
import time
import uuid
from contextlib import suppress
from datetime import UTC, datetime
from typing import Any, Mapping, Sequence

from synth_ai import __version__ as sdk_version

from .base_url import get_backend_from_env
from .http import http_request

Severity = str

_ENDPOINT_SUFFIX = "/api/v1/sdk-logs"
_MAX_MESSAGE_LENGTH = 4_096
_ALLOWED_LEVELS = {"debug", "info", "warning", "error", "critical"}
_ACTIVE_THREADS: set[threading.Thread] = set()
_ACTIVE_LOCK = threading.Lock()


def log_event(
    level: Severity,
    message: str,
    *,
    attributes: Mapping[str, Any] | None = None,
    ctx: Mapping[str, Any] | None = None,
    context: Mapping[str, Any] | None = None,
    request_id: str | None = None,
    batch_id: str | None = None,
) -> None:
    """Send a single log entry to the backend BetterStack proxy (best effort)."""

    try:
        payload_context = context if context is not None else ctx
        entry = _build_entry(level, message, attributes=attributes, context=payload_context, request_id=request_id)
        if entry is None:
            return
        _dispatch((entry,), batch_id=batch_id)
    except Exception:
        # Logging must never raise for callers
        return


def log_info(message: str, **kwargs: Any) -> None:
    log_event("info", message, **kwargs)


def log_warning(message: str, **kwargs: Any) -> None:
    log_event("warning", message, **kwargs)


def log_error(message: str, **kwargs: Any) -> None:
    log_event("error", message, **kwargs)


def log_batch(
    entries: Sequence[Mapping[str, Any]],
    *,
    batch_id: str | None = None
) -> None:
    """Send a batch of pre-built entries. Invalid entries are skipped silently."""

    try:
        prepared = [
            _build_entry(
                entry.get("level", "info"),
                entry.get("message", ""),
                attributes=entry.get("attributes"),
                context=entry.get("context"),
                request_id=entry.get("request_id"),
            )
            for entry in entries
            if isinstance(entry, Mapping)
        ]
        filtered = tuple(item for item in prepared if item)
        if filtered:
            _dispatch(filtered, batch_id=batch_id)
    except Exception:
        return


def _build_entry(
    level: Severity,
    message: str,
    *,
    attributes: Mapping[str, Any] | None,
    context: Mapping[str, Any] | None,
    request_id: str | None,
) -> dict[str, Any] | None:
    normalized_level = str(level or "").lower()
    if normalized_level not in _ALLOWED_LEVELS:
        normalized_level = "info"
    safe_message = (message or "").strip()
    if not safe_message:
        return None
    safe_message = safe_message[:_MAX_MESSAGE_LENGTH]
    return {
        "level": normalized_level,
        "message": safe_message,
        "timestamp": datetime.now(UTC).isoformat(),
        "sdk_version": sdk_version,
        "request_id": request_id or uuid.uuid4().hex,
        "attributes": _normalize_mapping(attributes),
        "context": _normalize_mapping(context),
    }


def _dispatch(
    entries: Sequence[Mapping[str, Any]],
    *,
    batch_id: str | None
) -> None:
    try:
        thread = threading.Thread(
            target=_thread_target,
            args=(tuple(entries), batch_id),
            daemon=True,
            name="sdk-logger",
        )
        with _ACTIVE_LOCK:
            _ACTIVE_THREADS.add(thread)
        thread.start()
    except Exception:
        return


def _thread_target(entries: Sequence[Mapping[str, Any]], batch_id: str | None) -> None:
    try:
        _post_entries(entries, batch_id)
    finally:
        with _ACTIVE_LOCK:
            _ACTIVE_THREADS.discard(threading.current_thread())


def _post_entries(
    entries: Sequence[Mapping[str, Any]],
    batch_id: str | None
) -> None:
    try:
        base_url, api_key = get_backend_from_env()
        base = (base_url or "").rstrip("/")
        key = (api_key or "").strip()
        if not base or not key:
            return
        url = f"{base}{_ENDPOINT_SUFFIX}"
        headers = {
            "authorization": f"Bearer {key}",
            "accept": "application/json",
            "content-type": "application/json",
        }
        payload: dict[str, Any] = {"entries": list(entries)}
        if batch_id:
            payload["batch_id"] = batch_id
        http_request("POST", url, headers=headers, body=payload)
    except Exception:
        return


def _normalize_mapping(values: Mapping[str, Any] | None) -> dict[str, Any]:
    if not values:
        return {}
    normalized: dict[str, Any] = {}
    for key, value in values.items():
        try:
            normalized[str(key)] = _coerce(value)
        except Exception:
            continue
    return normalized


def _coerce(value: Any) -> Any:
    if isinstance(value, str | int | float | bool) or value is None:
        return value
    if isinstance(value, Mapping):
        return {str(k): _coerce(v) for k, v in value.items()}
    if isinstance(value, list | tuple | set):
        return [_coerce(v) for v in value]
    return str(value)


def flush_logger(timeout: float | None = None) -> None:
    """Wait for any in-flight logging threads to finish."""

    deadline = time.time() + timeout if timeout is not None else None
    while True:
        with _ACTIVE_LOCK:
            threads = [t for t in _ACTIVE_THREADS if t.is_alive()]
        if not threads:
            return
        remaining = None
        if deadline is not None:
            remaining = max(0.0, deadline - time.time())
            if remaining == 0:
                return
        for thread in threads:
            with suppress(Exception):
                thread.join(remaining)
