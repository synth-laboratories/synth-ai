import contextlib
import os
import queue
import threading
import time
import uuid
from datetime import UTC, datetime
from typing import Any, Mapping, Sequence

from synth_ai.core.env import get_backend_from_env
from synth_ai.core.http import http_request


def _get_sdk_version() -> str:
    """Lazy import of SDK version to avoid circular imports."""
    try:
        from synth_ai import __version__

        return __version__
    except ImportError:
        return "0.0.0"


_ENDPOINT_SUFFIX = "/api/v1/sdk-logs"
_MAX_MESSAGE_LENGTH = 4_096
_ALLOWED_LEVELS = {"debug", "info", "warning", "error", "critical"}

# Queue and batching configuration
_QUEUE_SIZE = 1000
_BATCH_SIZE = 50
_FLUSH_INTERVAL = 5.0
_MAX_RETRIES = 3
_BACKOFF_BASE = 1.0

# Global state
_LOG_QUEUE: queue.Queue[dict[str, Any]] = queue.Queue(maxsize=_QUEUE_SIZE)
_worker_thread: threading.Thread | None = None
_worker_lock = threading.Lock()
_shutdown_event = threading.Event()

DISABLE_LOGGING = os.getenv("SYNTH_ENABLE_SDK_LOGGING", None) is None  # Disabled by default


def _start_worker() -> None:
    """Lazy-start the background worker thread on first log."""
    global _worker_thread
    with _worker_lock:
        if _worker_thread is not None and _worker_thread.is_alive():
            return
        _shutdown_event.clear()
        _worker_thread = threading.Thread(
            target=_worker_loop,
            daemon=True,
            name="sdk-logger",
        )
        _worker_thread.start()


def _enqueue(entry: dict[str, Any]) -> None:
    """Add entry to queue, dropping oldest if full."""
    _start_worker()
    try:
        _LOG_QUEUE.put_nowait(entry)
    except queue.Full:
        # Drop oldest entry and try again
        with contextlib.suppress(queue.Empty):
            _LOG_QUEUE.get_nowait()
        with contextlib.suppress(queue.Full):
            _LOG_QUEUE.put_nowait(entry)
            # Still full (race condition), just drop this entry


def _worker_loop() -> None:
    """Main worker loop: collect batch, post with retry."""
    pending_batch: list[dict[str, Any]] = []
    last_flush = time.time()

    while not _shutdown_event.is_set():
        try:
            # Calculate timeout for next flush
            elapsed = time.time() - last_flush
            timeout = max(0.1, _FLUSH_INTERVAL - elapsed)

            try:
                entry = _LOG_QUEUE.get(timeout=timeout)
                pending_batch.append(entry)
                # Limit pending batch size to prevent memory issues
                while len(pending_batch) > _QUEUE_SIZE:
                    pending_batch.pop(0)
            except queue.Empty:
                pass

            # Check if we should try to flush
            now = time.time()
            should_flush = len(pending_batch) >= _BATCH_SIZE or (
                pending_batch and now - last_flush >= _FLUSH_INTERVAL
            )

            if should_flush and pending_batch:
                success = _post_with_retry(pending_batch)
                if success:
                    pending_batch = []
                # If not success (no API key), keep pending_batch for next cycle
                last_flush = now

        except Exception:
            # Worker must never crash
            continue

    # Drain remaining entries on shutdown
    while True:
        try:
            entry = _LOG_QUEUE.get_nowait()
            pending_batch.append(entry)
        except queue.Empty:
            break

    if pending_batch:
        _post_with_retry(pending_batch)


def _post_with_retry(entries: list[dict[str, Any]]) -> bool:
    """POST entries with exponential backoff retry.

    Returns True if successful or if retries exhausted (batch should be dropped).
    Returns False if no API key available (batch should be held for later).
    """
    try:
        base_url, api_key = get_backend_from_env()
        base = (base_url or "").rstrip("/")
        key = (api_key or "").strip()

        # If no API key, signal to hold batch for next cycle
        if not base or not key:
            return False

        url = f"{base}{_ENDPOINT_SUFFIX}"
        headers = {
            "authorization": f"Bearer {key}",
            "accept": "application/json",
            "content-type": "application/json",
        }
        payload: dict[str, Any] = {
            "entries": entries,
            "batch_id": uuid.uuid4().hex,
        }

        for attempt in range(_MAX_RETRIES):
            try:
                http_request("POST", url, headers=headers, body=payload)
                return True  # Success
            except Exception:
                if attempt < _MAX_RETRIES - 1:
                    backoff = _BACKOFF_BASE * (2**attempt)
                    time.sleep(backoff)

        # All retries failed, drop batch
        return True
    except Exception:
        # Unexpected error, drop batch to avoid blocking
        return True


def log_event(
    level: str,
    msg: str,
    *,
    attributes: Mapping[str, Any] | None = None,
    ctx: Mapping[str, Any] | None = None,
    req_id: str | None = None,
    batch_id: str | None = None,
) -> None:
    """Send a single log entry to the backend (best effort, batched async)."""
    if DISABLE_LOGGING:
        return
    try:
        entry = _build_entry(
            level, msg, attributes=attributes, ctx=ctx, req_id=req_id
        )
        if entry is None:
            return
        _enqueue(entry)
    except Exception:
        # Logging must never raise for callers
        return


def log_info(msg: str, **kwargs: Any) -> None:
    log_event("info", msg, **kwargs)


def log_warning(msg: str, **kwargs: Any) -> None:
    log_event("warning", msg, **kwargs)


def log_error(msg: str, **kwargs: Any) -> None:
    log_event("error", msg, **kwargs)


def log_batch(
    entries: Sequence[Mapping[str, Any]], *, batch_id: str | None = None
) -> None:
    """Send a batch of pre-built entries. Invalid entries are skipped silently."""
    if DISABLE_LOGGING:
        return
    try:
        for entry in entries:
            if not isinstance(entry, Mapping):
                continue
            prepared = _build_entry(
                entry.get("level", "info"),
                entry.get("message", ""),
                attributes=entry.get("attributes"),
                ctx=entry.get("context"),
                req_id=entry.get("request_id"),
            )
            if prepared:
                _enqueue(prepared)
    except Exception:
        return


def _build_entry(
    level: str,
    msg: str,
    *,
    attributes: Mapping[str, Any] | None,
    ctx: Mapping[str, Any] | None,
    req_id: str | None,
) -> dict[str, Any] | None:
    normalized_level = str(level or "").lower()
    if normalized_level not in _ALLOWED_LEVELS:
        normalized_level = "info"
    safe_message = (msg or "").strip()
    if not safe_message:
        return None
    safe_message = safe_message[:_MAX_MESSAGE_LENGTH]
    return {
        "level": normalized_level,
        "message": safe_message,
        "timestamp": datetime.now(UTC).isoformat(),
        "sdk_version": _get_sdk_version(),
        "request_id": req_id or uuid.uuid4().hex,
        "attributes": _normalize_mapping(attributes),
        "context": _normalize_mapping(ctx),
    }


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
    """Signal shutdown and wait for worker to drain queue and finish."""
    global _worker_thread

    with _worker_lock:
        thread = _worker_thread

    if thread is None or not thread.is_alive():
        return

    _shutdown_event.set()
    thread.join(timeout=timeout)
