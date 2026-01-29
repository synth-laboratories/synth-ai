from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import random
import sys
import time
from dataclasses import dataclass
from typing import Any, AsyncIterator, Iterable, Sequence

from synth_ai.core.rust_core.http import RustCoreHttpClient, sleep
from synth_ai.core.rust_core.sse import stream_sse_events

from .config import StreamConfig
from .handlers import StreamHandler
from .types import StreamMessage, StreamType

logger = logging.getLogger(__name__)

TERMINAL_STATUSES = {"succeeded", "failed", "cancelled", "canceled", "completed"}
TERMINAL_STATUS_GRACE_SECONDS = 6.0
TERMINAL_HANDLER_GRACE_SECONDS = 8.0
STALE_STATUS_TIMEOUT_SECONDS = 150.0  # 2.5 minutes with no progress = stale

# Terminal success events - canonical format only
# Format: <activity>.<target>.<algorithm?>.<entity>.<action>
TERMINAL_EVENT_SUCCESS = {
    # Eval events
    "eval.policy.job.completed",
    "eval.verifier.rlm.job.completed",
    # Learning events
    "learning.policy.gepa.job.completed",
    "learning.policy.mipro.job.completed",
    "learning.policy.rl.job.completed",
    "learning.policy.sft.job.completed",
    "learning.graph.gepa.job.completed",
    # Completions events
    "completions.policy.job.completed",
    "completions.graph.job.completed",
    "completions.verifier.rlm.job.completed",
    "completions.rlm.job.completed",
}

# Terminal failure events - canonical format only
TERMINAL_EVENT_FAILURE = {
    # Eval events
    "eval.policy.job.failed",
    "eval.verifier.rlm.job.failed",
    # Learning events
    "learning.policy.gepa.job.failed",
    "learning.policy.mipro.job.failed",
    "learning.policy.rl.job.failed",
    "learning.policy.sft.job.failed",
    "learning.graph.gepa.job.failed",
    # Completions events
    "completions.policy.job.failed",
    "completions.graph.job.failed",
    "completions.verifier.rlm.job.failed",
    "completions.rlm.job.failed",
}


def is_terminal_success_event(event_type: str) -> bool:
    """Check if event_type indicates terminal success.

    Uses both exact matching against TERMINAL_EVENT_SUCCESS and flexible
    suffix matching for `job.completed` patterns.
    """
    event_type = event_type.lower()
    if event_type in TERMINAL_EVENT_SUCCESS:
        return True
    # Flexible matching: any event ending with job.completed
    if event_type.endswith("job.completed") or event_type.endswith(".job.completed"):
        return True
    return False


def is_terminal_failure_event(event_type: str) -> bool:
    """Check if event_type indicates terminal failure.

    Uses both exact matching against TERMINAL_EVENT_FAILURE and flexible
    suffix matching for `job.failed` patterns.
    """
    event_type = event_type.lower()
    if event_type in TERMINAL_EVENT_FAILURE:
        return True
    # Flexible matching: any event ending with job.failed
    if event_type.endswith("job.failed") or event_type.endswith(".job.failed"):
        return True
    return False


def check_terminal_event_typed(event_data: dict[str, Any]) -> tuple[bool, str | None]:
    """Check if an event is terminal using the typed event system.

    This provides an alternative to the string-based TERMINAL_EVENT_SUCCESS/FAILURE
    sets, using the OpenResponses-aligned event parser.

    Args:
        event_data: Raw event dictionary from backend

    Returns:
        A tuple of (is_terminal, status) where:
        - is_terminal: True if this is a terminal event
        - status: "succeeded", "failed", or None if not terminal
    """
    try:
        import synth_ai_py
    except Exception as exc:
        raise RuntimeError("synth_ai_py is required for streaming event parsing.") from exc

    parsed = synth_ai_py.parse_orchestration_event(event_data)
    category = (parsed or {}).get("category")
    if category in {"complete", "termination"}:
        event_type = str(event_data.get("type") or "").lower()
        if "fail" in event_type:
            return True, "failed"
        if "cancel" in event_type:
            return True, "cancelled"
        return True, "succeeded"
    return False, None


@dataclass(slots=True)
class StreamEndpoints:
    """Collection of endpoint paths (with optional fallbacks) to poll for a job."""

    status: str | None
    events: str | None = None
    metrics: str | None = None
    timeline: str | None = None
    status_fallbacks: tuple[str, ...] = ()
    event_fallbacks: tuple[str, ...] = ()
    metric_fallbacks: tuple[str, ...] = ()
    timeline_fallbacks: tuple[str, ...] = ()

    @classmethod
    def learning(cls, job_id: str) -> StreamEndpoints:
        base = f"/learning/jobs/{job_id}"
        return cls(
            status=base,
            events=f"{base}/events",
            metrics=f"{base}/metrics",
            timeline=f"{base}/timeline",
        )

    @classmethod
    def prompt_learning(cls, job_id: str) -> StreamEndpoints:
        """Endpoints for prompt learning jobs (GEPA)."""
        base = f"/policy-optimization/online/jobs/{job_id}"
        return cls(
            status=base,
            events=f"{base}/events",
            metrics=f"{base}/metrics",
            timeline=None,
            status_fallbacks=(
                f"/learning/jobs/{job_id}",
                f"/orchestration/jobs/{job_id}",
            ),
            event_fallbacks=(
                f"/learning/jobs/{job_id}/events",
                f"/orchestration/jobs/{job_id}/events",
            ),
        )

    @property
    def events_stream_url(self) -> str | None:
        """Get the SSE streaming URL for events if available."""
        if self.events and self.events.endswith("/events"):
            return f"{self.events}/stream"
        return None

    @classmethod
    def rl(cls, job_id: str) -> StreamEndpoints:
        base = f"/rl/jobs/{job_id}"
        return cls(
            status=base,
            events=f"{base}/events",
            metrics=f"{base}/metrics",
            timeline=f"{base}/timeline",
            status_fallbacks=(
                f"/learning/jobs/{job_id}",
                f"/orchestration/jobs/{job_id}",
            ),
            event_fallbacks=(
                f"/learning/jobs/{job_id}/events",
                f"/orchestration/jobs/{job_id}/events",
            ),
            metric_fallbacks=(f"/learning/jobs/{job_id}/metrics",),
            timeline_fallbacks=(f"/learning/jobs/{job_id}/timeline",),
        )

    @classmethod
    def graph_evolve(cls, job_id: str) -> StreamEndpoints:
        """Endpoints for Graph Evolve workflow optimization jobs.

        Prefer /api/graph-evolve/jobs/{job_id} with legacy /api/graph_evolve and /api/graphgen fallbacks.
        """
        base = f"/graph-evolve/jobs/{job_id}"
        return cls(
            status=base,
            events=f"{base}/events",
            metrics=f"{base}/metrics",
            timeline=None,
            status_fallbacks=(f"/graph_evolve/jobs/{job_id}", f"/graphgen/jobs/{job_id}"),
            event_fallbacks=(
                f"/graph_evolve/jobs/{job_id}/events",
                f"/graphgen/jobs/{job_id}/events",
            ),
            metric_fallbacks=(
                f"/graph_evolve/jobs/{job_id}/metrics",
                f"/graphgen/jobs/{job_id}/metrics",
            ),
        )

    @classmethod
    def graphgen(cls, job_id: str) -> StreamEndpoints:
        """Legacy alias for Graph Evolve stream endpoints."""
        return cls.graph_evolve(job_id)

    @classmethod
    def eval(cls, job_id: str) -> StreamEndpoints:
        """Endpoints for eval jobs.

        Eval jobs use /api/eval/jobs/{job_id} endpoints.
        No fallbacks needed - eval endpoints are standalone.
        """
        base = f"/eval/jobs/{job_id}"
        return cls(
            status=base,
            events=f"{base}/events",
            metrics=None,  # Eval jobs don't have a metrics endpoint
            timeline=None,
        )


class JobStreamer:
    """Poll job endpoints and dispatch messages to configured handlers."""

    def __init__(
        self,
        *,
        base_url: str,
        api_key: str,
        job_id: str,
        endpoints: StreamEndpoints | None = None,
        config: StreamConfig | None = None,
        handlers: Sequence[StreamHandler] | None = None,
        interval_seconds: float = 2.0,
        timeout_seconds: float | None = None,
        http_timeout: float = 60.0,
        http_client: RustCoreHttpClient | None = None,
        sleep_fn=sleep,
        debug: bool = False,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.job_id = job_id
        self.endpoints = endpoints or StreamEndpoints.learning(job_id)
        self.config = config or StreamConfig.default()
        self.handlers: list[StreamHandler] = list(handlers or [])
        self.interval_seconds = interval_seconds
        self.timeout_seconds = timeout_seconds
        self.http_timeout = http_timeout
        self._http = http_client
        self._sleep = sleep_fn
        self.debug = debug

        status_sources: list[str | None] = [self.endpoints.status]
        status_sources.extend(self.endpoints.status_fallbacks)
        self._status_paths = [p for p in status_sources if p]

        event_sources: list[str | None] = [self.endpoints.events]
        event_sources.extend(self.endpoints.event_fallbacks)
        self._event_paths = [p for p in event_sources if p]

        metric_sources: list[str | None] = [self.endpoints.metrics]
        metric_sources.extend(self.endpoints.metric_fallbacks)
        self._metric_paths = [p for p in metric_sources if p]

        timeline_sources: list[str | None] = [self.endpoints.timeline]
        timeline_sources.extend(self.endpoints.timeline_fallbacks)
        self._timeline_paths = [p for p in timeline_sources if p]

        self._last_seq_by_stream: dict[str, int] = {}
        self._metric_cursors: dict[str, tuple[int | None, str]] = {}
        self._seen_messages: set[str] = set()
        self._last_status_payload: dict[str, Any] | None = None
        self._last_status_value: str | None = None
        self._terminal_seen = False
        self._terminal_event_status: str | None = None
        self._consecutive_terminal_status_polls = 0
        self._terminal_status_seen_at: float | None = None
        self._terminal_status_value: str | None = None
        self._force_event_backfill = False
        self._last_progress_time: float = time.time()  # Track last meaningful progress

        if not self.handlers:
            from .handlers import CLIHandler

            self.handlers = [CLIHandler()]

    async def stream_until_terminal(self) -> dict[str, Any]:
        """Stream configured endpoints until the job reaches a terminal state."""
        http_cm = self._http or RustCoreHttpClient(
            self.base_url, self.api_key, timeout=self.http_timeout
        )
        start_time = time.time()
        async with http_cm as http:
            # Use SSE streaming exclusively for events (prompt learning jobs)
            # SSE provides real-time event delivery from Redis, avoiding empty polling responses
            sse_url = self.endpoints.events_stream_url
            if sse_url and StreamType.EVENTS in self.config.enabled_streams:
                # SSE streaming - real-time event delivery, with concurrent status polling

                # Create a queue to receive events from SSE stream
                event_queue: asyncio.Queue = asyncio.Queue()
                sse_done = asyncio.Event()

                async def sse_reader():
                    """Read SSE events and put them in the queue."""
                    try:
                        async for event_msg in self._stream_events_sse(sse_url):
                            await event_queue.put(event_msg)
                            if self._terminal_seen:
                                break
                    except Exception as e:
                        await event_queue.put(e)  # Put exception in queue
                    finally:
                        sse_done.set()

                async def status_poller():
                    """Periodically poll status while SSE stream is active.

                    Note: Status polling can terminate the stream if the status
                    endpoint reports a terminal state.
                    """
                    while not sse_done.is_set() and not self._terminal_seen:
                        await asyncio.sleep(2.0)  # Check every 2 seconds
                        if self._terminal_seen or sse_done.is_set():
                            break

                        await self._refresh_status(http)

                        metric_messages = await self._poll_metrics(http)
                        timeline_messages = await self._poll_timeline(http)
                        self._dispatch(metric_messages + timeline_messages)
                        if self._force_event_backfill:
                            event_messages = await self._poll_events(http)
                            self._dispatch(event_messages)

                # Start both tasks concurrently
                sse_task = asyncio.create_task(sse_reader())
                status_task = asyncio.create_task(status_poller())

                try:
                    # Process events from queue
                    while not self._terminal_seen:
                        # Wait for event or timeout
                        try:
                            item = await asyncio.wait_for(event_queue.get(), timeout=1.0)
                        except TimeoutError:
                            # No event received, check if SSE is done or terminal
                            if sse_done.is_set() or self._terminal_seen:
                                break
                            # Check for stale status (no progress for too long)
                            if time.time() - self._last_progress_time > STALE_STATUS_TIMEOUT_SECONDS:
                                logger.warning(
                                    "No progress for %.0f seconds - treating as stale/failed",
                                    STALE_STATUS_TIMEOUT_SECONDS,
                                )
                                self._terminal_seen = True
                                self._terminal_event_status = "failed"
                                break
                            continue

                        # Handle exception from SSE reader
                        if isinstance(item, Exception):
                            raise item

                        # Update progress time - we received an event
                        self._last_progress_time = time.time()

                        # Process event
                        self._dispatch([item])
                        self._update_backfill_from_handlers()

                        # Poll metrics/timeline after each event
                        metric_messages = await self._poll_metrics(http)
                        timeline_messages = await self._poll_timeline(http)
                        self._dispatch(metric_messages + timeline_messages)
                        if self._force_event_backfill:
                            event_messages = await self._poll_events(http)
                            self._dispatch(event_messages)

                        # Check for terminal status
                        if self._terminal_seen:
                            break
                finally:
                    # Cancel tasks
                    sse_task.cancel()
                    status_task.cancel()
                    with contextlib.suppress(Exception):
                        await asyncio.gather(sse_task, status_task, return_exceptions=True)
                # If SSE ended before terminal status, fall back to polling
                if not self._terminal_seen:
                    logger.debug(
                        "SSE stream ended before terminal status; continuing with status polling."
                    )
                while not self._terminal_seen:
                    if (
                        self.timeout_seconds is not None
                        and time.time() - start_time > self.timeout_seconds
                    ):
                        logger.debug(
                            "Stream timeout reached while waiting for terminal event."
                        )
                        break
                    status = await self._refresh_status(http)
                    if status:
                        logger.debug(
                            "Status polling: %s (elapsed=%.1fs)", status, time.time() - start_time
                        )
                    # Poll events - terminal events (job.completed) will set _terminal_seen
                    event_messages = await self._poll_events(http)
                    metric_messages = await self._poll_metrics(http)
                    timeline_messages = await self._poll_timeline(http)

                    # Update progress time if we received any messages
                    all_messages = event_messages + metric_messages + timeline_messages
                    if all_messages:
                        self._last_progress_time = time.time()

                    self._dispatch(all_messages)
                    self._update_backfill_from_handlers()

                    # Check for stale status (no progress for too long)
                    if time.time() - self._last_progress_time > STALE_STATUS_TIMEOUT_SECONDS:
                        logger.warning(
                            "No progress for %.0f seconds - treating as stale/failed",
                            STALE_STATUS_TIMEOUT_SECONDS,
                        )
                        self._terminal_seen = True
                        self._terminal_event_status = "failed"
                        break

                    # _terminal_seen is set by _poll_events/_dispatch when terminal event received
                    await self._sleep(self.interval_seconds)
            else:
                # No SSE endpoint available - use polling for events
                # Terminal state is determined by terminal EVENTS, not status endpoint
                while not self._terminal_seen:
                    if (
                        self.timeout_seconds is not None
                        and time.time() - start_time > self.timeout_seconds
                    ):
                        logger.debug(
                            "Stream timeout reached while waiting for terminal event."
                        )
                        break

                    await self._refresh_status(http)

                    # Poll events - terminal events (job.completed) will set _terminal_seen
                    event_messages = await self._poll_events(http)
                    metric_messages = await self._poll_metrics(http)
                    timeline_messages = await self._poll_timeline(http)

                    # Update progress time if we received any messages
                    all_messages = event_messages + metric_messages + timeline_messages
                    if all_messages:
                        self._last_progress_time = time.time()

                    self._dispatch(all_messages)
                    self._update_backfill_from_handlers()

                    # Check for stale status (no progress for too long)
                    if time.time() - self._last_progress_time > STALE_STATUS_TIMEOUT_SECONDS:
                        logger.warning(
                            "No progress for %.0f seconds - treating as stale/failed",
                            STALE_STATUS_TIMEOUT_SECONDS,
                        )
                        self._terminal_seen = True
                        self._terminal_event_status = "failed"
                        break

                    # _terminal_seen is set by _poll_events/_dispatch when terminal event received
                    await self._sleep(self.interval_seconds)

        for handler in self.handlers:
            with contextlib.suppress(Exception):
                handler.flush()

        final_status = self._terminal_event_status or self._last_status_value or "unknown"
        if self.debug:
            print(f"[STREAM DEBUG] Stream exiting: terminal_seen={self._terminal_seen}, "
                  f"terminal_event_status={self._terminal_event_status}, "
                  f"last_status_value={self._last_status_value}, final={final_status}")
        if self._last_status_payload:
            self._last_status_payload["status"] = final_status
            return self._last_status_payload
        return {"job_id": self.job_id, "status": final_status}

    async def _stream_events_sse(self, sse_url: str) -> AsyncIterator[StreamMessage]:
        """Stream events via Server-Sent Events (SSE).

        The backend SSE stream will:
        1. Send events as they occur
        2. Detect terminal events (job.completed, job.failed) and exit cleanly
        3. Periodically check job status and exit if terminal
        4. Send sse.stream.ended event and [DONE] signal when finished
        """
        url = f"{self.base_url.rstrip('/')}/{sse_url.lstrip('/')}"

        # Get last sequence number for reconnection support
        last_seq = self._last_seq_by_stream.get(sse_url, 0)

        headers = {
            "Accept": "text/event-stream",
            "Cache-Control": "no-cache",
            "authorization": f"Bearer {self.api_key}",
        }

        # Include Last-Event-ID header for reconnection
        if last_seq > 0:
            headers["Last-Event-ID"] = str(last_seq)

        # Create a separate session for SSE (long-lived connection)
        timeout = aiohttp.ClientTimeout(total=None)  # No timeout for SSE
        async with (
            aiohttp.ClientSession(headers=headers, timeout=timeout) as session,
            session.get(url) as resp,
        ):
            if resp.status != 200:
                raise Exception(f"SSE endpoint returned {resp.status}: {await resp.text()}")

            print(f"[DEBUG] SSE stream connected to {url}, status={resp.status}", file=sys.stderr)
            buffer = ""
            event_count = 0
            last_event_time = time.time()
            no_events_warning_printed = False

            # Read SSE stream in chunks and parse events
            async for chunk in resp.content.iter_chunked(8192):
                current_time = time.time()

                # Warn if no events received for 10 seconds (events should be streaming)
                if (
                    event_count == 1
                    and current_time - last_event_time > 10
                    and not no_events_warning_printed
                ):
                    print(
                        "[DEBUG] WARNING: No events received via SSE for 10s after connection. Backend may not be publishing to Redis (check SSE_USE_REDIS env var).",
                        file=sys.stderr,
                    )
                    no_events_warning_printed = True

                buffer += chunk.decode("utf-8", errors="ignore")

                # SSE events are separated by double newlines
                while "\n\n" in buffer:
                    event_block, buffer = buffer.split("\n\n", 1)
                    event_block = event_block.strip()

                    if not event_block:
                        continue

                    event_data = {}

                    # Parse SSE event block line by line
                    for event_line in event_block.split("\n"):
                        event_line = event_line.strip()
                        if not event_line or event_line.startswith(":"):
                            continue  # Skip comments/empty lines
                        if event_line.startswith("id:"):
                            pass  # SSE event id (not currently used)
                        elif event_line.startswith("event:"):
                            pass  # SSE event type (not currently used)
                        elif event_line.startswith("data:"):
                            data_str = event_line[5:].strip()
                            if data_str == "[DONE]":
                                print(
                                    "[DEBUG] SSE stream received [DONE] - stream completed gracefully",
                                    file=sys.stderr,
                                )
                                # Stream ended gracefully - if we already saw terminal event,
                                # the job is complete. Otherwise, we'll fall back to polling.
                                return
                            try:
                                event_data = json.loads(data_str)
                            except json.JSONDecodeError as e:
                                print(
                                    f"[DEBUG] Failed to parse SSE data: {e}, data={data_str[:200]}",
                                    file=sys.stderr,
                                )
                                continue

                    # Debug: log what we parsed
                    if event_data:
                        event_count += 1
                        last_event_time = time.time()
                        print(
                            f"[DEBUG] Parsed SSE event #{event_count}: type={event_data.get('type')}, seq={event_data.get('seq')}",
                            file=sys.stderr,
                        )

                    if event_data and "type" in event_data:
                        event_type = str(event_data.get("type", "")).lower()

                        # Handle stream lifecycle events from backend
                        if event_type == "sse.stream.ended":
                            # Backend signaled stream end - this is a graceful termination
                            print(
                                "[DEBUG] SSE stream received sse.stream.ended event",
                                file=sys.stderr,
                            )
                            # Don't yield this event, just return
                            return

            event_job_id = event_data.get("job_id") or self.job_id
            msg = StreamMessage.from_event(event_job_id, event_data)

            seq = event_data.get("seq")
            if seq is not None:
                try:
                    seq_int = int(seq)
                    if (
                        sse_url not in self._last_seq_by_stream
                        or seq_int > self._last_seq_by_stream[sse_url]
                    ):
                        self._last_seq_by_stream[sse_url] = seq_int
                except (TypeError, ValueError):
                    pass

                # Check for terminal events
                if event_type in TERMINAL_EVENT_SUCCESS:
                    self._terminal_seen = True
                    self._terminal_event_status = "succeeded"
                    print(
                        f"[DEBUG] Terminal success event detected: {event_type}",
                        file=sys.stderr,
                    )
                elif event_type in TERMINAL_EVENT_FAILURE:
                    self._terminal_seen = True
                    self._terminal_event_status = "failed"
                    print(
                        f"[DEBUG] Terminal failure event detected: {event_type}",
                        file=sys.stderr,
                    )

                # Also check event data for status (synthetic events from status check)
                event_status = str(event_data.get("data", {}).get("status", "")).lower()
                if event_status in TERMINAL_STATUSES and not self._terminal_seen:
                    self._terminal_seen = True
                    self._terminal_event_status = (
                        "succeeded"
                        if event_status in ("succeeded", "completed")
                        else "failed"
                    )
                    print(
                        f"[DEBUG] Terminal status detected in event data: {event_status}",
                        file=sys.stderr,
                    )

            yield msg

    async def _refresh_status(self, http: AsyncHttpClient) -> str:
        status_payload = await self._poll_status(http)
        if status_payload:
            self._last_status_payload = status_payload
            status = str(status_payload.get("status") or status_payload.get("state") or "").lower()
            if status:
                self._last_status_value = status
                if status in TERMINAL_STATUSES:
                    # Treat status as authoritative for terminal state.
                    self._terminal_seen = True
                    if status in {"failed", "cancelled", "canceled"}:
                        self._terminal_event_status = "failed"
                    else:
                        self._terminal_event_status = "succeeded"
                    if self.debug:
                        print(
                            f"[STREAM DEBUG] STATUS TERMINAL: {status}"
                        )
                    self._consecutive_terminal_status_polls = 0
                    self._terminal_status_seen_at = None
                    self._terminal_status_value = None
                else:
                    self._consecutive_terminal_status_polls = 0
                    self._terminal_status_seen_at = None
                    self._terminal_status_value = None
            return status
        return self._last_status_value or ""

    async def _poll_status(self, http: RustCoreHttpClient) -> dict[str, Any] | None:
        if StreamType.STATUS not in self.config.enabled_streams or not self._status_paths:
            return None

        last_error: Exception | None = None
        for path in self._status_paths:
            try:
                # Add cache-busting query param to ensure fresh status
                # Use a timestamp to prevent any caching
                params = {"_t": int(time.time() * 1000)}
                data = await http.get(path, params=params)
            except Exception as exc:
                last_error = exc
                # Try next fallback path
                continue
            if isinstance(data, dict):
                message = StreamMessage.from_status(self.job_id, data)
                self._dispatch([message])
                self._update_backfill_from_handlers()
                return data

        # If all paths failed, log the error for debugging
        if last_error is not None:
            logger = logging.getLogger(__name__)
            logger.debug(f"Status polling failed for all paths: {last_error}")
        return None

    async def _poll_events(self, http: RustCoreHttpClient) -> list[StreamMessage]:
        if StreamType.EVENTS not in self.config.enabled_streams or not self._event_paths:
            return []
        messages: list[StreamMessage] = []
        total = 0
        for path in self._event_paths:
            since = self._last_seq_by_stream.get(path, 0)
            # Increase limit to capture more events per poll
            limit = (
                1000
                if self.config.max_events_per_poll and self.config.max_events_per_poll > 200
                else 200
            )
            params = {"since_seq": since, "limit": limit}
            try:
                data = await http.get(path, params=params)
                # Debug: Always log what we got from API
                logger.debug("Polling %s with since_seq=%s, limit=%s", path, since, limit)
                logger.debug(
                    "Got response from %s, type=%s, keys=%s",
                    path,
                    type(data).__name__,
                    list(data.keys()) if isinstance(data, dict) else "not dict",
                )
                if isinstance(data, dict):
                    # Check for next_seq to see if we should update our tracking
                    if "next_seq" in data:
                        logger.debug(
                            "Response has next_seq=%s, current since=%s",
                            data.get("next_seq"),
                            since,
                        )
                    # Show what keys are in the response
                    for key in data:
                        val = data[key]
                        if isinstance(val, list):
                            logger.debug("Response[%s] is list with %d items", key, len(val))
                            if len(val) > 0:
                                logger.debug(
                                    "First item in %s: %s",
                                    key,
                                    list(val[0].keys()) if isinstance(val[0], dict) else type(val[0]),
                                )
                        elif isinstance(val, dict):
                            logger.debug(
                                "Response[%s] is dict with keys: %s", key, list(val.keys())[:5]
                            )
            except Exception as e:
                error_str = str(e)
                logger.debug("Error polling %s: %s", path, e)
                # Fail fast if we get 404 on GraphGen and fallback endpoints (indicates job ID mapping issue)
                if (
                    "404" in error_str
                    and (
                        "graphgen" in path.lower()
                        or "policy-optimization" in path.lower()
                        or "prompt-learning" in path.lower()
                    )
                    and path == self._event_paths[-1]  # Last fallback path
                ):
                    raise RuntimeError(
                        f"Failed to poll events: All endpoints returned 404. "
                        f"This likely indicates a job ID mapping issue. "
                        f"GraphGen endpoints need the GraphGen job ID; GEPA fallback endpoints need the GEPA job ID. "
                        f"Last error: {error_str}"
                    ) from e
                continue
            raw_events = _extract_list(data, "events")
            # Debug: Always log what we extracted
            logger.debug("Extracted %d events from %s using _extract_list", len(raw_events), path)
            # Update last_seq using next_seq if available
            if isinstance(data, dict) and "next_seq" in data:
                next_seq = data.get("next_seq")
                if next_seq is not None:
                    try:
                        next_seq_int = int(next_seq)
                        if next_seq_int > since:
                            self._last_seq_by_stream[path] = next_seq_int
                            logger.debug("Updated last_seq for %s to %d", path, next_seq_int)
                    except (TypeError, ValueError):
                        pass
            if raw_events and len(raw_events) > 0:
                # Log first event type for debugging
                first_event_type = raw_events[0].get("type", "unknown")
                logger.debug("First event type: %s", first_event_type)
            for event in raw_events:
                seq_raw = event.get("seq")
                try:
                    seq_value = int(seq_raw)  # type: ignore[arg-type]
                except (TypeError, ValueError):
                    seq_value = None
                last_seq = self._last_seq_by_stream.get(path, 0)
                seq = seq_value if seq_value is not None else last_seq + 1
                if seq <= last_seq:
                    continue
                if seq_value is None:
                    event["seq"] = seq
                self._last_seq_by_stream[path] = seq
                # Bypass filtering - show ALL events
                # if not self.config.should_include_event(event):
                #     continue
                event_job_id = event.get("job_id") or self.job_id
                event_message = StreamMessage.from_event(event_job_id, event)
                event_type = str(event.get("type") or "").lower()
                if is_terminal_success_event(event_type):
                    self._terminal_seen = True
                    self._terminal_event_status = "succeeded"
                    if self.debug:
                        print(f"[STREAM DEBUG] POLL TERMINAL SUCCESS: {event_type}")
                elif is_terminal_failure_event(event_type):
                    self._terminal_seen = True
                    self._terminal_event_status = "failed"
                    if self.debug:
                        print(f"[STREAM DEBUG] POLL TERMINAL FAILURE: {event_type}")
                messages.append(event_message)
                total += 1
                if self.config.max_events_per_poll and total >= self.config.max_events_per_poll:
                    return messages
        return messages

    async def _poll_metrics(self, http: RustCoreHttpClient) -> list[StreamMessage]:
        if StreamType.METRICS not in self.config.enabled_streams or not self._metric_paths:
            return []
        messages: list[StreamMessage] = []
        for path in self._metric_paths:
            params = {"limit": 200}
            try:
                data = await http.get(path, params=params)
            except Exception:
                continue
            points = _extract_list(data, "points")
            for point in points:
                name = str(point.get("name") or "")
                if not name:
                    continue
                step, fingerprint = _metric_cursor(point)
                last_step, last_fingerprint = self._metric_cursors.get(name, (None, ""))
                if step is not None:
                    if last_step is not None and step <= last_step:
                        continue
                elif fingerprint and fingerprint == last_fingerprint:
                    continue
                self._metric_cursors[name] = (step, fingerprint)
                if not self.config.should_include_metric(point):
                    continue
                metric_job_id = point.get("job_id") or self.job_id
                messages.append(StreamMessage.from_metric(metric_job_id, point))
        return messages

    async def _poll_timeline(self, http: RustCoreHttpClient) -> list[StreamMessage]:
        if StreamType.TIMELINE not in self.config.enabled_streams or not self._timeline_paths:
            return []
        messages: list[StreamMessage] = []
        for path in self._timeline_paths:
            try:
                data = await http.get(path)
            except Exception:
                continue

            timeline_entries = _extract_list(data, "events")
            for entry in timeline_entries:
                if not self.config.should_include_timeline(entry):
                    continue
                timeline_job_id = entry.get("job_id") or self.job_id
                phase = str(entry.get("phase") or "").lower()
                if phase in TERMINAL_STATUSES:
                    self._terminal_seen = True
                    if phase in {"failed", "cancelled", "canceled"}:
                        self._terminal_event_status = "failed"
                        if self.debug:
                            print(f"[STREAM DEBUG] TIMELINE TERMINAL FAILURE: phase={phase}")
                    elif phase:
                        self._terminal_event_status = "succeeded"
                        if self.debug:
                            print(f"[STREAM DEBUG] TIMELINE TERMINAL SUCCESS: phase={phase}")
                messages.append(StreamMessage.from_timeline(timeline_job_id, entry))
        return messages

    def _dispatch(self, messages: Iterable[StreamMessage]) -> None:
        message_list = list(messages)
        for message in message_list:
            if message.stream_type == StreamType.EVENTS and message.data:
                event_type = str(message.data.get("type") or "").lower()
                if (
                    event_type.startswith("learning.policy.gepa.")
                    and message.data.get("run_id") is None
                    and any(
                        token in event_type
                        for token in (
                            "candidate.evaluated",
                            "candidate_scored",
                            "proposal.scored",
                            "generation.started",
                            "generation.completed",
                        )
                    )
                ):
                    if self.debug:
                        print(f"[STREAM DEBUG] filtered GEPA event without run_id: {event_type}")
                    continue
            dedupe_keys = [message.key]
            if message.stream_type == StreamType.EVENTS and message.data:
                data = message.data.get("data")
                if isinstance(data, dict) and data.get("source") == "status_check":
                    continue
            if (
                self.config.deduplicate
                and self.config.dedupe_events
                and message.stream_type == StreamType.EVENTS
                and message.data
            ):
                dedupe_keys.extend(_event_dedupe_keys(message.data))
                fingerprint = _event_dedupe_fingerprint(message.data)
                if fingerprint:
                    fp_key = f"event:fp:{fingerprint}"
                    if message.seq is None:
                        dedupe_keys = [fp_key]
                    else:
                        dedupe_keys.append(fp_key)
            if self.config.deduplicate and any(key in self._seen_messages for key in dedupe_keys):
                continue
            if self.config.sample_rate < 1.0 and random.random() > self.config.sample_rate:
                continue
            if self.config.deduplicate:
                self._seen_messages.update(dedupe_keys)

            # Debug: print all events
            if self.debug and message.stream_type == StreamType.EVENTS and message.data:
                event_type = str(message.data.get("type", ""))
                print(f"[STREAM DEBUG] event: {event_type}")

            # Check for terminal events in dispatch (belt-and-suspenders)
            if message.stream_type == StreamType.EVENTS and message.data:
                event_type = str(message.data.get("type", "")).lower()
                if is_terminal_success_event(event_type):
                    self._terminal_seen = True
                    self._terminal_event_status = "succeeded"
                    if self.debug:
                        print(f"[STREAM DEBUG] *** TERMINAL SUCCESS: {event_type} ***")
                elif is_terminal_failure_event(event_type):
                    self._terminal_seen = True
                    self._terminal_event_status = "failed"
                    if self.debug:
                        print(f"[STREAM DEBUG] *** TERMINAL FAILURE: {event_type} ***")

            for handler in self.handlers:
                try:
                    if handler.should_handle(message):
                        handler.handle(message)
                except Exception:
                    pass

    def _update_backfill_from_handlers(self) -> None:
        if self._force_event_backfill:
            return
        for handler in self.handlers:
            wants = getattr(handler, "wants_event_backfill", None)
            if callable(wants) and wants():
                self._force_event_backfill = True
                break
        if self._terminal_seen:
            return
        for handler in self.handlers:
            hint = getattr(handler, "terminal_hint_ready", None)
            if callable(hint) and hint(grace_seconds=TERMINAL_HANDLER_GRACE_SECONDS):
                self._terminal_seen = True
                self._terminal_event_status = "succeeded"
                if self.debug:
                    print(
                        f"[STREAM DEBUG] TERMINAL HINT: handler signaled completion after {TERMINAL_HANDLER_GRACE_SECONDS:.0f}s grace"
                    )
                break


def _metric_cursor(point: dict[str, Any]) -> tuple[int | None, str]:
    raw_step = point.get("step")
    step_value: int | None = None
    if raw_step is not None:
        try:
            step_value = int(raw_step)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            step_value = None

    fingerprint = ""
    for key in ("created_at", "updated_at", "timestamp"):
        ts_val = point.get(key)
        if ts_val is not None and ts_val != "":
            fingerprint = str(ts_val)
            break
    if not fingerprint:
        try:
            fingerprint = json.dumps(point, sort_keys=True, default=str)
        except Exception:
            fingerprint = repr(point)
    return step_value, fingerprint


def _event_dedupe_keys(event_data: dict[str, Any]) -> list[str]:
    event_type = str(event_data.get("type") or "").lower()
    data = event_data.get("data") or {}
    if not isinstance(data, dict):
        data = {}

    keys: list[str] = []
    run_id = event_data.get("run_id") or data.get("run_id")

    def _with_run(key: str) -> str:
        return f"run:{run_id}:{key}" if run_id else key

    event_id = data.get("event_id") or event_data.get("event_id")
    if event_id:
        keys.append(_with_run(f"event_id:{event_id}"))

    if event_type.startswith("learning.policy.gepa."):
        candidate_id = data.get("candidate_id") or data.get("version_id")
        if not candidate_id and isinstance(data.get("program_candidate"), dict):
            candidate_id = data.get("program_candidate", {}).get("candidate_id")
        if candidate_id:
            keys.append(_with_run(f"{event_type}:candidate:{candidate_id}"))
        generation = data.get("generation")
        if generation is not None and event_type.endswith((".generation.started", ".generation.completed")):
            keys.append(_with_run(f"{event_type}:generation:{generation}"))

    if event_type.endswith((".job.completed", ".job.failed", ".job.cancelled", ".job.canceled")):
        keys.append(_with_run(f"{event_type}:terminal"))

    return keys


def _event_dedupe_fingerprint(event_data: dict[str, Any]) -> str:
    drop_keys = {
        "id",
        "job_id",
        "run_id",
        "seq",
        "created_at",
        "updated_at",
        "timestamp",
        "inserted_at",
        "emitted_at",
    }
    drop_data_keys = {
        "event_id",
        "created_at",
        "updated_at",
        "timestamp",
        "timestamp_ms",
        "inserted_at",
        "emitted_at",
        "run_id",
        "workflow_id",
        "workflow_run_id",
        "activity_id",
        "attempt",
        "task_queue",
    }

    def scrub(value: Any) -> Any:
        if isinstance(value, dict):
            return {k: scrub(v) for k, v in value.items() if k not in drop_data_keys}
        if isinstance(value, list):
            return [scrub(item) for item in value]
        return value

    cleaned = {key: value for key, value in event_data.items() if key not in drop_keys}
    if "data" in cleaned:
        cleaned["data"] = scrub(cleaned["data"])
    try:
        return json.dumps(cleaned, sort_keys=True, default=str)
    except Exception:
        return repr(cleaned)


def _extract_list(data: Any, field: str) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    seen_items: set[int] = set()
    stack: list[Any] = [data]
    seen_containers: set[int] = set()

    fallback_keys = {
        "data",
        "result",
        "results",
        "items",
        "payload",
        "records",
        "entries",
        "values",
    }

    while stack:
        current = stack.pop()
        if current is None:
            continue
        current_id = id(current)
        if current_id in seen_containers:
            continue
        seen_containers.add(current_id)

        if isinstance(current, list):
            for item in current:
                if isinstance(item, dict):
                    item_id = id(item)
                    if item_id not in seen_items:
                        seen_items.add(item_id)
                        results.append(item)
        elif isinstance(current, dict):
            if field in current:
                stack.append(current[field])
            for key in fallback_keys:
                if key in current:
                    stack.append(current[key])
    return results


__all__ = ["JobStreamer", "StreamEndpoints"]
