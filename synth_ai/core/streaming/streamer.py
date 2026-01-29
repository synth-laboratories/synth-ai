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

TERMINAL_STATUSES = {"succeeded", "failed", "cancelled", "canceled", "completed"}

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

        Prefer /api/graph_evolve/jobs/{job_id} with legacy /api/graphgen fallbacks.
        """
        base = f"/graph_evolve/jobs/{job_id}"
        return cls(
            status=base,
            events=f"{base}/events",
            metrics=f"{base}/metrics",
            timeline=None,
            status_fallbacks=(f"/graphgen/jobs/{job_id}",),
            event_fallbacks=(f"/graphgen/jobs/{job_id}/events",),
            metric_fallbacks=(f"/graphgen/jobs/{job_id}/metrics",),
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
                    """Periodically poll status while SSE stream is active."""
                    while not sse_done.is_set() and not self._terminal_seen:
                        await asyncio.sleep(2.0)  # Check every 2 seconds
                        if self._terminal_seen or sse_done.is_set():
                            break

                        status = await self._refresh_status(http)

                        metric_messages = await self._poll_metrics(http)
                        timeline_messages = await self._poll_timeline(http)
                        self._dispatch(metric_messages + timeline_messages)

                        # Check for terminal status
                        if status and status.lower() in TERMINAL_STATUSES:
                            self._terminal_seen = True
                            break

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
                            continue

                        # Handle exception from SSE reader
                        if isinstance(item, Exception):
                            raise item

                        # Process event
                        self._dispatch([item])

                        # Poll metrics/timeline after each event
                        metric_messages = await self._poll_metrics(http)
                        timeline_messages = await self._poll_timeline(http)
                        self._dispatch(metric_messages + timeline_messages)

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
                    print(
                        "[SDK] SSE stream ended before terminal status; continuing with status polling.",
                        flush=True,
                    )
                while not self._terminal_seen:
                    if (
                        self.timeout_seconds is not None
                        and time.time() - start_time > self.timeout_seconds
                    ):
                        print(
                            "[SDK] Stream timeout reached while waiting for terminal status.",
                            flush=True,
                        )
                        break
                    status = await self._refresh_status(http)
                    if status:
                        print(
                            f"[SDK] Status polling: {status} (elapsed={time.time() - start_time:.1f}s)",
                            flush=True,
                        )
                    metric_messages = await self._poll_metrics(http)
                    timeline_messages = await self._poll_timeline(http)
                    self._dispatch(metric_messages + timeline_messages)
                    if status and status.lower() in TERMINAL_STATUSES:
                        self._terminal_seen = True
                        break
                    await self._sleep(self.interval_seconds)
            else:
                # No SSE endpoint available - use polling for events
                while True:
                    status = await self._refresh_status(http)

                    # Check status FIRST before polling events/metrics
                    if status and status.lower() in TERMINAL_STATUSES:
                        self._terminal_seen = True
                        break
                    if self._terminal_seen:
                        break

                    event_messages = await self._poll_events(http)
                    metric_messages = await self._poll_metrics(http)
                    timeline_messages = await self._poll_timeline(http)

                    self._dispatch(event_messages + metric_messages + timeline_messages)

                    # Check again after polling (terminal events might have been received)
                    if self._terminal_seen:
                        break
                    if status and status.lower() in TERMINAL_STATUSES:
                        self._terminal_seen = True
                        break

                    await self._sleep(self.interval_seconds)

        for handler in self.handlers:
            with contextlib.suppress(Exception):
                handler.flush()

        final_status = self._terminal_event_status or self._last_status_value or "unknown"
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

        print(f"[DEBUG] SSE stream connecting to {url}", file=sys.stderr)
        event_count = 0
        start_time = time.time()
        no_events_warning_printed = False

        async for event_data in stream_sse_events(url, headers=headers, timeout=None):
            if not event_data:
                continue
            if event_count == 0 and not no_events_warning_printed:
                elapsed = time.time() - start_time
                if elapsed > 10:
                    print(
                        "[DEBUG] WARNING: No events received via SSE for 10s after connection. Backend may not be publishing to Redis (check SSE_USE_REDIS env var).",
                        file=sys.stderr,
                    )
                    no_events_warning_printed = True
            event_count += 1
            print(
                f"[DEBUG] Parsed SSE event #{event_count}: type={event_data.get('type')}, seq={event_data.get('seq')}",
                file=sys.stderr,
            )

            event_type = str(event_data.get("type", "")).lower()
            if event_type == "sse.stream.ended":
                print("[DEBUG] SSE stream received sse.stream.ended event", file=sys.stderr)
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

            event_status = str(event_data.get("data", {}).get("status", "")).lower()
            if event_status in TERMINAL_STATUSES and not self._terminal_seen:
                self._terminal_seen = True
                self._terminal_event_status = (
                    "succeeded" if event_status in ("succeeded", "completed") else "failed"
                )
                print(
                    f"[DEBUG] Terminal status detected in event data: {event_status}",
                    file=sys.stderr,
                )

            yield msg

    async def _refresh_status(self, http: RustCoreHttpClient) -> str:
        status_payload = await self._poll_status(http)
        if status_payload:
            self._last_status_payload = status_payload
            status = str(status_payload.get("status") or status_payload.get("state") or "").lower()
            if status:
                self._last_status_value = status
                if status in TERMINAL_STATUSES:
                    self._terminal_seen = True
                    print(f"[SDK] Terminal status detected: {status}", flush=True)
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
                print(
                    f"[DEBUG] Polling {path} with since_seq={since}, limit={limit}",
                    file=sys.stderr,
                )
                print(
                    f"[DEBUG] Got response from {path}, type={type(data).__name__}, keys={list(data.keys()) if isinstance(data, dict) else 'not dict'}",
                    file=sys.stderr,
                )
                if isinstance(data, dict):
                    # Check for next_seq to see if we should update our tracking
                    if "next_seq" in data:
                        print(
                            f"[DEBUG] Response has next_seq={data.get('next_seq')}, current since={since}",
                            file=sys.stderr,
                        )
                    # Show what keys are in the response
                    for key in data:
                        val = data[key]
                        if isinstance(val, list):
                            print(
                                f"[DEBUG] Response[{key}] is list with {len(val)} items",
                                file=sys.stderr,
                            )
                            if len(val) > 0:
                                print(
                                    f"[DEBUG] First item in {key}: {list(val[0].keys()) if isinstance(val[0], dict) else type(val[0])}",
                                    file=sys.stderr,
                                )
                        elif isinstance(val, dict):
                            print(
                                f"[DEBUG] Response[{key}] is dict with keys: {list(val.keys())[:5]}",
                                file=sys.stderr,
                            )
            except Exception as e:
                error_str = str(e)
                print(f"[DEBUG] Error polling {path}: {e}", file=sys.stderr)
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
            print(
                f"[DEBUG] Extracted {len(raw_events)} events from {path} using _extract_list",
                file=sys.stderr,
            )
            # Update last_seq using next_seq if available
            if isinstance(data, dict) and "next_seq" in data:
                next_seq = data.get("next_seq")
                if next_seq is not None:
                    try:
                        next_seq_int = int(next_seq)
                        if next_seq_int > since:
                            self._last_seq_by_stream[path] = next_seq_int
                            print(
                                f"[DEBUG] Updated last_seq for {path} to {next_seq_int}",
                                file=sys.stderr,
                            )
                    except (TypeError, ValueError):
                        pass
            if raw_events and len(raw_events) > 0:
                # Log first event type for debugging
                first_event_type = raw_events[0].get("type", "unknown")
                print(f"[DEBUG] First event type: {first_event_type}", file=sys.stderr)
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
                if event_type in TERMINAL_EVENT_SUCCESS:
                    self._terminal_seen = True
                    self._terminal_event_status = "succeeded"
                elif event_type in TERMINAL_EVENT_FAILURE:
                    self._terminal_seen = True
                    self._terminal_event_status = "failed"
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
                    elif phase:
                        self._terminal_event_status = "succeeded"
                messages.append(StreamMessage.from_timeline(timeline_job_id, entry))
        return messages

    def _dispatch(self, messages: Iterable[StreamMessage]) -> None:
        message_list = list(messages)
        for message in message_list:
            if self.config.deduplicate and message.key in self._seen_messages:
                continue
            if self.config.sample_rate < 1.0 and random.random() > self.config.sample_rate:
                continue
            if self.config.deduplicate:
                self._seen_messages.add(message.key)

            # Check for terminal events in dispatch (belt-and-suspenders)
            if message.stream_type == StreamType.EVENTS and message.data:
                event_type = str(message.data.get("type", "")).lower()
                if event_type in TERMINAL_EVENT_SUCCESS:
                    self._terminal_seen = True
                    self._terminal_event_status = "succeeded"
                elif event_type in TERMINAL_EVENT_FAILURE:
                    self._terminal_seen = True
                    self._terminal_event_status = "failed"

            for handler in self.handlers:
                try:
                    if handler.should_handle(message):
                        handler.handle(message)
                except Exception:
                    pass


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
