from __future__ import annotations

import contextlib
import random
from dataclasses import dataclass
from typing import Any, Iterable, Sequence

from synth_ai.http import AsyncHttpClient, sleep

from .config import StreamConfig
from .handlers import StreamHandler
from .types import StreamMessage, StreamType

TERMINAL_STATUSES = {"succeeded", "failed", "cancelled", "canceled", "completed"}
TERMINAL_EVENT_SUCCESS = {
    "sft.job.completed",
    "rl.train.completed",
    "rl.job.completed",
    "workflow.completed",
    "training.completed",
}
TERMINAL_EVENT_FAILURE = {
    "sft.job.failed",
    "rl.train.failed",
    "rl.job.failed",
    "workflow.failed",
    "training.failed",
}


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
        """Endpoints for prompt learning jobs (MIPRO/GEPA)."""
        base = f"/prompt-learning/online/jobs/{job_id}"
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
            metric_fallbacks=(
                f"/learning/jobs/{job_id}/metrics",
            ),
            timeline_fallbacks=(
                f"/learning/jobs/{job_id}/timeline",
            ),
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
        http_client: AsyncHttpClient | None = None,
        sleep_fn= sleep,
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
        self._last_step_by_metric: dict[str, int] = {}
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
        http_cm = self._http or AsyncHttpClient(self.base_url, self.api_key, timeout=self.http_timeout)
        async with http_cm as http:
            while True:
                status = await self._refresh_status(http)

                event_messages = await self._poll_events(http)
                metric_messages = await self._poll_metrics(http)
                timeline_messages = await self._poll_timeline(http)

                self._dispatch(event_messages + metric_messages + timeline_messages)

                if self._terminal_seen or (status and status in TERMINAL_STATUSES):
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

    async def _refresh_status(self, http: AsyncHttpClient) -> str:
        status_payload = await self._poll_status(http)
        if status_payload:
            self._last_status_payload = status_payload
            status = str(status_payload.get("status") or status_payload.get("state") or "").lower()
            if status:
                self._last_status_value = status
                if status in TERMINAL_STATUSES:
                    self._terminal_seen = True
            return status
        return self._last_status_value or ""

    async def _poll_status(self, http: AsyncHttpClient) -> dict[str, Any] | None:
        if StreamType.STATUS not in self.config.enabled_streams or not self._status_paths:
            return None

        for path in self._status_paths:
            try:
                data = await http.get(path)
            except Exception:
                continue
            if isinstance(data, dict):
                message = StreamMessage.from_status(self.job_id, data)
                self._dispatch([message])
                return data
        return None

    async def _poll_events(self, http: AsyncHttpClient) -> list[StreamMessage]:
        if StreamType.EVENTS not in self.config.enabled_streams or not self._event_paths:
            return []
        messages: list[StreamMessage] = []
        total = 0
        for path in self._event_paths:
            since = self._last_seq_by_stream.get(path, 0)
            params = {"since_seq": since, "limit": 200}
            try:
                data = await http.get(path, params=params)
            except Exception:
                continue
            raw_events = _extract_list(data, "events")
            for event in raw_events:
                seq = int(event.get("seq") or 0)
                if seq <= self._last_seq_by_stream.get(path, 0):
                    continue
                if not self.config.should_include_event(event):
                    continue
                self._last_seq_by_stream[path] = seq
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

    async def _poll_metrics(self, http: AsyncHttpClient) -> list[StreamMessage]:
        if StreamType.METRICS not in self.config.enabled_streams or not self._metric_paths:
            return []
        messages: list[StreamMessage] = []
        for path in self._metric_paths:
            after = max(self._last_step_by_metric.values()) if self._last_step_by_metric else -1
            params = {"after_step": after, "limit": 200}
            try:
                data = await http.get(path, params=params)
            except Exception:
                continue
            points = _extract_list(data, "points")
            for point in points:
                name = point.get("name", "")
                step = int(point.get("step") or -1)
                if step <= self._last_step_by_metric.get(name, -1):
                    continue
                if not self.config.should_include_metric(point):
                    continue
                self._last_step_by_metric[name] = step
                metric_job_id = point.get("job_id") or self.job_id
                messages.append(StreamMessage.from_metric(metric_job_id, point))
        return messages

    async def _poll_timeline(self, http: AsyncHttpClient) -> list[StreamMessage]:
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
        for message in messages:
            if self.config.deduplicate and message.key in self._seen_messages:
                continue
            if self.config.sample_rate < 1.0 and random.random() > self.config.sample_rate:
                continue
            if self.config.deduplicate:
                self._seen_messages.add(message.key)

            for handler in self.handlers:
                try:
                    if handler.should_handle(message):
                        handler.handle(message)
                except Exception:
                    pass


def _extract_list(data: Any, field: str) -> list[dict[str, Any]]:
    raw = (data or {}).get(field) if isinstance(data, dict) else None
    if isinstance(raw, list):
        return [item for item in raw if isinstance(item, dict)]
    return []


__all__ = ["JobStreamer", "StreamEndpoints"]
