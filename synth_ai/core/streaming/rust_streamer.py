"""Rust-backed JobStreamer wrapper."""

from __future__ import annotations

import asyncio
import time
from typing import Any, Iterable, Optional

from .config import StreamConfig
from .streamer import TERMINAL_STATUSES
from .types import StreamMessage, StreamType

try:
    import synth_ai_py
except Exception as exc:  # pragma: no cover
    raise RuntimeError("synth_ai_py is required for core.streaming.rust_streamer.") from exc


def _require() -> Any:
    if synth_ai_py is None:
        raise RuntimeError("synth_ai_py not available")
    return synth_ai_py


def _stream_type_name(stream_type: StreamType) -> str:
    if stream_type == StreamType.STATUS:
        return "status"
    if stream_type == StreamType.EVENTS:
        return "events"
    if stream_type == StreamType.METRICS:
        return "metrics"
    if stream_type == StreamType.TIMELINE:
        return "timeline"
    return "events"


def _parse_stream_type(value: Any) -> StreamType:
    if isinstance(value, StreamType):
        return value
    if isinstance(value, str):
        lowered = value.lower()
        if lowered == "status":
            return StreamType.STATUS
        if lowered == "events":
            return StreamType.EVENTS
        if lowered == "metrics":
            return StreamType.METRICS
        if lowered == "timeline":
            return StreamType.TIMELINE
    return StreamType.EVENTS


def _to_stream_message(item: Any) -> Optional[StreamMessage]:
    if isinstance(item, StreamMessage):
        return item
    if not isinstance(item, dict):
        return None
    stream_type = _parse_stream_type(item.get("stream_type"))
    data = item.get("data")
    if not isinstance(data, dict):
        data = dict(data) if isinstance(data, dict) else (data or {})
    return StreamMessage(
        stream_type=stream_type,
        timestamp=item.get("timestamp", "") or "",
        job_id=item.get("job_id", "") or "",
        data=data if isinstance(data, dict) else {},
        seq=item.get("seq"),
        step=item.get("step"),
        phase=item.get("phase"),
    )


def _build_rust_config(config: StreamConfig, interval_seconds: float | None = None) -> Any:
    rust = _require().StreamConfig.all()
    enabled = set(config.enabled_streams)
    for st in StreamType:
        name = _stream_type_name(st)
        if st in enabled:
            rust.enable_stream(name)
        else:
            rust.disable_stream(name)

    if config.event_types:
        for event_type in config.event_types:
            rust.include_event_type(event_type)
    if config.event_types_exclude:
        for event_type in config.event_types_exclude:
            rust.exclude_event_type(event_type)
    if config.event_levels:
        rust.with_levels(list(config.event_levels))
    rust.with_sample_rate(config.sample_rate)
    rust.without_deduplication() if not config.deduplicate else None
    if config.max_events_per_poll is not None:
        rust.set_max_events_per_poll(config.max_events_per_poll)
    if interval_seconds is not None:
        rust.with_interval(interval_seconds)
    return rust


def _build_rust_endpoints(endpoints: Any) -> Any:
    rust = _require().StreamEndpoints.custom(
        status=endpoints.status,
        events=endpoints.events,
        metrics=endpoints.metrics,
        timeline=endpoints.timeline,
    )
    for fallback in getattr(endpoints, "status_fallbacks", ()):
        rust.with_status_fallback(fallback)
    for fallback in getattr(endpoints, "event_fallbacks", ()):
        rust.with_event_fallback(fallback)
    return rust


class JobStreamer:
    """Rust-backed streamer API compatible with core.streaming.JobStreamer."""

    def __init__(
        self,
        *,
        base_url: str,
        api_key: str,
        job_id: str,
        endpoints: Any | None = None,
        config: StreamConfig | None = None,
        handlers: Iterable[Any] | None = None,
        interval_seconds: float | None = None,
        **_: Any,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.job_id = job_id
        self.endpoints = endpoints
        self.config = config or StreamConfig.default()
        self.handlers = list(handlers or [])

        rust = _require()
        rust_endpoints = _build_rust_endpoints(self.endpoints) if self.endpoints else None
        rust_config = _build_rust_config(self.config, interval_seconds)

        self._rust = rust.JobStreamer(
            self.base_url,
            self.api_key,
            self.job_id,
            endpoints=rust_endpoints,
            config=rust_config,
        )

    def _dispatch(self, messages: Iterable[StreamMessage]) -> None:
        for message in messages:
            for handler in self.handlers:
                try:
                    if handler.should_handle(message):
                        handler.handle(message)
                except Exception:
                    continue

    async def poll_status(self) -> dict[str, Any] | None:
        return await asyncio.to_thread(self._rust.poll_status)

    async def poll_events(self) -> list[StreamMessage]:
        raw = await asyncio.to_thread(self._rust.poll_events)
        messages = [m for m in (_to_stream_message(item) for item in raw) if m is not None]
        return messages

    async def poll_metrics(self) -> list[StreamMessage]:
        raw = await asyncio.to_thread(self._rust.poll_metrics)
        messages = [m for m in (_to_stream_message(item) for item in raw) if m is not None]
        return messages

    async def stream_until_terminal(self) -> dict[str, Any]:
        start_time = time.time()
        while True:
            status_payload = await self.poll_status()
            if status_payload:
                status_message = StreamMessage.from_status(self.job_id, status_payload)
                self._dispatch([status_message])
                status_val = str(status_payload.get("status") or status_payload.get("state") or "")
                if status_val.lower() in TERMINAL_STATUSES:
                    return status_payload

            event_messages = await self.poll_events()
            metric_messages = await self.poll_metrics()
            self._dispatch([*event_messages, *metric_messages])

            # Fallback: if any event indicates terminal completion
            for message in event_messages:
                event_type = str(message.data.get("type") or "").lower()
                if event_type.endswith("job.completed") or event_type.endswith("job.failed"):
                    return {
                        "job_id": self.job_id,
                        "status": "succeeded" if "completed" in event_type else "failed",
                    }

            # Avoid tight loop
            elapsed = time.time() - start_time
            if elapsed < 0:
                break
            await asyncio.sleep(2.0)
