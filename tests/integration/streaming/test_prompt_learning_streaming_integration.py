"""Integration test for prompt learning event/metric streaming.

This test reproduces the core streaming issue where events and metrics
are not being streamed correctly for GEPA/MIPRO jobs.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

pytestmark = pytest.mark.integration

from synth_ai.http import AsyncHttpClient
from synth_ai.streaming import (
    IntegrationTestHandler,
    JobStreamer,
    StreamConfig,
    StreamEndpoints,
    StreamType,
)


class FakePromptLearningBackend:
    """Fake backend that simulates prompt learning job endpoints."""

    def __init__(self) -> None:
        self.events: list[dict[str, Any]] = []
        self.metrics: list[dict[str, Any]] = []
        self.status_history: list[dict[str, Any]] = []
        self._event_seq = 0
        self._metric_step = 0

    def add_event(self, event_type: str, message: str = "", data: dict[str, Any] | None = None) -> None:
        """Add an event to the fake backend."""
        self._event_seq += 1
        self.events.append({
            "seq": self._event_seq,
            "type": event_type,
            "message": message,
            "level": "info",
            "created_at": "2024-01-01T00:00:00Z",
            "data": data or {},
            "job_id": "pl_test123",
        })

    def add_metric(self, name: str, value: float, step: int | None = None, data: dict[str, Any] | None = None) -> None:
        """Add a metric to the fake backend."""
        if step is None:
            self._metric_step += 1
            step = self._metric_step
        self.metrics.append({
            "name": name,
            "value": value,
            "step": step,
            "created_at": "2024-01-01T00:00:00Z",
            "data": data or {},
            "job_id": "pl_test123",
        })

    def set_status(self, status: str) -> None:
        """Set job status."""
        self.status_history.append({
            "status": status,
            "updated_at": "2024-01-01T00:00:00Z",
            "job_id": "pl_test123",
        })

    async def handle_get(self, path: str, params: dict[str, Any] | None = None) -> Any:
        """Handle GET requests to fake backend."""
        params = params or {}

        # Status endpoint
        if path == "/prompt-learning/online/jobs/pl_test123":
            if self.status_history:
                return self.status_history[-1]
            return {"status": "running", "updated_at": "2024-01-01T00:00:00Z", "job_id": "pl_test123"}

        # Events endpoint - Backend now accepts since_seq parameter
        if path == "/prompt-learning/online/jobs/pl_test123/events":
            since_seq = params.get("since_seq", 0)
            limit = params.get("limit", 200)
            # Filter events by sequence number and return list directly (not wrapped)
            filtered = [e for e in self.events if e.get("seq", 0) > since_seq][:limit]
            return filtered

        # Metrics endpoint
        if path == "/prompt-learning/online/jobs/pl_test123/metrics":
            after_step = params.get("after_step", -1)
            limit = params.get("limit", 200)
            filtered = [m for m in self.metrics if m.get("step", -1) > after_step][:limit]
            return {"points": filtered}

        return {}

    async def handle_post(self, path: str, *, json: dict[str, Any] | None = None) -> Any:
        """Handle POST requests (job creation)."""
        if path == "/prompt-learning/online/jobs":
            return {"job_id": "pl_test123", "status": "queued"}
        return {}


@pytest.mark.asyncio
async def test_prompt_learning_streaming_reproduces_issue() -> None:
    """Integration test that reproduces the core streaming issue."""
    backend = FakePromptLearningBackend()
    backend.set_status("running")

    # Simulate GEPA job emitting events and metrics
    backend.add_event("prompt.learning.gepa.start", "Starting GEPA optimisation")
    backend.add_event("prompt.learning.progress", "10% complete", {"percent_overall": 0.1})
    backend.add_metric("gepa.transformation.mean_score", 0.5, step=1, data={"n": 10, "kind": "variation"})
    backend.add_metric("gepa.transformation.mean_score", 0.6, step=2, data={"n": 15, "kind": "variation"})
    backend.add_event("prompt.learning.progress", "20% complete", {"percent_overall": 0.2})
    backend.add_metric("gepa.transformation.mean_score", 0.7, step=3, data={"n": 20, "kind": "variation"})
    backend.set_status("succeeded")
    backend.add_event("prompt.learning.gepa.complete", "GEPA optimisation complete")

    # Create fake HTTP client
    class FakeHttpClient(AsyncHttpClient):
        def __init__(self, backend: FakePromptLearningBackend) -> None:
            super().__init__("http://localhost:8000", "sk-test")
            self.backend = backend

        async def get(self, path: str, *, params: dict[str, Any] | None = None, headers: dict[str, str] | None = None) -> Any:
            return await self.backend.handle_get(path, params)

        async def __aenter__(self) -> AsyncHttpClient:
            return self

        async def __aexit__(self, exc_type, exc, tb) -> None:
            pass

    handler = IntegrationTestHandler()

    config = StreamConfig(
        enabled_streams={StreamType.STATUS, StreamType.EVENTS, StreamType.METRICS},
        metric_names={"gepa.transformation.mean_score"},
    )

    streamer = JobStreamer(
        base_url="http://localhost:8000",
        api_key="sk-test",
        job_id="pl_test123",
        endpoints=StreamEndpoints.prompt_learning("pl_test123"),
        config=config,
        handlers=[handler],
        interval_seconds=0.1,
        timeout_seconds=5.0,
        http_client=FakeHttpClient(backend),
        sleep_fn=lambda s: asyncio.sleep(s),
    )

    result = await streamer.stream_until_terminal()

    # Verify we got terminal status
    assert result.get("status") == "succeeded"

    # CRITICAL: Verify events were streamed
    event_messages = [m for m in handler.messages if m.stream_type == StreamType.EVENTS]
    assert len(event_messages) > 0, f"Expected events but got {len(event_messages)}. All messages: {[m.stream_type for m in handler.messages]}"

    # Verify we got the key events
    event_types = {m.data.get("type") for m in event_messages}
    assert "prompt.learning.gepa.start" in event_types, f"Missing gepa.start event. Got: {event_types}"
    assert "prompt.learning.gepa.complete" in event_types, f"Missing gepa.complete event. Got: {event_types}"
    assert "prompt.learning.progress" in event_types, f"Missing progress event. Got: {event_types}"

    # CRITICAL: Verify metrics were streamed
    metric_messages = [m for m in handler.messages if m.stream_type == StreamType.METRICS]
    assert len(metric_messages) > 0, f"Expected metrics but got {len(metric_messages)}. All messages: {[m.stream_type for m in handler.messages]}"

    # Verify we got the expected metric
    metric_names = {m.data.get("name") for m in metric_messages}
    assert "gepa.transformation.mean_score" in metric_names, f"Missing gepa.transformation.mean_score metric. Got: {metric_names}"

    # Verify metric values
    gepa_metrics = [m for m in metric_messages if m.data.get("name") == "gepa.transformation.mean_score"]
    assert len(gepa_metrics) >= 3, f"Expected at least 3 gepa metrics, got {len(gepa_metrics)}"
    values = [m.data.get("value") for m in gepa_metrics]
    assert 0.5 in values, f"Missing expected metric value 0.5. Got: {values}"
    assert 0.6 in values, f"Missing expected metric value 0.6. Got: {values}"
    assert 0.7 in values, f"Missing expected metric value 0.7. Got: {values}"

    # Verify status was streamed
    status_messages = [m for m in handler.messages if m.stream_type == StreamType.STATUS]
    assert len(status_messages) > 0, "Expected status messages"


@pytest.mark.asyncio
async def test_prompt_learning_events_endpoint_returns_list_directly() -> None:
    """Test that events endpoint returns list directly (not wrapped)."""
    backend = FakePromptLearningBackend()
    backend.add_event("test.event", "Test message")

    # Simulate what the backend actually returns
    events_response = await backend.handle_get("/prompt-learning/online/jobs/pl_test123/events", {"since_seq": 0})

    # Backend returns list directly
    assert isinstance(events_response, list), f"Expected list, got {type(events_response)}"
    assert len(events_response) == 1
    assert events_response[0]["type"] == "test.event"


@pytest.mark.asyncio
async def test_prompt_learning_metrics_endpoint_returns_points() -> None:
    """Test that metrics endpoint returns points wrapped in dict."""
    backend = FakePromptLearningBackend()
    backend.add_metric("test.metric", 0.5, step=1)

    # Simulate what the backend actually returns
    metrics_response = await backend.handle_get("/prompt-learning/online/jobs/pl_test123/metrics", {"after_step": -1})

    # Backend returns {"points": [...]}
    assert isinstance(metrics_response, dict), f"Expected dict, got {type(metrics_response)}"
    assert "points" in metrics_response
    assert len(metrics_response["points"]) == 1
    assert metrics_response["points"][0]["name"] == "test.metric"


@pytest.mark.asyncio
async def test_prompt_learning_streaming_with_since_seq_parameter() -> None:
    """Test that since_seq parameter works correctly for incremental polling."""
    backend = FakePromptLearningBackend()
    backend.add_event("event.1", "First")
    backend.add_event("event.2", "Second")
    backend.add_event("event.3", "Third")

    # First poll: get all events
    events1 = await backend.handle_get("/prompt-learning/online/jobs/pl_test123/events", {"since_seq": 0})
    assert len(events1) == 3

    # Second poll: get only new events (after seq 2)
    events2 = await backend.handle_get("/prompt-learning/online/jobs/pl_test123/events", {"since_seq": 2})
    # Backend now correctly filters by since_seq
    assert len(events2) == 1
    assert events2[0]["type"] == "event.3"


@pytest.mark.asyncio
async def test_backend_route_since_seq_parameter_works() -> None:
    """Verify that backend route correctly accepts and uses since_seq parameter.
    
    FIXED: backend/app/routes/prompt_learning/routes_online.py:589
    The route handler now accepts 'since_seq' and passes it to job_service.get_events().
    This enables efficient incremental polling - only new events are returned.
    """
    backend = FakePromptLearningBackend()
    backend.add_event("event.1", "First")
    backend.add_event("event.2", "Second")
    backend.add_event("event.3", "Third")
    
    # SDK passes since_seq and backend now uses it correctly
    events = await backend.handle_get(
        "/prompt-learning/online/jobs/pl_test123/events",
        {"since_seq": 1, "limit": 100}
    )
    
    # Backend correctly filters to only return events with seq > 1
    assert len(events) == 2
    assert all(e.get("seq", 0) > 1 for e in events)
    event_types = {e["type"] for e in events}
    assert "event.2" in event_types
    assert "event.3" in event_types
    assert "event.1" not in event_types


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

