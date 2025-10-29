from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

import pytest
from synth_ai.http import AsyncHttpClient
from synth_ai.streaming import (
    BufferedHandler,
    CallbackHandler,
    CLIHandler,
    IntegrationTestHandler,
    JobStreamer,
    JSONHandler,
    LossCurveHandler,
    StreamConfig,
    StreamEndpoints,
    StreamMessage,
    StreamType,
)


def _make_event(seq: int = 1, event_type: str = "sft.progress") -> dict[str, Any]:
    return {
        "seq": seq,
        "type": event_type,
        "level": "info",
        "message": "step update",
        "created_at": "2024-10-28T19:22:00Z",
        "data": {"step": 5, "total_steps": 10},
    }


def test_stream_type_endpoint_path() -> None:
    assert StreamType.STATUS.endpoint_path == ""
    assert StreamType.EVENTS.endpoint_path == "/events"
    assert StreamType.METRICS.endpoint_path == "/metrics"
    assert StreamType.TIMELINE.endpoint_path == "/timeline"


def test_stream_message_key_variants() -> None:
    status = StreamMessage.from_status("job_123", {"updated_at": "now", "status": "running"})
    event = StreamMessage.from_event("job_123", _make_event(seq=7))
    metric = StreamMessage.from_metric(
        "job_123", {"name": "train.loss", "value": 0.42, "step": 11, "created_at": "now"}
    )
    timeline = StreamMessage.from_timeline(
        "job_123", {"phase": "training", "created_at": "now", "metadata": {}}
    )

    assert status.key.startswith("status:")
    assert event.key == "event:7"
    assert metric.key == "metric:train.loss:11"
    assert timeline.key == "timeline:training:now"


def test_stream_config_filters() -> None:
    cfg = StreamConfig(
        enabled_streams={StreamType.STATUS, StreamType.EVENTS},
        event_types={"sft.progress"},
        metric_names={"train.loss"},
        timeline_phases={"training"},
    )

    assert not cfg.should_include_event(_make_event(event_type="sft.training.started"))
    assert cfg.should_include_event(_make_event(event_type="sft.progress"))
    assert cfg.should_include_metric({"name": "train.loss", "phase": "train"})
    assert not cfg.should_include_metric({"name": "val.loss", "phase": "eval"})
    assert cfg.should_include_timeline({"phase": "training"})
    assert not cfg.should_include_timeline({"phase": "completed"})


def test_cli_handler_outputs_status(capsys: pytest.CaptureFixture[str]) -> None:
    handler = CLIHandler()
    message = StreamMessage.from_status("job_123", {"updated_at": "now", "status": "running"})
    handler.handle(message)
    captured = capsys.readouterr()
    assert "status=running" in captured.out


def test_cli_handler_hidden_events(capsys: pytest.CaptureFixture[str]) -> None:
    handler = CLIHandler(hidden_event_types={"sft.running"})
    handler.handle(StreamMessage.from_event("job", _make_event(event_type="sft.running")))
    assert capsys.readouterr().out == ""


def test_cli_handler_hidden_substrings(capsys: pytest.CaptureFixture[str]) -> None:
    handler = CLIHandler(hidden_event_substrings={"modal"})
    event = _make_event(event_type="rl.runtime.modal.started")
    event["message"] = "Modal container started"
    handler.handle(StreamMessage.from_event("job", event))
    assert capsys.readouterr().out == ""


def test_callback_handler_invokes_callbacks() -> None:
    calls: dict[str, list[dict[str, Any]]] = {}

    def _record(kind: str, payload: dict[str, Any]) -> None:
        calls.setdefault(kind, []).append(payload)

    handler = CallbackHandler(
        on_status=lambda data: _record("status", data),
        on_event=lambda data: _record("event", data),
    )

    handler.handle(StreamMessage.from_status("job_1", {"updated_at": "now", "status": "running"}))
    handler.handle(StreamMessage.from_event("job_1", _make_event()))
    assert "status" in calls and "event" in calls


def test_json_handler_writes_file(tmp_path: Path) -> None:
    output = tmp_path / "stream.jsonl"
    handler = JSONHandler(output_file=str(output))
    handler.handle(StreamMessage.from_metric("job_123", {"step": 1, "name": "loss", "value": 1.2}))
    handler.flush()

    content = output.read_text(encoding="utf-8").strip()
    assert content
    parsed = json.loads(content)
    assert parsed["stream_type"] == "METRICS"
    assert parsed["data"]["name"] == "loss"


def test_buffered_handler_batches_messages() -> None:
    processed: list[list[StreamMessage]] = []

    class RecordingBufferedHandler(BufferedHandler):
        def process_batch(self, messages: list[StreamMessage]) -> None:
            processed.append(list(messages))

    handler = RecordingBufferedHandler(max_buffer_size=2, flush_interval=60.0)
    handler.handle(StreamMessage.from_event("job", _make_event(seq=1)))
    assert not processed
    handler.handle(StreamMessage.from_event("job", _make_event(seq=2)))
    assert processed and len(processed[0]) == 2


def test_integration_test_handler_collects_messages() -> None:
    handler = IntegrationTestHandler()
    msg = StreamMessage.from_event("job", _make_event(seq=42))
    handler.handle(msg)
    assert handler.messages == [msg]
    handler.clear()
    assert handler.messages == []


def test_loss_curve_handler_renders_chart(monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("rich")

    class DummyLive:
        def __init__(self) -> None:
            self.started = False
            self.updated = []

        def start(self) -> None:
            self.started = True

        def stop(self) -> None:
            self.started = False

        def update(self, renderable) -> None:
            self.updated.append(renderable)

    dummy_live = DummyLive()
    handler = LossCurveHandler(width=5, live=dummy_live)
    handler.handle(StreamMessage.from_status("job", {"status": "running", "updated_at": "t"}))
    for idx, loss in enumerate([2.0, 1.9, 1.8], start=1):
        handler.handle(
            StreamMessage.from_metric(
                "job",
                {"name": "train.loss", "value": loss, "step": idx, "created_at": "t"},
            )
        )

    assert dummy_live.started
    assert dummy_live.updated, "expected live updates"
    # Latest panel should include latest loss value
    panel = dummy_live.updated[-1]
    assert "1.8000" in str(getattr(panel, "renderable", panel))
    handler.flush()
    assert not dummy_live.started


@pytest.mark.asyncio
async def test_job_streamer_streams_until_terminal() -> None:
    class FakeHttp:
        def __init__(self) -> None:
            self.status_calls = 0

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        async def get(self, path: str, *, params=None, headers=None):
            if path.endswith("/events"):
                return {"events": [_make_event(seq=1)]}
            if path.endswith("/metrics"):
                return {"points": [{"name": "train.loss", "value": 0.42, "step": 1, "created_at": "t"}]}
            if path.endswith("/timeline"):
                return {"events": [{"phase": "training", "created_at": "t"}]}
            self.status_calls += 1
            if self.status_calls < 2:
                return {"status": "running", "updated_at": "t0"}
            return {"status": "succeeded", "updated_at": "t1"}

    captured = IntegrationTestHandler()

    async def _noop_sleep(_: float) -> None:
        return None

    streamer = JobStreamer(
        base_url="https://api.example.com/api",
        api_key="sk-test",
        job_id="job_1",
        endpoints=StreamEndpoints.learning("job_1"),
        handlers=[captured],
        http_client=cast(AsyncHttpClient, FakeHttp()),
        sleep_fn=_noop_sleep,
        interval_seconds=0.01,
    )

    assert StreamType.EVENTS in streamer.config.enabled_streams
    result = await streamer.stream_until_terminal()
    assert result["status"] == "succeeded"
    types = {msg.stream_type for msg in captured.messages}
    assert StreamType.STATUS in types
    assert StreamType.METRICS in types


@pytest.mark.asyncio
async def test_job_streamer_rl_fallbacks_and_terminal_event() -> None:
    class FakeHttp:
        def __init__(self) -> None:
            self.status_calls = 0
            self.event_calls = 0

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        async def get(self, path: str, *, params=None, headers=None):
            if path.endswith("/rl/jobs/job_rl"):
                self.status_calls += 1
                # Simulate status lagging behind terminal event
                return {"status": "running", "updated_at": f"t{self.status_calls}"}
            if path.endswith("/rl/jobs/job_rl/events"):
                return {"events": []}
            if path.endswith("/learning/jobs/job_rl/events"):
                self.event_calls += 1
                if self.event_calls == 1:
                    return {
                        "events": [
                            {
                                "seq": 1,
                                "type": "rl.train.completed",
                                "created_at": "t-final",
                                "job_id": "job_rl",
                            }
                        ]
                    }
                return {"events": []}
            if path.endswith("/learning/jobs/job_rl"):
                # Fallback status eventually reflects completion
                return {"status": "succeeded", "updated_at": "t-final"}
            if path.endswith("/learning/jobs/job_rl/metrics"):
                return {"points": []}
            if path.endswith("/learning/jobs/job_rl/timeline"):
                return {"events": []}
            return {}

    handler = IntegrationTestHandler()

    async def _noop_sleep(_: float) -> None:
        return None

    streamer = JobStreamer(
        base_url="https://api.example.com/api",
        api_key="sk-test",
        job_id="job_rl",
        endpoints=StreamEndpoints.rl("job_rl"),
        handlers=[handler],
        http_client=cast(AsyncHttpClient, FakeHttp()),
        sleep_fn=_noop_sleep,
        interval_seconds=0.01,
    )

    result = await streamer.stream_until_terminal()
    assert result["status"] == "succeeded"
    assert any(msg.stream_type is StreamType.EVENTS for msg in handler.messages)


def test_rich_handler_optional_dependency(monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("rich")
    from synth_ai.streaming.handlers import RichHandler

    handler = RichHandler()
    msg = StreamMessage.from_event("job", _make_event(seq=5))
    handler.handle(msg)
    handler.flush()
