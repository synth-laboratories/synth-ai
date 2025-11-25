"""Unit tests for prompt learning metrics streaming."""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.unit

from synth_ai.sdk.streaming import CLIHandler, StreamMessage, StreamType


class TestCLIHandlerMetricsDisplay:
    """Test CLIHandler metrics display for prompt learning."""

    def test_cli_handler_displays_metrics(self, capsys: pytest.CaptureFixture[str]) -> None:
        """Test that CLIHandler displays metrics correctly."""
        handler = CLIHandler()
        message = StreamMessage.from_metric(
            "pl_test123",
            {
                "name": "gepa.transformation.mean_score",
                "value": 0.5714,
                "step": 5,
                "created_at": "2024-11-04T20:30:00Z",
            },
        )
        handler.handle(message)
        captured = capsys.readouterr()
        assert "[metric]" in captured.out
        assert "gepa.transformation.mean_score" in captured.out
        assert "0.5714" in captured.out
        assert "step=5" in captured.out

    def test_cli_handler_metrics_with_n_value(self, capsys: pytest.CaptureFixture[str]) -> None:
        """Test metrics display includes n value from data field."""
        handler = CLIHandler()
        message = StreamMessage.from_metric(
            "pl_test123",
            {
                "name": "gepa.transformation.mean_score",
                "value": 0.4,
                "step": 3,
                "data": {"n": 15, "kind": "variation", "index": 2},
                "created_at": "2024-11-04T20:30:00Z",
            },
        )
        handler.handle(message)
        captured = capsys.readouterr()
        assert "n=15" in captured.out
        assert "0.4000" in captured.out

    def test_cli_handler_metrics_without_step(self, capsys: pytest.CaptureFixture[str]) -> None:
        """Test metrics display when step is not provided."""
        handler = CLIHandler()
        message = StreamMessage.from_metric(
            "pl_test123",
            {
                "name": "gepa.transformation.mean_score",
                "value": 0.6667,
                "created_at": "2024-11-04T20:30:00Z",
            },
        )
        handler.handle(message)
        captured = capsys.readouterr()
        assert "0.6667" in captured.out
        assert "step=" not in captured.out

    def test_cli_handler_metrics_non_numeric_value(self, capsys: pytest.CaptureFixture[str]) -> None:
        """Test metrics display with non-numeric values."""
        handler = CLIHandler()
        message = StreamMessage.from_metric(
            "pl_test123",
            {
                "name": "gepa.status",
                "value": "running",
                "created_at": "2024-11-04T20:30:00Z",
            },
        )
        handler.handle(message)
        captured = capsys.readouterr()
        assert "gepa.status=running" in captured.out

    def test_cli_handler_metrics_int_value(self, capsys: pytest.CaptureFixture[str]) -> None:
        """Test metrics display with integer values."""
        handler = CLIHandler()
        message = StreamMessage.from_metric(
            "pl_test123",
            {
                "name": "gepa.transformation.count",
                "value": 42,
                "step": 10,
                "created_at": "2024-11-04T20:30:00Z",
            },
        )
        handler.handle(message)
        captured = capsys.readouterr()
        assert "42.0000" in captured.out or "42" in captured.out

    def test_cli_handler_metrics_empty_data(self, capsys: pytest.CaptureFixture[str]) -> None:
        """Test metrics display with empty data field."""
        handler = CLIHandler()
        message = StreamMessage.from_metric(
            "pl_test123",
            {
                "name": "gepa.transformation.mean_score",
                "value": 0.5,
                "step": 1,
                "data": {},
                "created_at": "2024-11-04T20:30:00Z",
            },
        )
        handler.handle(message)
        captured = capsys.readouterr()
        assert "0.5000" in captured.out
        assert "n=" not in captured.out


class TestCLIHandlerEventFiltering:
    """Test CLIHandler event filtering for prompt learning."""

    def test_cli_handler_hides_policy_tokens_event(self, capsys: pytest.CaptureFixture[str]) -> None:
        """Test that policy.tokens events are hidden by default for prompt learning."""
        handler = CLIHandler(hidden_event_types={"prompt.learning.policy.tokens"})
        message = StreamMessage.from_event(
            "pl_test123",
            {
                "seq": 10,
                "type": "prompt.learning.policy.tokens",
                "level": "info",
                "message": "policy tokens: prompt=2240 completion=2446 total=4686",
                "created_at": "2024-11-04T20:30:00Z",
            },
        )
        handler.handle(message)
        captured = capsys.readouterr()
        assert "policy.tokens" not in captured.out
        assert captured.out.strip() == ""

    def test_cli_handler_shows_eval_summary(self, capsys: pytest.CaptureFixture[str]) -> None:
        """Test that eval.summary events are shown."""
        handler = CLIHandler(hidden_event_types={"prompt.learning.policy.tokens"})
        message = StreamMessage.from_event(
            "pl_test123",
            {
                "seq": 15,
                "type": "prompt.learning.eval.summary",
                "level": "info",
                "message": "mean=0.133 (N=10/15) min=0.0 max=1.0",
                "created_at": "2024-11-04T20:30:00Z",
            },
        )
        handler.handle(message)
        captured = capsys.readouterr()
        assert "eval.summary" in captured.out
        assert "mean=0.133" in captured.out

    def test_cli_handler_shows_progress(self, capsys: pytest.CaptureFixture[str]) -> None:
        """Test that progress events are shown."""
        handler = CLIHandler(hidden_event_types={"prompt.learning.policy.tokens"})
        message = StreamMessage.from_event(
            "pl_test123",
            {
                "seq": 14,
                "type": "prompt.learning.progress",
                "level": "info",
                "message": "3% complete; rollouts=30/1000; tokens=0.0M/NA; elapsed=49s, eta=26.8min",
                "created_at": "2024-11-04T20:30:00Z",
            },
        )
        handler.handle(message)
        captured = capsys.readouterr()
        assert "progress" in captured.out
        assert "3% complete" in captured.out
        assert "eta=26.8min" in captured.out

    def test_cli_handler_multiple_hidden_events(self, capsys: pytest.CaptureFixture[str]) -> None:
        """Test that multiple event types can be hidden."""
        handler = CLIHandler(
            hidden_event_types={
                "prompt.learning.policy.tokens",
                "prompt.learning.worker.alive",
            }
        )
        
        # Test hidden events
        for event_type in ["prompt.learning.policy.tokens", "prompt.learning.worker.alive"]:
            message = StreamMessage.from_event(
                "pl_test123",
                {
                    "seq": 1,
                    "type": event_type,
                    "level": "info",
                    "message": "test",
                    "created_at": "2024-11-04T20:30:00Z",
                },
            )
            handler.handle(message)
            captured = capsys.readouterr()
            assert captured.out.strip() == ""

        # Test visible event
        message = StreamMessage.from_event(
            "pl_test123",
            {
                "seq": 2,
                "type": "prompt.learning.gepa.start",
                "level": "info",
                "message": "Starting GEPA optimisation",
                "created_at": "2024-11-04T20:30:00Z",
            },
        )
        handler.handle(message)
        captured = capsys.readouterr()
        assert "gepa.start" in captured.out


class TestPromptLearningStreamConfig:
    """Test StreamConfig for prompt learning jobs."""

    def test_prompt_learning_stream_config_metrics_enabled(self) -> None:
        """Test that prompt learning stream config enables metrics."""
        from synth_ai.sdk.streaming import StreamConfig, StreamType

        config = StreamConfig(
            enabled_streams={StreamType.STATUS, StreamType.EVENTS, StreamType.METRICS},
            metric_names={"gepa.transformation.mean_score"},
        )

        assert StreamType.METRICS in config.enabled_streams
        assert "gepa.transformation.mean_score" in config.metric_names

    def test_prompt_learning_stream_config_event_types(self) -> None:
        """Test that prompt learning stream config includes correct event types."""
        from synth_ai.sdk.streaming import StreamConfig, StreamType

        config = StreamConfig(
            enabled_streams={StreamType.STATUS, StreamType.EVENTS, StreamType.METRICS},
            event_types={
                "prompt.learning.progress",
                "prompt.learning.gepa.start",
                "prompt.learning.gepa.complete",
            },
            metric_names={"gepa.transformation.mean_score"},
        )

        assert "prompt.learning.progress" in config.event_types
        assert "prompt.learning.gepa.start" in config.event_types
        assert "prompt.learning.gepa.complete" in config.event_types

    def test_prompt_learning_stream_config_filters_metrics(self) -> None:
        """Test that stream config correctly filters metrics."""
        from synth_ai.sdk.streaming import StreamConfig, StreamType

        config = StreamConfig(
            enabled_streams={StreamType.STATUS, StreamType.EVENTS, StreamType.METRICS},
            metric_names={"gepa.transformation.mean_score"},
        )

        assert config.should_include_metric({"name": "gepa.transformation.mean_score", "step": 1})
        assert not config.should_include_metric({"name": "train.loss", "step": 1})
        assert not config.should_include_metric({"name": "val.accuracy", "step": 1})


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


