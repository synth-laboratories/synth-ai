"""Unit tests for prompt learning stream configuration."""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.unit

from synth_ai.sdk.api.train.cli import _DEFAULT_PROMPT_LEARNING_HIDDEN_EVENTS
from synth_ai.sdk.streaming import CLIHandler, LossCurveHandler, StreamConfig, StreamEndpoints, StreamType


class TestPromptLearningStreamConfigCLI:
    """Test stream configuration for prompt learning in CLI mode."""

    def test_cli_config_enables_metrics(self) -> None:
        """Test that CLI config enables metrics streaming."""
        config = StreamConfig(
            enabled_streams={StreamType.STATUS, StreamType.EVENTS, StreamType.METRICS},
            metric_names={"gepa.transformation.mean_score"},
        )
        
        assert StreamType.METRICS in config.enabled_streams
        assert "gepa.transformation.mean_score" in config.metric_names

    def test_cli_config_handlers_hide_policy_tokens(self) -> None:
        """Test that CLI handlers hide policy tokens events."""
        handler = CLIHandler(
            hidden_event_types=_DEFAULT_PROMPT_LEARNING_HIDDEN_EVENTS,
            hidden_event_substrings={"modal", "hatchet"},
        )
        
        assert "prompt.learning.policy.tokens" in handler._hidden_event_types

    def test_cli_config_allows_eval_summary(self) -> None:
        """Test that eval summary events are not hidden."""
        handler = CLIHandler(
            hidden_event_types=_DEFAULT_PROMPT_LEARNING_HIDDEN_EVENTS,
        )
        
        assert "prompt.learning.eval.summary" not in handler._hidden_event_types

    def test_cli_config_allows_progress(self) -> None:
        """Test that progress events are not hidden."""
        handler = CLIHandler(
            hidden_event_types=_DEFAULT_PROMPT_LEARNING_HIDDEN_EVENTS,
        )
        
        assert "prompt.learning.progress" not in handler._hidden_event_types


class TestPromptLearningStreamConfigChart:
    """Test stream configuration for prompt learning in chart mode."""

    def test_chart_config_enables_metrics(self) -> None:
        """Test that chart config enables metrics streaming."""
        config = StreamConfig(
            enabled_streams={StreamType.STATUS, StreamType.EVENTS, StreamType.METRICS},
            event_types={
                "prompt.learning.progress",
                "prompt.learning.gepa.start",
                "prompt.learning.gepa.complete",
            },
            metric_names={"gepa.transformation.mean_score"},
        )
        
        assert StreamType.METRICS in config.enabled_streams
        assert "gepa.transformation.mean_score" in config.metric_names

    def test_chart_config_uses_loss_curve_handler(self) -> None:
        """Test that chart mode uses LossCurveHandler."""
        handlers = [LossCurveHandler()]
        
        assert len(handlers) == 1
        assert isinstance(handlers[0], LossCurveHandler)

    def test_chart_config_event_types(self) -> None:
        """Test that chart config includes correct event types."""
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


class TestPromptLearningEndpoints:
    """Test prompt learning stream endpoints."""

    def test_prompt_learning_endpoints_includes_metrics(self) -> None:
        """Test that prompt learning endpoints include metrics."""
        endpoints = StreamEndpoints.prompt_learning("pl_test123")
        
        assert endpoints.metrics is not None
        assert endpoints.metrics == "/prompt-learning/online/jobs/pl_test123/metrics"

    def test_prompt_learning_endpoints_status(self) -> None:
        """Test prompt learning status endpoint."""
        endpoints = StreamEndpoints.prompt_learning("pl_test123")
        
        assert endpoints.status == "/prompt-learning/online/jobs/pl_test123"

    def test_prompt_learning_endpoints_events(self) -> None:
        """Test prompt learning events endpoint."""
        endpoints = StreamEndpoints.prompt_learning("pl_test123")
        
        assert endpoints.events == "/prompt-learning/online/jobs/pl_test123/events"

    def test_prompt_learning_endpoints_no_timeline(self) -> None:
        """Test that prompt learning does not have timeline endpoint."""
        endpoints = StreamEndpoints.prompt_learning("pl_test123")
        
        assert endpoints.timeline is None


class TestStreamConfigFiltering:
    """Test stream config filtering for prompt learning."""

    def test_config_filters_correct_metrics(self) -> None:
        """Test that config filters metrics correctly."""
        config = StreamConfig(
            enabled_streams={StreamType.METRICS},
            metric_names={"gepa.transformation.mean_score"},
        )
        
        assert config.should_include_metric({
            "name": "gepa.transformation.mean_score",
            "step": 1,
            "value": 0.5,
        })
        
        assert not config.should_include_metric({
            "name": "train.loss",
            "step": 1,
            "value": 0.3,
        })

    def test_config_filters_correct_events(self) -> None:
        """Test that config filters events correctly."""
        config = StreamConfig(
            enabled_streams={StreamType.EVENTS},
            event_types={"prompt.learning.progress", "prompt.learning.gepa.start"},
        )
        
        assert config.should_include_event({
            "type": "prompt.learning.progress",
            "seq": 1,
        })
        
        assert config.should_include_event({
            "type": "prompt.learning.gepa.start",
            "seq": 2,
        })
        
        assert not config.should_include_event({
            "type": "prompt.learning.policy.tokens",
            "seq": 3,
        })


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


