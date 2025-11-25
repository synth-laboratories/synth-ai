"""Unit tests for prompt learning streaming endpoints."""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.unit

from synth_ai.sdk.streaming.streamer import StreamEndpoints


class TestPromptLearningStreamEndpoints:
    """Test streaming endpoint configuration for prompt learning jobs."""

    def test_prompt_learning_endpoints_basic(self) -> None:
        """Test basic prompt learning endpoints structure."""
        job_id = "pl_abc123"
        endpoints = StreamEndpoints.prompt_learning(job_id)

        assert endpoints.status == "/prompt-learning/online/jobs/pl_abc123"
        assert endpoints.events == "/prompt-learning/online/jobs/pl_abc123/events"
        assert endpoints.metrics == "/prompt-learning/online/jobs/pl_abc123/metrics"
        assert endpoints.timeline is None

    def test_prompt_learning_endpoints_fallbacks(self) -> None:
        """Test that prompt learning endpoints have fallbacks configured."""
        job_id = "pl_test456"
        endpoints = StreamEndpoints.prompt_learning(job_id)

        assert endpoints.status_fallbacks is not None
        assert len(endpoints.status_fallbacks) > 0
        assert any("learning" in fallback for fallback in endpoints.status_fallbacks)

        assert endpoints.event_fallbacks is not None
        assert len(endpoints.event_fallbacks) > 0
        assert any("learning" in fallback for fallback in endpoints.event_fallbacks)

    def test_prompt_learning_vs_rl_endpoints(self) -> None:
        """Test that prompt learning endpoints differ from RL endpoints."""
        job_id = "test_job"
        pl_endpoints = StreamEndpoints.prompt_learning(job_id)
        rl_endpoints = StreamEndpoints.rl(job_id)

        # Prompt learning uses different base path
        assert pl_endpoints.status != rl_endpoints.status
        assert "/prompt-learning/" in pl_endpoints.status
        assert "/rl/" in rl_endpoints.status

    def test_prompt_learning_job_id_with_prefix(self) -> None:
        """Test endpoints with various job ID formats."""
        # Standard prompt learning job ID with pl_ prefix
        endpoints = StreamEndpoints.prompt_learning("pl_123abc456def")
        assert "pl_123abc456def" in endpoints.status
        assert "pl_123abc456def" in endpoints.events

        # Job ID without prefix (should still work)
        endpoints2 = StreamEndpoints.prompt_learning("abc123")
        assert "abc123" in endpoints2.status
        assert "abc123" in endpoints2.events

    def test_prompt_learning_events_endpoint(self) -> None:
        """Test events endpoint path construction."""
        job_id = "pl_gepa_test"
        endpoints = StreamEndpoints.prompt_learning(job_id)

        expected_events = f"/prompt-learning/online/jobs/{job_id}/events"
        assert endpoints.events == expected_events

    def test_prompt_learning_has_metrics_endpoint(self) -> None:
        """Test that metrics endpoint exists for prompt learning."""
        endpoints = StreamEndpoints.prompt_learning("pl_mipro_job")

        # Prompt learning now has a metrics endpoint
        assert endpoints.metrics is not None
        assert endpoints.metrics == "/prompt-learning/online/jobs/pl_mipro_job/metrics"

    def test_prompt_learning_no_timeline_endpoint(self) -> None:
        """Test that timeline endpoint is None for prompt learning."""
        endpoints = StreamEndpoints.prompt_learning("pl_test")

        # Prompt learning may not have a timeline endpoint
        assert endpoints.timeline is None


class TestStreamEndpointsFactory:
    """Test StreamEndpoints factory methods for different job types."""

    def test_all_job_type_factories_exist(self) -> None:
        """Test that factory methods exist for all job types."""
        job_id = "test"

        # All factory methods should exist and return StreamEndpoints
        assert hasattr(StreamEndpoints, "rl")
        assert hasattr(StreamEndpoints, "learning")
        assert hasattr(StreamEndpoints, "prompt_learning")

        rl_endpoints = StreamEndpoints.rl(job_id)
        learning_endpoints = StreamEndpoints.learning(job_id)
        pl_endpoints = StreamEndpoints.prompt_learning(job_id)

        assert isinstance(rl_endpoints, StreamEndpoints)
        assert isinstance(learning_endpoints, StreamEndpoints)
        assert isinstance(pl_endpoints, StreamEndpoints)

    def test_factory_methods_return_different_endpoints(self) -> None:
        """Test that different factory methods return different endpoint configurations."""
        job_id = "test"

        rl = StreamEndpoints.rl(job_id)
        learning = StreamEndpoints.learning(job_id)
        pl = StreamEndpoints.prompt_learning(job_id)

        # Each should have different status endpoints
        endpoints = {rl.status, learning.status, pl.status}
        assert len(endpoints) == 3, "All job types should have unique status endpoints"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

