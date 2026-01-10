import importlib

import pytest

from synth_ai.sdk.task.trace_correlation_helpers import extract_trace_correlation_id


def _load_grpo_module():
    from synth_ai.sdk.task.apps import registry

    registry.clear()
    return importlib.import_module("examples.task_apps.crafter.task_app.grpo_crafter")


def _load_utils_module():
    return importlib.import_module("examples.task_apps.crafter.task_app.synth_envs_hosted.utils")


def test_extract_trace_correlation_id_from_modal_url():
    utils_mod = _load_utils_module()
    extractor = getattr(utils_mod, "extract_trace_correlation_id")
    url = (
        "https://synth-labs--crafter.modal.run/v1/chat/completions"
        "?cid=trace_run-1234abcd&foo=bar"
    )

    assert extractor(url, mode="rl") == "trace_run-1234abcd"


@pytest.mark.parametrize("mode", ["rl"])
def test_resolve_trace_correlation_id_uses_inference_url(mode):
    grpo_mod = _load_grpo_module()
    resolver = getattr(grpo_mod, "_resolve_trace_correlation_id")
    policy_cfg = {
        "model": "Qwen/Qwen3-4B",
        "inference_url": (
            "https://ta-01k8skyc1cgtrtep1avyyqhs3s-8000.modal.host"
            "?cid=trace_run-e2f1b3da"
        ),
    }

    assert resolver(policy_cfg, mode=mode) == "trace_run-e2f1b3da"


class TestExtractTraceCorrelationIdPathBased:
    """Test path-based correlation ID extraction (OpenAI SDK compatible format)."""

    def test_extract_from_path_basic(self):
        """Test extraction from path: /v1/{trial_id}/{correlation_id}/chat/completions"""
        policy_cfg = {}
        inference_url = "http://localhost:8115/v1/baseline-0-abc123/trace_test_789/chat/completions"
        result = extract_trace_correlation_id(policy_cfg, inference_url, mode="rl")
        assert result == "trace_test_789"

    def test_extract_from_path_full_interceptor_url(self):
        """Test extraction from full interceptor URL path."""
        policy_cfg = {}
        inference_url = "https://agent-learning.onrender.com/api/interceptor/v1/baseline-0-def456/trace_validation-0-xyz123/chat/completions"
        result = extract_trace_correlation_id(policy_cfg, inference_url, mode="rl")
        assert result == "trace_validation-0-xyz123"

    def test_extract_path_takes_precedence_over_query(self):
        """Test that path-based extraction takes precedence over query param."""
        policy_cfg = {}
        inference_url = "http://localhost:8115/v1/baseline-0-abc123/trace_path_abc/chat/completions?cid=trace_query_xyz"
        result = extract_trace_correlation_id(policy_cfg, inference_url, mode="rl")
        assert result == "trace_path_abc"

    def test_extract_from_query_param_fallback(self):
        """Test that query param is used when path doesn't have correlation ID."""
        policy_cfg = {}
        inference_url = "http://localhost:8115/v1/baseline-0-abc123/chat/completions?cid=trace_query_only"
        result = extract_trace_correlation_id(policy_cfg, inference_url, mode="rl")
        assert result == "trace_query_only"

    def test_policy_config_takes_precedence(self):
        """Test that policy_config trace_correlation_id takes precedence over URL."""
        policy_cfg = {"trace_correlation_id": "trace_from_config"}
        inference_url = "http://localhost:8115/v1/baseline-0-abc123/trace_from_path/chat/completions"
        result = extract_trace_correlation_id(policy_cfg, inference_url, mode="rl")
        assert result == "trace_from_config"
