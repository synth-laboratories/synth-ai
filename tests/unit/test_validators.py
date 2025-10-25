"""Tests for synth_ai.task.validators module."""

import pytest

from synth_ai.task.validators import normalize_inference_url


class TestNormalizeInferenceUrl:
    """Test suite for normalize_inference_url function."""
    
    def test_already_complete_url(self):
        """URLs that are already complete should be returned unchanged."""
        url = "https://api.openai.com/v1/chat/completions"
        assert normalize_inference_url(url) == url
        
    def test_url_with_groq_path(self):
        """Groq-style URLs with /openai/v1/chat/completions should be unchanged."""
        url = "https://api.groq.com/openai/v1/chat/completions"
        assert normalize_inference_url(url) == url
    
    def test_base_domain_only(self):
        """Base domain should get full path appended."""
        assert normalize_inference_url("https://api.groq.com") == \
            "https://api.groq.com/v1/chat/completions"
            
    def test_domain_with_v1_path(self):
        """Domain with /v1 should only append /chat/completions."""
        assert normalize_inference_url("https://api.openai.com/v1") == \
            "https://api.openai.com/v1/chat/completions"
            
    def test_domain_with_chat_path(self):
        """Domain with /chat should only append /completions."""
        assert normalize_inference_url("https://example.com/chat") == \
            "https://example.com/chat/completions"
    
    def test_url_with_query_params(self):
        """URL with query parameters should preserve them correctly.
        
        This is the critical test case that was causing the 404 errors in RL training.
        The query parameter should be preserved, not have the path appended after it.
        """
        # Base URL with tracing query parameter
        url = "https://modal.host?cid=trace_run-123"
        result = normalize_inference_url(url)
        assert result == "https://modal.host/v1/chat/completions?cid=trace_run-123"
        
        # URL with multiple query parameters
        url = "https://modal.host?cid=trace_run-123&debug=true"
        result = normalize_inference_url(url)
        assert result == "https://modal.host/v1/chat/completions?cid=trace_run-123&debug=true"
        
    def test_url_with_path_and_query_params(self):
        """URL with existing path and query parameters."""
        url = "https://api.groq.com/v1?cid=trace_456"
        result = normalize_inference_url(url)
        assert result == "https://api.groq.com/v1/chat/completions?cid=trace_456"
        
    def test_url_with_trailing_slash(self):
        """URLs with trailing slashes should be handled correctly."""
        assert normalize_inference_url("https://api.openai.com/v1/") == \
            "https://api.openai.com/v1/chat/completions"
            
    def test_url_with_custom_path(self):
        """URLs with custom paths should get full path appended."""
        assert normalize_inference_url("https://example.com/custom/path") == \
            "https://example.com/custom/path/v1/chat/completions"
            
    def test_none_url_uses_default(self):
        """None URL should use the default."""
        default = "https://api.openai.com/v1/chat/completions"
        result = normalize_inference_url(None)
        assert result == default
        
    def test_empty_url_uses_default(self):
        """Empty string should use the default."""
        default = "https://api.openai.com/v1/chat/completions"
        result = normalize_inference_url("")
        assert result == default
        
    def test_custom_default(self):
        """Custom default should be used when URL is None/empty."""
        custom_default = "https://custom.api.com/v1/chat/completions"
        assert normalize_inference_url(None, default=custom_default) == custom_default
        assert normalize_inference_url("", default=custom_default) == custom_default
        
    def test_url_with_port(self):
        """URLs with port numbers should be handled correctly."""
        assert normalize_inference_url("https://localhost:8000") == \
            "https://localhost:8000/v1/chat/completions"
            
    def test_url_with_port_and_query(self):
        """URLs with port and query parameters."""
        url = "https://localhost:8000?cid=test"
        result = normalize_inference_url(url)
        assert result == "https://localhost:8000/v1/chat/completions?cid=test"
        
    def test_url_with_fragment(self):
        """URLs with fragments (though uncommon for APIs) should preserve them."""
        url = "https://api.example.com/v1#section"
        result = normalize_inference_url(url)
        assert result == "https://api.example.com/v1/chat/completions#section"
        
    def test_whitespace_handling(self):
        """URLs with leading/trailing whitespace should be trimmed."""
        url = "  https://api.groq.com  "
        result = normalize_inference_url(url)
        assert result == "https://api.groq.com/v1/chat/completions"
        
    def test_real_world_modal_url(self):
        """Test with actual Modal.com style URL from RL training."""
        # This is the actual format the backend uses for RL training
        url = "https://ta-01k8bqnc9hp7gk8s8f4m8jt057-8000.wo-q4g3xwhug6md9nd914kyjiia4.w.modal.host?cid=trace_run-d2de4d76-cbd5-4679-9ebe-5186a8c5998e"
        result = normalize_inference_url(url)
        expected = "https://ta-01k8bqnc9hp7gk8s8f4m8jt057-8000.wo-q4g3xwhug6md9nd914kyjiia4.w.modal.host/v1/chat/completions?cid=trace_run-d2de4d76-cbd5-4679-9ebe-5186a8c5998e"
        assert result == expected
        
    def test_completions_only_endpoint(self):
        """URLs ending in /chat/completions should be preserved."""
        url = "https://api.example.com/chat/completions"
        assert normalize_inference_url(url) == url

