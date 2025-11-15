"""Integration tests for JobPoller URL normalization to prevent double /api/api paths."""

from unittest.mock import MagicMock, patch

import pytest

from synth_ai.api.train.pollers import PromptLearningJobPoller


class TestPollersURLNormalization:
    """Test that pollers correctly normalize URLs to avoid double /api/api paths."""

    @patch("synth_ai.api.train.pollers.http_get")
    @patch("synth_ai.api.train.pollers.sleep")
    def test_prompt_learning_poller_no_double_api(
        self, mock_sleep: MagicMock, mock_http_get: MagicMock
    ) -> None:
        """Test that PromptLearningJobPoller doesn't create double /api/api URLs.
        
        When base_url is "http://localhost:8000", ensure_api_base() adds /api,
        making it "http://localhost:8000/api". Then poll_job() uses path "/api/prompt-learning/...",
        which should result in "http://localhost:8000/api/prompt-learning/..." (not /api/api/...).
        """
        # Mock http_get to return a successful response immediately
        mock_response = MagicMock()
        mock_response.json.return_value = {"status": "completed", "best_score": 0.95}
        mock_response.headers = {"content-type": "application/json"}
        mock_http_get.return_value = mock_response
        
        # Create poller with base_url that doesn't have /api
        poller = PromptLearningJobPoller(
            base_url="http://localhost:8000",
            api_key="test-key",
            timeout=1.0,  # Short timeout for test
            interval=0.1,  # Short interval for test
        )
        
        # Call poll_job - this should NOT create /api/api/... URLs
        job_id = "pl_test123"
        outcome = poller.poll_job(job_id)
        
        # Verify the outcome
        assert outcome.status == "completed"
        assert outcome.payload["best_score"] == 0.95
        
        # Verify http_get was called
        assert mock_http_get.called
        
        # Get the URL that was actually called
        call_args = mock_http_get.call_args
        actual_url = call_args[0][0]  # First positional argument
        
        # CRITICAL: Verify NO double /api/api in the URL
        assert "/api/api" not in actual_url, f"Found double /api/api in URL: {actual_url}"
        
        # Verify the URL is correct
        expected_url = "http://localhost:8000/api/prompt-learning/online/jobs/pl_test123"
        assert actual_url == expected_url, f"Expected {expected_url}, got {actual_url}"

    @patch("synth_ai.api.train.pollers.http_get")
    @patch("synth_ai.api.train.pollers.sleep")
    def test_prompt_learning_poller_with_existing_api(
        self, mock_sleep: MagicMock, mock_http_get: MagicMock
    ) -> None:
        """Test that PromptLearningJobPoller handles base_url that already has /api.
        
        When base_url is "http://localhost:8000/api", ensure_api_base() keeps it as-is.
        Then poll_job() uses path "/api/prompt-learning/...", which should be normalized
        to "http://localhost:8000/api/prompt-learning/..." (not /api/api/...).
        """
        # Mock http_get to return a successful response immediately
        mock_response = MagicMock()
        mock_response.json.return_value = {"status": "completed", "best_score": 0.95}
        mock_response.headers = {"content-type": "application/json"}
        mock_http_get.return_value = mock_response
        
        # Create poller with base_url that already has /api
        poller = PromptLearningJobPoller(
            base_url="http://localhost:8000/api",  # Already has /api
            api_key="test-key",
            timeout=1.0,
            interval=0.1,
        )
        
        # Call poll_job
        job_id = "pl_test456"
        outcome = poller.poll_job(job_id)
        
        # Verify the outcome
        assert outcome.status == "completed"
        
        # Get the URL that was actually called
        call_args = mock_http_get.call_args
        actual_url = call_args[0][0]
        
        # CRITICAL: Verify NO double /api/api in the URL
        assert "/api/api" not in actual_url, f"Found double /api/api in URL: {actual_url}"
        
        # Verify the URL is correct (should still be normalized)
        expected_url = "http://localhost:8000/api/prompt-learning/online/jobs/pl_test456"
        assert actual_url == expected_url, f"Expected {expected_url}, got {actual_url}"

    @patch("synth_ai.api.train.pollers.http_get")
    @patch("synth_ai.api.train.pollers.sleep")
    def test_prompt_learning_poller_path_without_api_prefix(
        self, mock_sleep: MagicMock, mock_http_get: MagicMock
    ) -> None:
        """Test that paths without /api prefix still work correctly."""
        # Mock http_get
        mock_response = MagicMock()
        mock_response.json.return_value = {"status": "running"}
        mock_response.headers = {"content-type": "application/json"}
        mock_http_get.return_value = mock_response
        
        poller = PromptLearningJobPoller(
            base_url="http://localhost:8000",
            api_key="test-key",
            timeout=0.5,  # Will timeout quickly
            interval=0.1,
        )
        
        # Manually call poll() with a path that doesn't start with /api
        # (This shouldn't happen in practice, but test the normalization logic)
        outcome = poller.poll("prompt-learning/online/jobs/pl_test789")
        
        # Get the URL that was called
        call_args = mock_http_get.call_args
        actual_url = call_args[0][0]
        
        # Should still work correctly (base_url has /api, path doesn't)
        expected_url = "http://localhost:8000/api/prompt-learning/online/jobs/pl_test789"
        assert actual_url == expected_url, f"Expected {expected_url}, got {actual_url}"
        assert "/api/api" not in actual_url

