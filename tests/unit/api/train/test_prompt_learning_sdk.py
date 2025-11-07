"""Unit tests for Prompt Learning SDK."""

from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import patch

import pytest
from synth_ai.api.train.prompt_learning import (
    PromptLearningJob,
    PromptLearningJobConfig,
    PromptLearningJobPoller,
)


class TestPromptLearningJobConfig:
    """Tests for PromptLearningJobConfig."""
    
    def test_config_validation_missing_file(self) -> None:
        """Test that missing config file raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            PromptLearningJobConfig(
                config_path=Path("/nonexistent.toml"),
                backend_url="https://api.usesynth.ai",
                api_key="test-key",
            )
    
    def test_config_validation_missing_backend(self) -> None:
        """Test that missing backend_url raises ValueError."""
        with pytest.raises(ValueError, match="backend_url is required"):
            PromptLearningJobConfig(
                config_path=Path(__file__),  # Use this file as dummy
                backend_url="",
                api_key="test-key",
            )
    
    def test_config_validation_missing_api_key(self) -> None:
        """Test that missing api_key raises ValueError."""
        with pytest.raises(ValueError, match="api_key is required"):
            PromptLearningJobConfig(
                config_path=Path(__file__),
                backend_url="https://api.usesynth.ai",
                api_key="",
            )
    
    def test_config_auto_resolve_task_app_key(self) -> None:
        """Test that task_app_api_key is resolved from environment."""
        with patch.dict(os.environ, {"ENVIRONMENT_API_KEY": "env-key"}):
            config = PromptLearningJobConfig(
                config_path=Path(__file__),
                backend_url="https://api.usesynth.ai",
                api_key="test-key",
            )
            assert config.task_app_api_key == "env-key"
    
    def test_config_explicit_task_app_key(self) -> None:
        """Test that explicit task_app_api_key is used."""
        config = PromptLearningJobConfig(
            config_path=Path(__file__),
            backend_url="https://api.usesynth.ai",
            api_key="test-key",
            task_app_api_key="explicit-key",
        )
        assert config.task_app_api_key == "explicit-key"


class TestPromptLearningJob:
    """Tests for PromptLearningJob."""
    
    def test_from_config_missing_api_key(self) -> None:
        """Test that from_config raises ValueError if API key missing."""
        with patch.dict(os.environ, {}, clear=True), pytest.raises(
            ValueError, match="api_key is required"
        ):
            PromptLearningJob.from_config(
                config_path=Path(__file__),
                backend_url="https://api.usesynth.ai",
            )
    
    def test_from_config_resolves_env(self) -> None:
        """Test that from_config resolves backend and API key from env."""
        with patch.dict(
            os.environ,
            {
                "SYNTH_API_KEY": "test-key",
                "BACKEND_BASE_URL": "https://custom.backend.com",
            },
        ), patch("synth_ai.config.base_url.get_backend_from_env") as mock_get:
            mock_get.return_value = ("https://api.usesynth.ai", "key")
            job = PromptLearningJob.from_config(config_path=Path(__file__))
            assert job.config.api_key == "test-key"
            # Backend URL is stored as-is (normalization happens when used)
            assert job.config.backend_url == "https://custom.backend.com"
    
    def test_from_job_id(self) -> None:
        """Test creating job from existing job ID."""
        with patch.dict(os.environ, {"SYNTH_API_KEY": "test-key"}), patch(
            "synth_ai.config.base_url.get_backend_from_env"
        ) as mock_get:
            mock_get.return_value = ("https://api.usesynth.ai", "key")
            job = PromptLearningJob.from_job_id(
                job_id="pl_1234567890",
                backend_url="https://api.usesynth.ai",
                api_key="test-key",
            )
            assert job.job_id == "pl_1234567890"
            assert job.config.backend_url == "https://api.usesynth.ai"
    
    def test_submit_requires_config(self) -> None:
        """Test that submit() requires config file for new jobs."""
        with patch.dict(
            os.environ, {"SYNTH_API_KEY": "test-key", "ENVIRONMENT_API_KEY": "env-key"}
        ), patch("synth_ai.config.base_url.get_backend_from_env") as mock_get:
            mock_get.return_value = ("https://api.usesynth.ai", "key")
            # Create job from job_id (no config file)
            # This creates a job with a job_id, so submit() will raise "already submitted"
            job = PromptLearningJob.from_job_id(
                job_id="pl_1234567890",
                backend_url="https://api.usesynth.ai",
                api_key="test-key",
            )
            # Should fail because job already has an ID (can't submit twice)
            with pytest.raises(RuntimeError, match="already submitted"):
                job.submit()
            
            # To test "Cannot build payload", we need a job without a job_id but with /dev/null config
            # The check in _build_payload looks for "/dev/null" name specifically
            job2 = PromptLearningJob(
                config=PromptLearningJobConfig(
                    config_path=Path("/dev/null"),  # /dev/null exists but triggers "Cannot build payload"
                    backend_url="https://api.usesynth.ai",
                    api_key="test-key",
                    task_app_api_key="env-key",
                ),
            )
            # This will fail when trying to build payload because we check for "/dev/null" name
            # OR because /dev/null isn't valid TOML (which raises ClickException from validator)
            import click
            with pytest.raises((RuntimeError, click.ClickException), match="Cannot build payload|Invalid prompt learning config"):
                job2.submit()
    
    def test_submit_already_submitted(self) -> None:
        """Test that submit() raises error if already submitted."""
        job = PromptLearningJob(
            config=PromptLearningJobConfig(
                config_path=Path(__file__),
                backend_url="https://api.usesynth.ai",
                api_key="test-key",
                task_app_api_key="env-key",
            ),
            job_id="pl_existing",
        )
        with pytest.raises(RuntimeError, match="already submitted"):
            job.submit()
    
    def test_get_status_requires_submission(self) -> None:
        """Test that get_status() requires job to be submitted."""
        job = PromptLearningJob(
            config=PromptLearningJobConfig(
                config_path=Path(__file__),
                backend_url="https://api.usesynth.ai",
                api_key="test-key",
                task_app_api_key="env-key",
            ),
        )
        with pytest.raises(RuntimeError, match="not yet submitted"):
            job.get_status()
    
    def test_poll_until_complete_requires_submission(self) -> None:
        """Test that poll_until_complete() requires job to be submitted."""
        job = PromptLearningJob(
            config=PromptLearningJobConfig(
                config_path=Path(__file__),
                backend_url="https://api.usesynth.ai",
                api_key="test-key",
                task_app_api_key="env-key",
            ),
        )
        with pytest.raises(RuntimeError, match="not yet submitted"):
            job.poll_until_complete()
    
    def test_get_results_requires_submission(self) -> None:
        """Test that get_results() requires job to be submitted."""
        job = PromptLearningJob(
            config=PromptLearningJobConfig(
                config_path=Path(__file__),
                backend_url="https://api.usesynth.ai",
                api_key="test-key",
                task_app_api_key="env-key",
            ),
        )
        with pytest.raises(RuntimeError, match="not yet submitted"):
            job.get_results()


class TestPromptLearningJobPoller:
    """Tests for PromptLearningJobPoller."""
    
    def test_poll_job_path(self) -> None:
        """Test that poll_job uses correct endpoint path."""
        poller = PromptLearningJobPoller(
            base_url="https://api.usesynth.ai",
            api_key="test-key",
        )
        # Check that the path is correct (we can't easily test the full poll without mocking)
        # ensure_api_base normalizes the URL in __init__
        assert poller.base_url.endswith("/api")
        assert poller.api_key == "test-key"

