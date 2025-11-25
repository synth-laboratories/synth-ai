"""Unit tests for RL SDK."""

from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import patch

import pytest
from synth_ai.sdk.api.train.rl import RLJob, RLJobConfig


class TestRLJobConfig:
    """Tests for RLJobConfig."""
    
    def test_config_validation_missing_file(self) -> None:
        """Test that missing config file raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            RLJobConfig(
                config_path=Path("/nonexistent.toml"),
                backend_url="https://api.usesynth.ai",
                api_key="test-key",
            )
    
    def test_config_validation_missing_backend(self) -> None:
        """Test that missing backend_url raises ValueError."""
        with pytest.raises(ValueError, match="backend_url is required"):
            RLJobConfig(
                config_path=Path(__file__),
                backend_url="",
                api_key="test-key",
            )
    
    def test_config_validation_missing_api_key(self) -> None:
        """Test that missing api_key raises ValueError."""
        with pytest.raises(ValueError, match="api_key is required"):
            RLJobConfig(
                config_path=Path(__file__),
                backend_url="https://api.usesynth.ai",
                api_key="",
            )
    
    def test_config_auto_resolve_task_app_key(self) -> None:
        """Test that task_app_api_key is resolved from environment."""
        with patch.dict(os.environ, {"ENVIRONMENT_API_KEY": "env-key"}):
            config = RLJobConfig(
                config_path=Path(__file__),
                backend_url="https://api.usesynth.ai",
                api_key="test-key",
            )
            assert config.task_app_api_key == "env-key"
    
    def test_config_explicit_task_app_key(self) -> None:
        """Test that explicit task_app_api_key is used."""
        config = RLJobConfig(
            config_path=Path(__file__),
            backend_url="https://api.usesynth.ai",
            api_key="test-key",
            task_app_api_key="explicit-key",
        )
        assert config.task_app_api_key == "explicit-key"
    
    def test_config_auto_resolve_task_app_url(self) -> None:
        """Test that task_app_url is resolved from environment."""
        with patch.dict(os.environ, {"TASK_APP_URL": "https://task.app"}):
            config = RLJobConfig(
                config_path=Path(__file__),
                backend_url="https://api.usesynth.ai",
                api_key="test-key",
            )
            assert config.task_app_url == "https://task.app"
    
    def test_config_explicit_task_app_url(self) -> None:
        """Test that explicit task_app_url is used."""
        config = RLJobConfig(
            config_path=Path(__file__),
            backend_url="https://api.usesynth.ai",
            api_key="test-key",
            task_app_url="https://explicit.task.app",
        )
        assert config.task_app_url == "https://explicit.task.app"


class TestRLJob:
    """Tests for RLJob."""
    
    def test_from_config_missing_api_key(self) -> None:
        """Test that from_config raises ValueError if API key missing."""
        with patch.dict(os.environ, {}, clear=True), pytest.raises(
            ValueError, match="api_key is required"
        ):
            RLJob.from_config(
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
                "ENVIRONMENT_API_KEY": "env-key",
            },
            clear=True,
        ), patch("synth_ai.core.env.get_backend_from_env") as mock_get:
            mock_get.return_value = ("https://api.usesynth.ai", "key")
            job = RLJob.from_config(config_path=Path(__file__))
            assert job.config.api_key == "test-key"
            assert job.config.backend_url == "https://custom.backend.com"
    
    def test_from_job_id(self) -> None:
        """Test creating job from existing job ID."""
        with patch.dict(
            os.environ,
            {
                "SYNTH_API_KEY": "test-key",
                "ENVIRONMENT_API_KEY": "env-key",
            },
            clear=True,
        ), patch("synth_ai.core.env.get_backend_from_env") as mock_get:
            mock_get.return_value = ("https://api.usesynth.ai", "key")
            job = RLJob.from_job_id(
                job_id="rl_1234567890",
                backend_url="https://api.usesynth.ai",
                api_key="test-key",
            )
            assert job.job_id == "rl_1234567890"
            assert job.config.backend_url == "https://api.usesynth.ai"
    
    def test_submit_requires_config(self) -> None:
        """Test that submit() requires config file for new jobs."""
        with patch.dict(os.environ, {"SYNTH_API_KEY": "test-key", "ENVIRONMENT_API_KEY": "env-key"}), patch(
            "synth_ai.core.env.get_backend_from_env"
        ) as mock_get:
            mock_get.return_value = ("https://api.usesynth.ai", "key")
            # Create job from job_id (no config file)
            job = RLJob.from_job_id(
                job_id="rl_1234567890",
                backend_url="https://api.usesynth.ai",
                api_key="test-key",
            )
            # Should fail because job already has an ID (can't submit twice)
            with pytest.raises(RuntimeError, match="already submitted"):
                job.submit()
            
            # To test "Cannot build payload", we need a job without a job_id but with /dev/null config
            job2 = RLJob(
                config=RLJobConfig(
                    config_path=Path("/dev/null"),
                    backend_url="https://api.usesynth.ai",
                    api_key="test-key",
                    task_app_api_key="env-key",
                ),
            )
            # This will fail when trying to build payload
            with pytest.raises(RuntimeError, match="Cannot build payload"):
                job2.submit()
    
    def test_submit_already_submitted(self) -> None:
        """Test that submit() raises error if already submitted."""
        job = RLJob(
            config=RLJobConfig(
                config_path=Path(__file__),
                backend_url="https://api.usesynth.ai",
                api_key="test-key",
                task_app_api_key="env-key",
            ),
            job_id="rl_existing",
        )
        with pytest.raises(RuntimeError, match="already submitted"):
            job.submit()
    
    def test_get_status_requires_submission(self) -> None:
        """Test that get_status() requires job to be submitted."""
        job = RLJob(
            config=RLJobConfig(
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
        job = RLJob(
            config=RLJobConfig(
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
        job = RLJob(
            config=RLJobConfig(
                config_path=Path(__file__),
                backend_url="https://api.usesynth.ai",
                api_key="test-key",
                task_app_api_key="env-key",
            ),
        )
        with pytest.raises(RuntimeError, match="not yet submitted"):
            job.get_results()


