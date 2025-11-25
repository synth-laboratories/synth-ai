"""Unit tests for SFT SDK."""

from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import patch

import pytest
from synth_ai.sdk.api.train.sft import SFTJob, SFTJobConfig


class TestSFTJobConfig:
    """Tests for SFTJobConfig."""
    
    def test_config_validation_missing_file(self) -> None:
        """Test that missing config file raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            SFTJobConfig(
                config_path=Path("/nonexistent.toml"),
                backend_url="https://api.usesynth.ai",
                api_key="test-key",
            )
    
    def test_config_validation_missing_backend(self) -> None:
        """Test that missing backend_url raises ValueError."""
        with pytest.raises(ValueError, match="backend_url is required"):
            SFTJobConfig(
                config_path=Path(__file__),
                backend_url="",
                api_key="test-key",
            )
    
    def test_config_validation_missing_api_key(self) -> None:
        """Test that missing api_key raises ValueError."""
        with pytest.raises(ValueError, match="api_key is required"):
            SFTJobConfig(
                config_path=Path(__file__),
                backend_url="https://api.usesynth.ai",
                api_key="",
            )


class TestSFTJob:
    """Tests for SFTJob."""
    
    def test_from_config_missing_api_key(self) -> None:
        """Test that from_config raises ValueError if API key missing."""
        with patch.dict(os.environ, {}, clear=True), pytest.raises(
            ValueError, match="api_key is required"
        ):
            SFTJob.from_config(
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
            job = SFTJob.from_config(config_path=Path(__file__))
            assert job.config.api_key == "test-key"
            # Backend URL is stored as-is (normalization happens when used)
            assert job.config.backend_url == "https://custom.backend.com"
    
    def test_from_job_id(self) -> None:
        """Test creating job from existing job ID."""
        with patch.dict(os.environ, {"SYNTH_API_KEY": "test-key"}), patch(
            "synth_ai.config.base_url.get_backend_from_env"
        ) as mock_get:
            mock_get.return_value = ("https://api.usesynth.ai", "key")
            job = SFTJob.from_job_id(
                job_id="sft_1234567890",
                backend_url="https://api.usesynth.ai",
                api_key="test-key",
            )
            assert job.job_id == "sft_1234567890"
            assert job.config.backend_url == "https://api.usesynth.ai"
    
    def test_submit_requires_config(self) -> None:
        """Test that submit() requires config file for new jobs."""
        with patch.dict(os.environ, {"SYNTH_API_KEY": "test-key"}), patch(
            "synth_ai.config.base_url.get_backend_from_env"
        ) as mock_get:
            mock_get.return_value = ("https://api.usesynth.ai", "key")
            # Create job from job_id (no config file)
            # This creates a job with a job_id, so submit() will raise "already submitted"
            job = SFTJob.from_job_id(
                job_id="sft_1234567890",
                backend_url="https://api.usesynth.ai",
                api_key="test-key",
            )
            # Should fail because job already has an ID (can't submit twice)
            with pytest.raises(RuntimeError, match="already submitted"):
                job.submit()
            
            # To test "Cannot build payload", we need a job without a job_id but with /dev/null config
            # The check in _build_payload looks for "/dev/null" name specifically
            job2 = SFTJob(
                config=SFTJobConfig(
                    config_path=Path("/dev/null"),  # /dev/null exists but triggers "Cannot build payload"
                    backend_url="https://api.usesynth.ai",
                    api_key="test-key",
                ),
            )
            # This will fail when trying to build payload because we check for "/dev/null" name
            # OR because /dev/null isn't valid TOML (which raises TrainError/ValidationError from builder)
            import click
            from pydantic import ValidationError
            from synth_ai.sdk.api.train.utils import TrainError
            with pytest.raises(
                (RuntimeError, click.ClickException, TrainError, ValidationError, ValueError),
                match="Cannot build payload|Config|validation|not found|Failed to parse",
            ):
                job2.submit()
    
    def test_submit_already_submitted(self) -> None:
        """Test that submit() raises error if already submitted."""
        job = SFTJob(
            config=SFTJobConfig(
                config_path=Path(__file__),
                backend_url="https://api.usesynth.ai",
                api_key="test-key",
            ),
            job_id="sft_existing",
        )
        with pytest.raises(RuntimeError, match="already submitted"):
            job.submit()
    
    def test_get_status_requires_submission(self) -> None:
        """Test that get_status() requires job to be submitted."""
        job = SFTJob(
            config=SFTJobConfig(
                config_path=Path(__file__),
                backend_url="https://api.usesynth.ai",
                api_key="test-key",
            ),
        )
        with pytest.raises(RuntimeError, match="not yet submitted"):
            job.get_status()
    
    def test_poll_until_complete_requires_submission(self) -> None:
        """Test that poll_until_complete() requires job to be submitted."""
        job = SFTJob(
            config=SFTJobConfig(
                config_path=Path(__file__),
                backend_url="https://api.usesynth.ai",
                api_key="test-key",
            ),
        )
        with pytest.raises(RuntimeError, match="not yet submitted"):
            job.poll_until_complete()

