"""Integration tests for SFT SDK and CLI."""

from __future__ import annotations

import os
import subprocess
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest
from synth_ai.api.train.sft import SFTJob

pytestmark = pytest.mark.integration


def _get_test_sft_config() -> dict:
    """Get a minimal valid SFT config for testing."""
    return {
        "job": {
            "model": "Qwen/Qwen3-0.6B",
            "data": "tests/artifacts/datasets/small.jsonl",
        },
        "hyperparameters": {
            "n_epochs": 1,
            "batch_size": 1,
        },
    }


def _create_test_config_file(config_data: dict) -> Path:
    """Create a temporary TOML config file."""
    try:
        import tomli_w
        
        with tempfile.NamedTemporaryFile(mode="wb", suffix=".toml", delete=False) as f:
            tomli_w.dump(config_data, f)
            return Path(f.name)
    except ImportError:
        # Fallback: write as JSON
        with tempfile.NamedTemporaryFile(mode="w", suffix=".toml", delete=False) as f:
            import json
            json.dump(config_data, f, indent=2)
            return Path(f.name)


class TestSFTSDKIntegration:
    """Integration tests for SFTJob SDK."""
    
    def test_from_config_creates_job(self) -> None:
        """Test that from_config creates a job instance."""
        config_file = _create_test_config_file(_get_test_sft_config())
        try:
            with patch.dict(os.environ, {"SYNTH_API_KEY": "test-key"}), patch(
                "synth_ai.config.base_url.get_backend_from_env"
            ) as mock_get:
                mock_get.return_value = ("https://api.usesynth.ai", "key")
                job = SFTJob.from_config(
                    config_path=config_file,
                    backend_url="https://api.usesynth.ai",
                    api_key="test-key",
                )
                assert job.config.config_path == config_file
                assert job.config.backend_url == "https://api.usesynth.ai"
                assert job.job_id is None  # Not yet submitted
        finally:
            config_file.unlink(missing_ok=True)
    
    def test_from_job_id_resumes_job(self) -> None:
        """Test that from_job_id creates a job for existing job."""
        with patch.dict(os.environ, {"SYNTH_API_KEY": "test-key"}), patch(
            "synth_ai.config.base_url.get_backend_from_env"
        ) as mock_get:
            mock_get.return_value = ("https://api.usesynth.ai", "key")
            job = SFTJob.from_job_id(
                job_id="sft_test123",
                backend_url="https://api.usesynth.ai",
                api_key="test-key",
            )
            assert job.job_id == "sft_test123"


class TestSFTCLIIntegration:
    """Integration tests for SFT CLI."""
    
    def test_cli_train_sft_help(self) -> None:
        """Test that CLI help works."""
        result = subprocess.run(
            ["uvx", "synth-ai", "train", "--help"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        assert result.returncode == 0
        assert "sft" in result.stdout.lower()

