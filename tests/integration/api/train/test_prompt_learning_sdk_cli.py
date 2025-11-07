"""Integration tests for Prompt Learning SDK and CLI."""

from __future__ import annotations

import os
import subprocess
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest
from synth_ai.api.train.prompt_learning import PromptLearningJob

pytestmark = pytest.mark.integration


def _get_test_config() -> dict:
    """Get a minimal valid GEPA config for testing."""
    return {
        "prompt_learning": {
            "algorithm": "gepa",
            "task_app_url": "http://localhost:8000",
            "task_app_id": "test-app",
            "policy": {
                "inference_mode": "synth_hosted",
                "model": "gpt-4o-mini",
                "provider": "openai",
            },
            "gepa": {
                "num_generations": 1,
                "initial_population_size": 2,
                "children_per_generation": 1,
                "max_concurrent_rollouts": 1,
            },
        }
    }


def _create_test_config_file(config_data: dict) -> Path:
    """Create a temporary TOML config file."""
    try:
        import tomli_w
        
        with tempfile.NamedTemporaryFile(mode="wb", suffix=".toml", delete=False) as f:
            tomli_w.dump(config_data, f)
            return Path(f.name)
    except ImportError:
        # Fallback: write as JSON (tests will handle it)
        with tempfile.NamedTemporaryFile(mode="w", suffix=".toml", delete=False) as f:
            import json
            json.dump(config_data, f, indent=2)
            return Path(f.name)


class TestPromptLearningSDKIntegration:
    """Integration tests for PromptLearningJob SDK."""
    
    def test_from_config_creates_job(self) -> None:
        """Test that from_config creates a job instance."""
        config_file = _create_test_config_file(_get_test_config())
        try:
            with patch.dict(os.environ, {"SYNTH_API_KEY": "test-key"}), patch(
                "synth_ai.config.base_url.get_backend_from_env"
            ) as mock_get:
                mock_get.return_value = ("https://api.usesynth.ai", "key")
                job = PromptLearningJob.from_config(
                    config_path=config_file,
                    backend_url="https://api.usesynth.ai",
                    api_key="test-key",
                    task_app_api_key="env-key",
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
            job = PromptLearningJob.from_job_id(
                job_id="pl_test123",
                backend_url="https://api.usesynth.ai",
                api_key="test-key",
            )
            assert job.job_id == "pl_test123"
    
    @pytest.mark.skipif(
        not os.getenv("SYNTH_API_KEY") or not os.getenv("ENVIRONMENT_API_KEY"),
        reason="Requires SYNTH_API_KEY and ENVIRONMENT_API_KEY",
    )
    def test_submit_job_real_backend(self) -> None:
        """Test submitting a job to real backend (requires API keys)."""
        config_file = _create_test_config_file(_get_test_config())
        try:
            backend_url = os.getenv("BACKEND_BASE_URL", "https://api.usesynth.ai")
            job = PromptLearningJob.from_config(
                config_path=config_file,
                backend_url=backend_url,
                api_key=os.environ["SYNTH_API_KEY"],
                task_app_api_key=os.environ.get("ENVIRONMENT_API_KEY"),
            )
            # Note: This will fail if task app is not running, which is expected
            # We're just testing that the SDK method exists and can be called
            try:
                job_id = job.submit()
                assert job_id.startswith("pl_")
                assert job.job_id == job_id
            except (RuntimeError, ValueError) as e:
                # Expected if task app is not available
                assert "health check" in str(e).lower() or "task app" in str(e).lower()
        finally:
            config_file.unlink(missing_ok=True)


class TestPromptLearningCLIIntegration:
    """Integration tests for Prompt Learning CLI."""
    
    def test_cli_train_prompt_learning_help(self) -> None:
        """Test that CLI help works."""
        result = subprocess.run(
            ["uvx", "synth-ai", "train", "--help"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        assert result.returncode == 0
        assert "prompt_learning" in result.stdout or "prompt-learning" in result.stdout
    
    def test_cli_train_prompt_learning_invalid_config(self) -> None:
        """Test that CLI validates config file."""
        config_file = _create_test_config_file({"invalid": "config"})
        try:
            result = subprocess.run(
                ["uvx", "synth-ai", "train", "--type", "prompt_learning", "--config", str(config_file), "--no-poll"],
                capture_output=True,
                text=True,
                timeout=30,
            )
            # Should fail with validation error
            assert result.returncode != 0
            assert "prompt_learning" in result.stderr.lower() or "invalid" in result.stderr.lower()
        finally:
            config_file.unlink(missing_ok=True)
    
    @pytest.mark.skipif(
        not os.getenv("SYNTH_API_KEY"),
        reason="Requires SYNTH_API_KEY",
    )
    def test_cli_train_prompt_learning_dry_run(self) -> None:
        """Test CLI with valid config (dry run - just validates)."""
        config_file = _create_test_config_file(_get_test_config())
        try:
            # Use --no-poll to avoid waiting
            result = subprocess.run(
                [
                    "uvx", "synth-ai", "train",
                    "--type", "prompt_learning",
                    "--config", str(config_file),
                    "--no-poll",
                ],
                capture_output=True,
                text=True,
                timeout=60,
                env={**os.environ, "SYNTH_API_KEY": os.environ.get("SYNTH_API_KEY", "")},
            )
            # May fail if task app not available, but should at least validate config
            # Check that it got past config validation
            if result.returncode == 0:
                assert "job" in result.stdout.lower() or "submitted" in result.stdout.lower()
            else:
                # Should be a runtime error (task app), not a config validation error
                assert "health check" in result.stderr.lower() or "task app" in result.stderr.lower()
        finally:
            config_file.unlink(missing_ok=True)

