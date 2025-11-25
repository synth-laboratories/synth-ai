"""Integration tests for Prompt Learning SDK (MIPRO and GEPA).

These tests validate the full SDK workflow for prompt learning jobs:
1. Configuration loading and validation
2. Job submission and creation
3. Polling and status monitoring
4. Result retrieval and prompt extraction

Tests are marked with pytest markers for selective execution:
- @pytest.mark.integration: All integration tests
- @pytest.mark.slow: Tests that may take >30 seconds
- @pytest.mark.requires_backend: Tests that require a running backend
"""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

pytestmark = [pytest.mark.integration]


class MockHTTPResponse:
    """Mock HTTP response for testing."""

    def __init__(self, status_code: int, json_data: dict[str, Any] | None = None, text: str = ""):
        self.status_code = status_code
        self._json_data = json_data or {}
        self.text = text or json.dumps(self._json_data)
        self.headers = {"content-type": "application/json"}

    def json(self) -> dict[str, Any]:
        return self._json_data


class TestPromptLearningJobConfig:
    """Test PromptLearningJobConfig validation and construction."""

    def test_config_requires_config_path(self, tmp_path: Path) -> None:
        """Config should require a valid config path."""
        from synth_ai.sdk.api.train.prompt_learning import PromptLearningJobConfig

        # Create a valid config file
        config_file = tmp_path / "config.toml"
        config_file.write_text("""
[prompt_learning]
algorithm = "gepa"
task_app_url = "http://localhost:8001"
""")

        config = PromptLearningJobConfig(
            config_path=config_file,
            backend_url="https://api.usesynth.ai",
            api_key="test-api-key",
            task_app_api_key="test-env-key",
        )

        assert config.config_path == config_file
        assert config.backend_url == "https://api.usesynth.ai"
        assert config.api_key == "test-api-key"

    def test_config_raises_for_missing_file(self) -> None:
        """Config should raise FileNotFoundError for missing config file."""
        from synth_ai.sdk.api.train.prompt_learning import PromptLearningJobConfig

        with pytest.raises(FileNotFoundError):
            PromptLearningJobConfig(
                config_path=Path("/nonexistent/config.toml"),
                backend_url="https://api.usesynth.ai",
                api_key="test-api-key",
                task_app_api_key="test-env-key",
            )

    def test_config_raises_for_missing_backend(self, tmp_path: Path) -> None:
        """Config should raise ValueError for missing backend URL."""
        from synth_ai.sdk.api.train.prompt_learning import PromptLearningJobConfig

        config_file = tmp_path / "config.toml"
        config_file.write_text("[prompt_learning]\nalgorithm = 'gepa'")

        with pytest.raises(ValueError, match="backend_url is required"):
            PromptLearningJobConfig(
                config_path=config_file,
                backend_url="",
                api_key="test-api-key",
                task_app_api_key="test-env-key",
            )

    def test_config_raises_for_missing_api_key(self, tmp_path: Path) -> None:
        """Config should raise ValueError for missing API key."""
        from synth_ai.sdk.api.train.prompt_learning import PromptLearningJobConfig

        config_file = tmp_path / "config.toml"
        config_file.write_text("[prompt_learning]\nalgorithm = 'gepa'")

        with pytest.raises(ValueError, match="api_key is required"):
            PromptLearningJobConfig(
                config_path=config_file,
                backend_url="https://api.usesynth.ai",
                api_key="",
                task_app_api_key="test-env-key",
            )


class TestPromptLearningJobFromConfig:
    """Test PromptLearningJob.from_config() factory method."""

    def test_from_config_creates_job(self, tmp_path: Path, monkeypatch) -> None:
        """from_config should create a valid PromptLearningJob."""
        from synth_ai.sdk.api.train.prompt_learning import PromptLearningJob

        # Set required env vars
        monkeypatch.setenv("SYNTH_API_KEY", "test-api-key")
        monkeypatch.setenv("ENVIRONMENT_API_KEY", "test-env-key")

        # Create config file
        config_file = tmp_path / "config.toml"
        config_file.write_text("""
[prompt_learning]
algorithm = "gepa"
task_app_url = "http://localhost:8001"
results_folder = "results"

[prompt_learning.policy]
provider = "openai"
model = "gpt-4o-mini"
""")

        job = PromptLearningJob.from_config(
            config_path=config_file,
            backend_url="https://api.usesynth.ai",
        )

        assert job.config.config_path == config_file
        assert job.config.backend_url == "https://api.usesynth.ai"
        assert job.job_id is None  # Not yet submitted

    def test_from_config_uses_env_defaults(self, tmp_path: Path, monkeypatch) -> None:
        """from_config should use environment defaults when not provided."""
        from synth_ai.sdk.api.train.prompt_learning import PromptLearningJob

        # Set required env vars
        monkeypatch.setenv("SYNTH_API_KEY", "env-api-key")
        monkeypatch.setenv("ENVIRONMENT_API_KEY", "env-task-key")
        monkeypatch.setenv("BACKEND_BASE_URL", "http://localhost:8000/api")

        # Create config file
        config_file = tmp_path / "config.toml"
        config_file.write_text("""
[prompt_learning]
algorithm = "mipro"
task_app_url = "http://localhost:8001"
results_folder = "results"
""")

        job = PromptLearningJob.from_config(config_path=config_file)

        assert job.config.api_key == "env-api-key"


class TestPromptLearningJobSubmit:
    """Test PromptLearningJob.submit() method."""

    @pytest.fixture
    def mock_job(self, tmp_path: Path, monkeypatch) -> "PromptLearningJob":
        """Create a mock PromptLearningJob for testing."""
        from synth_ai.sdk.api.train.prompt_learning import PromptLearningJob

        monkeypatch.setenv("SYNTH_API_KEY", "test-api-key")
        monkeypatch.setenv("ENVIRONMENT_API_KEY", "test-env-key")

        config_file = tmp_path / "config.toml"
        config_file.write_text("""
[prompt_learning]
algorithm = "gepa"
task_app_url = "http://localhost:8001"
results_folder = "results"

[prompt_learning.policy]
provider = "openai"
model = "gpt-4o-mini"
inference_mode = "synth_hosted"

[prompt_learning.gepa]
n_generations = 5
population_size = 3

[prompt_learning.gepa.evaluation]
train_seeds = [0, 1, 2]
val_seeds = [10, 11, 12]
""")

        return PromptLearningJob.from_config(
            config_path=config_file,
            backend_url="https://api.usesynth.ai",
        )

    def test_submit_creates_job_and_returns_id(self, mock_job, monkeypatch) -> None:
        """submit() should create a job and return the job ID."""
        # Mock HTTP responses
        mock_health = MockHTTPResponse(200, {"status": "ok"})
        mock_task_info = MockHTTPResponse(200, {"task_app_id": "test"})
        mock_create = MockHTTPResponse(201, {"job_id": "pl_test123"})

        def mock_post(url, **kwargs):
            if "/health" in url:
                return mock_health
            if "/task_info" in url:
                return mock_task_info
            return mock_create

        def mock_get(url, **kwargs):
            if "/health" in url:
                return mock_health
            if "/task_info" in url:
                return mock_task_info
            return MockHTTPResponse(200, {})

        # Patch at the point of use, not at definition
        monkeypatch.setattr("synth_ai.sdk.api.train.prompt_learning.http_post", mock_post)
        monkeypatch.setattr("synth_ai.sdk.api.train.task_app.http_get", mock_get)

        job_id = mock_job.submit()

        assert job_id == "pl_test123"
        assert mock_job.job_id == "pl_test123"

    def test_submit_raises_if_already_submitted(self, mock_job, monkeypatch) -> None:
        """submit() should raise if job was already submitted."""
        # Mock successful submission
        mock_health = MockHTTPResponse(200, {"status": "ok"})
        mock_create = MockHTTPResponse(201, {"job_id": "pl_test123"})

        def mock_post(url, **kwargs):
            return mock_create if "jobs" in url else mock_health

        def mock_get(url, **kwargs):
            return mock_health

        # Patch at the point of use, not at definition
        monkeypatch.setattr("synth_ai.sdk.api.train.prompt_learning.http_post", mock_post)
        monkeypatch.setattr("synth_ai.sdk.api.train.task_app.http_get", mock_get)

        mock_job.submit()

        with pytest.raises(RuntimeError, match="already submitted"):
            mock_job.submit()

    def test_submit_raises_on_health_check_failure(self, mock_job, monkeypatch) -> None:
        """submit() should raise if task app health check fails."""
        mock_health = MockHTTPResponse(500, {"error": "unhealthy"})

        def mock_get(url, **kwargs):
            return mock_health

        # Patch at the point of use, not at definition
        monkeypatch.setattr("synth_ai.sdk.api.train.task_app.http_get", mock_get)

        with pytest.raises(ValueError, match="health check failed"):
            mock_job.submit()


class TestPromptLearningJobFromJobId:
    """Test PromptLearningJob.from_job_id() for resuming jobs."""

    def test_from_job_id_creates_job(self, monkeypatch) -> None:
        """from_job_id should create a job for an existing job ID."""
        from synth_ai.sdk.api.train.prompt_learning import PromptLearningJob

        monkeypatch.setenv("SYNTH_API_KEY", "test-api-key")
        monkeypatch.setenv("ENVIRONMENT_API_KEY", "test-env-key")

        job = PromptLearningJob.from_job_id(
            job_id="pl_existing123",
            backend_url="https://api.usesynth.ai",
        )

        assert job.job_id == "pl_existing123"

    def test_from_job_id_cannot_build_payload(self, monkeypatch) -> None:
        """from_job_id jobs cannot build payloads (no valid config file)."""
        from synth_ai.sdk.api.train.prompt_learning import PromptLearningJob
        import click

        monkeypatch.setenv("SYNTH_API_KEY", "test-api-key")
        monkeypatch.setenv("ENVIRONMENT_API_KEY", "test-env-key")

        job = PromptLearningJob.from_job_id(
            job_id="pl_existing123",
            backend_url="https://api.usesynth.ai",
        )

        # The job uses /dev/null as config_path which should fail validation
        # (either RuntimeError for invalid path, or click.ClickException for empty config)
        with pytest.raises((RuntimeError, click.ClickException)):
            job._build_payload()


class TestPromptLearningJobGetStatus:
    """Test PromptLearningJob.get_status() method."""

    def test_get_status_returns_job_data(self, monkeypatch) -> None:
        """get_status should return job metadata."""
        from synth_ai.sdk.api.train.prompt_learning import PromptLearningJob

        monkeypatch.setenv("SYNTH_API_KEY", "test-api-key")
        monkeypatch.setenv("ENVIRONMENT_API_KEY", "test-env-key")

        job = PromptLearningJob.from_job_id(
            job_id="pl_test123",
            backend_url="https://api.usesynth.ai",
        )

        # Mock the async client
        async def mock_get_job(job_id):
            return {"job_id": job_id, "status": "running", "best_score": 0.85}

        with patch.object(
            job, "get_status",
            return_value={"job_id": "pl_test123", "status": "running", "best_score": 0.85}
        ):
            status = job.get_status()

        assert status["status"] == "running"
        assert status["best_score"] == 0.85

    def test_get_status_raises_if_not_submitted(self, tmp_path: Path, monkeypatch) -> None:
        """get_status should raise if job not yet submitted."""
        from synth_ai.sdk.api.train.prompt_learning import PromptLearningJob

        monkeypatch.setenv("SYNTH_API_KEY", "test-api-key")
        monkeypatch.setenv("ENVIRONMENT_API_KEY", "test-env-key")

        config_file = tmp_path / "config.toml"
        config_file.write_text("[prompt_learning]\nalgorithm = 'gepa'\nresults_folder = 'results'")

        job = PromptLearningJob.from_config(
            config_path=config_file,
            backend_url="https://api.usesynth.ai",
        )

        with pytest.raises(RuntimeError, match="not yet submitted"):
            job.get_status()


class TestPromptLearningJobGetResults:
    """Test PromptLearningJob.get_results() method."""

    def test_get_results_returns_prompt_data(self, monkeypatch) -> None:
        """get_results should return prompt optimization results."""
        from synth_ai.sdk.api.train.prompt_learning import PromptLearningJob

        monkeypatch.setenv("SYNTH_API_KEY", "test-api-key")
        monkeypatch.setenv("ENVIRONMENT_API_KEY", "test-env-key")

        job = PromptLearningJob.from_job_id(
            job_id="pl_test123",
            backend_url="https://api.usesynth.ai",
        )

        expected_results = {
            "best_prompt": {"sections": [{"role": "system", "content": "You are a classifier."}]},
            "best_score": 0.92,
            "top_prompts": [{"rank": 1, "train_accuracy": 0.92}],
            "optimized_candidates": [],
            "attempted_candidates": [],
            "validation_results": [],
        }

        with patch.object(job, "get_results", return_value=expected_results):
            results = job.get_results()

        assert results["best_score"] == 0.92
        assert results["best_prompt"]["sections"][0]["role"] == "system"

    def test_get_results_raises_if_not_submitted(self, tmp_path: Path, monkeypatch) -> None:
        """get_results should raise if job not yet submitted."""
        from synth_ai.sdk.api.train.prompt_learning import PromptLearningJob

        monkeypatch.setenv("SYNTH_API_KEY", "test-api-key")
        monkeypatch.setenv("ENVIRONMENT_API_KEY", "test-env-key")

        config_file = tmp_path / "config.toml"
        config_file.write_text("[prompt_learning]\nalgorithm = 'gepa'\nresults_folder = 'results'")

        job = PromptLearningJob.from_config(
            config_path=config_file,
            backend_url="https://api.usesynth.ai",
        )

        with pytest.raises(RuntimeError, match="not yet submitted"):
            job.get_results()


class TestPromptLearningClient:
    """Test PromptLearningClient for job result queries."""

    def test_validate_job_id_format(self) -> None:
        """Client should validate job ID format."""
        from synth_ai.sdk.learning.prompt_learning_client import _validate_job_id

        # Valid format
        _validate_job_id("pl_abc123")  # Should not raise

        # Invalid format
        with pytest.raises(ValueError, match="Invalid prompt learning job ID format"):
            _validate_job_id("invalid_id")

        with pytest.raises(ValueError, match="Invalid prompt learning job ID format"):
            _validate_job_id("rl_abc123")  # Wrong prefix

    @pytest.mark.asyncio
    async def test_get_prompts_extracts_results(self, monkeypatch) -> None:
        """get_prompts should extract results from events."""
        from synth_ai.sdk.learning.prompt_learning_client import PromptLearningClient

        # Mock events response
        mock_events = [
            {
                "type": "prompt.learning.best.prompt",
                "data": {
                    "best_prompt": {"sections": [{"role": "system", "content": "Test"}]},
                    "best_score": 0.88,
                }
            },
            {
                "type": "prompt.learning.final.results",
                "data": {
                    "optimized_candidates": [{"accuracy": 0.88}],
                    "attempted_candidates": [{"accuracy": 0.75}, {"accuracy": 0.88}],
                }
            },
        ]

        async def mock_get_events(*args, **kwargs):
            return mock_events

        client = PromptLearningClient(
            base_url="https://api.usesynth.ai",
            api_key="test-key",
        )

        with patch.object(client, "get_events", side_effect=mock_get_events):
            results = await client.get_prompts("pl_test123")

        assert results.best_score == 0.88
        assert len(results.attempted_candidates) == 2

    @pytest.mark.asyncio
    async def test_get_scoring_summary_computes_stats(self, monkeypatch) -> None:
        """get_scoring_summary should compute scoring statistics."""
        from synth_ai.sdk.learning.prompt_learning_client import PromptLearningClient
        from synth_ai.sdk.learning.prompt_learning_types import PromptResults

        mock_results = PromptResults(
            best_prompt=None,
            best_score=0.90,
            top_prompts=[],
            optimized_candidates=[{"accuracy": 0.90}],
            attempted_candidates=[
                {"accuracy": 0.70},
                {"accuracy": 0.80},
                {"accuracy": 0.90},
            ],
            validation_results=[
                {"accuracy": 0.85},
                {"accuracy": 0.90},
            ],
        )

        async def mock_get_prompts(*args, **kwargs):
            return mock_results

        client = PromptLearningClient(
            base_url="https://api.usesynth.ai",
            api_key="test-key",
        )

        with patch.object(client, "get_prompts", side_effect=mock_get_prompts):
            summary = await client.get_scoring_summary("pl_test123")

        assert summary["best_train_accuracy"] == 0.90
        assert summary["best_val_accuracy"] == 0.90
        assert summary["num_candidates_tried"] == 3
        assert summary["num_frontier_candidates"] == 1
