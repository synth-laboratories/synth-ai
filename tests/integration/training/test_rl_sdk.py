"""Integration tests for RL (Reinforcement Learning) SDK.

These tests validate the full SDK workflow for RL jobs:
1. Configuration loading and validation
2. Task app health verification
3. Job submission and creation
4. Polling and status monitoring
5. Metrics retrieval

Tests are marked with pytest markers for selective execution:
- @pytest.mark.integration: All integration tests
- @pytest.mark.slow: Tests that may take >30 seconds
- @pytest.mark.requires_backend: Tests that require a running backend
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import patch

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


class TestRLConfigValidation:
    """Test RL configuration validation."""

    def test_validate_rl_config_requires_algorithm(self) -> None:
        """RL config requires algorithm section."""
        from synth_ai.cli.commands.train.validation import validate_rl_config
        from synth_ai.cli.commands.train.errors import MissingAlgorithmError

        with pytest.raises(MissingAlgorithmError):
            validate_rl_config({})

    def test_validate_rl_config_requires_task_url(self) -> None:
        """RL config requires task_app_url."""
        from synth_ai.cli.commands.train.validation import validate_rl_config
        from synth_ai.cli.commands.train.errors import MissingTaskURLError

        with pytest.raises(MissingTaskURLError):
            validate_rl_config({
                "algorithm": {"type": "online", "variety": "grpo"},
            })

    def test_validate_rl_config_requires_model(self) -> None:
        """RL config requires model in algorithm or policy section."""
        from synth_ai.cli.commands.train.validation import validate_rl_config
        from synth_ai.cli.commands.train.errors import MissingModelError

        with pytest.raises(MissingModelError):
            validate_rl_config({
                "algorithm": {"type": "online", "variety": "grpo"},
                "task_app_url": "http://localhost:8001",
            })

    def test_validate_rl_config_valid(self) -> None:
        """RL config should pass with all required fields."""
        from synth_ai.cli.commands.train.validation import validate_rl_config

        # Should not raise
        validate_rl_config({
            "algorithm": {"type": "online", "variety": "grpo"},
            "task_app_url": "http://localhost:8001",
            "policy": {"model": "Qwen/Qwen3-0.6B"},
        })


class TestRLPayloadBuilder:
    """Test RL payload building from config."""

    def test_build_rl_payload_from_config(self, tmp_path: Path) -> None:
        """build_rl_payload should construct valid payload."""
        from synth_ai.sdk.api.train.builders import build_rl_payload

        config = tmp_path / "rl.toml"
        config.write_text("""
[algorithm]
type = "online"
variety = "grpo"

[policy]
model = "Qwen/Qwen3-0.6B"
provider = "synth"

[hyperparameters]
n_rollouts = 100
batch_size = 4
learning_rate = 1e-5
""")

        build = build_rl_payload(
            config_path=config,
            task_url="http://localhost:8001",
        )

        assert build.task_url == "http://localhost:8001"
        assert build.payload["policy"]["model"] == "Qwen/Qwen3-0.6B"
        assert "hyperparameters" in build.payload

    def test_build_rl_payload_with_model_override(self, tmp_path: Path) -> None:
        """build_rl_payload should accept model override."""
        from synth_ai.sdk.api.train.builders import build_rl_payload

        config = tmp_path / "rl.toml"
        config.write_text("""
[algorithm]
type = "online"
variety = "grpo"

[policy]
model = "Qwen/Qwen3-0.6B"
provider = "synth"
""")

        build = build_rl_payload(
            config_path=config,
            task_url="http://localhost:8001",
            overrides={"model": "Qwen/Qwen3-1.7B"},
        )

        # Model override should be applied
        assert build.payload["policy"]["model"] == "Qwen/Qwen3-1.7B"


class TestRLClientJobCreation:
    """Test RLClient job creation functionality."""

    @pytest.mark.asyncio
    async def test_create_rl_job_with_valid_config(self, monkeypatch) -> None:
        """create_rl_job should create job with valid config."""
        from synth_ai.sdk.learning.rl.client import RLClient

        class MockHTTPClient:
            def __init__(self, *args, **kwargs):
                self.json_calls = []

            async def __aenter__(self):
                return self

            async def __aexit__(self, *args):
                pass

            async def post_json(self, url, json):
                self.json_calls.append((url, json))
                return {"job_id": "rl_test123", "status": "created"}

        mock_clients = []

        def make_mock(*args, **kwargs):
            client = MockHTTPClient(*args, **kwargs)
            mock_clients.append(client)
            return client

        monkeypatch.setattr("synth_ai.sdk.learning.rl.client.AsyncHttpClient", make_mock)

        client = RLClient(base_url="https://api.usesynth.ai", api_key="test-key")
        response = await client.create_rl_job(
            task_url="http://localhost:8001",
            policy_config={
                "model": "Qwen/Qwen3-0.6B",
                "provider": "synth",
            },
            hyperparameters={
                "n_rollouts": 100,
                "batch_size": 4,
            },
        )

        assert response["job_id"] == "rl_test123"
        assert len(mock_clients) == 1
        assert len(mock_clients[0].json_calls) == 1


class TestRLClientJobStatus:
    """Test RLClient job status functionality."""

    @pytest.mark.asyncio
    async def test_get_job_status_returns_data(self, monkeypatch) -> None:
        """get_job_status should return job data."""
        from synth_ai.sdk.learning.rl.client import RLClient

        class MockHTTPClient:
            async def __aenter__(self):
                return self

            async def __aexit__(self, *args):
                pass

            async def get_json(self, url):
                return {
                    "job_id": "rl_test123",
                    "status": "running",
                    "progress": {"current_rollout": 50, "total_rollouts": 100},
                }

        def make_mock(*args, **kwargs):
            return MockHTTPClient()

        monkeypatch.setattr("synth_ai.sdk.learning.rl.client.AsyncHttpClient", make_mock)

        client = RLClient(base_url="https://api.usesynth.ai", api_key="test-key")
        status = await client.get_job_status("rl_test123")

        assert status["status"] == "running"
        assert status["progress"]["current_rollout"] == 50

    @pytest.mark.asyncio
    async def test_poll_until_terminal_returns_on_complete(self, monkeypatch) -> None:
        """poll_until_terminal should return when status is terminal."""
        from synth_ai.sdk.learning.rl.client import RLClient

        call_count = [0]

        class MockHTTPClient:
            async def __aenter__(self):
                return self

            async def __aexit__(self, *args):
                pass

            async def get_json(self, url):
                call_count[0] += 1
                if call_count[0] < 3:
                    return {"job_id": "rl_test123", "status": "running"}
                return {"job_id": "rl_test123", "status": "completed", "final_reward": 0.85}

        def make_mock(*args, **kwargs):
            return MockHTTPClient()

        monkeypatch.setattr("synth_ai.sdk.learning.rl.client.AsyncHttpClient", make_mock)

        client = RLClient(base_url="https://api.usesynth.ai", api_key="test-key")
        result = await client.poll_until_terminal("rl_test123", interval=0.01)

        assert result["status"] == "completed"
        assert result["final_reward"] == 0.85
        assert call_count[0] >= 3


class TestRLClientMetrics:
    """Test RLClient metrics retrieval."""

    @pytest.mark.asyncio
    async def test_get_metrics_returns_training_data(self, monkeypatch) -> None:
        """get_metrics should return training metrics."""
        from synth_ai.sdk.learning.rl.client import RLClient

        class MockHTTPClient:
            async def __aenter__(self):
                return self

            async def __aexit__(self, *args):
                pass

            async def get_json(self, url):
                return {
                    "metrics": [
                        {"step": 0, "reward": 0.1, "loss": 2.5},
                        {"step": 10, "reward": 0.3, "loss": 2.0},
                        {"step": 20, "reward": 0.5, "loss": 1.5},
                    ]
                }

        def make_mock(*args, **kwargs):
            return MockHTTPClient()

        monkeypatch.setattr("synth_ai.sdk.learning.rl.client.AsyncHttpClient", make_mock)

        client = RLClient(base_url="https://api.usesynth.ai", api_key="test-key")
        metrics = await client.get_metrics("rl_test123")

        assert len(metrics["metrics"]) == 3
        assert metrics["metrics"][2]["reward"] == 0.5


class TestTaskAppHealthCheck:
    """Test task app health check functionality."""

    def test_check_task_app_health_success(self, monkeypatch) -> None:
        """check_task_app_health should return OK for healthy app."""
        from synth_ai.sdk.api.train.task_app import check_task_app_health

        def mock_get(url, **kwargs):
            if "/health" in url:
                return MockHTTPResponse(200, {"status": "ok"})
            if "/task_info" in url:
                return MockHTTPResponse(200, {"task_app_id": "test"})
            return MockHTTPResponse(404, {})

        monkeypatch.setattr("synth_ai.sdk.api.train.utils.http_get", mock_get)

        result = check_task_app_health("http://localhost:8001", "test-env-key")

        assert result.ok is True
        assert result.health_status == 200
        assert result.task_info_status == 200

    def test_check_task_app_health_failure_health(self, monkeypatch) -> None:
        """check_task_app_health should return not OK for unhealthy app."""
        from synth_ai.sdk.api.train.task_app import check_task_app_health

        def mock_get(url, **kwargs):
            return MockHTTPResponse(500, {"error": "Internal error"})

        monkeypatch.setattr("synth_ai.sdk.api.train.utils.http_get", mock_get)

        result = check_task_app_health("http://localhost:8001", "test-env-key")

        assert result.ok is False
        assert result.health_status == 500

    def test_check_task_app_health_failure_task_info(self, monkeypatch) -> None:
        """check_task_app_health should return not OK if task_info fails."""
        from synth_ai.sdk.api.train.task_app import check_task_app_health

        def mock_get(url, **kwargs):
            if "/health" in url:
                return MockHTTPResponse(200, {"status": "ok"})
            return MockHTTPResponse(401, {"error": "Unauthorized"})

        monkeypatch.setattr("synth_ai.sdk.api.train.utils.http_get", mock_get)

        result = check_task_app_health("http://localhost:8001", "test-env-key")

        assert result.ok is False
        assert result.task_info_status == 401


class TestRLTerminalStates:
    """Test RL terminal state detection."""

    def test_is_terminal_state_completed(self) -> None:
        """completed should be a terminal state."""
        from synth_ai.sdk.learning.rl.client import _is_terminal_state

        assert _is_terminal_state("completed") is True
        assert _is_terminal_state("COMPLETED") is True

    def test_is_terminal_state_failed(self) -> None:
        """failed should be a terminal state."""
        from synth_ai.sdk.learning.rl.client import _is_terminal_state

        assert _is_terminal_state("failed") is True
        assert _is_terminal_state("FAILED") is True

    def test_is_terminal_state_cancelled(self) -> None:
        """cancelled should be a terminal state."""
        from synth_ai.sdk.learning.rl.client import _is_terminal_state

        assert _is_terminal_state("cancelled") is True
        assert _is_terminal_state("canceled") is True

    def test_is_terminal_state_running(self) -> None:
        """running should not be a terminal state."""
        from synth_ai.sdk.learning.rl.client import _is_terminal_state

        assert _is_terminal_state("running") is False
        assert _is_terminal_state("pending") is False
        assert _is_terminal_state("queued") is False
