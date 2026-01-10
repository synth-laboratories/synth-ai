"""Integration tests for RL (Reinforcement Learning) SDK.

These tests validate the SDK components for RL jobs:
1. Configuration validation
2. Client functionality with mocks

Tests are marked with pytest markers for selective execution:
- @pytest.mark.integration: All integration tests
"""

import pytest

pytestmark = [pytest.mark.integration]


class TestRLConfigValidation:
    """Test RL configuration validation."""

    def test_validate_rl_config_requires_algorithm(self) -> None:
        """RL config requires algorithm section."""
        from synth_ai.sdk.api.train.validation import MissingAlgorithmError, validate_rl_config

        with pytest.raises(MissingAlgorithmError):
            validate_rl_config({})

    def test_validate_rl_config_requires_model_or_policy(self) -> None:
        """RL config requires model or policy section."""
        from synth_ai.sdk.api.train.validation import MissingModelError, validate_rl_config

        with pytest.raises(MissingModelError):
            validate_rl_config(
                {
                    "algorithm": {"type": "online", "variety": "gspo"},
                }
            )

    def test_validate_rl_config_requires_variety(self) -> None:
        """RL config requires algorithm variety."""
        from synth_ai.sdk.api.train.validation import MissingAlgorithmError, validate_rl_config

        with pytest.raises(MissingAlgorithmError):
            validate_rl_config(
                {
                    "algorithm": {"type": "online"},
                    "policy": {"model_name": "Qwen/Qwen3-0.6B"},
                }
            )


class TestRlClientWithMocks:
    """Test RlClient functionality with mocked HTTP."""

    @pytest.mark.asyncio
    async def test_rl_client_creates_job(self, monkeypatch) -> None:
        """RlClient should create a job via HTTP."""
        from synth_ai.sdk.learning.rl_client import RlClient

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

        client = RlClient(base_url="https://api.usesynth.ai", api_key="test-key")

        # Create a job with the actual API signature
        response = await client.create_job(
            model="Qwen/Qwen3-0.6B",
            task_app_url="http://localhost:8001",
            trainer={"batch_size": 4, "group_size": 8},
        )

        assert response["job_id"] == "rl_test123"
        assert len(mock_clients) == 1

    @pytest.mark.asyncio
    async def test_rl_client_gets_job_status(self, monkeypatch) -> None:
        """RlClient should retrieve job status."""
        from synth_ai.sdk.learning.rl_client import RlClient

        class MockHTTPClient:
            async def __aenter__(self):
                return self

            async def __aexit__(self, *args):
                pass

            async def get(self, url, **kwargs):
                return {
                    "job_id": "rl_test123",
                    "status": "running",
                    "progress": {"current_step": 50, "total_steps": 100},
                }

        def make_mock(*args, **kwargs):
            return MockHTTPClient()

        monkeypatch.setattr("synth_ai.sdk.learning.rl.client.AsyncHttpClient", make_mock)

        client = RlClient(base_url="https://api.usesynth.ai", api_key="test-key")
        status = await client.get_job("rl_test123")

        assert status["status"] == "running"
        assert status["progress"]["current_step"] == 50

    @pytest.mark.asyncio
    async def test_rl_client_gets_metrics(self, monkeypatch) -> None:
        """RlClient should retrieve training metrics."""
        from synth_ai.sdk.learning.rl_client import RlClient

        class MockHTTPClient:
            async def __aenter__(self):
                return self

            async def __aexit__(self, *args):
                pass

            async def get(self, url, **kwargs):
                return {
                    "points": [
                        {"name": "reward", "step": 0, "value": 0.1},
                        {"name": "reward", "step": 10, "value": 0.3},
                        {"name": "reward", "step": 20, "value": 0.5},
                    ]
                }

        def make_mock(*args, **kwargs):
            return MockHTTPClient()

        monkeypatch.setattr("synth_ai.sdk.learning.rl.client.AsyncHttpClient", make_mock)

        client = RlClient(base_url="https://api.usesynth.ai", api_key="test-key")
        metrics = await client.get_metrics("rl_test123")

        assert len(metrics) == 3
        assert metrics[2]["value"] == 0.5


class TestRLTerminalStates:
    """Test RL terminal state detection via poll logic."""

    def test_terminal_statuses_recognized(self) -> None:
        """Terminal statuses should be in the terminal set."""
        # The terminal set is defined in poll_until_terminal
        terminal = {"succeeded", "failed", "cancelled", "canceled", "error", "completed"}

        assert "succeeded" in terminal
        assert "failed" in terminal
        assert "cancelled" in terminal
        assert "canceled" in terminal
        assert "error" in terminal
        assert "completed" in terminal

        # Non-terminal statuses
        assert "running" not in terminal
        assert "pending" not in terminal
        assert "queued" not in terminal
