"""Integration tests for Judge Client.

These tests validate the judge client functionality:
1. Client initialization
2. Scoring with trace data
3. Error handling

Tests are marked with pytest markers for selective execution:
- @pytest.mark.integration: All integration tests
"""

from __future__ import annotations

import os
from typing import Any

import pytest

# Silence experimental warnings for tests
os.environ["SYNTH_SILENCE_EXPERIMENTAL"] = "1"

pytestmark = [pytest.mark.integration]


class TestJudgeClientInit:
    """Test JudgeClient initialization."""

    def test_client_creates_with_valid_params(self) -> None:
        """Client should create with valid parameters."""
        from synth_ai.sdk.judging.client import JudgeClient

        client = JudgeClient(
            base_url="https://api.usesynth.ai",
            api_key="test-key",
        )

        assert client is not None

    def test_client_accepts_custom_timeout(self) -> None:
        """Client should accept custom timeout."""
        from synth_ai.sdk.judging.client import JudgeClient

        client = JudgeClient(
            base_url="https://api.usesynth.ai",
            api_key="test-key",
            timeout=120.0,
        )

        assert client is not None


class TestJudgeClientScore:
    """Test JudgeClient scoring functionality."""

    @pytest.fixture
    def client(self):
        """Create JudgeClient instance."""
        from synth_ai.sdk.judging.client import JudgeClient

        return JudgeClient(
            base_url="https://api.usesynth.ai",
            api_key="test-key",
        )

    @pytest.mark.asyncio
    async def test_score_returns_result(self, client, monkeypatch) -> None:
        """score should return scoring result."""

        class MockHTTPClient:
            async def __aenter__(self):
                return self

            async def __aexit__(self, *args):
                pass

            async def post_json(self, url, json):
                return {
                    "status": "ok",
                    "outcome_reward": {"value": 0.85, "reasoning": "Accurate response"},
                    "event_rewards": [],
                }

        monkeypatch.setattr("synth_ai.sdk.judging.client.AsyncHttpClient", lambda *a, **kw: MockHTTPClient())

        # Create a minimal trace
        trace = {
            "session_id": "test-session",
            "turns": [
                {"role": "user", "content": "What is 2+2?"},
                {"role": "assistant", "content": "4"},
            ],
        }

        result = await client.score(
            trace=trace,
            policy_name="test_policy",
            task_app_id="math",
            options={"outcome": True, "rubric_id": "accuracy"},
        )

        assert result["status"] == "ok"
        assert result["outcome_reward"]["value"] == 0.85

    @pytest.mark.asyncio
    async def test_score_captures_request_params(self, client, monkeypatch) -> None:
        """score should send correct request parameters."""
        captured_payload = {}

        class MockHTTPClient:
            async def __aenter__(self):
                return self

            async def __aexit__(self, *args):
                pass

            async def post_json(self, url, json):
                captured_payload.update(json)
                return {"status": "ok", "outcome_reward": {}, "event_rewards": []}

        monkeypatch.setattr("synth_ai.sdk.judging.client.AsyncHttpClient", lambda *a, **kw: MockHTTPClient())

        trace = {"session_id": "test", "turns": []}

        await client.score(
            trace=trace,
            policy_name="my_policy",
            task_app_id="heartdisease",
            options={"outcome": True, "provider": "groq"},
        )

        assert captured_payload["policy_name"] == "my_policy"
        assert captured_payload["task_app"]["id"] == "heartdisease"
        assert captured_payload["options"]["provider"] == "groq"


class TestJudgeClientErrorHandling:
    """Test JudgeClient error handling."""

    @pytest.fixture
    def client(self):
        """Create JudgeClient instance."""
        from synth_ai.sdk.judging.client import JudgeClient

        return JudgeClient(
            base_url="https://api.usesynth.ai",
            api_key="test-key",
        )

    @pytest.mark.asyncio
    async def test_handles_validation_error(self, client, monkeypatch) -> None:
        """Should handle validation errors (400/422)."""
        from synth_ai.core.http import HTTPError

        class MockHTTPClient:
            async def __aenter__(self):
                return self

            async def __aexit__(self, *args):
                pass

            async def post_json(self, url, json):
                raise HTTPError(
                    status=400,
                    url="/api/judge/v1/score",
                    message="validation_error",
                    body_snippet="Invalid rubric_id",
                )

        monkeypatch.setattr("synth_ai.sdk.judging.client.AsyncHttpClient", lambda *a, **kw: MockHTTPClient())

        with pytest.raises(ValueError, match="judge_validation_error"):
            await client.score(
                trace={},
                policy_name="test",
                task_app_id="test",
                options={},
            )

    @pytest.mark.asyncio
    async def test_handles_auth_error(self, client, monkeypatch) -> None:
        """Should handle auth errors (401/403)."""
        from synth_ai.core.http import HTTPError

        class MockHTTPClient:
            async def __aenter__(self):
                return self

            async def __aexit__(self, *args):
                pass

            async def post_json(self, url, json):
                raise HTTPError(
                    status=401,
                    url="/api/judge/v1/score",
                    message="unauthorized",
                    body_snippet="Invalid API key",
                )

        monkeypatch.setattr("synth_ai.sdk.judging.client.AsyncHttpClient", lambda *a, **kw: MockHTTPClient())

        with pytest.raises(PermissionError, match="judge_auth_error"):
            await client.score(
                trace={},
                policy_name="test",
                task_app_id="test",
                options={},
            )

    @pytest.mark.asyncio
    async def test_handles_not_found_error(self, client, monkeypatch) -> None:
        """Should handle not found errors (404)."""
        from synth_ai.core.http import HTTPError

        class MockHTTPClient:
            async def __aenter__(self):
                return self

            async def __aexit__(self, *args):
                pass

            async def post_json(self, url, json):
                raise HTTPError(
                    status=404,
                    url="/api/judge/v1/score",
                    message="not_found",
                    body_snippet="Task app not found",
                )

        monkeypatch.setattr("synth_ai.sdk.judging.client.AsyncHttpClient", lambda *a, **kw: MockHTTPClient())

        with pytest.raises(FileNotFoundError, match="judge_route_not_found"):
            await client.score(
                trace={},
                policy_name="test",
                task_app_id="nonexistent",
                options={},
            )


class TestJudgeOptions:
    """Test JudgeOptions TypedDict."""

    def test_judge_options_accepts_valid_fields(self) -> None:
        """JudgeOptions should accept valid fields."""
        from synth_ai.sdk.judging.client import JudgeOptions

        options: JudgeOptions = {
            "event": True,
            "outcome": True,
            "rubric_id": "accuracy",
            "provider": "groq",
            "model": "llama-3.1-8b-instant",
        }

        assert options["event"] is True
        assert options["rubric_id"] == "accuracy"
