"""Integration tests for Judge Client.

These tests validate the judge client functionality:
1. Judge configuration and validation
2. Scoring requests and responses
3. Custom judge creation
4. Error handling

Tests are marked with pytest markers for selective execution:
- @pytest.mark.integration: All integration tests
- @pytest.mark.judging: Judging-specific tests
"""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

pytestmark = [pytest.mark.integration, pytest.mark.judging]


class MockHTTPResponse:
    """Mock HTTP response for testing."""

    def __init__(self, status_code: int, json_data: dict[str, Any] | None = None, text: str = ""):
        self.status_code = status_code
        self._json_data = json_data or {}
        self.text = text or json.dumps(self._json_data)
        self.headers = {"content-type": "application/json"}

    def json(self) -> dict[str, Any]:
        return self._json_data


class TestJudgeClientInit:
    """Test JudgeClient initialization."""

    def test_client_creates_with_valid_params(self) -> None:
        """Client should create with valid parameters."""
        from synth_ai.sdk.judging.client import JudgeClient

        client = JudgeClient(
            base_url="https://api.usesynth.ai",
            api_key="test-key",
        )

        assert client.base_url == "https://api.usesynth.ai"

    def test_client_accepts_custom_timeout(self) -> None:
        """Client should accept custom timeout."""
        from synth_ai.sdk.judging.client import JudgeClient

        client = JudgeClient(
            base_url="https://api.usesynth.ai",
            api_key="test-key",
            timeout=60.0,
        )

        assert client.timeout == 60.0


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
                    "score": 0.85,
                    "reasoning": "The response is accurate and helpful.",
                    "metadata": {"judge_model": "gpt-4o-mini"},
                }

        monkeypatch.setattr("synth_ai.sdk.judging.client.AsyncHttpClient", lambda *a, **kw: MockHTTPClient())

        result = await client.score(
            judge_id="accuracy",
            input_text="What is 2+2?",
            output_text="4",
        )

        assert result["score"] == 0.85
        assert "reasoning" in result

    @pytest.mark.asyncio
    async def test_score_with_reference(self, client, monkeypatch) -> None:
        """score should accept reference for comparison."""
        captured_payload = {}

        class MockHTTPClient:
            async def __aenter__(self):
                return self

            async def __aexit__(self, *args):
                pass

            async def post_json(self, url, json):
                captured_payload.update(json)
                return {"score": 0.95}

        monkeypatch.setattr("synth_ai.sdk.judging.client.AsyncHttpClient", lambda *a, **kw: MockHTTPClient())

        await client.score(
            judge_id="similarity",
            input_text="What is 2+2?",
            output_text="The answer is 4.",
            reference_text="4",
        )

        assert captured_payload["reference_text"] == "4"

    @pytest.mark.asyncio
    async def test_score_with_custom_criteria(self, client, monkeypatch) -> None:
        """score should accept custom criteria."""
        captured_payload = {}

        class MockHTTPClient:
            async def __aenter__(self):
                return self

            async def __aexit__(self, *args):
                pass

            async def post_json(self, url, json):
                captured_payload.update(json)
                return {"score": 0.8}

        monkeypatch.setattr("synth_ai.sdk.judging.client.AsyncHttpClient", lambda *a, **kw: MockHTTPClient())

        await client.score(
            judge_id="custom",
            input_text="Write a poem",
            output_text="Roses are red...",
            criteria="Evaluate creativity and originality",
        )

        assert captured_payload["criteria"] == "Evaluate creativity and originality"


class TestJudgeClientBatchScore:
    """Test JudgeClient batch scoring functionality."""

    @pytest.fixture
    def client(self):
        """Create JudgeClient instance."""
        from synth_ai.sdk.judging.client import JudgeClient

        return JudgeClient(
            base_url="https://api.usesynth.ai",
            api_key="test-key",
        )

    @pytest.mark.asyncio
    async def test_batch_score_returns_results(self, client, monkeypatch) -> None:
        """batch_score should return multiple results."""

        class MockHTTPClient:
            async def __aenter__(self):
                return self

            async def __aexit__(self, *args):
                pass

            async def post_json(self, url, json):
                return {
                    "results": [
                        {"score": 0.9, "reasoning": "Correct"},
                        {"score": 0.7, "reasoning": "Partially correct"},
                        {"score": 0.5, "reasoning": "Needs improvement"},
                    ]
                }

        monkeypatch.setattr("synth_ai.sdk.judging.client.AsyncHttpClient", lambda *a, **kw: MockHTTPClient())

        results = await client.batch_score(
            judge_id="accuracy",
            items=[
                {"input": "2+2?", "output": "4"},
                {"input": "3+3?", "output": "7"},
                {"input": "4+4?", "output": "9"},
            ],
        )

        assert len(results["results"]) == 3
        assert results["results"][0]["score"] == 0.9


class TestBuiltInJudges:
    """Test built-in judge types."""

    @pytest.fixture
    def client(self):
        """Create JudgeClient instance."""
        from synth_ai.sdk.judging.client import JudgeClient

        return JudgeClient(
            base_url="https://api.usesynth.ai",
            api_key="test-key",
        )

    @pytest.mark.asyncio
    async def test_accuracy_judge(self, client, monkeypatch) -> None:
        """Accuracy judge should score correctness."""

        class MockHTTPClient:
            async def __aenter__(self):
                return self

            async def __aexit__(self, *args):
                pass

            async def post_json(self, url, json):
                return {"score": 1.0, "reasoning": "Exact match"}

        monkeypatch.setattr("synth_ai.sdk.judging.client.AsyncHttpClient", lambda *a, **kw: MockHTTPClient())

        result = await client.score(
            judge_id="accuracy",
            input_text="What is the capital of France?",
            output_text="Paris",
            reference_text="Paris",
        )

        assert result["score"] == 1.0

    @pytest.mark.asyncio
    async def test_helpfulness_judge(self, client, monkeypatch) -> None:
        """Helpfulness judge should score utility."""

        class MockHTTPClient:
            async def __aenter__(self):
                return self

            async def __aexit__(self, *args):
                pass

            async def post_json(self, url, json):
                return {"score": 0.9, "reasoning": "Very helpful response"}

        monkeypatch.setattr("synth_ai.sdk.judging.client.AsyncHttpClient", lambda *a, **kw: MockHTTPClient())

        result = await client.score(
            judge_id="helpfulness",
            input_text="How do I make coffee?",
            output_text="Here's a step-by-step guide: 1. Grind beans... 2. Add water...",
        )

        assert result["score"] == 0.9

    @pytest.mark.asyncio
    async def test_safety_judge(self, client, monkeypatch) -> None:
        """Safety judge should detect harmful content."""

        class MockHTTPClient:
            async def __aenter__(self):
                return self

            async def __aexit__(self, *args):
                pass

            async def post_json(self, url, json):
                return {"score": 1.0, "reasoning": "No harmful content detected"}

        monkeypatch.setattr("synth_ai.sdk.judging.client.AsyncHttpClient", lambda *a, **kw: MockHTTPClient())

        result = await client.score(
            judge_id="safety",
            input_text="Tell me a joke",
            output_text="Why did the chicken cross the road?",
        )

        assert result["score"] == 1.0


class TestCustomJudgeCreation:
    """Test custom judge creation."""

    @pytest.fixture
    def client(self):
        """Create JudgeClient instance."""
        from synth_ai.sdk.judging.client import JudgeClient

        return JudgeClient(
            base_url="https://api.usesynth.ai",
            api_key="test-key",
        )

    @pytest.mark.asyncio
    async def test_create_custom_judge(self, client, monkeypatch) -> None:
        """Should be able to create custom judge."""

        class MockHTTPClient:
            async def __aenter__(self):
                return self

            async def __aexit__(self, *args):
                pass

            async def post_json(self, url, json):
                return {
                    "judge_id": "custom_poetry_judge",
                    "status": "created",
                }

        monkeypatch.setattr("synth_ai.sdk.judging.client.AsyncHttpClient", lambda *a, **kw: MockHTTPClient())

        result = await client.create_judge(
            name="custom_poetry_judge",
            criteria="Evaluate the poem for meter, rhyme, and emotional impact.",
            scoring_rubric={
                "0": "Not a poem",
                "0.5": "Basic poem structure",
                "1.0": "Excellent poetry",
            },
        )

        assert result["judge_id"] == "custom_poetry_judge"


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
    async def test_handles_invalid_judge_id(self, client, monkeypatch) -> None:
        """Should handle invalid judge ID."""

        class MockHTTPClient:
            async def __aenter__(self):
                return self

            async def __aexit__(self, *args):
                pass

            async def post_json(self, url, json):
                raise Exception("Judge not found: invalid_judge")

        monkeypatch.setattr("synth_ai.sdk.judging.client.AsyncHttpClient", lambda *a, **kw: MockHTTPClient())

        with pytest.raises(Exception, match="Judge not found"):
            await client.score(
                judge_id="invalid_judge",
                input_text="Test",
                output_text="Test",
            )

    @pytest.mark.asyncio
    async def test_handles_missing_input(self, client, monkeypatch) -> None:
        """Should validate required inputs."""
        # This test depends on client-side validation
        with pytest.raises((ValueError, TypeError)):
            await client.score(
                judge_id="accuracy",
                input_text="",  # Empty input
                output_text="Response",
            )


class TestJudgeScoreTypes:
    """Test different score output types."""

    @pytest.fixture
    def client(self):
        """Create JudgeClient instance."""
        from synth_ai.sdk.judging.client import JudgeClient

        return JudgeClient(
            base_url="https://api.usesynth.ai",
            api_key="test-key",
        )

    @pytest.mark.asyncio
    async def test_binary_score(self, client, monkeypatch) -> None:
        """Binary judge should return 0 or 1."""

        class MockHTTPClient:
            async def __aenter__(self):
                return self

            async def __aexit__(self, *args):
                pass

            async def post_json(self, url, json):
                return {"score": 1, "binary": True}

        monkeypatch.setattr("synth_ai.sdk.judging.client.AsyncHttpClient", lambda *a, **kw: MockHTTPClient())

        result = await client.score(
            judge_id="binary_accuracy",
            input_text="Is Python a programming language?",
            output_text="Yes",
            score_type="binary",
        )

        assert result["score"] in (0, 1)

    @pytest.mark.asyncio
    async def test_continuous_score(self, client, monkeypatch) -> None:
        """Continuous judge should return float in [0, 1]."""

        class MockHTTPClient:
            async def __aenter__(self):
                return self

            async def __aexit__(self, *args):
                pass

            async def post_json(self, url, json):
                return {"score": 0.73, "continuous": True}

        monkeypatch.setattr("synth_ai.sdk.judging.client.AsyncHttpClient", lambda *a, **kw: MockHTTPClient())

        result = await client.score(
            judge_id="quality",
            input_text="Write something creative",
            output_text="The moon danced...",
            score_type="continuous",
        )

        assert 0 <= result["score"] <= 1

    @pytest.mark.asyncio
    async def test_categorical_score(self, client, monkeypatch) -> None:
        """Categorical judge should return category label."""

        class MockHTTPClient:
            async def __aenter__(self):
                return self

            async def __aexit__(self, *args):
                pass

            async def post_json(self, url, json):
                return {"score": "positive", "category": "sentiment"}

        monkeypatch.setattr("synth_ai.sdk.judging.client.AsyncHttpClient", lambda *a, **kw: MockHTTPClient())

        result = await client.score(
            judge_id="sentiment",
            input_text="How do you feel about this product?",
            output_text="I love it! Best purchase ever!",
            score_type="categorical",
        )

        assert result["score"] in ("positive", "negative", "neutral")
