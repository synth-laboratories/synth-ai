"""Integration tests for Inference Client.

These tests validate the inference client functionality:
1. Chat completion requests
2. Streaming responses
3. Error handling
4. Model selection

Tests are marked with pytest markers for selective execution:
- @pytest.mark.integration: All integration tests
- @pytest.mark.inference: Inference-specific tests
"""

from __future__ import annotations

import json
from typing import Any, AsyncIterator
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

pytestmark = [pytest.mark.integration, pytest.mark.inference]


class MockHTTPResponse:
    """Mock HTTP response for testing."""

    def __init__(self, status_code: int, json_data: dict[str, Any] | None = None, text: str = ""):
        self.status_code = status_code
        self._json_data = json_data or {}
        self.text = text or json.dumps(self._json_data)
        self.headers = {"content-type": "application/json"}

    def json(self) -> dict[str, Any]:
        return self._json_data


class TestInferenceClientInit:
    """Test InferenceClient initialization."""

    def test_client_requires_base_url(self) -> None:
        """Client should require base_url."""
        from synth_ai.sdk.inference.client import InferenceClient

        with pytest.raises(ValueError):
            InferenceClient(base_url="", api_key="test-key")

    def test_client_requires_api_key(self) -> None:
        """Client should require api_key."""
        from synth_ai.sdk.inference.client import InferenceClient

        with pytest.raises(ValueError):
            InferenceClient(base_url="https://api.usesynth.ai", api_key="")

    def test_client_accepts_valid_params(self) -> None:
        """Client should accept valid parameters."""
        from synth_ai.sdk.inference.client import InferenceClient

        client = InferenceClient(
            base_url="https://api.usesynth.ai",
            api_key="test-key",
        )

        assert client.base_url == "https://api.usesynth.ai"


class TestInferenceClientChatCompletion:
    """Test InferenceClient chat completion."""

    @pytest.fixture
    def client(self):
        """Create InferenceClient instance."""
        from synth_ai.sdk.inference.client import InferenceClient

        return InferenceClient(
            base_url="https://api.usesynth.ai",
            api_key="test-key",
        )

    @pytest.mark.asyncio
    async def test_create_chat_completion_basic(self, client, monkeypatch) -> None:
        """create_chat_completion should make basic request."""

        class MockHTTPClient:
            async def __aenter__(self):
                return self

            async def __aexit__(self, *args):
                pass

            async def post_json(self, url, json):
                return {
                    "id": "chatcmpl-123",
                    "object": "chat.completion",
                    "model": "gpt-4o-mini",
                    "choices": [
                        {
                            "index": 0,
                            "message": {
                                "role": "assistant",
                                "content": "Hello! How can I help you?",
                            },
                            "finish_reason": "stop",
                        }
                    ],
                }

        monkeypatch.setattr("synth_ai.sdk.inference.client.AsyncHttpClient", lambda *a, **kw: MockHTTPClient())

        response = await client.create_chat_completion(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "Hello"}],
        )

        assert response["id"] == "chatcmpl-123"
        assert response["choices"][0]["message"]["content"] == "Hello! How can I help you?"

    @pytest.mark.asyncio
    async def test_create_chat_completion_with_system(self, client, monkeypatch) -> None:
        """create_chat_completion should accept system message."""
        captured_payload = {}

        class MockHTTPClient:
            async def __aenter__(self):
                return self

            async def __aexit__(self, *args):
                pass

            async def post_json(self, url, json):
                captured_payload.update(json)
                return {
                    "id": "chatcmpl-123",
                    "choices": [{"message": {"content": "OK"}, "finish_reason": "stop"}],
                }

        monkeypatch.setattr("synth_ai.sdk.inference.client.AsyncHttpClient", lambda *a, **kw: MockHTTPClient())

        await client.create_chat_completion(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Hello"},
            ],
        )

        assert len(captured_payload["messages"]) == 2
        assert captured_payload["messages"][0]["role"] == "system"

    @pytest.mark.asyncio
    async def test_create_chat_completion_with_options(self, client, monkeypatch) -> None:
        """create_chat_completion should accept options like temperature."""
        captured_payload = {}

        class MockHTTPClient:
            async def __aenter__(self):
                return self

            async def __aexit__(self, *args):
                pass

            async def post_json(self, url, json):
                captured_payload.update(json)
                return {"id": "chatcmpl-123", "choices": [{"message": {"content": "OK"}}]}

        monkeypatch.setattr("synth_ai.sdk.inference.client.AsyncHttpClient", lambda *a, **kw: MockHTTPClient())

        await client.create_chat_completion(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "Hello"}],
            temperature=0.7,
            max_tokens=100,
        )

        assert captured_payload["temperature"] == 0.7
        assert captured_payload["max_tokens"] == 100


class TestInferenceClientStreaming:
    """Test InferenceClient streaming responses."""

    @pytest.fixture
    def client(self):
        """Create InferenceClient instance."""
        from synth_ai.sdk.inference.client import InferenceClient

        return InferenceClient(
            base_url="https://api.usesynth.ai",
            api_key="test-key",
        )

    @pytest.mark.asyncio
    async def test_create_chat_completion_stream(self, client, monkeypatch) -> None:
        """create_chat_completion_stream should yield chunks."""

        async def mock_stream() -> AsyncIterator[dict]:
            chunks = [
                {"choices": [{"delta": {"content": "Hello"}}]},
                {"choices": [{"delta": {"content": " there"}}]},
                {"choices": [{"delta": {"content": "!"}}]},
            ]
            for chunk in chunks:
                yield chunk

        class MockHTTPClient:
            async def __aenter__(self):
                return self

            async def __aexit__(self, *args):
                pass

            def stream_json_post(self, url, json):
                return mock_stream()

        monkeypatch.setattr("synth_ai.sdk.inference.client.AsyncHttpClient", lambda *a, **kw: MockHTTPClient())

        chunks = []
        async for chunk in client.create_chat_completion_stream(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "Hello"}],
        ):
            chunks.append(chunk)

        assert len(chunks) == 3
        assert chunks[0]["choices"][0]["delta"]["content"] == "Hello"


class TestInferenceClientErrorHandling:
    """Test InferenceClient error handling."""

    @pytest.fixture
    def client(self):
        """Create InferenceClient instance."""
        from synth_ai.sdk.inference.client import InferenceClient

        return InferenceClient(
            base_url="https://api.usesynth.ai",
            api_key="test-key",
        )

    @pytest.mark.asyncio
    async def test_handles_rate_limit_error(self, client, monkeypatch) -> None:
        """Should handle rate limit errors."""

        class MockHTTPClient:
            async def __aenter__(self):
                return self

            async def __aexit__(self, *args):
                pass

            async def post_json(self, url, json):
                raise Exception("Rate limit exceeded")

        monkeypatch.setattr("synth_ai.sdk.inference.client.AsyncHttpClient", lambda *a, **kw: MockHTTPClient())

        with pytest.raises(Exception, match="Rate limit"):
            await client.create_chat_completion(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": "Hello"}],
            )

    @pytest.mark.asyncio
    async def test_handles_invalid_model_error(self, client, monkeypatch) -> None:
        """Should handle invalid model errors."""

        class MockHTTPClient:
            async def __aenter__(self):
                return self

            async def __aexit__(self, *args):
                pass

            async def post_json(self, url, json):
                raise Exception("Invalid model: unknown-model")

        monkeypatch.setattr("synth_ai.sdk.inference.client.AsyncHttpClient", lambda *a, **kw: MockHTTPClient())

        with pytest.raises(Exception, match="Invalid model"):
            await client.create_chat_completion(
                model="unknown-model",
                messages=[{"role": "user", "content": "Hello"}],
            )


class TestInferenceClientModelSelection:
    """Test InferenceClient model selection."""

    @pytest.fixture
    def client(self):
        """Create InferenceClient instance."""
        from synth_ai.sdk.inference.client import InferenceClient

        return InferenceClient(
            base_url="https://api.usesynth.ai",
            api_key="test-key",
        )

    @pytest.mark.asyncio
    async def test_supports_openai_models(self, client, monkeypatch) -> None:
        """Should support OpenAI model names."""
        captured_model = []

        class MockHTTPClient:
            async def __aenter__(self):
                return self

            async def __aexit__(self, *args):
                pass

            async def post_json(self, url, json):
                captured_model.append(json.get("model"))
                return {"id": "chatcmpl-123", "choices": [{"message": {"content": "OK"}}]}

        monkeypatch.setattr("synth_ai.sdk.inference.client.AsyncHttpClient", lambda *a, **kw: MockHTTPClient())

        await client.create_chat_completion(
            model="gpt-4o",
            messages=[{"role": "user", "content": "Hello"}],
        )

        assert captured_model[0] == "gpt-4o"

    @pytest.mark.asyncio
    async def test_supports_synth_models(self, client, monkeypatch) -> None:
        """Should support Synth-specific model names."""
        captured_model = []

        class MockHTTPClient:
            async def __aenter__(self):
                return self

            async def __aexit__(self, *args):
                pass

            async def post_json(self, url, json):
                captured_model.append(json.get("model"))
                return {"id": "chatcmpl-123", "choices": [{"message": {"content": "OK"}}]}

        monkeypatch.setattr("synth_ai.sdk.inference.client.AsyncHttpClient", lambda *a, **kw: MockHTTPClient())

        await client.create_chat_completion(
            model="Qwen/Qwen3-0.6B",
            messages=[{"role": "user", "content": "Hello"}],
        )

        assert captured_model[0] == "Qwen/Qwen3-0.6B"


class TestInferenceClientFineTunedModels:
    """Test InferenceClient with fine-tuned models."""

    @pytest.fixture
    def client(self):
        """Create InferenceClient instance."""
        from synth_ai.sdk.inference.client import InferenceClient

        return InferenceClient(
            base_url="https://api.usesynth.ai",
            api_key="test-key",
        )

    @pytest.mark.asyncio
    async def test_supports_fine_tuned_model_id(self, client, monkeypatch) -> None:
        """Should support fine-tuned model IDs."""
        captured_model = []

        class MockHTTPClient:
            async def __aenter__(self):
                return self

            async def __aexit__(self, *args):
                pass

            async def post_json(self, url, json):
                captured_model.append(json.get("model"))
                return {"id": "chatcmpl-123", "choices": [{"message": {"content": "OK"}}]}

        monkeypatch.setattr("synth_ai.sdk.inference.client.AsyncHttpClient", lambda *a, **kw: MockHTTPClient())

        await client.create_chat_completion(
            model="ft:Qwen/Qwen3-0.6B:my-fine-tune:abc123",
            messages=[{"role": "user", "content": "Hello"}],
        )

        assert captured_model[0] == "ft:Qwen/Qwen3-0.6B:my-fine-tune:abc123"
