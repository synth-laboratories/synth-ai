"""Integration tests for Inference Client.

These tests validate the inference client functionality:
1. Client initialization
2. Chat completion requests with mocks
3. Model validation

Tests are marked with pytest markers for selective execution:
- @pytest.mark.integration: All integration tests
"""

from __future__ import annotations

import json
from typing import Any

import pytest

pytestmark = [pytest.mark.integration]


class TestInferenceClientInit:
    """Test InferenceClient initialization."""

    def test_client_initializes_with_params(self) -> None:
        """Client should initialize with valid parameters."""
        from synth_ai.sdk.inference.client import InferenceClient

        client = InferenceClient(
            base_url="https://api.usesynth.ai",
            api_key="test-key",
        )

        assert client is not None

    def test_client_accepts_timeout(self) -> None:
        """Client should accept custom timeout."""
        from synth_ai.sdk.inference.client import InferenceClient

        client = InferenceClient(
            base_url="https://api.usesynth.ai",
            api_key="test-key",
            timeout=60.0,
        )

        assert client is not None


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
                    "model": "Qwen/Qwen3-0.6B",
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
            model="Qwen/Qwen3-0.6B",
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
            model="Qwen/Qwen3-0.6B",
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
            model="Qwen/Qwen3-0.6B",
            messages=[{"role": "user", "content": "Hello"}],
            temperature=0.7,
            max_tokens=100,
        )

        assert captured_payload["temperature"] == 0.7
        assert captured_payload["max_tokens"] == 100


class TestInferenceClientModelValidation:
    """Test model validation."""

    @pytest.fixture
    def client(self):
        """Create InferenceClient instance."""
        from synth_ai.sdk.inference.client import InferenceClient

        return InferenceClient(
            base_url="https://api.usesynth.ai",
            api_key="test-key",
        )

    @pytest.mark.asyncio
    async def test_rejects_unsupported_model(self, client) -> None:
        """Should reject unsupported model names."""
        with pytest.raises(ValueError, match="not supported"):
            await client.create_chat_completion(
                model="unknown-model-xyz",
                messages=[{"role": "user", "content": "Hello"}],
            )

    @pytest.mark.asyncio
    async def test_accepts_supported_model(self, client, monkeypatch) -> None:
        """Should accept supported model names."""

        class MockHTTPClient:
            async def __aenter__(self):
                return self

            async def __aexit__(self, *args):
                pass

            async def post_json(self, url, json):
                return {"id": "chatcmpl-123", "choices": [{"message": {"content": "OK"}}]}

        monkeypatch.setattr("synth_ai.sdk.inference.client.AsyncHttpClient", lambda *a, **kw: MockHTTPClient())

        # Should not raise
        await client.create_chat_completion(
            model="Qwen/Qwen3-0.6B",
            messages=[{"role": "user", "content": "Hello"}],
        )


class TestSupportedModels:
    """Test supported model functions."""

    def test_supported_model_ids_returns_list(self) -> None:
        """supported_model_ids should return list of models."""
        from synth_ai.sdk.api.models.supported import supported_model_ids

        models = supported_model_ids()

        assert isinstance(models, list)
        assert len(models) > 0
        assert "Qwen/Qwen3-0.6B" in models

    def test_normalize_model_identifier(self) -> None:
        """normalize_model_identifier should handle valid models."""
        from synth_ai.sdk.api.models.supported import normalize_model_identifier

        result = normalize_model_identifier("Qwen/Qwen3-0.6B")
        assert result == "Qwen/Qwen3-0.6B"

    def test_normalize_rejects_unknown_model(self) -> None:
        """normalize_model_identifier should reject unknown models."""
        from synth_ai.sdk.api.models.supported import (
            UnsupportedModelError,
            normalize_model_identifier,
        )

        with pytest.raises(UnsupportedModelError):
            normalize_model_identifier("unknown/model")
