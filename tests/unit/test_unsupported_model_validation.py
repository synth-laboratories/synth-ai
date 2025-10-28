

import pytest

pytestmark = pytest.mark.unit

from synth_ai.api.models.supported import UnsupportedModelError
from synth_ai.inference.client import InferenceClient
from synth_ai.learning.client import LearningClient


@pytest.mark.fast
def test_inference_client_rejects_unsupported_model_before_http(monkeypatch):
    """Test that InferenceClient validates model before making HTTP requests."""
    
    # Mock HTTP client to ensure no requests are made
    async def mock_http_factory(*args, **kwargs):
        raise AssertionError("HTTP client should not be constructed for unsupported models")
    
    monkeypatch.setattr("synth_ai.inference.client.AsyncHttpClient", mock_http_factory)
    
    client = InferenceClient(base_url="https://api.example.com", api_key="sk-test")
    
    with pytest.raises(ValueError, match="Model 'gpt-4' is not supported"):
        # This should fail before any HTTP request is made
        import asyncio
        asyncio.run(client.create_chat_completion(
            model="gpt-4",
            messages=[{"role": "user", "content": "Hello"}]
        ))


@pytest.mark.fast
def test_inference_client_rejects_empty_model_before_http(monkeypatch):
    """Test that InferenceClient rejects empty model identifiers."""
    
    async def mock_http_factory(*args, **kwargs):
        raise AssertionError("HTTP client should not be constructed for empty models")
    
    monkeypatch.setattr("synth_ai.inference.client.AsyncHttpClient", mock_http_factory)
    
    client = InferenceClient(base_url="https://api.example.com", api_key="sk-test")
    
    with pytest.raises(ValueError, match="Model identifier is empty"):
        import asyncio
        asyncio.run(client.create_chat_completion(
            model="",
            messages=[{"role": "user", "content": "Hello"}]
        ))


@pytest.mark.fast
def test_inference_client_rejects_unknown_provider_model_before_http(monkeypatch):
    """Test that InferenceClient rejects models from unsupported providers."""
    
    async def mock_http_factory(*args, **kwargs):
        raise AssertionError("HTTP client should not be constructed for unsupported models")
    
    monkeypatch.setattr("synth_ai.inference.client.AsyncHttpClient", mock_http_factory)
    
    client = InferenceClient(base_url="https://api.example.com", api_key="sk-test")
    
    with pytest.raises(ValueError, match="Model 'claude-3-opus' is not supported"):
        import asyncio
        asyncio.run(client.create_chat_completion(
            model="claude-3-opus",
            messages=[{"role": "user", "content": "Hello"}]
        ))


@pytest.mark.fast
def test_learning_client_rejects_unsupported_model_for_sft_before_http(monkeypatch):
    """Test that LearningClient validates model for SFT training before HTTP requests."""
    
    async def mock_http_factory(*args, **kwargs):
        raise AssertionError("HTTP client should not be constructed for unsupported models")
    
    monkeypatch.setattr("synth_ai.learning.client.AsyncHttpClient", mock_http_factory)
    
    client = LearningClient(base_url="https://api.example.com", api_key="sk-test")
    
    with pytest.raises(ValueError, match="Model 'gpt-3.5-turbo' is not supported"):
        import asyncio
        asyncio.run(client.create_job(
            training_type="sft_offline",
            model="gpt-3.5-turbo",
            training_file_id="file-123",
            hyperparameters={"n_epochs": 1}
        ))


@pytest.mark.fast
def test_learning_client_rejects_unsupported_model_for_rl_before_http(monkeypatch):
    """Test that LearningClient validates model for RL training before HTTP requests."""
    
    async def mock_http_factory(*args, **kwargs):
        raise AssertionError("HTTP client should not be constructed for unsupported models")
    
    monkeypatch.setattr("synth_ai.learning.client.AsyncHttpClient", mock_http_factory)
    
    client = LearningClient(base_url="https://api.example.com", api_key="sk-test")
    
    with pytest.raises(ValueError, match="Model 'llama-2-7b' is not supported"):
        import asyncio
        asyncio.run(client.create_job(
            training_type="rl",
            model="llama-2-7b",
            training_file_id="file-123",
            hyperparameters={"batch_size": 1}
        ))


@pytest.mark.fast
def test_learning_client_rejects_empty_model_before_http(monkeypatch):
    """Test that LearningClient rejects empty model identifiers."""
    
    async def mock_http_factory(*args, **kwargs):
        raise AssertionError("HTTP client should not be constructed for empty models")
    
    monkeypatch.setattr("synth_ai.learning.client.AsyncHttpClient", mock_http_factory)
    
    client = LearningClient(base_url="https://api.example.com", api_key="sk-test")
    
    with pytest.raises(ValueError, match="Model identifier is empty"):
        import asyncio
        asyncio.run(client.create_job(
            training_type="sft_offline",
            model="",
            training_file_id="file-123",
            hyperparameters={"n_epochs": 1}
        ))


@pytest.mark.fast
def test_learning_client_rejects_unsupported_finetuned_prefix_before_http(monkeypatch):
    """Test that LearningClient rejects unsupported fine-tuned model prefixes."""
    
    async def mock_http_factory(*args, **kwargs):
        raise AssertionError("HTTP client should not be constructed for unsupported models")
    
    monkeypatch.setattr("synth_ai.learning.client.AsyncHttpClient", mock_http_factory)
    
    client = LearningClient(base_url="https://api.example.com", api_key="sk-test")
    
    with pytest.raises(ValueError, match="Model 'ft:gpt-4:job-123' is not supported"):
        import asyncio
        asyncio.run(client.create_job(
            training_type="sft_offline",
            model="ft:gpt-4:job-123",  # gpt-4 base model is not supported
            training_file_id="file-123",
            hyperparameters={"n_epochs": 1}
        ))


@pytest.mark.fast
def test_learning_client_accepts_supported_model_for_sft():
    """Test that LearningClient accepts supported models for SFT training."""
    
    class MockHTTP:
        def __init__(self, *args, **kwargs):
            self.calls = []
        
        async def __aenter__(self):
            return self
        
        async def __aexit__(self, exc_type, exc, tb):
            return None
        
        async def post_json(self, url, json):
            self.calls.append((url, json))
            return {"id": "job-123"}
    
    client = LearningClient(base_url="https://api.example.com", api_key="sk-test")
    
    # Mock the HTTP client to avoid actual requests
    import synth_ai.learning.client
    original_client = synth_ai.learning.client.AsyncHttpClient
    synth_ai.learning.client.AsyncHttpClient = MockHTTP
    
    try:
        import asyncio
        result = asyncio.run(client.create_job(
            training_type="sft_offline",
            model="Qwen/Qwen3-0.6B",  # This is a supported model
            training_file_id="file-123",
            hyperparameters={"n_epochs": 1}
        ))
        
        assert result == {"id": "job-123"}
        assert len(MockHTTP().calls) == 0  # HTTP client was mocked, no actual calls
    finally:
        synth_ai.learning.client.AsyncHttpClient = original_client


@pytest.mark.fast
def test_inference_client_accepts_supported_model():
    """Test that InferenceClient accepts supported models."""
    
    class MockHTTP:
        def __init__(self, *args, **kwargs):
            self.calls = []
        
        async def __aenter__(self):
            return self
        
        async def __aexit__(self, exc_type, exc, tb):
            return None
        
        async def post_json(self, url, json):
            self.calls.append((url, json))
            return {"choices": [{"message": {"content": "Hello"}}]}
    
    client = InferenceClient(base_url="https://api.example.com", api_key="sk-test")
    
    # Mock the HTTP client to avoid actual requests
    import synth_ai.inference.client
    original_client = synth_ai.inference.client.AsyncHttpClient
    synth_ai.inference.client.AsyncHttpClient = MockHTTP
    
    try:
        import asyncio
        result = asyncio.run(client.create_chat_completion(
            model="Qwen/Qwen3-0.6B",  # This is a supported model
            messages=[{"role": "user", "content": "Hello"}]
        ))
        
        assert "choices" in result
        assert len(MockHTTP().calls) == 0  # HTTP client was mocked, no actual calls
    finally:
        synth_ai.inference.client.AsyncHttpClient = original_client

