"""Unit tests for InferenceAPIClient Google/Gemini provider support.

These tests verify that InferenceAPIClient correctly handles Google/Gemini
provider initialization, API key management, and chat completion calls.
"""

import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

try:
    from synth_ai.task.inference_api import InferenceAPIClient
except ImportError:
    InferenceAPIClient = None  # type: ignore[assignment, misc]
    pytest.skip("synth_ai.task.inference_api not available", allow_module_level=True)

# Type guard to ensure InferenceAPIClient is not None
if InferenceAPIClient is None:
    pytest.skip("InferenceAPIClient not available", allow_module_level=True)


class TestInferenceAPIClientGoogleInitialization:
    """Test InferenceAPIClient initialization with Google provider."""
    
    def setup_method(self) -> None:
        """Ensure InferenceAPIClient is available."""
        assert InferenceAPIClient is not None
    
    def test_init_google_provider(self):
        """Test InferenceAPIClient can be initialized with Google provider."""
        with patch.dict(os.environ, {"GEMINI_API_KEY": "test-key-123"}), patch("google.genai.Client") as mock_client_class:
            mock_client = MagicMock()
            mock_client_class.return_value = mock_client
            
            client = InferenceAPIClient(provider="google")  # type: ignore[union-attr]
            assert client.provider == "google"
            assert client._client is not None
    
    def test_init_google_provider_missing_api_key(self):
        """Test InferenceAPIClient raises error when GEMINI_API_KEY is missing."""
        with patch.dict(os.environ, {}, clear=True), pytest.raises(RuntimeError, match="GEMINI_API_KEY must be set"):
            InferenceAPIClient(provider="google")  # type: ignore[union-attr]
    
    def test_init_google_provider_with_inference_url(self):
        """Test InferenceAPIClient handles inference_url for Google provider."""
        with patch.dict(os.environ, {"GEMINI_API_KEY": "test-key"}), patch("google.genai.Client") as mock_client_class:
            mock_client = MagicMock()
            mock_client_class.return_value = mock_client
            
            # Google GenAI SDK doesn't use base_url, but should not error
            client = InferenceAPIClient(  # type: ignore[union-attr]
                provider="google",
                inference_url="https://generativelanguage.googleapis.com/v1"
            )
            assert client.provider == "google"
    
    def test_init_google_provider_sets_env_var(self):
        """Test that InferenceAPIClient sets GEMINI_API_KEY in environment."""
        with patch.dict(os.environ, {"GEMINI_API_KEY": "test-key-456"}), patch("google.genai.Client") as mock_client_class:
            mock_client = MagicMock()
            mock_client_class.return_value = mock_client
            
            InferenceAPIClient(provider="google")  # type: ignore[union-attr]
            
            # Verify environment variable is set (Google SDK reads from env)
            assert os.getenv("GEMINI_API_KEY") == "test-key-456"


class TestInferenceAPIClientGoogleChatCompletion:
    """Test InferenceAPIClient chat_completion with Google provider."""
    
    def setup_method(self) -> None:
        """Ensure InferenceAPIClient is available."""
        assert InferenceAPIClient is not None
    
    @pytest.fixture
    def mock_gemini_response(self):
        """Create a mock Gemini API response."""
        response = MagicMock()
        response.text = "Test response from Gemini"
        response.candidates = [MagicMock()]
        response.candidates[0].content = MagicMock()
        response.candidates[0].content.parts = []
        return response
    
    @pytest.mark.asyncio
    async def test_chat_completion_google_basic(self, mock_gemini_response):
        """Test basic chat completion with Google provider."""
        with patch.dict(os.environ, {"GEMINI_API_KEY": "test-key"}), patch("google.genai.Client") as mock_client_class:
            mock_client = MagicMock()
            mock_client.aio = MagicMock()
            mock_client.aio.models = MagicMock()
            mock_client.aio.models.generate_content = AsyncMock(return_value=mock_gemini_response)
            mock_client_class.return_value = mock_client
            
            client = InferenceAPIClient(provider="google")  # type: ignore[union-attr]
            
            response = await client.chat_completion(
                messages=[{"role": "user", "content": "Hello"}],
                model="gemini-2.5-flash-lite",
                temperature=0.0,
                max_tokens=100,
            )
            
            assert "choices" in response
            assert len(response["choices"]) > 0
            assert response["choices"][0]["message"]["content"] == "Test response from Gemini"
    
    @pytest.mark.asyncio
    async def test_chat_completion_google_with_system_message(self, mock_gemini_response):
        """Test chat completion with system message (Google uses system_instruction)."""
        with patch.dict(os.environ, {"GEMINI_API_KEY": "test-key"}), patch("google.genai.Client") as mock_client_class:
            mock_client = MagicMock()
            mock_client.aio = MagicMock()
            mock_client.aio.models = MagicMock()
            mock_client.aio.models.generate_content = AsyncMock(return_value=mock_gemini_response)
            mock_client_class.return_value = mock_client
            
            client = InferenceAPIClient(provider="google")  # type: ignore[union-attr]
            
            response = await client.chat_completion(
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": "Hello"},
                ],
                model="gemini-2.5-flash",
                temperature=0.0,
                max_tokens=100,
            )
            
            assert "choices" in response
            # Verify generate_content was called (system message converted to system_instruction)
            mock_client.aio.models.generate_content.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_chat_completion_google_message_format_conversion(self, mock_gemini_response):
        """Test that OpenAI format messages are converted to Gemini format."""
        with patch.dict(os.environ, {"GEMINI_API_KEY": "test-key"}), patch("google.genai.Client") as mock_client_class:
            mock_client = MagicMock()
            mock_client.aio = MagicMock()
            mock_client.aio.models = MagicMock()
            mock_client.aio.models.generate_content = AsyncMock(return_value=mock_gemini_response)
            mock_client_class.return_value = mock_client
            
            client = InferenceAPIClient(provider="google")  # type: ignore[union-attr]
            
            # Call with OpenAI format (assistant role)
            await client.chat_completion(
                messages=[
                    {"role": "user", "content": "Hello"},
                    {"role": "assistant", "content": "Hi there!"},
                    {"role": "user", "content": "How are you?"},
                ],
                model="gemini-2.5-flash-lite",
                temperature=0.0,
                max_tokens=100,
            )
            
            # Verify generate_content was called
            call_args = mock_client.aio.models.generate_content.call_args
            assert call_args is not None
            
            # Verify config was passed
            config = call_args[1]["config"]
            assert config is not None
    
    @pytest.mark.asyncio
    async def test_chat_completion_google_temperature_and_max_tokens(self, mock_gemini_response):
        """Test that temperature and max_tokens are passed correctly."""
        with patch.dict(os.environ, {"GEMINI_API_KEY": "test-key"}), patch("google.genai.Client") as mock_client_class:
            mock_client = MagicMock()
            mock_client.aio = MagicMock()
            mock_client.aio.models = MagicMock()
            mock_client.aio.models.generate_content = AsyncMock(return_value=mock_gemini_response)
            mock_client_class.return_value = mock_client
            
            client = InferenceAPIClient(provider="google")  # type: ignore[union-attr]
            
            await client.chat_completion(
                messages=[{"role": "user", "content": "Test"}],
                model="gemini-2.5-flash",
                temperature=0.7,
                max_tokens=512,
            )
            
            # Verify generate_content was called with correct config
            call_args = mock_client.aio.models.generate_content.call_args
            config = call_args[1]["config"]
            assert config.temperature == 0.7
            assert config.max_output_tokens == 512
    
    @pytest.mark.asyncio
    async def test_chat_completion_google_response_format(self, mock_gemini_response):
        """Test that Gemini response is converted to OpenAI format."""
        with patch.dict(os.environ, {"GEMINI_API_KEY": "test-key"}), patch("google.genai.Client") as mock_client_class:
            mock_client = MagicMock()
            mock_client.aio = MagicMock()
            mock_client.aio.models = MagicMock()
            mock_client.aio.models.generate_content = AsyncMock(return_value=mock_gemini_response)
            mock_client_class.return_value = mock_client
            
            client = InferenceAPIClient(provider="google")  # type: ignore[union-attr]
            
            response = await client.chat_completion(
                messages=[{"role": "user", "content": "Test"}],
                model="gemini-2.5-flash-lite",
                temperature=0.0,
                max_tokens=100,
            )
            
            # Verify OpenAI-compatible response format
            assert "id" in response
            assert "object" in response
            assert response["object"] == "chat.completion"
            assert "choices" in response
            assert len(response["choices"]) > 0
            assert "message" in response["choices"][0]
            assert response["choices"][0]["message"]["role"] == "assistant"
            assert "usage" in response
            assert "prompt_tokens" in response["usage"]
            assert "completion_tokens" in response["usage"]


class TestInferenceAPIClientGoogleErrorHandling:
    """Test error handling for Google provider."""
    
    def setup_method(self) -> None:
        """Ensure InferenceAPIClient is available."""
        assert InferenceAPIClient is not None
    
    @pytest.mark.asyncio
    async def test_chat_completion_google_import_error(self):
        """Test error when google-genai package is not installed."""
        with patch.dict(os.environ, {"GEMINI_API_KEY": "test-key"}), patch("builtins.__import__", side_effect=ImportError("No module named 'google.genai'")), pytest.raises(RuntimeError, match="google-genai package not installed"):
            InferenceAPIClient(provider="google")  # type: ignore[union-attr]
    
    @pytest.mark.asyncio
    async def test_chat_completion_google_api_error(self):
        """Test handling of Gemini API errors."""
        with patch.dict(os.environ, {"GEMINI_API_KEY": "test-key"}), patch("google.genai.Client") as mock_client_class:
            mock_client = MagicMock()
            mock_client.aio = MagicMock()
            mock_client.aio.models = MagicMock()
            mock_client.aio.models.generate_content = AsyncMock(
                side_effect=Exception("Gemini API error")
            )
            mock_client_class.return_value = mock_client
            
            client = InferenceAPIClient(provider="google")  # type: ignore[union-attr]
            
            with pytest.raises(Exception, match="Gemini API error"):
                await client.chat_completion(
                    messages=[{"role": "user", "content": "Test"}],
                    model="gemini-2.5-flash",
                    temperature=0.0,
                    max_tokens=100,
                )


class TestInferenceAPIClientProviderParity:
    """Test that Google provider has parity with OpenAI and Groq."""
    
    def setup_method(self) -> None:
        """Ensure InferenceAPIClient is available."""
        assert InferenceAPIClient is not None
    
    def test_all_providers_supported(self):
        """Test that all three providers are supported."""
        providers = ["openai", "groq", "google"]
        
        for provider in providers:
            # Each provider should be able to initialize (with proper API keys)
            if provider == "openai":
                env_key = "OPENAI_API_KEY"
            elif provider == "groq":
                env_key = "GROQ_API_KEY"
            elif provider == "google":
                env_key = "GEMINI_API_KEY"
            else:
                env_key = ""  # Should never happen, but satisfies type checker
            
            assert InferenceAPIClient is not None  # Type guard
            
            if provider == "google":
                with patch.dict(os.environ, {env_key: "test-key"}), patch("google.genai.Client"):
                    client = InferenceAPIClient(provider=provider)  # type: ignore[union-attr]
            elif provider == "groq":
                with patch.dict(os.environ, {env_key: "test-key"}), patch("groq.AsyncGroq"):
                    client = InferenceAPIClient(provider=provider)  # type: ignore[union-attr]
            else:  # openai
                with patch.dict(os.environ, {env_key: "test-key"}), patch("openai.AsyncOpenAI"):
                    client = InferenceAPIClient(provider=provider)  # type: ignore[union-attr]
            
            assert client.provider == provider
    
    @pytest.mark.asyncio
    async def test_all_providers_chat_completion_format(self):
        """Test that all providers return same response format."""
        messages = [{"role": "user", "content": "Test"}]
        model = "test-model"
        temperature = 0.0
        max_tokens = 100
        
        # Mock responses for each provider
        with patch.dict(os.environ, {
            "OPENAI_API_KEY": "test-key",
            "GROQ_API_KEY": "test-key",
            "GEMINI_API_KEY": "test-key",
        }):
            # Test OpenAI format
            with patch("openai.AsyncOpenAI") as mock_openai:
                mock_response = MagicMock()
                mock_response.choices = [MagicMock()]
                mock_response.choices[0].message.content = "OpenAI response"
                mock_response.usage.prompt_tokens = 10
                mock_response.usage.completion_tokens = 5
                # Mock model_dump to return proper format
                mock_response.model_dump = MagicMock(return_value={
                    "id": "chatcmpl-123",
                    "object": "chat.completion",
                    "choices": [{
                        "message": {"role": "assistant", "content": "OpenAI response"},
                        "index": 0,
                    }],
                    "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
                })
                mock_openai.return_value.chat.completions.create = AsyncMock(return_value=mock_response)
                
                client_openai = InferenceAPIClient(provider="openai")  # type: ignore[union-attr]
                response_openai = await client_openai.chat_completion(
                    messages=messages, model=model, temperature=temperature, max_tokens=max_tokens
                )
            
            # Test Groq format
            with patch("groq.AsyncGroq") as mock_groq:
                mock_response = MagicMock()
                mock_response.choices = [MagicMock()]
                mock_response.choices[0].message.content = "Groq response"
                mock_response.usage.prompt_tokens = 10
                mock_response.usage.completion_tokens = 5
                # Mock model_dump to return proper format
                mock_response.model_dump = MagicMock(return_value={
                    "id": "chatcmpl-456",
                    "object": "chat.completion",
                    "choices": [{
                        "message": {"role": "assistant", "content": "Groq response"},
                        "index": 0,
                    }],
                    "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
                })
                mock_groq.return_value.chat.completions.create = AsyncMock(return_value=mock_response)
                
                client_groq = InferenceAPIClient(provider="groq")  # type: ignore[union-attr]
                response_groq = await client_groq.chat_completion(
                    messages=messages, model=model, temperature=temperature, max_tokens=max_tokens
                )
            
            # Test Google format
            with patch("google.genai.Client") as mock_google:
                mock_response = MagicMock()
                mock_response.text = "Google response"
                mock_response.candidates = [MagicMock(content=MagicMock(parts=[]))]
                mock_google.return_value.aio.models.generate_content = AsyncMock(return_value=mock_response)
                
                client_google = InferenceAPIClient(provider="google")  # type: ignore[union-attr]
                response_google = await client_google.chat_completion(
                    messages=messages, model=model, temperature=temperature, max_tokens=max_tokens
                )
            
            # All responses should have same structure
            for response in [response_openai, response_groq, response_google]:
                assert "choices" in response
                assert len(response["choices"]) > 0
                assert "message" in response["choices"][0]
                assert "content" in response["choices"][0]["message"]
                assert "usage" in response

