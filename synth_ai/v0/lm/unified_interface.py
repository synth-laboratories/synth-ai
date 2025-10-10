"""
Unified interface for LM providers.
Provides a consistent API for OpenAI and Synth backends.
"""

import logging
from abc import ABC, abstractmethod
from typing import Any

from .config import OpenAIConfig, SynthConfig

logger = logging.getLogger(__name__)


class UnifiedLMProvider(ABC):
    """Abstract base class for LM providers."""

    @abstractmethod
    async def create_chat_completion(
        self, model: str, messages: list[dict[str, Any]], **kwargs
    ) -> dict[str, Any]:
        """Create a chat completion."""
        pass

    @abstractmethod
    async def warmup(self, model: str, **kwargs) -> bool:
        """Warm up the model (optional for providers)."""
        pass

    @abstractmethod
    async def close(self):
        """Clean up resources."""
        pass


class OpenAIProvider(UnifiedLMProvider):
    """OpenAI provider implementation."""

    def __init__(self, api_key: str | None = None, **kwargs):
        """
        Initialize OpenAI provider.

        Args:
            api_key: Optional API key. If not provided, uses OPENAI_API_KEY env var.
            **kwargs: Additional OpenAI client configuration
        """
        try:
            from openai import AsyncOpenAI
        except ImportError as err:
            raise ImportError("OpenAI package not installed. Run: pip install openai") from err

        # Use provided key or load from environment
        if api_key is None:
            config = OpenAIConfig.from_env()
            api_key = config.api_key

        self.client = AsyncOpenAI(api_key=api_key, **kwargs)
        logger.info("Initialized OpenAI provider")

    async def create_chat_completion(
        self, model: str, messages: list[dict[str, Any]], **kwargs
    ) -> dict[str, Any]:
        """Create a chat completion using OpenAI."""
        response = await self.client.chat.completions.create(
            model=model, messages=messages, **kwargs
        )
        # Convert to dict for consistency
        return response.model_dump()

    async def warmup(self, model: str, **kwargs) -> bool:
        """OpenAI doesn't need warmup."""
        logger.debug(f"OpenAI model {model} doesn't require warmup")
        return True

    async def close(self):
        """Close the OpenAI client."""
        if hasattr(self.client, "close"):
            await self.client.close()


class SynthProvider(UnifiedLMProvider):
    """Synth provider implementation."""

    def __init__(self, config: SynthConfig | None = None, **kwargs):
        """
        Initialize Synth provider.

        Args:
            config: Optional SynthConfig. If not provided, loads from environment.
            **kwargs: Additional configuration
        """
        from .vendors.synth_client import AsyncSynthClient

        self.config = config or SynthConfig.from_env()
        self.client = AsyncSynthClient(self.config)

    async def create_chat_completion(
        self, model: str, messages: list[dict[str, Any]], **kwargs
    ) -> dict[str, Any]:
        """Create a chat completion using Synth."""
        return await self.client.chat_completions_create(model=model, messages=messages, **kwargs)

    async def warmup(self, model: str, **kwargs) -> bool:
        """Warm up the Synth model."""
        from .warmup import warmup_synth_model

        return await warmup_synth_model(model, config=self.config, **kwargs)

    async def close(self):
        """Close the Synth client."""
        await self.client.close()


# Factory function
def create_provider(provider_type: str, **config) -> UnifiedLMProvider:
    """
    Create a provider instance.

    Args:
        provider_type: "openai" or "synth"
        **config: Provider-specific configuration

    Returns:
        UnifiedLMProvider instance

    Examples:
        # Create OpenAI provider
        provider = create_provider("openai")

        # Create Synth provider with custom config
        provider = create_provider("synth", timeout=60.0)

        # Create provider based on environment variable
        provider_type = os.getenv("LM_PROVIDER", "synth")
        provider = create_provider(provider_type)
    """
    if provider_type.lower() == "openai":
        return OpenAIProvider(**config)
    elif provider_type.lower() == "synth":
        return SynthProvider(**config)
    else:
        raise ValueError(f"Unknown provider: {provider_type}. Supported providers: openai, synth")


class UnifiedLMClient:
    """
    High-level client that can switch between providers dynamically.
    """

    def __init__(self, default_provider: str = "synth"):
        """
        Initialize unified client.

        Args:
            default_provider: Default provider to use ("openai" or "synth")
        """
        self.default_provider = default_provider
        self._providers: dict[str, UnifiedLMProvider] = {}

    async def _get_provider(self, provider: str | None = None) -> UnifiedLMProvider:
        """Get or create a provider instance."""
        provider_name = provider or self.default_provider

        if provider_name not in self._providers:
            self._providers[provider_name] = create_provider(provider_name)

        return self._providers[provider_name]

    async def create_chat_completion(
        self, model: str, messages: list[dict[str, Any]], provider: str | None = None, **kwargs
    ) -> dict[str, Any]:
        """
        Create a chat completion using specified or default provider.

        Args:
            model: Model identifier
            messages: List of message dicts
            provider: Optional provider override
            **kwargs: Additional parameters

        Returns:
            OpenAI-compatible response dict
        """
        provider_instance = await self._get_provider(provider)
        return await provider_instance.create_chat_completion(model, messages, **kwargs)

    async def warmup(self, model: str, provider: str | None = None, **kwargs) -> bool:
        """Warm up a model on specified provider."""
        provider_instance = await self._get_provider(provider)
        return await provider_instance.warmup(model, **kwargs)

    async def close(self):
        """Close all provider connections."""
        for provider in self._providers.values():
            await provider.close()
        self._providers.clear()

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        await self.close()
