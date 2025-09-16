"""
Vendor client selection and routing.

This module handles the logic for selecting the appropriate vendor client
based on model names or explicit provider specifications.
"""

import re
from re import Pattern
from typing import Any

from synth_ai.lm.core.all import (
    AnthropicClient,
    CustomEndpointClient,
    DeepSeekClient,
    GeminiClient,
    GrokClient,
    GroqClient,
    # OpenAIClient,
    OpenAIStructuredOutputClient,
    OpenRouterClient,
    TogetherClient,
)
from synth_ai.lm.core.synth_models import SYNTH_SUPPORTED_MODELS

# Regular expressions to match model names to their respective providers
openai_naming_regexes: list[Pattern] = [
    re.compile(r"^(ft:)?(o[1,3,4](-.*)?|gpt-.*)$"),
]
openai_formatting_model_regexes: list[Pattern] = [
    re.compile(r"^(ft:)?gpt-4o(-.*)?$"),
]
anthropic_naming_regexes: list[Pattern] = [
    re.compile(r"^claude-.*$"),
]
gemini_naming_regexes: list[Pattern] = [
    re.compile(r"^gemini-.*$"),
    re.compile(r"^gemma[2-9].*$"),
]
deepseek_naming_regexes: list[Pattern] = [
    re.compile(r"^deepseek-.*$"),
]
# Synth-specific model patterns (Qwen3 and fine-tuned models)
synth_naming_regexes: list[Pattern] = [
    re.compile(r"^ft:.*$"),  # Fine-tuned models (ft:model-name)
    re.compile(r"^Qwen/Qwen3.*$"),  # Qwen3 models specifically (Qwen/Qwen3-*)
]

groq_naming_regexes: list[Pattern] = [
    re.compile(r"^llama-3.3-70b-versatile$"),
    re.compile(r"^llama-3.1-8b-instant$"),
    re.compile(r"^qwen-2.5-32b$"),
    re.compile(r"^deepseek-r1-distill-qwen-32b$"),
    re.compile(r"^deepseek-r1-distill-llama-70b-specdec$"),
    re.compile(r"^deepseek-r1-distill-llama-70b$"),
    re.compile(r"^llama-3.3-70b-specdec$"),
    re.compile(r"^llama-3.2-1b-preview$"),
    re.compile(r"^llama-3.2-3b-preview$"),
    re.compile(r"^llama-3.2-11b-vision-preview$"),
    re.compile(r"^llama-3.2-90b-vision-preview$"),
    re.compile(r"^meta-llama/llama-4-scout-17b-16e-instruct$"),
    re.compile(r"^meta-llama/llama-4-maverick-17b-128e-instruct$"),
    re.compile(r"^qwen/qwen3-32b$"),
    re.compile(r"^moonshotai/kimi-k2-instruct$"),
]

grok_naming_regexes: list[Pattern] = [
    re.compile(r"^grok-3-beta$"),
    re.compile(r"^grok-3-mini-beta$"),
    re.compile(r"^grok-beta$"),
    re.compile(r"^grok-.*$"),  # Catch-all for future Grok models
]


openrouter_naming_regexes: list[Pattern] = [
    re.compile(r"^openrouter/.*$"),  # openrouter/model-name pattern
]

openrouter_naming_regexes: list[Pattern] = [
    re.compile(r"^openrouter/.*$"),  # openrouter/model-name pattern
]

# Custom endpoint patterns - check these before generic patterns
custom_endpoint_naming_regexes: list[Pattern] = [
    # Generic domain patterns for custom endpoints
    re.compile(r"^[a-zA-Z0-9\-]+\.[a-zA-Z0-9\-]+\.[a-zA-Z]+$"),  # domain.tld
    re.compile(r"^[a-zA-Z0-9\-]+\.[a-zA-Z0-9\-]+\.[a-zA-Z]+\/[a-zA-Z0-9\-\/]+$"),  # domain.tld/path
]

# Provider mapping for explicit provider overrides
PROVIDER_MAP: dict[str, Any] = {
    "openai": OpenAIStructuredOutputClient,
    "anthropic": AnthropicClient,
    "groq": GroqClient,
    "gemini": GeminiClient,
    "deepseek": DeepSeekClient,
    "grok": GrokClient,
    "openrouter": OpenRouterClient,
    "together": TogetherClient,
    "synth": OpenAIStructuredOutputClient,  # Synth uses OpenAI-compatible API
    "custom_endpoint": CustomEndpointClient,
}


def get_client(
    model_name: str,
    with_formatting: bool = False,
    synth_logging: bool = True,
    provider: str | None = None,
) -> Any:
    """
    Get a vendor client for the specified model.

    Args:
        model_name: The name of the model to use.
        with_formatting: Whether to use formatting capabilities.
        synth_logging: Whether to enable Synth logging.
        provider: Optional provider override. If specified, forces the use of a specific vendor
                 implementation regardless of model name.

    Returns:
        A vendor client instance.

    Raises:
        ValueError: If the provider is unsupported or model name is invalid.
    """
    # print("With formatting", with_formatting)

    # If provider is explicitly specified, use it
    if provider:
        if provider not in PROVIDER_MAP:
            raise ValueError(
                f"Unsupported provider: '{provider}'. Supported providers are: {', '.join(PROVIDER_MAP.keys())}"
            )

        # Log the provider override
        client_class = PROVIDER_MAP[provider]

        # Special handling for OpenAI and Synth with formatting
        if provider in ["openai", "synth"]:
            return client_class(synth_logging=synth_logging)
        # Special handling for Anthropic with formatting
        elif provider == "anthropic" and with_formatting:
            client = client_class()
            client._hit_api_async_structured_output = OpenAIStructuredOutputClient(
                synth_logging=synth_logging
            )._hit_api_async
            return client
        # Custom endpoint needs the model_name as endpoint_url
        elif provider == "custom_endpoint":
            return CustomEndpointClient(endpoint_url=model_name)
        else:
            return client_class()

    # Original regex-based detection
    if any(regex.match(model_name) for regex in openai_naming_regexes):
        # print("Returning OpenAIStructuredOutputClient")
        return OpenAIStructuredOutputClient(
            synth_logging=synth_logging,
        )
    elif any(regex.match(model_name) for regex in anthropic_naming_regexes):
        if with_formatting:
            client = AnthropicClient()
            client._hit_api_async_structured_output = OpenAIStructuredOutputClient(
                synth_logging=synth_logging
            )._hit_api_async
            return client
        else:
            return AnthropicClient()
    elif any(regex.match(model_name) for regex in gemini_naming_regexes):
        return GeminiClient()
    elif any(regex.match(model_name) for regex in deepseek_naming_regexes):
        return DeepSeekClient()
    elif any(regex.match(model_name) for regex in groq_naming_regexes):
        return GroqClient()
    elif any(regex.match(model_name) for regex in grok_naming_regexes):
        return GrokClient()
    elif any(regex.match(model_name) for regex in openrouter_naming_regexes):
        return OpenRouterClient()
    elif any(regex.match(model_name) for regex in custom_endpoint_naming_regexes):
        # Custom endpoints are passed as the endpoint URL
        return CustomEndpointClient(endpoint_url=model_name)
    elif (any(regex.match(model_name) for regex in synth_naming_regexes) or
          model_name in SYNTH_SUPPORTED_MODELS):
        # Synth models use OpenAI-compatible client with custom endpoint
        return OpenAIStructuredOutputClient(synth_logging=synth_logging)
    else:
        raise ValueError(f"Invalid model name: {model_name}")
