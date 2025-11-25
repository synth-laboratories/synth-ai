"""Inference client for model inference via Synth AI.

This module provides a client for making inference requests through Synth AI's
inference proxy, which routes requests to appropriate model providers (OpenAI,
Groq, etc.) based on the model identifier.

Example:
    >>> from synth_ai.sdk.inference import InferenceClient
    >>> 
    >>> client = InferenceClient(
    ...     base_url="https://api.usesynth.ai",
    ...     api_key=os.environ["SYNTH_API_KEY"],
    ... )
    >>> 
    >>> response = await client.create_chat_completion(
    ...     model="gpt-4o-mini",
    ...     messages=[
    ...         {"role": "user", "content": "Hello!"}
    ...     ],
    ...     temperature=0.7,
    ...     max_tokens=100,
    ... )
    >>> 
    >>> print(response["choices"][0]["message"]["content"])
"""

from __future__ import annotations

from typing import Any

from synth_ai.core._utils.http import AsyncHttpClient
from synth_ai.sdk.api.models.supported import (
    UnsupportedModelError,
    normalize_model_identifier,
)


class InferenceClient:
    """Client for making inference requests through Synth AI's inference proxy.
    
    This client provides a unified interface for calling LLMs through Synth AI's
    backend, which handles routing to appropriate providers (OpenAI, Groq, etc.)
    based on the model identifier.
    
    Example:
        >>> from synth_ai.sdk.inference import InferenceClient
        >>> 
        >>> client = InferenceClient(
        ...     base_url="https://api.usesynth.ai",
        ...     api_key=os.environ["SYNTH_API_KEY"],
        ... )
        >>> 
        >>> # Simple completion
        >>> response = await client.create_chat_completion(
        ...     model="gpt-4o-mini",
        ...     messages=[{"role": "user", "content": "Hello!"}],
        ... )
        >>> 
        >>> # With options
        >>> response = await client.create_chat_completion(
        ...     model="Qwen/Qwen3-4B",
        ...     messages=[{"role": "user", "content": "Explain RL"}],
        ...     temperature=0.7,
        ...     max_tokens=500,
        ...     thinking_budget=512,
        ... )
    """
    
    def __init__(self, base_url: str, api_key: str, *, timeout: float = 30.0) -> None:
        """Initialize the inference client.
        
        Args:
            base_url: Base URL for the Synth AI API
            api_key: API key for authentication
            timeout: Request timeout in seconds (default: 30.0)
        """
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key
        self._timeout = timeout

    async def create_chat_completion(
        self, *, model: str, messages: list[dict], **kwargs: Any
    ) -> dict[str, Any]:
        """Create a chat completion request.
        
        This method sends a chat completion request to the Synth AI inference proxy,
        which routes it to the appropriate provider based on the model identifier.
        
        Args:
            model: Model identifier (e.g., "gpt-4o-mini", "Qwen/Qwen3-4B")
            messages: List of message dicts with "role" and "content" keys
            **kwargs: Additional OpenAI-compatible parameters:
                - temperature: Sampling temperature (0.0-2.0)
                - max_tokens: Maximum tokens to generate
                - thinking_budget: Budget for thinking tokens (default: 256)
                - top_p: Nucleus sampling parameter
                - frequency_penalty: Frequency penalty (-2.0 to 2.0)
                - presence_penalty: Presence penalty (-2.0 to 2.0)
                - stop: Stop sequences
                - tools: Function calling tools
                - tool_choice: Tool choice strategy
                - stream: Whether to stream responses
                - ... (other OpenAI API parameters)
                
        Returns:
            Completion response dict with:
                - id: Request ID
                - choices: List of completion choices
                - usage: Token usage statistics
                - ... (other OpenAI-compatible fields)
                
        Raises:
            ValueError: If model is not supported or request is invalid
            HTTPError: If the API request fails
        """
        try:
            normalized_model = normalize_model_identifier(model)
        except UnsupportedModelError as exc:
            raise ValueError(str(exc)) from exc

        body: dict[str, Any] = {"model": normalized_model, "messages": messages}
        body.update(kwargs)
        # Backend now expects an explicit thinking_budget; provide a sensible default if omitted
        if "thinking_budget" not in body:
            body["thinking_budget"] = 256
        async with AsyncHttpClient(self._base_url, self._api_key, timeout=self._timeout) as http:
            # Route through backend inference proxy to Modal
            return await http.post_json("/api/inference/v1/chat/completions", json=body)
