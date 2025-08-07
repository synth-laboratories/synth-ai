"""
OpenAI-compatible client for Synth backend.
Provides async and sync interfaces matching OpenAI's API.
"""

import asyncio
import httpx
import json
import logging
from typing import List, Dict, Any, Optional, Union

from ..config import SynthConfig

logger = logging.getLogger(__name__)


class AsyncSynthClient:
    """Async client with OpenAI-compatible interface."""

    def __init__(self, config: Optional[SynthConfig] = None):
        """Initialize with config from environment if not provided."""
        self.config = config or SynthConfig.from_env()
        self._client = None

    async def __aenter__(self):
        self._client = httpx.AsyncClient(
            timeout=self.config.timeout,
            headers={
                "Authorization": f"Bearer {self.config.api_key}",
                "Content-Type": "application/json",
            },
        )
        return self

    async def __aexit__(self, *args):
        if self._client:
            await self._client.aclose()

    async def _ensure_client(self):
        """Ensure client is initialized."""
        if self._client is None:
            self._client = httpx.AsyncClient(
                timeout=self.config.timeout,
                headers={
                    "Authorization": f"Bearer {self.config.api_key}",
                    "Content-Type": "application/json",
                },
            )

    async def responses_create(
        self,
        model: str,
        messages: List[Dict[str, Any]],
        previous_response_id: Optional[str] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: Optional[Union[str, Dict[str, Any]]] = "auto",
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Create response using Synth Responses API.
        
        Args:
            model: Model identifier
            messages: List of message dicts with 'role' and 'content'
            previous_response_id: Optional ID of previous response for thread management
            tools: List of available tools
            tool_choice: How to choose tools
            **kwargs: Additional parameters
            
        Returns:
            Responses API-compatible response dict
        """
        await self._ensure_client()
        
        # Build payload for Responses API
        payload = {
            "model": model,
            "messages": messages,
        }
        
        # Add optional parameters
        if previous_response_id is not None:
            payload["previous_response_id"] = previous_response_id
        if tools is not None:
            payload["tools"] = tools
            payload["tool_choice"] = tool_choice
            
        # Add any additional kwargs
        payload.update(kwargs)
        
        # Retry logic
        for attempt in range(self.config.max_retries):
            try:
                url = f"{self.config.get_base_url_without_v1()}/v1/responses"
                response = await self._client.post(url, json=payload)
                
                if response.status_code == 200:
                    return response.json()
                    
                # Handle rate limits with exponential backoff
                if response.status_code == 429:
                    wait_time = 2**attempt
                    await asyncio.sleep(wait_time)
                    continue
                    
                # Other errors
                response.raise_for_status()
                
            except Exception as e:
                if attempt == self.config.max_retries - 1:
                    logger.error(f"Failed after {self.config.max_retries} attempts: {e}")
                    raise
                await asyncio.sleep(2**attempt)
                
        raise Exception(f"Failed to create response after {self.config.max_retries} attempts")

    async def chat_completions_create(
        self,
        model: str,
        messages: List[Dict[str, Any]],
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        top_p: float = 1.0,
        frequency_penalty: float = 0.0,
        presence_penalty: float = 0.0,
        stop: Optional[Union[str, List[str]]] = None,
        stream: bool = False,
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: Optional[Union[str, Dict[str, Any]]] = "auto",
        response_format: Optional[Dict[str, Any]] = None,
        seed: Optional[int] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Create chat completion with OpenAI-compatible API.

        Args:
            model: Model identifier
            messages: List of message dicts with 'role' and 'content'
            temperature: Sampling temperature (0-2)
            max_tokens: Maximum tokens to generate
            top_p: Nucleus sampling parameter
            frequency_penalty: Frequency penalty (-2 to 2)
            presence_penalty: Presence penalty (-2 to 2)
            stop: Stop sequences
            stream: Whether to stream responses
            tools: List of available tools
            tool_choice: How to choose tools
            response_format: Response format constraints
            seed: Random seed for deterministic output
            **kwargs: Additional parameters

        Returns:
            OpenAI-compatible response dict
        """
        await self._ensure_client()

        # Build payload
        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "top_p": top_p,
            "frequency_penalty": frequency_penalty,
            "presence_penalty": presence_penalty,
            "stream": stream,
        }

        # Add optional parameters
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        if stop is not None:
            payload["stop"] = stop
        if tools is not None:
            payload["tools"] = tools
            payload["tool_choice"] = tool_choice
        if response_format is not None:
            payload["response_format"] = response_format
        if seed is not None:
            payload["seed"] = seed

        # Add any additional kwargs
        payload.update(kwargs)

        # Retry logic
        for attempt in range(self.config.max_retries):
            try:
                url = f"{self.config.get_base_url_without_v1()}/v1/chat/completions"
                print(f"🔍 SYNTH DEBUG: Making request to URL: {url}")
                print(f"🔍 SYNTH DEBUG: Payload keys: {list(payload.keys())}")
                if 'tools' in payload:
                    print(f"🔍 SYNTH DEBUG: Tools in payload: {len(payload['tools'])} tools")
                    print(f"🔍 SYNTH DEBUG: First tool: {json.dumps(payload['tools'][0], indent=2)}")
                
                response = await self._client.post(url, json=payload)
                
                print(f"🔍 SYNTH DEBUG: Response status: {response.status_code}")
                
                if response.status_code == 200:
                    result = response.json()
                    print(f"🔍 SYNTH DEBUG: Response keys: {list(result.keys())}")
                    if 'choices' in result and result['choices']:
                        choice = result['choices'][0]
                        print(f"🔍 SYNTH DEBUG: Choice keys: {list(choice.keys())}")
                        if 'message' in choice:
                            message = choice['message']
                            print(f"🔍 SYNTH DEBUG: Message keys: {list(message.keys())}")
                            if 'tool_calls' in message:
                                print(f"🔍 SYNTH DEBUG: Tool calls: {message['tool_calls']}")
                            else:
                                print(f"🔍 SYNTH DEBUG: No tool_calls in message")
                                print(f"🔍 SYNTH DEBUG: Message content: {message.get('content', 'N/A')[:200]}...")
                    return result

                # Handle rate limits with exponential backoff
                if response.status_code == 429:
                    wait_time = 2**attempt
                    logger.warning(f"Rate limited, waiting {wait_time}s...")
                    await asyncio.sleep(wait_time)
                    continue

                # Other errors
                error_msg = f"API error {response.status_code}: {response.text}"
                logger.error(error_msg)
                raise Exception(error_msg)

            except httpx.TimeoutException:
                if attempt < self.config.max_retries - 1:
                    logger.warning(f"Timeout on attempt {attempt + 1}, retrying...")
                    continue
                raise
            except Exception as e:
                if attempt < self.config.max_retries - 1 and "rate" in str(e).lower():
                    wait_time = 2**attempt
                    logger.warning(f"Error on attempt {attempt + 1}, waiting {wait_time}s...")
                    await asyncio.sleep(wait_time)
                    continue
                raise

        raise Exception(f"Failed after {self.config.max_retries} attempts")

    async def close(self):
        """Close the client."""
        if self._client:
            await self._client.aclose()


class SyncSynthClient:
    """Sync client with OpenAI-compatible interface."""

    def __init__(self, config: Optional[SynthConfig] = None):
        """Initialize with config from environment if not provided."""
        self.config = config or SynthConfig.from_env()
        self._client = None

    def __enter__(self):
        self._client = httpx.Client(
            timeout=self.config.timeout,
            headers={
                "Authorization": f"Bearer {self.config.api_key}",
                "Content-Type": "application/json",
            },
        )
        return self

    def __exit__(self, *args):
        if self._client:
            self._client.close()

    def _ensure_client(self):
        """Ensure client is initialized."""
        if self._client is None:
            self._client = httpx.Client(
                timeout=self.config.timeout,
                headers={
                    "Authorization": f"Bearer {self.config.api_key}",
                    "Content-Type": "application/json",
                },
            )

    def responses_create(
        self,
        model: str,
        messages: List[Dict[str, Any]],
        previous_response_id: Optional[str] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: Optional[Union[str, Dict[str, Any]]] = "auto",
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Create response using Synth Responses API (sync version).
        
        See AsyncSynthClient.responses_create for full parameter documentation.
        """
        self._ensure_client()
        
        # Build payload for Responses API
        payload = {
            "model": model,
            "messages": messages,
        }
        
        # Add optional parameters
        if previous_response_id is not None:
            payload["previous_response_id"] = previous_response_id
        if tools is not None:
            payload["tools"] = tools
            payload["tool_choice"] = tool_choice
            
        # Add any additional kwargs
        payload.update(kwargs)
        
        # Retry logic
        for attempt in range(self.config.max_retries):
            try:
                response = self._client.post(
                    f"{self.config.get_base_url_without_v1()}/v1/responses", json=payload
                )
                
                if response.status_code == 200:
                    return response.json()
                    
                # Handle rate limits
                if response.status_code == 429:
                    wait_time = 2**attempt
                    logger.warning(f"Rate limited, waiting {wait_time}s...")
                    import time
                    time.sleep(wait_time)
                    continue
                    
                # Other errors
                error_msg = f"API error {response.status_code}: {response.text}"
                logger.error(error_msg)
                raise Exception(error_msg)
                
            except httpx.TimeoutException:
                if attempt < self.config.max_retries - 1:
                    logger.warning(f"Timeout on attempt {attempt + 1}, retrying...")
                    continue
                raise
                
        raise Exception(f"Failed after {self.config.max_retries} attempts")

    def chat_completions_create(
        self, model: str, messages: List[Dict[str, Any]], **kwargs
    ) -> Dict[str, Any]:
        """
        Create chat completion with OpenAI-compatible API (sync version).

        See AsyncSynthClient.chat_completions_create for full parameter documentation.
        """
        self._ensure_client()

        # Build payload (same as async version)
        payload = {"model": model, "messages": messages, **kwargs}

        # Retry logic
        for attempt in range(self.config.max_retries):
            try:
                response = self._client.post(
                    f"{self.config.get_base_url_without_v1()}/v1/chat/completions", json=payload
                )

                if response.status_code == 200:
                    return response.json()

                # Handle rate limits
                if response.status_code == 429:
                    wait_time = 2**attempt
                    logger.warning(f"Rate limited, waiting {wait_time}s...")
                    import time

                    time.sleep(wait_time)
                    continue

                # Other errors
                error_msg = f"API error {response.status_code}: {response.text}"
                logger.error(error_msg)
                raise Exception(error_msg)

            except httpx.TimeoutException:
                if attempt < self.config.max_retries - 1:
                    logger.warning(f"Timeout on attempt {attempt + 1}, retrying...")
                    continue
                raise

        raise Exception(f"Failed after {self.config.max_retries} attempts")

    def close(self):
        """Close the client."""
        if self._client:
            self._client.close()


# Factory functions for easy instantiation
def create_async_client(config: Optional[SynthConfig] = None) -> AsyncSynthClient:
    """
    Create async Synth client.

    Args:
        config: Optional SynthConfig. If not provided, loads from environment.

    Returns:
        AsyncSynthClient instance
    """
    return AsyncSynthClient(config)


def create_sync_client(config: Optional[SynthConfig] = None) -> SyncSynthClient:
    """
    Create sync Synth client.

    Args:
        config: Optional SynthConfig. If not provided, loads from environment.

    Returns:
        SyncSynthClient instance
    """
    return SyncSynthClient(config)


# Convenience functions for one-off requests
async def create_chat_completion_async(
    model: str, messages: List[Dict[str, Any]], config: Optional[SynthConfig] = None, **kwargs
) -> Dict[str, Any]:
    """
    Create a chat completion with automatic client management.

    Args:
        model: Model identifier
        messages: List of message dicts
        config: Optional SynthConfig
        **kwargs: Additional parameters for chat completion

    Returns:
        OpenAI-compatible response dict
    """
    async with create_async_client(config) as client:
        return await client.chat_completions_create(model, messages, **kwargs)


def create_chat_completion_sync(
    model: str, messages: List[Dict[str, Any]], config: Optional[SynthConfig] = None, **kwargs
) -> Dict[str, Any]:
    """
    Create a chat completion with automatic client management (sync version).

    Args:
        model: Model identifier
        messages: List of message dicts
        config: Optional SynthConfig
        **kwargs: Additional parameters for chat completion

    Returns:
        OpenAI-compatible response dict
    """
    with create_sync_client(config) as client:
        return client.chat_completions_create(model, messages, **kwargs)
