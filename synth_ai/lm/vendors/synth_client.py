"""
OpenAI-compatible client for Synth backend.
Provides async and sync interfaces matching OpenAI's API.
"""

import asyncio
import json
import logging
import os
from typing import Any, Optional

import httpx

from ..config import SynthConfig

logger = logging.getLogger(__name__)


class ChatInterface:
    """Nested interface to match OpenAI client structure."""

    def __init__(self, client):
        self._client = client
        self.completions = self

    async def create(self, **kwargs):
        """Create chat completion - matches OpenAI interface."""
        result = await self._client.chat_completions_create(**kwargs)
        # If streaming was requested and the result is an async-iterable, return it directly
        if kwargs.get("stream") and hasattr(result, "__aiter__"):
            return result
        # Convert dict response to object-like structure for OpenAI compatibility
        return OpenAIResponse(result)


class OpenAIResponse:
    """Wrapper to make dict response behave like OpenAI response object."""

    def __init__(self, data: dict):
        self._data = data

    @property
    def choices(self):
        return [OpenAIChoice(choice) for choice in self._data.get("choices", [])]

    @property
    def usage(self):
        return self._data.get("usage")

    @property
    def id(self):
        return self._data.get("id")

    @property
    def model(self):
        return self._data.get("model")

    @property
    def object(self):
        return self._data.get("object")


class OpenAIChoice:
    """Wrapper for choice objects."""

    def __init__(self, data: dict):
        self._data = data

    @property
    def message(self):
        return OpenAIMessage(self._data.get("message", {}))

    @property
    def finish_reason(self):
        return self._data.get("finish_reason")


class OpenAIMessage:
    """Wrapper for message objects."""

    def __init__(self, data: dict):
        self._data = data

    @property
    def role(self):
        return self._data.get("role")

    @property
    def content(self):
        return self._data.get("content")

    @property
    def tool_calls(self):
        return self._data.get("tool_calls")


class StreamDelta:
    """Wrapper for stream delta objects."""

    def __init__(self, data: dict):
        self._data = data or {}

    @property
    def content(self) -> Optional[str]:
        return self._data.get("content")


class StreamChoice:
    """Wrapper for stream choice objects."""

    def __init__(self, data: dict):
        self._data = data or {}

    @property
    def delta(self) -> StreamDelta:
        return StreamDelta(self._data.get("delta", {}))


class StreamChunk:
    """Wrapper for stream chunk to expose .choices[0].delta.content."""

    def __init__(self, data: dict):
        self._data = data or {}

    @property
    def choices(self):
        return [StreamChoice(c) for c in self._data.get("choices", [])]


def _wrap_stream_chunk(data: dict) -> StreamChunk:
    return StreamChunk(data)


class AsyncSynthClient:
    """Async client with OpenAI-compatible interface."""

    def __init__(
        self,
        config: SynthConfig | None = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        **_: Any,
    ):
        """Initialize with config or OpenAI-style parameters/env.

        Precedence: explicit args -> OPENAI_* env -> SYNTH_* env -> SynthConfig.from_env().
        """
        if config is None and (api_key or base_url):
            config = SynthConfig(
                base_url=base_url or os.getenv("OPENAI_API_BASE") or os.getenv("SYNTH_BASE_URL"),
                api_key=api_key or os.getenv("OPENAI_API_KEY") or os.getenv("SYNTH_API_KEY"),
            )
        elif config is None and (os.getenv("OPENAI_API_BASE") and os.getenv("OPENAI_API_KEY")):
            config = SynthConfig(
                base_url=os.getenv("OPENAI_API_BASE"),
                api_key=os.getenv("OPENAI_API_KEY"),
            )
        self.config = config or SynthConfig.from_env()
        self._client = None

        # Create nested OpenAI-style interface
        self.chat = ChatInterface(self)
        self.completions = self.chat  # Alias for backward compatibility

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
        messages: list[dict[str, Any]],
        previous_response_id: str | None = None,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | dict[str, Any] | None = "auto",
        **kwargs,
    ) -> dict[str, Any]:
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
        messages: list[dict[str, Any]],
        temperature: float = 0.7,
        max_tokens: int | None = None,
        top_p: float = 1.0,
        frequency_penalty: float = 0.0,
        presence_penalty: float = 0.0,
        stop: str | list[str] | None = None,
        stream: bool = False,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | dict[str, Any] | None = "auto",
        response_format: dict[str, Any] | None = None,
        seed: int | None = None,
        **kwargs,
    ) -> dict[str, Any]:
        """
        Create chat completion with OpenAI-compatible API.
        This method provides the OpenAI client interface structure.
        """
        return await self._chat_completions_create(
            model, messages, temperature, max_tokens, top_p, frequency_penalty,
            presence_penalty, stop, stream, tools, tool_choice, response_format, seed, **kwargs
        )

    async def _chat_completions_create(
        self,
        model: str,
        messages: list[dict[str, Any]],
        temperature: float = 0.7,
        max_tokens: int | None = None,
        top_p: float = 1.0,
        frequency_penalty: float = 0.0,
        presence_penalty: float = 0.0,
        stop: str | list[str] | None = None,
        stream: bool = False,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | dict[str, Any] | None = "auto",
        response_format: dict[str, Any] | None = None,
        seed: int | None = None,
        **kwargs,
    ) -> dict[str, Any]:
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

        # Add any additional kwargs (including thinking_mode and thinking_budget)
        payload.update(kwargs)

        # Apply env defaults for thinking if not set explicitly
        try:
            if "thinking_mode" not in payload:
                env_mode = os.getenv("SYNTH_THINKING_MODE")
                if env_mode in ("think", "no_think"):
                    payload["thinking_mode"] = env_mode
            if "thinking_budget" not in payload:
                env_budget = os.getenv("SYNTH_THINKING_BUDGET")
                if env_budget and str(env_budget).strip().isdigit():
                    payload["thinking_budget"] = int(env_budget)
        except Exception:
            pass

        # Local warn if budget exceeds max_tokens (do not mutate payload)
        try:
            bt = payload.get("thinking_budget")
            mt = payload.get("max_tokens")
            if isinstance(bt, int) and isinstance(mt, int) and bt > mt:
                logger.warning(
                    "thinking_budget (%s) exceeds max_tokens (%s) – forwarding as-is",
                    str(bt), str(mt)
                )
        except Exception:
            pass

        # Retry logic
        for attempt in range(self.config.max_retries):
            try:
                url = f"{self.config.get_base_url_without_v1()}/v1/chat/completions"
                _debug_client = os.getenv("SYNTH_CLIENT_DEBUG") == "1"
                if _debug_client:
                    print(f"🔍 SYNTH DEBUG: Making request to URL: {url}")
                    print(f"🔍 SYNTH DEBUG: Payload keys: {list(payload.keys())}")
                    if "tools" in payload:
                        # Only print counts, avoid dumping tool schemas unless explicitly enabled
                        print(f"🔍 SYNTH DEBUG: Tools in payload: {len(payload['tools'])} tools")

                # If streaming requested, return an async stream adapter
                if stream:
                    async def _astream():
                        await self._ensure_client()
                        async with self._client.stream("POST", url, json=payload) as r:  # type: ignore
                            r.raise_for_status()
                            async for line in r.aiter_lines():
                                if not line:
                                    continue
                                if line.startswith("data:"):
                                    data_line = line[len("data:") :].strip()
                                    if data_line == "[DONE]":
                                        return
                                    try:
                                        chunk = json.loads(data_line)
                                        yield _wrap_stream_chunk(chunk)
                                    except json.JSONDecodeError:
                                        logger.debug("Non-JSON stream line: %s", data_line)

                    class _AsyncStream:
                        def __aiter__(self):
                            return _astream()

                        async def __aenter__(self):
                            return self

                        async def __aexit__(self, *exc):
                            return False

                    return _AsyncStream()

                response = await self._client.post(url, json=payload)

                if _debug_client:
                    print(f"🔍 SYNTH DEBUG: Response status: {response.status_code}")

                if response.status_code == 200:
                    result = response.json()
                    if _debug_client:
                        print(f"🔍 SYNTH DEBUG: Response keys: {list(result.keys())}")
                        if "choices" in result and result["choices"]:
                            choice = result["choices"][0]
                            print(f"🔍 SYNTH DEBUG: Choice keys: {list(choice.keys())}")
                            if "message" in choice:
                                message = choice["message"]
                                print(f"🔍 SYNTH DEBUG: Message keys: {list(message.keys())}")
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


class SyncChatInterface:
    """Nested interface to match OpenAI client structure (sync version)."""

    def __init__(self, client):
        self._client = client
        self.completions = self

    def create(self, **kwargs):
        """Create chat completion - matches OpenAI interface."""
        result = self._client.chat_completions_create(**kwargs)
        # Convert dict response to object-like structure for OpenAI compatibility
        return OpenAIResponse(result)


class SyncSynthClient:
    """Sync client with OpenAI-compatible interface."""

    def __init__(
        self,
        config: SynthConfig | None = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        **_: Any,
    ):
        """Initialize with config or OpenAI-style parameters/env."""
        if config is None and (api_key or base_url):
            config = SynthConfig(
                base_url=base_url or os.getenv("OPENAI_API_BASE") or os.getenv("SYNTH_BASE_URL"),
                api_key=api_key or os.getenv("OPENAI_API_KEY") or os.getenv("SYNTH_API_KEY"),
            )
        elif config is None and (os.getenv("OPENAI_API_BASE") and os.getenv("OPENAI_API_KEY")):
            config = SynthConfig(
                base_url=os.getenv("OPENAI_API_BASE"),
                api_key=os.getenv("OPENAI_API_KEY"),
            )
        self.config = config or SynthConfig.from_env()
        self._client = None

        # Create nested OpenAI-style interface
        self.chat = SyncChatInterface(self)
        self.completions = self.chat  # Alias for backward compatibility

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
        messages: list[dict[str, Any]],
        previous_response_id: str | None = None,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | dict[str, Any] | None = "auto",
        **kwargs,
    ) -> dict[str, Any]:
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
        self, model: str, messages: list[dict[str, Any]], **kwargs
    ) -> dict[str, Any]:
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
def create_async_client(config: SynthConfig | None = None) -> AsyncSynthClient:
    """
    Create async Synth client.

    Args:
        config: Optional SynthConfig. If not provided, loads from environment.

    Returns:
        AsyncSynthClient instance
    """
    return AsyncSynthClient(config)


def create_sync_client(config: SynthConfig | None = None) -> SyncSynthClient:
    """
    Create sync Synth client.

    Args:
        config: Optional SynthConfig. If not provided, loads from environment.

    Returns:
        SyncSynthClient instance
    """
    return SyncSynthClient(config)


# Drop-in replacements for OpenAI clients
# These allow Synth to be used as a complete replacement for OpenAI

class AsyncOpenAI(AsyncSynthClient):
    """
    Drop-in replacement for openai.AsyncOpenAI.

    Use Synth backend instead of OpenAI while maintaining the same API.

    Example:
        from synth_ai.lm.vendors.synth_client import AsyncOpenAI

        client = AsyncOpenAI(
            api_key="sk_live_...",
            base_url="https://synth-backend-dev-docker.onrender.com/api"
        )

        # Works exactly like openai.AsyncOpenAI!
        response = await client.chat.completions.create(
            model="Qwen/Qwen3-0.6B",
            messages=[{"role": "user", "content": "Hello"}]
        )
    """

    def __init__(self, api_key: str | None = None, base_url: str | None = None, **kwargs):
        """
        Initialize AsyncOpenAI-compatible Synth client.

        Args:
            api_key: Synth API key (if not provided, uses SYNTH_API_KEY env var)
            base_url: Synth base URL (if not provided, uses OPENAI_API_BASE env var)
            **kwargs: Additional arguments passed to AsyncSynthClient
        """
        # Handle OpenAI-style initialization
        from ..config import SynthConfig
        if api_key or base_url:
            config = SynthConfig(
                base_url=base_url or os.getenv("OPENAI_API_BASE", "https://synth-backend-dev-docker.onrender.com/api"),
                api_key=api_key or os.getenv("OPENAI_API_KEY", "")
            )
        else:
            # Fallback to environment variables (OPENAI_* first, then SYNTH_*)
            env_base = os.getenv("OPENAI_API_BASE") or os.getenv("SYNTH_BASE_URL")
            env_key = os.getenv("OPENAI_API_KEY") or os.getenv("SYNTH_API_KEY")
            config = SynthConfig(base_url=env_base, api_key=env_key) if env_base and env_key else None

        super().__init__(config, **kwargs)


class OpenAI(SyncSynthClient):
    """
    Drop-in replacement for openai.OpenAI.

    Synchronous version of AsyncOpenAI for Synth backend.
    """

    def __init__(self, api_key: str | None = None, base_url: str | None = None, **kwargs):
        """
        Initialize OpenAI-compatible Synth client.

        Args:
            api_key: Synth API key (if not provided, uses SYNTH_API_KEY env var)
            base_url: Synth base URL (if not provided, uses OPENAI_API_BASE env var)
            **kwargs: Additional arguments passed to SyncSynthClient
        """
        # Handle OpenAI-style initialization
        from ..config import SynthConfig
        if api_key or base_url:
            config = SynthConfig(
                base_url=base_url or os.getenv("OPENAI_API_BASE", "https://synth-backend-dev-docker.onrender.com/api"),
                api_key=api_key or os.getenv("OPENAI_API_KEY", "")
            )
        else:
            env_base = os.getenv("OPENAI_API_BASE") or os.getenv("SYNTH_BASE_URL")
            env_key = os.getenv("OPENAI_API_KEY") or os.getenv("SYNTH_API_KEY")
            config = SynthConfig(base_url=env_base, api_key=env_key) if env_base and env_key else None

        super().__init__(config, **kwargs)


# Convenience imports for easy usage
__all__ = [
    "AsyncSynthClient",
    "SyncSynthClient",
    "AsyncOpenAI",  # Drop-in replacement for openai.AsyncOpenAI
    "OpenAI",       # Drop-in replacement for openai.OpenAI
    "create_async_client",
    "create_sync_client",
    "create_chat_completion_async",
    "create_chat_completion_sync",
    "SynthConfig",
]


# Convenience functions for one-off requests
async def create_chat_completion_async(
    model: str, messages: list[dict[str, Any]], config: SynthConfig | None = None, **kwargs
) -> dict[str, Any]:
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
    model: str, messages: list[dict[str, Any]], config: SynthConfig | None = None, **kwargs
) -> dict[str, Any]:
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
