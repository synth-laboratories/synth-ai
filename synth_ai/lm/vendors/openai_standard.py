from typing import Any, Dict, List, Optional, Union

import groq
import openai
import pydantic_core
from pydantic import BaseModel

from synth_ai.lm.caching.initialize import (
    get_cache_handler,
)
from synth_ai.lm.tools.base import BaseTool
from synth_ai.lm.vendors.base import BaseLMResponse, VendorBase
from synth_ai.lm.constants import SPECIAL_BASE_TEMPS
from synth_ai.lm.vendors.retries import MAX_BACKOFF
import backoff

DEFAULT_EXCEPTIONS_TO_RETRY = (
    pydantic_core._pydantic_core.ValidationError,
    openai.APIConnectionError,
    openai.APITimeoutError,
    groq.InternalServerError,
    groq.APITimeoutError,
    groq.APIConnectionError,
)


def special_orion_transform(model: str, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Transform messages for O1 series models which don't support system messages.
    
    Args:
        model: Model name to check
        messages: Original messages list
        
    Returns:
        Transformed messages list with system content merged into user message
    """
    if "o1-" in model:
        messages = [
            {
                "role": "user",
                "content": f"<instructions>{messages[0]['content']}</instructions><information>{messages[1]}</information>",
            }
        ]
    return messages


def _silent_backoff_handler(_details):
    """No-op handler to keep stdout clean while still allowing visibility via logging if desired."""
    pass


class OpenAIStandard(VendorBase):
    """
    Standard OpenAI-compatible vendor implementation.
    
    This class provides a standard implementation for OpenAI-compatible APIs,
    including proper retry logic, caching, and support for various model features.
    
    Attributes:
        used_for_structured_outputs: Whether this client supports structured outputs
        exceptions_to_retry: List of exceptions that trigger automatic retries
        sync_client: Synchronous API client
        async_client: Asynchronous API client
    """
    used_for_structured_outputs: bool = True
    exceptions_to_retry: List = DEFAULT_EXCEPTIONS_TO_RETRY
    sync_client: Any
    async_client: Any

    def __init__(
        self,
        sync_client: Any,
        async_client: Any,
        exceptions_to_retry: List[Exception] = DEFAULT_EXCEPTIONS_TO_RETRY,
        used_for_structured_outputs: bool = False,
    ):
        self.sync_client = sync_client
        self.async_client = async_client
        self.used_for_structured_outputs = used_for_structured_outputs
        self.exceptions_to_retry = exceptions_to_retry

    @backoff.on_exception(
        backoff.expo,
        DEFAULT_EXCEPTIONS_TO_RETRY,
        max_time=MAX_BACKOFF,
        jitter=backoff.full_jitter,
        on_backoff=_silent_backoff_handler,
    )
    async def _hit_api_async(
        self,
        model: str,
        messages: List[Dict[str, Any]],
        lm_config: Dict[str, Any],
        use_ephemeral_cache_only: bool = False,
        reasoning_effort: str = "high",
        tools: Optional[List[BaseTool]] = None,
    ) -> BaseLMResponse:
        assert lm_config.get("response_model", None) is None, (
            "response_model is not supported for standard calls"
        )
        messages = special_orion_transform(model, messages)
        used_cache_handler = get_cache_handler(use_ephemeral_cache_only)
        lm_config["reasoning_effort"] = reasoning_effort
        cache_result = used_cache_handler.hit_managed_cache(
            model, messages, lm_config=lm_config, tools=tools
        )
        if cache_result:
            return cache_result

        # Common API call params
        api_params = {
            "model": model,
            "messages": messages,
        }

        # Add tools if provided
        if tools and all(isinstance(tool, BaseTool) for tool in tools):
            api_params["tools"] = [tool.to_openai_tool() for tool in tools]
        elif tools:
            api_params["tools"] = tools

        # Only add temperature for non o1/o3 models
        if not any(prefix in model for prefix in ["o1-", "o3-"]):
            api_params["temperature"] = lm_config.get(
                "temperature", SPECIAL_BASE_TEMPS.get(model, 0)
            )

        # Add reasoning_effort only for o3-mini
        if model in ["o3-mini"]:
            print("Reasoning effort:", reasoning_effort)
            api_params["reasoning_effort"] = reasoning_effort

        output = await self.async_client.chat.completions.create(**api_params)
        message = output.choices[0].message

        # Convert tool calls to dict format
        tool_calls = None
        if message.tool_calls:
            tool_calls = [
                {
                    "id": tc.id,
                    "type": tc.type,
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    },
                }
                for tc in message.tool_calls
            ]

        lm_response = BaseLMResponse(
            raw_response=message.content or "",  # Use empty string if no content
            structured_output=None,
            tool_calls=tool_calls,
        )
        lm_config["reasoning_effort"] = reasoning_effort
        used_cache_handler.add_to_managed_cache(
            model, messages, lm_config=lm_config, output=lm_response, tools=tools
        )
        return lm_response

    @backoff.on_exception(
        backoff.expo,
        DEFAULT_EXCEPTIONS_TO_RETRY,
        max_time=MAX_BACKOFF,
        jitter=backoff.full_jitter,
        on_backoff=_silent_backoff_handler,
    )
    def _hit_api_sync(
        self,
        model: str,
        messages: List[Dict[str, Any]],
        lm_config: Dict[str, Any],
        use_ephemeral_cache_only: bool = False,
        reasoning_effort: str = "high",
        tools: Optional[List[BaseTool]] = None,
    ) -> BaseLMResponse:
        assert lm_config.get("response_model", None) is None, (
            "response_model is not supported for standard calls"
        )
        messages = special_orion_transform(model, messages)
        used_cache_handler = get_cache_handler(use_ephemeral_cache_only=use_ephemeral_cache_only)
        lm_config["reasoning_effort"] = reasoning_effort
        cache_result = used_cache_handler.hit_managed_cache(
            model, messages, lm_config=lm_config, tools=tools
        )
        if cache_result:
            return cache_result

        # Common API call params
        api_params = {
            "model": model,
            "messages": messages,
        }

        # Add tools if provided
        if tools and all(isinstance(tool, BaseTool) for tool in tools):
            api_params["tools"] = [tool.to_openai_tool() for tool in tools]
        elif tools:
            api_params["tools"] = tools

        # Only add temperature for non o1/o3 models
        if not any(prefix in model for prefix in ["o1-", "o3-"]):
            api_params["temperature"] = lm_config.get(
                "temperature", SPECIAL_BASE_TEMPS.get(model, 0)
            )

        # Add reasoning_effort only for o3-mini
        if model in ["o3-mini"]:
            api_params["reasoning_effort"] = reasoning_effort

        output = self.sync_client.chat.completions.create(**api_params)
        message = output.choices[0].message

        # Convert tool calls to dict format
        tool_calls = None
        if message.tool_calls:
            tool_calls = [
                {
                    "id": tc.id,
                    "type": tc.type,
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    },
                }
                for tc in message.tool_calls
            ]

        lm_response = BaseLMResponse(
            raw_response=message.content or "",  # Use empty string if no content
            structured_output=None,
            tool_calls=tool_calls,
        )
        lm_config["reasoning_effort"] = reasoning_effort
        used_cache_handler.add_to_managed_cache(
            model, messages, lm_config=lm_config, output=lm_response, tools=tools
        )
        return lm_response

    async def _hit_api_async_structured_output(
        self,
        model: str,
        messages: List[Dict[str, Any]],
        response_model: BaseModel,
        temperature: float,
        use_ephemeral_cache_only: bool = False,
        reasoning_effort: str = "high",
        tools: Optional[List[BaseTool]] = None,
    ) -> BaseLMResponse:
        lm_config = {
            "temperature": temperature,
            "response_model": response_model,
            "reasoning_effort": reasoning_effort,
        }
        used_cache_handler = get_cache_handler(use_ephemeral_cache_only)
        cache_result: Union[BaseLMResponse, None] = used_cache_handler.hit_managed_cache(
            model, messages, lm_config=lm_config, tools=tools
        )
        if cache_result is not None:
            return cache_result

        # Common API call params
        api_params = {
            "model": model,
            "messages": messages,
        }

        # Add tools if provided
        if tools and all(isinstance(tool, BaseTool) for tool in tools):
            api_params["tools"] = [tool.to_openai_tool() for tool in tools]
        elif tools:
            api_params["tools"] = tools

        # Only add temperature for non o1/o3 models
        if not any(prefix in model for prefix in ["o1-", "o3-"]):
            api_params["temperature"] = lm_config.get(
                "temperature", SPECIAL_BASE_TEMPS.get(model, 0)
            )

        # Add reasoning_effort only for o3-mini
        if model in ["o3-mini"]:
            api_params["reasoning_effort"] = reasoning_effort

        output = await self.async_client.chat.completions.create(**api_params)

        structured_output_api_result = response_model(**output.choices[0].message.content)
        tool_calls = output.choices[0].message.tool_calls
        lm_response = BaseLMResponse(
            raw_response=output.choices[0].message.content,
            structured_output=structured_output_api_result,
            tool_calls=tool_calls,
        )
        lm_config["reasoning_effort"] = reasoning_effort
        used_cache_handler.add_to_managed_cache(
            model, messages, lm_config=lm_config, output=lm_response, tools=tools
        )
        return lm_response

    def _hit_api_sync_structured_output(
        self,
        model: str,
        messages: List[Dict[str, Any]],
        response_model: BaseModel,
        temperature: float,
        use_ephemeral_cache_only: bool = False,
        reasoning_effort: str = "high",
        tools: Optional[List[BaseTool]] = None,
    ) -> BaseLMResponse:
        lm_config = {
            "temperature": temperature,
            "response_model": response_model,
            "reasoning_effort": reasoning_effort,
        }
        used_cache_handler = get_cache_handler(use_ephemeral_cache_only)
        cache_result: Union[BaseLMResponse, None] = used_cache_handler.hit_managed_cache(
            model, messages, lm_config=lm_config, tools=tools
        )
        if cache_result is not None:
            return cache_result

        # Common API call params
        api_params = {
            "model": model,
            "messages": messages,
        }

        # Add tools if provided
        if tools and all(isinstance(tool, BaseTool) for tool in tools):
            api_params["tools"] = [tool.to_openai_tool() for tool in tools]
        elif tools:
            api_params["tools"] = tools

        # Only add temperature for non o1/o3 models
        if not any(prefix in model for prefix in ["o1-", "o3-"]):
            api_params["temperature"] = lm_config.get(
                "temperature", SPECIAL_BASE_TEMPS.get(model, 0)
            )

        # Add reasoning_effort only for o3-mini
        if model in ["o3-mini"]:
            api_params["reasoning_effort"] = reasoning_effort

        output = self.sync_client.chat.completions.create(**api_params)

        structured_output_api_result = response_model(**output.choices[0].message.content)
        tool_calls = output.choices[0].message.tool_calls
        lm_response = BaseLMResponse(
            raw_response=output.choices[0].message.content,
            structured_output=structured_output_api_result,
            tool_calls=tool_calls,
        )
        lm_config["reasoning_effort"] = reasoning_effort
        used_cache_handler.add_to_managed_cache(
            model, messages, lm_config=lm_config, output=lm_response, tools=tools
        )
        return lm_response
