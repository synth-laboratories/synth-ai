from typing import Any, Dict, List, Optional, Union

import groq
import openai
import pydantic_core
from pydantic import BaseModel

from synth_ai.zyk.lms.caching.initialize import (
    get_cache_handler,
)
from synth_ai.zyk.lms.tools.base import BaseTool
from synth_ai.zyk.lms.vendors.base import BaseLMResponse, VendorBase
from synth_ai.zyk.lms.constants import SPECIAL_BASE_TEMPS

DEFAULT_EXCEPTIONS_TO_RETRY = (
    pydantic_core._pydantic_core.ValidationError,
    openai.APIConnectionError,
    openai.APITimeoutError,
    groq.InternalServerError,
    groq.APITimeoutError,
    groq.APIConnectionError,
)


def special_orion_transform(
    model: str, messages: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    if "o1-" in model:
        messages = [
            {
                "role": "user",
                "content": f"<instructions>{messages[0]['content']}</instructions><information>{messages[1]}</information>",
            }
        ]
    return messages


def on_backoff_handler_async(details):
    # Print every 5th retry attempt, excluding the first retry
    if details["tries"] > 1 and (details["tries"] - 1) % 5 == 0:
        print(f"Retrying async API call (attempt {details['tries'] - 1})")


def on_backoff_handler_sync(details):
    # Print every 5th retry attempt, excluding the first retry
    if details["tries"] > 1 and (details["tries"] - 1) % 5 == 0:
        print(f"Retrying sync API call (attempt {details['tries'] - 1})")


class OpenAIStandard(VendorBase):
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

    # @backoff.on_exception(
    #     backoff.expo,
    #     exceptions_to_retry,
    #     max_tries=BACKOFF_TOLERANCE,
    #     on_backoff=on_backoff_handler_async,
    #     on_giveup=lambda e: print(e),
    # )
    async def _hit_api_async(
        self,
        model: str,
        messages: List[Dict[str, Any]],
        lm_config: Dict[str, Any],
        use_ephemeral_cache_only: bool = False,
        reasoning_effort: str = "high",
        tools: Optional[List[BaseTool]] = None,
    ) -> BaseLMResponse:
        assert (
            lm_config.get("response_model", None) is None
        ), "response_model is not supported for standard calls"
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

    # @backoff.on_exception(
    #     backoff.expo,
    #     exceptions_to_retry,
    #     max_tries=BACKOFF_TOLERANCE,
    #     on_backoff=on_backoff_handler_sync,
    #     on_giveup=lambda e: print(e),
    # )
    def _hit_api_sync(
        self,
        model: str,
        messages: List[Dict[str, Any]],
        lm_config: Dict[str, Any],
        use_ephemeral_cache_only: bool = False,
        reasoning_effort: str = "high",
        tools: Optional[List[BaseTool]] = None,
    ) -> BaseLMResponse:
        assert (
            lm_config.get("response_model", None) is None
        ), "response_model is not supported for standard calls"
        messages = special_orion_transform(model, messages)
        used_cache_handler = get_cache_handler(
            use_ephemeral_cache_only=use_ephemeral_cache_only
        )
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
        lm_config = {"temperature": temperature, "response_model": response_model, "reasoning_effort": reasoning_effort}
        used_cache_handler = get_cache_handler(use_ephemeral_cache_only)
        cache_result: Union[BaseLMResponse, None] = (
            used_cache_handler.hit_managed_cache(
                model, messages, lm_config=lm_config, tools=tools
            )
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

        structured_output_api_result = response_model(
            **output.choices[0].message.content
        )
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
        lm_config = {"temperature": temperature, "response_model": response_model, "reasoning_effort": reasoning_effort}
        used_cache_handler = get_cache_handler(use_ephemeral_cache_only)
        cache_result: Union[BaseLMResponse, None] = (
            used_cache_handler.hit_managed_cache(
                model, messages, lm_config=lm_config, tools=tools
            )
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

        structured_output_api_result = response_model(
            **output.choices[0].message.content
        )
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
