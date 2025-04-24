import json
from typing import Any, Dict, List, Optional, Tuple, Type

import anthropic
import pydantic
from pydantic import BaseModel

from synth_ai.zyk.lms.caching.initialize import (
    get_cache_handler,
)
from synth_ai.zyk.lms.tools.base import BaseTool
from synth_ai.zyk.lms.vendors.base import BaseLMResponse, VendorBase
from synth_ai.zyk.lms.constants import SPECIAL_BASE_TEMPS, CLAUDE_REASONING_MODELS, SONNET_37_BUDGETS
from synth_ai.zyk.lms.vendors.core.openai_api import OpenAIStructuredOutputClient

ANTHROPIC_EXCEPTIONS_TO_RETRY: Tuple[Type[Exception], ...] = (anthropic.APIError,)


class AnthropicAPI(VendorBase):
    used_for_structured_outputs: bool = True
    exceptions_to_retry: Tuple = ANTHROPIC_EXCEPTIONS_TO_RETRY
    sync_client: Any
    async_client: Any

    def __init__(
        self,
        exceptions_to_retry: Tuple[
            Type[Exception], ...
        ] = ANTHROPIC_EXCEPTIONS_TO_RETRY,
        used_for_structured_outputs: bool = False,
        reasoning_effort: str = "high",
    ):
        self.sync_client = anthropic.Anthropic()
        self.async_client = anthropic.AsyncAnthropic()
        self.used_for_structured_outputs = used_for_structured_outputs
        self.exceptions_to_retry = exceptions_to_retry
        self._openai_fallback = None
        self.reasoning_effort = reasoning_effort

    # @backoff.on_exception(
    #     backoff.expo,
    #     exceptions_to_retry,
    #     max_tries=BACKOFF_TOLERANCE,
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
        **vendor_params: Dict[str, Any],
    ) -> BaseLMResponse:
        assert (
            lm_config.get("response_model", None) is None
        ), "response_model is not supported for standard calls"
        used_cache_handler = get_cache_handler(use_ephemeral_cache_only)
        lm_config["reasoning_effort"] = reasoning_effort
        cache_result = used_cache_handler.hit_managed_cache(
            model, messages, lm_config=lm_config, tools=tools
        )
        if cache_result:
            return cache_result

        # Common API parameters
        api_params = {
            "system": messages[0]["content"],
            "messages": messages[1:],
            "model": model,
            "max_tokens": lm_config.get("max_tokens", 4096),
            "temperature": lm_config.get(
                "temperature", SPECIAL_BASE_TEMPS.get(model, 0)
            ),
        }

        # Add tools if provided
        if tools:
            api_params["tools"] = [tool.to_anthropic_tool() for tool in tools]

        # Only try to add thinking if supported by the SDK
        try:
            import inspect

            create_sig = inspect.signature(self.async_client.messages.create)
            if "thinking" in create_sig.parameters and model in CLAUDE_REASONING_MODELS:
                if reasoning_effort in ["high", "medium"]:
                    budget = SONNET_37_BUDGETS[reasoning_effort]    
                    api_params["thinking"] = {
                        "type": "enabled",
                        "budget_tokens": budget,
                    }
                    api_params["max_tokens"] = budget+4096
                    api_params["temperature"] = 1
        except (ImportError, AttributeError, TypeError):
            pass

        # Make the API call
        response = await self.async_client.messages.create(**api_params)

        # Extract text content and tool calls
        raw_response = ""
        tool_calls = []

        for content in response.content:
            if content.type == "text":
                raw_response += content.text
            elif content.type == "tool_use":
                tool_calls.append(
                    {
                        "id": content.id,
                        "type": "function",
                        "function": {
                            "name": content.name,
                            "arguments": json.dumps(content.input),
                        },
                    }
                )

        lm_response = BaseLMResponse(
            raw_response=raw_response,
            structured_output=None,
            tool_calls=tool_calls if tool_calls else None,
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
        **vendor_params: Dict[str, Any],
    ) -> BaseLMResponse:
        assert (
            lm_config.get("response_model", None) is None
        ), "response_model is not supported for standard calls"
        used_cache_handler = get_cache_handler(
            use_ephemeral_cache_only=use_ephemeral_cache_only
        )
        lm_config["reasoning_effort"] = reasoning_effort
        cache_result = used_cache_handler.hit_managed_cache(
            model, messages, lm_config=lm_config, tools=tools
        )
        if cache_result:
            return cache_result

        # Common API parameters
        api_params = {
            "system": messages[0]["content"],
            "messages": messages[1:],
            "model": model,
            "max_tokens": lm_config.get("max_tokens", 4096),
            "temperature": lm_config.get(
                "temperature", SPECIAL_BASE_TEMPS.get(model, 0)
            ),
        }

        # Add tools if provided
        if tools:
            api_params["tools"] = [tool.to_anthropic_tool() for tool in tools]

        # Only try to add thinking if supported by the SDK
        try:
            import inspect

            create_sig = inspect.signature(self.sync_client.messages.create)
            if "thinking" in create_sig.parameters and model in CLAUDE_REASONING_MODELS:
                api_params["temperature"] = 1
                if reasoning_effort in ["high", "medium"]:
                    budgets = SONNET_37_BUDGETS
                    budget = budgets[reasoning_effort]
                    api_params["thinking"] = {
                        "type": "enabled",
                        "budget_tokens": budget,
                    }
                    api_params["max_tokens"] = budget+4096
                    api_params["temperature"] = 1
        except (ImportError, AttributeError, TypeError):
            pass

        # Make the API call
        response = self.sync_client.messages.create(**api_params)

        # Extract text content and tool calls
        raw_response = ""
        tool_calls = []

        for content in response.content:
            if content.type == "text":
                raw_response += content.text
            elif content.type == "tool_use":
                tool_calls.append(
                    {
                        "id": content.id,
                        "type": "function",
                        "function": {
                            "name": content.name,
                            "arguments": json.dumps(content.input),
                        },
                    }
                )

        lm_response = BaseLMResponse(
            raw_response=raw_response,
            structured_output=None,
            tool_calls=tool_calls if tool_calls else None,
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
        **vendor_params: Dict[str, Any],
    ) -> BaseLMResponse:
        try:
            # First try with Anthropic
            reasoning_effort = vendor_params.get("reasoning_effort", reasoning_effort)
            if model in CLAUDE_REASONING_MODELS:

                #if reasoning_effort in ["high", "medium"]:
                budgets = SONNET_37_BUDGETS
                budget = budgets[reasoning_effort]
                max_tokens = budget+4096
                temperature = 1
                
                response = await self.async_client.messages.create(
                    system=messages[0]["content"],
                    messages=messages[1:],
                    model=model,
                    max_tokens=max_tokens,
                    thinking={"type": "enabled", "budget_tokens": budget},
                    temperature=temperature,
                )
            else:
                response = await self.async_client.messages.create(
                    system=messages[0]["content"],
                    messages=messages[1:],
                    model=model,
                    max_tokens=max_tokens,
                    temperature=temperature,
                )
            result = response.content[0].text
            parsed = json.loads(result)
            lm_response = BaseLMResponse(
                raw_response="",
                structured_output=response_model(**parsed),
                tool_calls=None,
            )
            return lm_response
        except (json.JSONDecodeError, pydantic.ValidationError):
            # If Anthropic fails, fallback to OpenAI
            if self._openai_fallback is None:
                self._openai_fallback = OpenAIStructuredOutputClient()
            return await self._openai_fallback._hit_api_async_structured_output(
                model="gpt-4o",  # Fallback to GPT-4
                messages=messages,
                response_model=response_model,
                temperature=temperature,
                use_ephemeral_cache_only=use_ephemeral_cache_only,
            )

    def _hit_api_sync_structured_output(
        self,
        model: str,
        messages: List[Dict[str, Any]],
        response_model: BaseModel,
        temperature: float,
        use_ephemeral_cache_only: bool = False,
        reasoning_effort: str = "high",
        **vendor_params: Dict[str, Any],
    ) -> BaseLMResponse:
        try:
            # First try with Anthropic
            reasoning_effort = vendor_params.get("reasoning_effort", reasoning_effort)
            import time

            if model in CLAUDE_REASONING_MODELS:
                if reasoning_effort in ["high", "medium"]:
                    budgets = SONNET_37_BUDGETS
                    budget = budgets[reasoning_effort]
                    max_tokens = budget+4096
                    temperature = 1
                response = self.sync_client.messages.create(
                    system=messages[0]["content"],
                    messages=messages[1:],
                    model=model,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    thinking={"type": "enabled", "budget_tokens": budget},
                )
            else:
                response = self.sync_client.messages.create(
                    system=messages[0]["content"],
                    messages=messages[1:],
                    model=model,
                    max_tokens=max_tokens,
                    temperature=temperature,
                )
            # print("Time taken for API call", time.time() - t)
            result = response.content[0].text
            # Try to parse the result as JSON
            parsed = json.loads(result)
            lm_response = BaseLMResponse(
                raw_response="",
                structured_output=response_model(**parsed),
                tool_calls=None,
            )
            return lm_response
        except (json.JSONDecodeError, pydantic.ValidationError):
            # If Anthropic fails, fallback to OpenAI
            print("WARNING - Falling back to OpenAI - THIS IS SLOW")
            if self._openai_fallback is None:
                self._openai_fallback = OpenAIStructuredOutputClient()
            return self._openai_fallback._hit_api_sync_structured_output(
                model="gpt-4o",  # Fallback to GPT-4
                messages=messages,
                response_model=response_model,
                temperature=temperature,
                use_ephemeral_cache_only=use_ephemeral_cache_only,
            )

    async def _process_call_async(
        self,
        messages: List[Dict[str, Any]],
        model: str,
        response_model: BaseModel,
        api_call_method,
        temperature: float = 0.0,
        use_ephemeral_cache_only: bool = False,
        vendor_params: Dict[str, Any] = None,
    ) -> BaseModel:
        vendor_params = vendor_params or {}
        # Each vendor can filter parameters they support
        return await api_call_method(
            messages=messages,
            model=model,
            temperature=temperature,
            use_ephemeral_cache_only=use_ephemeral_cache_only,
            **vendor_params,  # Pass all vendor-specific params
        )
