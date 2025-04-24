import json
import os
from typing import Any, Dict, List, Optional, Tuple, Type

import pydantic
from mistralai import Mistral  # use Mistral as both sync and async client
from pydantic import BaseModel

from synth_ai.zyk.lms.caching.initialize import get_cache_handler
from synth_ai.zyk.lms.tools.base import BaseTool
from synth_ai.zyk.lms.vendors.base import BaseLMResponse, VendorBase
from synth_ai.zyk.lms.constants import SPECIAL_BASE_TEMPS
from synth_ai.zyk.lms.vendors.core.openai_api import OpenAIStructuredOutputClient

# Since the mistralai package doesn't expose an exceptions module,
# we fallback to catching all Exceptions for retry.
MISTRAL_EXCEPTIONS_TO_RETRY: Tuple[Type[Exception], ...] = (Exception,)


class MistralAPI(VendorBase):
    used_for_structured_outputs: bool = True
    exceptions_to_retry: Tuple = MISTRAL_EXCEPTIONS_TO_RETRY
    _openai_fallback: Any

    def __init__(
        self,
        exceptions_to_retry: Tuple[Type[Exception], ...] = MISTRAL_EXCEPTIONS_TO_RETRY,
        used_for_structured_outputs: bool = False,
    ):
        self.used_for_structured_outputs = used_for_structured_outputs
        self.exceptions_to_retry = exceptions_to_retry
        self._openai_fallback = None

    # @backoff.on_exception(
    #     backoff.expo,
    #     MISTRAL_EXCEPTIONS_TO_RETRY,
    #     max_tries=BACKOFF_TOLERANCE,
    #     on_giveup=lambda e: print(e),
    # )
    async def _hit_api_async(
        self,
        model: str,
        messages: List[Dict[str, Any]],
        lm_config: Dict[str, Any],
        response_model: Optional[BaseModel] = None,
        use_ephemeral_cache_only: bool = False,
        reasoning_effort: str = "high",
        tools: Optional[List[BaseTool]] = None,
    ) -> BaseLMResponse:
        assert (
            lm_config.get("response_model", None) is None
        ), "response_model is not supported for standard calls"
        assert not (response_model and tools), "Cannot provide both response_model and tools"
        used_cache_handler = get_cache_handler(use_ephemeral_cache_only)
        lm_config["reasoning_effort"] = reasoning_effort
        cache_result = used_cache_handler.hit_managed_cache(
            model, messages, lm_config=lm_config, tools=tools
        )
        if cache_result:
            assert type(cache_result) in [
                BaseLMResponse,
                str,
            ], f"Expected BaseLMResponse or str, got {type(cache_result)}"
            return (
                cache_result
                if type(cache_result) == BaseLMResponse
                else BaseLMResponse(
                    raw_response=cache_result, structured_output=None, tool_calls=None
                )
            )

        mistral_messages = [
            {"role": msg["role"], "content": msg["content"]} for msg in messages
        ]
        functions = [tool.to_mistral_tool() for tool in tools] if tools else None
        params = {
            "model": model,
            "messages": mistral_messages,
            "max_tokens": lm_config.get("max_tokens", 4096),
            "temperature": lm_config.get(
                "temperature", SPECIAL_BASE_TEMPS.get(model, 0)
            ),
            "stream": False,
            "tool_choice": "auto" if functions else None,
            
        }
        if response_model:
            params["response_format"] = response_model
        elif tools:
            params["tools"] = functions

        async with Mistral(api_key=os.getenv("MISTRAL_API_KEY", "")) as client:
            response = await client.chat.complete_async(**params)

        message = response.choices[0].message
        try:
            raw_response = message.content
        except AttributeError:
            raw_response = ""

        tool_calls = []
        try:
            if message.tool_calls:
                tool_calls = [
                    {
                        "id": call.id,
                        "type": "function",
                        "function": {
                            "name": call.function.name,
                            "arguments": call.function.arguments,
                        },
                    }
                    for call in message.tool_calls
                ]
        except AttributeError:
            pass

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
    #     MISTRAL_EXCEPTIONS_TO_RETRY,
    #     max_tries=BACKOFF_TOLERANCE,
    #     on_giveup=lambda e: print(e),
    # )
    def _hit_api_sync(
        self,
        model: str,
        messages: List[Dict[str, Any]],
        lm_config: Dict[str, Any],
        response_model: Optional[BaseModel] = None,
        use_ephemeral_cache_only: bool = False,
        reasoning_effort: str = "high",
        tools: Optional[List[BaseTool]] = None,
    ) -> BaseLMResponse:
        assert (
            lm_config.get("response_model", None) is None
        ), "response_model is not supported for standard calls"
        assert not (response_model and tools), "Cannot provide both response_model and tools"
       
        used_cache_handler = get_cache_handler(use_ephemeral_cache_only)
        lm_config["reasoning_effort"] = reasoning_effort
        cache_result = used_cache_handler.hit_managed_cache(
            model, messages, lm_config=lm_config, tools=tools
        )
        if cache_result:
            assert type(cache_result) in [
                BaseLMResponse,
                str,
            ], f"Expected BaseLMResponse or str, got {type(cache_result)}"
            return (
                cache_result
                if type(cache_result) == BaseLMResponse
                else BaseLMResponse(
                    raw_response=cache_result, structured_output=None, tool_calls=None
                )
            )

        mistral_messages = [
            {"role": msg["role"], "content": msg["content"]} for msg in messages
        ]
        functions = [tool.to_mistral_tool() for tool in tools] if tools else None
        
        params = {
            "model": model,
            "messages": mistral_messages,
            "max_tokens": lm_config.get("max_tokens", 4096),
            "temperature": lm_config.get(
                "temperature", SPECIAL_BASE_TEMPS.get(model, 0)
            ),
            "stream": False,
            "tool_choice": "auto" if functions else None,
            #"tools": functions,
        }
        if response_model:
            params["response_format"] = response_model
        elif tools:
            params["tools"] = functions

        with Mistral(api_key=os.getenv("MISTRAL_API_KEY", "")) as client:
            response = client.chat.complete(**params)

        message = response.choices[0].message
        try:
            raw_response = message.content
        except AttributeError:
            raw_response = ""

        tool_calls = []
        try:
            if message.tool_calls:
                tool_calls = [
                    {
                        "id": call.id,
                        "type": "function",
                        "function": {
                            "name": call.function.name,
                            "arguments": call.function.arguments,
                        },
                    }
                    for call in message.tool_calls
                ]
        except AttributeError:
            pass

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
    ) -> BaseLMResponse:
        try:
            mistral_messages = [
                {"role": msg["role"], "content": msg["content"]} for msg in messages
            ]
            async with Mistral(api_key=os.getenv("MISTRAL_API_KEY", "")) as client:
                response = await client.chat.complete_async(
                    model=model,
                    messages=mistral_messages,
                    max_tokens=4096,
                    temperature=temperature,
                    stream=False,
                )
            result = response.choices[0].message.content
            parsed = json.loads(result)
            lm_response = BaseLMResponse(
                raw_response="",
                structured_output=response_model(**parsed),
                tool_calls=None,
            )
            return lm_response
        except (json.JSONDecodeError, pydantic.ValidationError):
            if self._openai_fallback is None:
                self._openai_fallback = OpenAIStructuredOutputClient()
            return await self._openai_fallback._hit_api_async_structured_output(
                model="gpt-4o",
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
    ) -> BaseLMResponse:
        try:
            mistral_messages = [
                {"role": msg["role"], "content": msg["content"]} for msg in messages
            ]
            with Mistral(api_key=os.getenv("MISTRAL_API_KEY", "")) as client:
                response = client.chat.complete(
                    model=model,
                    messages=mistral_messages,
                    max_tokens=4096,
                    temperature=temperature,
                    stream=False,
                )
            result = response.choices[0].message.content
            parsed = json.loads(result)
            lm_response = BaseLMResponse(
                raw_response="",
                structured_output=response_model(**parsed),
                tool_calls=None,
            )
            return lm_response
        except (json.JSONDecodeError, pydantic.ValidationError):
            print("WARNING - Falling back to OpenAI - THIS IS SLOW")
            if self._openai_fallback is None:
                self._openai_fallback = OpenAIStructuredOutputClient()
            return self._openai_fallback._hit_api_sync_structured_output(
                model="gpt-4o",
                messages=messages,
                response_model=response_model,
                temperature=temperature,
                use_ephemeral_cache_only=use_ephemeral_cache_only,
            )


if __name__ == "__main__":
    import asyncio

    from pydantic import BaseModel

    class TestModel(BaseModel):
        name: str

    client = MistralAPI(used_for_structured_outputs=True, exceptions_to_retry=[])
    import time

    t = time.time()

    async def run_async():
        response = await client._hit_api_async_structured_output(
            model="mistral-large-latest",
            messages=[{"role": "user", "content": "What is the capital of the moon?"}],
            response_model=TestModel,
            temperature=0.0,
        )
        print(response)
        return response

    response = asyncio.run(run_async())
    t2 = time.time()
    print(f"Got {len(response.name)} chars in {t2-t} seconds")
