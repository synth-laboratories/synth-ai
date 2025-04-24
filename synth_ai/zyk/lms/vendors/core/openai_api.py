import json
from typing import Any, Dict, List, Optional, Tuple, Type

import openai
import pydantic_core

# from openai import AsyncOpenAI, OpenAI
from pydantic import BaseModel

from synth_ai.zyk.lms.caching.initialize import get_cache_handler
from synth_ai.zyk.lms.tools.base import BaseTool
from synth_ai.zyk.lms.vendors.base import BaseLMResponse
from synth_ai.zyk.lms.constants import SPECIAL_BASE_TEMPS, OPENAI_REASONING_MODELS
from synth_ai.zyk.lms.vendors.openai_standard import OpenAIStandard

OPENAI_EXCEPTIONS_TO_RETRY: Tuple[Type[Exception], ...] = (
    pydantic_core._pydantic_core.ValidationError,
    openai.OpenAIError,
    openai.APIConnectionError,
    openai.RateLimitError,
    openai.APIError,
    openai.Timeout,
    openai.InternalServerError,
    openai.APIConnectionError,
)


class OpenAIStructuredOutputClient(OpenAIStandard):
    def __init__(self, synth_logging: bool = True):
        if synth_logging:
            # print("Using synth logging - OpenAIStructuredOutputClient")
            from synth_sdk import AsyncOpenAI, OpenAI
        else:
            # print("Not using synth logging - OpenAIStructuredOutputClient")
            from openai import AsyncOpenAI, OpenAI

        super().__init__(
            used_for_structured_outputs=True,
            exceptions_to_retry=OPENAI_EXCEPTIONS_TO_RETRY,
            sync_client=OpenAI(),
            async_client=AsyncOpenAI(),
        )

    async def _hit_api_async_structured_output(
        self,
        model: str,
        messages: List[Dict[str, Any]],
        response_model: BaseModel,
        temperature: float,
        use_ephemeral_cache_only: bool = False,
        tools: Optional[List[BaseTool]] = None,
        reasoning_effort: str = "high",
    ) -> str:
        if tools:
            raise ValueError("Tools are not supported for async structured output")
        # "Hit client")
        lm_config = {"temperature": temperature, "response_model": response_model, "reasoning_effort": reasoning_effort}
        used_cache_handler = get_cache_handler(
            use_ephemeral_cache_only=use_ephemeral_cache_only
        )
        cache_result = used_cache_handler.hit_managed_cache(
            model, messages, lm_config=lm_config
        )
        if cache_result:
            # print("Hit cache")
            assert type(cache_result) in [
                dict,
                BaseLMResponse,
            ], f"Expected dict or BaseLMResponse, got {type(cache_result)}"
            return (
                cache_result["response"] if type(cache_result) == dict else cache_result
            )
        if model in OPENAI_REASONING_MODELS:
            output = await self.async_client.beta.chat.completions.parse(
                model=model,
                messages=messages,
                temperature=lm_config.get(
                    "temperature", SPECIAL_BASE_TEMPS.get(model, 0)
                ),
                response_format=response_model,
                reasoning_effort=reasoning_effort,
            )
        else:
            output = await self.async_client.beta.chat.completions.parse(
                model=model,
                messages=messages,
                response_format=response_model,
            )
        # "Output", output)
        api_result = response_model(**json.loads(output.choices[0].message.content))
        lm_response = BaseLMResponse(
            raw_response="",
            structured_output=api_result,
            tool_calls=None,
        )
        used_cache_handler.add_to_managed_cache(
            model, messages, lm_config, output=lm_response
        )
        return lm_response

    def _hit_api_sync_structured_output(
        self,
        model: str,
        messages: List[Dict[str, Any]],
        response_model: BaseModel,
        temperature: float,
        use_ephemeral_cache_only: bool = False,
        tools: Optional[List[BaseTool]] = None,
        reasoning_effort: str = "high",
    ) -> str:
        if tools:
            raise ValueError("Tools are not supported for sync structured output")
        lm_config = {"temperature": temperature, "response_model": response_model, "reasoning_effort": reasoning_effort}
        used_cache_handler = get_cache_handler(
            use_ephemeral_cache_only=use_ephemeral_cache_only
        )
        cache_result = used_cache_handler.hit_managed_cache(
            model, messages, lm_config=lm_config
        )
        if cache_result:
            assert type(cache_result) in [
                dict,
                BaseLMResponse,
            ], f"Expected dict or BaseLMResponse, got {type(cache_result)}"
            return (
                cache_result["response"] if type(cache_result) == dict else cache_result
            )
        if model in OPENAI_REASONING_MODELS:
            output = self.sync_client.beta.chat.completions.parse(
                model=model,
                messages=messages,
                temperature=lm_config.get(
                    "temperature", SPECIAL_BASE_TEMPS.get(model, 0)
                ),
                response_format=response_model,
                reasoning_effort=reasoning_effort,
            )
        else:
            output = self.sync_client.beta.chat.completions.parse(
                model=model,
                messages=messages,
                response_format=response_model,
            )
        api_result = response_model(**json.loads(output.choices[0].message.content))

        lm_response = BaseLMResponse(
            raw_response="",
            structured_output=api_result,
            tool_calls=None,
        )
        used_cache_handler.add_to_managed_cache(
            model, messages, lm_config=lm_config, output=lm_response
        )
        return lm_response


class OpenAIPrivate(OpenAIStandard):
    def __init__(self, synth_logging: bool = True):
        if synth_logging:
            # print("Using synth logging - OpenAIPrivate")
            from synth_sdk import AsyncOpenAI, OpenAI
        else:
            # print("Not using synth logging - OpenAIPrivate")
            from openai import AsyncOpenAI, OpenAI

        self.sync_client = OpenAI()
        self.async_client = AsyncOpenAI()


if __name__ == "__main__":
    client = OpenAIStructuredOutputClient(
        sync_client=openai.OpenAI(),
        async_client=openai.AsyncOpenAI(),
        used_for_structured_outputs=True,
        exceptions_to_retry=[],
    )

    class TestModel(BaseModel):
        name: str

    sync_model_response = client._hit_api_sync_structured_output(
        model="gpt-4o-mini-2024-07-18",
        messages=[{"role": "user", "content": " What is the capital of the moon?"}],
        response_model=TestModel,
        temperature=0.0,
    )
    # print(sync_model_response)
