"""
OpenAI API client implementation.

This module provides the OpenAI-specific implementation of the vendor base class,
supporting both standard and structured output modes.
"""

import json
import os
from typing import Any

import openai
import pydantic_core

# from openai import AsyncOpenAI, OpenAI
from pydantic import BaseModel

from synth_ai.config.base_url import PROD_BASE_URL_DEFAULT
from synth_ai.lm.caching.initialize import get_cache_handler
from synth_ai.lm.constants import OPENAI_REASONING_MODELS, SPECIAL_BASE_TEMPS
from synth_ai.lm.tools.base import BaseTool
from synth_ai.lm.vendors.base import BaseLMResponse
from synth_ai.lm.vendors.openai_standard import OpenAIStandard

# Exceptions that should trigger retry logic for OpenAI API calls
OPENAI_EXCEPTIONS_TO_RETRY: tuple[type[Exception], ...] = (
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
    """
    OpenAI client with support for structured outputs.

    This client extends the standard OpenAI client to support structured outputs
    using OpenAI's native structured output feature or response format parameter.
    """

    def __init__(self, synth_logging: bool = True):
        # Check if we should use Synth clients instead of OpenAI
        openai_base = os.getenv("OPENAI_API_BASE", "")
        prod_prefix = PROD_BASE_URL_DEFAULT.rstrip("/")
        use_synth = (
            openai_base.startswith("https://synth")
            or (prod_prefix and openai_base.startswith(prod_prefix))
            or os.getenv("SYNTH_BASE_URL")
            or os.getenv("MODAL_BASE_URL")
        )

        if use_synth:
            # Use Synth clients for Synth endpoints
            from synth_ai.lm.vendors.synth_client import AsyncSynthClient, SyncSynthClient
            from synth_ai.lm.config import SynthConfig

            # Create config from OPENAI_* environment variables if available
            openai_base = os.getenv("OPENAI_API_BASE")
            openai_key = os.getenv("OPENAI_API_KEY")

            if openai_base and openai_key:
                config = SynthConfig(base_url=openai_base, api_key=openai_key)
                sync_client = SyncSynthClient(config)
                async_client = AsyncSynthClient(config)
            else:
                # Fall back to default config loading
                sync_client = SyncSynthClient()
                async_client = AsyncSynthClient()
        elif synth_logging:
            # print("Using synth logging - OpenAIStructuredOutputClient")
            from synth_ai.lm.provider_support.openai import AsyncOpenAI, OpenAI
            sync_client = OpenAI()
            async_client = AsyncOpenAI()
        else:
            # print("Not using synth logging - OpenAIStructuredOutputClient")
            from openai import AsyncOpenAI, OpenAI
            sync_client = OpenAI()
            async_client = AsyncOpenAI()

        super().__init__(
            used_for_structured_outputs=True,
            exceptions_to_retry=OPENAI_EXCEPTIONS_TO_RETRY,
            sync_client=sync_client,
            async_client=async_client,
        )

    async def _hit_api_async_structured_output(
        self,
        model: str,
        messages: list[dict[str, Any]],
        response_model: BaseModel,
        temperature: float,
        use_ephemeral_cache_only: bool = False,
        tools: list[BaseTool] | None = None,
        reasoning_effort: str = "high",
    ) -> str:
        if tools:
            raise ValueError("Tools are not supported for async structured output")
        # "Hit client")
        lm_config = {
            "temperature": temperature,
            "response_model": response_model,
            "reasoning_effort": reasoning_effort,
        }
        used_cache_handler = get_cache_handler(use_ephemeral_cache_only=use_ephemeral_cache_only)
        cache_result = used_cache_handler.hit_managed_cache(model, messages, lm_config=lm_config)
        if cache_result:
            # print("Hit cache")
            assert type(cache_result) in [
                dict,
                BaseLMResponse,
            ], f"Expected dict or BaseLMResponse, got {type(cache_result)}"
            return cache_result["response"] if isinstance(cache_result, dict) else cache_result
        if model in OPENAI_REASONING_MODELS:
            output = await self.async_client.beta.chat.completions.parse(
                model=model,
                messages=messages,
                temperature=lm_config.get("temperature", SPECIAL_BASE_TEMPS.get(model, 0)),
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
        used_cache_handler.add_to_managed_cache(model, messages, lm_config, output=lm_response)
        return lm_response

    def _hit_api_sync_structured_output(
        self,
        model: str,
        messages: list[dict[str, Any]],
        response_model: BaseModel,
        temperature: float,
        use_ephemeral_cache_only: bool = False,
        tools: list[BaseTool] | None = None,
        reasoning_effort: str = "high",
    ) -> str:
        if tools:
            raise ValueError("Tools are not supported for sync structured output")
        lm_config = {
            "temperature": temperature,
            "response_model": response_model,
            "reasoning_effort": reasoning_effort,
        }
        used_cache_handler = get_cache_handler(use_ephemeral_cache_only=use_ephemeral_cache_only)
        cache_result = used_cache_handler.hit_managed_cache(model, messages, lm_config=lm_config)
        if cache_result:
            assert type(cache_result) in [
                dict,
                BaseLMResponse,
            ], f"Expected dict or BaseLMResponse, got {type(cache_result)}"
            return cache_result["response"] if isinstance(cache_result, dict) else cache_result
        if model in OPENAI_REASONING_MODELS:
            output = self.sync_client.beta.chat.completions.parse(
                model=model,
                messages=messages,
                temperature=lm_config.get("temperature", SPECIAL_BASE_TEMPS.get(model, 0)),
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
            from synth_ai.lm.provider_support.openai import AsyncOpenAI, OpenAI
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
