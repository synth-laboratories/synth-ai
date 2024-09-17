import json
from typing import Any, Dict, List, Tuple, Type

import openai
import pydantic_core
from openai import AsyncOpenAI, OpenAI
from pydantic import BaseModel

from zyk.src.lms.caching.initialize import get_cache_handler
from zyk.src.lms.vendors.constants import SPECIAL_BASE_TEMPS
from zyk.src.lms.vendors.openai_standard import OpenAIStandard

OPENAI_EXCEPTIONS_TO_RETRY: Tuple[Type[Exception], ...] = (
    pydantic_core._pydantic_core.ValidationError,
    openai.OpenAIError,
    openai.APIConnectionError,
    openai.RateLimitError,
    openai.APIError,
    openai.Timeout,
    openai.InternalServerError,
)

class OpenAIStructuredOutputClient(OpenAIStandard):

    def __init__(self):
        super().__init__(
            used_for_structured_outputs=True,
            exceptions_to_retry=OPENAI_EXCEPTIONS_TO_RETRY,
            sync_client=OpenAI(),
            async_client=AsyncOpenAI()
        )

    async def _hit_api_async_structured_output(self, model: str, messages: List[Dict[str, Any]], response_model: BaseModel, temperature: float, use_ephemeral_cache_only: bool = False) -> str:
        lm_config = {"temperature": temperature, "response_model": response_model}
        used_cache_handler = get_cache_handler(use_ephemeral_cache_only=use_ephemeral_cache_only)
        cache_result = used_cache_handler.hit_cache(model, messages, lm_config=lm_config)
        if cache_result:
            return cache_result["response"] if isinstance(cache_result, dict) else cache_result
        output = await self.async_client.beta.chat.completions.parse(
            model=model,
            messages=messages,
            temperature=lm_config.get("temperature", SPECIAL_BASE_TEMPS.get(model, 0)),
            response_format=response_model
        )
        api_result = response_model(**json.loads(output.choices[0].message.content))
        used_cache_handler.add_to_cache(model, messages, lm_config, output=output.choices[0].message.content)
        return api_result
    def _hit_api_sync_structured_output(self, model: str, messages: List[Dict[str, Any]], response_model: BaseModel, temperature: float, use_ephemeral_cache_only: bool = False) -> str:
        lm_config = {"temperature": temperature, "response_model": response_model}
        used_cache_handler = get_cache_handler(use_ephemeral_cache_only=use_ephemeral_cache_only)
        cache_result = used_cache_handler.hit_cache(model, messages, lm_config=lm_config)
        if cache_result:
            return cache_result["response"] if isinstance(cache_result, dict) else cache_result
        output = self.sync_client.beta.chat.completions.parse(
            model=model,
            messages=messages,
            temperature=lm_config.get("temperature", SPECIAL_BASE_TEMPS.get(model, 0)),
            response_format=response_model
        )
        api_result = response_model(**json.loads(output.choices[0].message.content))
        used_cache_handler.add_to_cache(model, messages, lm_config=lm_config, output=output.choices[0].message.content)
        return api_result
    
class OpenAIPrivate(OpenAIStandard):
    def __init__(self):
        self.sync_client = OpenAI()
        self.async_client = AsyncOpenAI()


if __name__ == "__main__":
    client = OpenAIStructuredOutputClient(
        sync_client=OpenAI(),
        async_client=AsyncOpenAI(),
        used_for_structured_outputs=True,
        exceptions_to_retry=[]
    )
    class TestModel(BaseModel):
        name: str
    sync_model_response = client._hit_api_sync_structured_output(
        model="gpt-4o-mini-2024-07-18",
        messages=[{"role": "user", "content": " What is the capital of the moon?"}],
        response_model=TestModel,
        temperature=0.0
    )
    print(sync_model_response)