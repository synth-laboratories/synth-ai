from typing import Any, Dict, List, Tuple, Type

import anthropic

from zyk.src.lms.caching.initialize import (
    get_cache_handler,
)
from zyk.src.lms.vendors.base import VendorBase
from zyk.src.lms.vendors.constants import SPECIAL_BASE_TEMPS
from zyk.src.lms.vendors.retries import BACKOFF_TOLERANCE, backoff

ANTHROPIC_EXCEPTIONS_TO_RETRY: Tuple[Type[Exception], ...] = (
    anthropic.APIError,
)

class AnthropicAPI(VendorBase):
    used_for_structured_outputs: bool = True
    exceptions_to_retry: Tuple = ANTHROPIC_EXCEPTIONS_TO_RETRY
    sync_client: Any
    async_client: Any

    def __init__(self, exceptions_to_retry: Tuple[Type[Exception], ...] = ANTHROPIC_EXCEPTIONS_TO_RETRY, used_for_structured_outputs: bool = False):
        self.sync_client = anthropic.Anthropic()
        self.async_client = anthropic.AsyncAnthropic()
        self.used_for_structured_outputs = used_for_structured_outputs
        self.exceptions_to_retry = exceptions_to_retry
    
    @backoff.on_exception(
        backoff.expo,
        exceptions_to_retry,
        max_tries=BACKOFF_TOLERANCE,
        on_giveup=lambda e: print(e)
    )
    async def _hit_api_async(self, model: str, messages: List[Dict[str, Any]], lm_config: Dict[str, Any], use_ephemeral_cache_only: bool = False) -> str:
        assert lm_config.get("response_model", None) is None, "response_model is not supported for standard calls"
        used_cache_handler = get_cache_handler(use_ephemeral_cache_only)
        cache_result = used_cache_handler.hit_managed_cache(model, messages, lm_config=lm_config)
        if cache_result:
            return cache_result["response"] if isinstance(cache_result, dict) else cache_result
        response = await self.async_client.messages.create(
            system=messages[0]["content"],
            messages=messages[1:],
            model=model,
            max_tokens=lm_config.get("max_tokens", 4096),
            temperature=lm_config.get("temperature", SPECIAL_BASE_TEMPS.get(model, 0)),
        )
        api_result = response.content[0].text
        used_cache_handler.add_to_managed_cache(model, messages, lm_config=lm_config, output=api_result)
        return api_result

    @backoff.on_exception(
        backoff.expo,
        exceptions_to_retry,
        max_tries=BACKOFF_TOLERANCE,
        on_giveup=lambda e: print(e)
    )
    def _hit_api_sync(self, model: str, messages: List[Dict[str, Any]], lm_config: Dict[str, Any], use_ephemeral_cache_only: bool = False) -> str:
        assert lm_config.get("response_model", None) is None, "response_model is not supported for standard calls"
        used_cache_handler = get_cache_handler(use_ephemeral_cache_only=use_ephemeral_cache_only)
        cache_result = used_cache_handler.hit_managed_cache(model, messages, lm_config=lm_config)
        if cache_result:
            return cache_result["response"] if isinstance(cache_result, dict) else cache_result
        response = self.sync_client.messages.create(
            system=messages[0]["content"],
            messages=messages[1:],
            model=model,
            max_tokens=lm_config.get("max_tokens", 4096),
            temperature=lm_config.get("temperature", SPECIAL_BASE_TEMPS.get(model, 0)),
        )
        api_result = response.content[0].text
        used_cache_handler.add_to_managed_cache(model, messages, lm_config=lm_config, output=api_result)
        return api_result