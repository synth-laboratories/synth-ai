import json
from typing import Any, Dict, List, Tuple, Type

import anthropic
import pydantic
from pydantic import BaseModel

from synth_ai.zyk.lms.caching.initialize import (
    get_cache_handler,
)
from synth_ai.zyk.lms.vendors.base import VendorBase
from synth_ai.zyk.lms.vendors.constants import SPECIAL_BASE_TEMPS
from synth_ai.zyk.lms.vendors.core.openai_api import OpenAIStructuredOutputClient
from synth_ai.zyk.lms.vendors.retries import BACKOFF_TOLERANCE, backoff

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

    @backoff.on_exception(
        backoff.expo,
        exceptions_to_retry,
        max_tries=BACKOFF_TOLERANCE,
        on_giveup=lambda e: print(e),
    )
    async def _hit_api_async(
        self,
        model: str,
        messages: List[Dict[str, Any]],
        lm_config: Dict[str, Any],
        use_ephemeral_cache_only: bool = False,
        reasoning_effort: str = "high",
        **vendor_params: Dict[str, Any],
    ) -> str:
        assert (
            lm_config.get("response_model", None) is None
        ), "response_model is not supported for standard calls"
        used_cache_handler = get_cache_handler(use_ephemeral_cache_only)
        cache_result = used_cache_handler.hit_managed_cache(
            model, messages, lm_config=lm_config
        )
        if cache_result:
            return (
                cache_result["response"]
                if isinstance(cache_result, dict)
                else cache_result
            )
            
        # Common API parameters
        api_params = {
            "system": messages[0]["content"],
            "messages": messages[1:],
            "model": model,
            "max_tokens": lm_config.get("max_tokens", 4096),
            "temperature": lm_config.get("temperature", SPECIAL_BASE_TEMPS.get(model, 0)),
        }
        
        # Only try to add thinking if supported by the SDK (check if Claude 3.7 and if reasoning_effort is set)
        # Try to detect capabilities without causing an error
        try:
            import inspect
            create_sig = inspect.signature(self.async_client.messages.create)
            if "thinking" in create_sig.parameters and "claude-3-7" in model:
                if reasoning_effort in ["high", "medium"]:
                    budgets = {
                        "high": 32000,
                        "medium": 16000,
                        "low": 8000,
                    }
                    budget = budgets[reasoning_effort]
                    api_params["thinking"] = {"type": "enabled", "budget_tokens": budget}
        except (ImportError, AttributeError, TypeError):
            # If we can't inspect or the parameter doesn't exist, just continue without it
            pass
        
        # Make the API call
        response = await self.async_client.messages.create(**api_params)
        
        api_result = response.content[0].text
        used_cache_handler.add_to_managed_cache(
            model, messages, lm_config=lm_config, output=api_result
        )
        return api_result

    @backoff.on_exception(
        backoff.expo,
        exceptions_to_retry,
        max_tries=BACKOFF_TOLERANCE,
        on_giveup=lambda e: print(e),
    )
    def _hit_api_sync(
        self,
        model: str,
        messages: List[Dict[str, Any]],
        lm_config: Dict[str, Any],
        use_ephemeral_cache_only: bool = False,
        reasoning_effort: str = "high",
        **vendor_params: Dict[str, Any],
    ) -> str:
        assert (
            lm_config.get("response_model", None) is None
        ), "response_model is not supported for standard calls"
        used_cache_handler = get_cache_handler(
            use_ephemeral_cache_only=use_ephemeral_cache_only
        )
        cache_result = used_cache_handler.hit_managed_cache(
            model, messages, lm_config=lm_config
        )
        if cache_result:
            return (
                cache_result["response"]
                if isinstance(cache_result, dict)
                else cache_result
            )
        
        # Common API parameters
        api_params = {
            "system": messages[0]["content"],
            "messages": messages[1:],
            "model": model,
            "max_tokens": lm_config.get("max_tokens", 4096),
            "temperature": lm_config.get("temperature", SPECIAL_BASE_TEMPS.get(model, 0)),
        }
        
        # Only try to add thinking if supported by the SDK (check if Claude 3.7 and if reasoning_effort is set)
        # Try to detect capabilities without causing an error
        try:
            import inspect
            create_sig = inspect.signature(self.sync_client.messages.create)
            if "thinking" in create_sig.parameters and "claude-3-7" in model:
                if reasoning_effort in ["high", "medium"]:
                    budgets = {
                        "high": 32000,
                        "medium": 16000,
                        "low": 8000,
                    }
                    budget = budgets[reasoning_effort]
                    api_params["thinking"] = {"type": "enabled", "budget_tokens": budget}
        except (ImportError, AttributeError, TypeError):
            # If we can't inspect or the parameter doesn't exist, just continue without it
            pass
            
        # Make the API call
        response = self.sync_client.messages.create(**api_params)
        
        api_result = response.content[0].text
        used_cache_handler.add_to_managed_cache(
            model, messages, lm_config=lm_config, output=api_result
        )
        return api_result

    async def _hit_api_async_structured_output(
        self,
        model: str,
        messages: List[Dict[str, Any]],
        response_model: BaseModel,
        temperature: float,
        use_ephemeral_cache_only: bool = False,
        reasoning_effort: str = "high",
        **vendor_params: Dict[str, Any],
    ) -> str:
        try:
            # First try with Anthropic
            reasoning_effort = vendor_params.get("reasoning_effort", reasoning_effort)
            if "claude-3-7" in model:
                if reasoning_effort in ["high", "medium"]:
                    budgets = {
                        "high": 32000,
                        "medium": 16000,
                        "low": 8000,
                    }
                    budget = budgets[reasoning_effort]
                response = await self.async_client.messages.create(
                    system=messages[0]["content"],
                    messages=messages[1:],
                    model=model,
                    max_tokens=4096,
                    thinking={"type": "enabled", "budget_tokens": budget},
                )
            else:
                response = await self.async_client.messages.create(
                    system=messages[0]["content"],
                    messages=messages[1:],
                    model=model,
                    max_tokens=4096,
                )
            result = response.content[0].text
            # Try to parse the result as JSON
            parsed = json.loads(result)
            return response_model(**parsed)
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
    ) -> str:
        try:
            # First try with Anthropic
            reasoning_effort = vendor_params.get("reasoning_effort", reasoning_effort)
            import time

            if "claude-3-7" in model:
                if reasoning_effort in ["high", "medium"]:
                    budgets = {
                        "high": 32000,
                        "medium": 16000,
                        "low": 8000,
                    }
                    budget = budgets[reasoning_effort]
                response = self.sync_client.messages.create(
                    system=messages[0]["content"],
                    messages=messages[1:],
                    model=model,
                    max_tokens=4096,
                    temperature=temperature,
                    thinking={"type": "enabled", "budget_tokens": budget},
                )
            else:
                response = self.sync_client.messages.create(
                    system=messages[0]["content"],
                    messages=messages[1:],
                    model=model,
                    max_tokens=4096,
                    temperature=temperature,
                )
            # print("Time taken for API call", time.time() - t)
            result = response.content[0].text
            # Try to parse the result as JSON
            parsed = json.loads(result)
            return response_model(**parsed)
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
