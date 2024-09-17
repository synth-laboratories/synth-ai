import logging
import os
import warnings
from typing import Any, Dict, List, Tuple, Type

import google.generativeai as genai
from google.api_core.exceptions import ResourceExhausted
from google.generativeai.types import HarmBlockThreshold, HarmCategory

from zyk.src.lms.caching.initialize import (
    get_cache_handler,
)
from zyk.src.lms.vendors.base import VendorBase
from zyk.src.lms.vendors.constants import SPECIAL_BASE_TEMPS
from zyk.src.lms.vendors.retries import BACKOFF_TOLERANCE, backoff

GEMINI_EXCEPTIONS_TO_RETRY: Tuple[Type[Exception], ...] = (
    ResourceExhausted,
)
logging.getLogger("google.generativeai").setLevel(logging.ERROR)
os.environ["GRPC_VERBOSITY"] = "ERROR"
# Suppress TensorFlow logging
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
warnings.filterwarnings("ignore")

SAFETY_SETTINGS = {
    HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
}

class GeminiAPI(VendorBase):
    used_for_structured_outputs: bool = True
    exceptions_to_retry: Tuple[Type[Exception], ...] = GEMINI_EXCEPTIONS_TO_RETRY

    def __init__(
        self,
        exceptions_to_retry: Tuple[Type[Exception], ...] = GEMINI_EXCEPTIONS_TO_RETRY,
        used_for_structured_outputs: bool = False,
    ):
        self.used_for_structured_outputs = used_for_structured_outputs
        self.exceptions_to_retry = exceptions_to_retry

    async def _private_request_async(
        self,
        messages: List[Dict],
        temperature: float = 0,
        model_name: str = "gemini-1.5-flash",
    ) -> str:
        code_generation_model = genai.GenerativeModel(
            model_name=model_name,
            generation_config={"temperature": temperature},
            system_instruction=messages[0]["content"],
        )
        result = await code_generation_model.generate_content_async(
            messages[1]["content"],
            safety_settings=SAFETY_SETTINGS,
        )
        return result.text

    def _private_request_sync(
        self,
        messages: List[Dict],
        temperature: float = 0,
        model_name: str = "gemini-1.5-flash",
    ) -> str:
        code_generation_model = genai.GenerativeModel(
            model_name=model_name,
            generation_config={"temperature": temperature},
            system_instruction=messages[0]["content"],
        )
        result = code_generation_model.generate_content(
            messages[1]["content"],
            safety_settings=SAFETY_SETTINGS,
        )
        return result.text

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
        api_result = await self._private_request_async(
            messages,
            temperature=lm_config.get(
                "temperature", SPECIAL_BASE_TEMPS.get(model, 0)
            ),
        )
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
        api_result = self._private_request_sync(
            messages,
            temperature=lm_config.get(
                "temperature", SPECIAL_BASE_TEMPS.get(model, 0)
            ),
        )
        used_cache_handler.add_to_managed_cache(
            model, messages, lm_config=lm_config, output=api_result
        )
        return api_result