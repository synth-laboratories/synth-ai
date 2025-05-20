import logging
import time
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Literal, Optional, Union

from pydantic import BaseModel

from synth_ai.zyk.lms.core.exceptions import StructuredOutputCoercionFailureException
from synth_ai.zyk.lms.structured_outputs.inject import (
    inject_structured_output_instructions,
)
from synth_ai.zyk.lms.structured_outputs.rehabilitate import (
    fix_errant_forced_async,
    fix_errant_forced_sync,
    pull_out_structured_output,
)
from synth_ai.zyk.lms.vendors.base import BaseLMResponse, VendorBase
from synth_ai.zyk.lms.constants import SPECIAL_BASE_TEMPS

logger = logging.getLogger(__name__)


class StructuredHandlerBase(ABC):
    core_client: VendorBase
    retry_client: VendorBase
    handler_params: Dict[str, Any]
    structured_output_mode: Literal["stringified_json", "forced_json"]

    def __init__(
        self,
        core_client: VendorBase,
        retry_client: VendorBase,
        handler_params: Optional[Dict[str, Any]] = None,
        structured_output_mode: Literal[
            "stringified_json", "forced_json"
        ] = "stringified_json",
    ):
        self.core_client = core_client
        self.retry_client = retry_client
        self.handler_params = (
            handler_params if handler_params is not None else {"retries": 3}
        )
        self.structured_output_mode = structured_output_mode

    async def call_async(
        self,
        messages: List[Dict[str, Any]],
        model: str,
        response_model: BaseModel,
        temperature: float = 0.0,
        use_ephemeral_cache_only: bool = False,
        reasoning_effort: str = "high",
    ) -> BaseLMResponse:
        if temperature == 0.0:
            temperature = SPECIAL_BASE_TEMPS.get(model, 0.0)
        # print("Calling from base")
        return await self._process_call_async(
            messages=messages,
            model=model,
            response_model=response_model,
            api_call_method=self.core_client._hit_api_async_structured_output
            if (not not response_model and self.structured_output_mode == "forced_json")
            else self.core_client._hit_api_async,
            temperature=temperature,
            use_ephemeral_cache_only=use_ephemeral_cache_only,
            reasoning_effort=reasoning_effort,
        )

    def call_sync(
        self,
        messages: List[Dict[str, Any]],
        response_model: BaseModel,
        model: str,
        temperature: float = 0.0,
        use_ephemeral_cache_only: bool = False,
        reasoning_effort: str = "high",
    ) -> BaseLMResponse:
        if temperature == 0.0:
            temperature = SPECIAL_BASE_TEMPS.get(model, 0.0)
        return self._process_call_sync(
            messages=messages,
            model=model,
            response_model=response_model,
            api_call_method=self.core_client._hit_api_sync_structured_output
            if (not not response_model and self.structured_output_mode == "forced_json")
            else self.core_client._hit_api_sync,
            temperature=temperature,
            use_ephemeral_cache_only=use_ephemeral_cache_only,
            reasoning_effort=reasoning_effort,
        )

    @abstractmethod
    async def _process_call_async(
        self,
        messages: List[Dict[str, Any]],
        model: str,
        response_model: BaseModel,
        api_call_method,
        use_ephemeral_cache_only: bool = False,
        reasoning_effort: str = "high",
    ) -> BaseLMResponse:
        pass

    @abstractmethod
    def _process_call_sync(
        self,
        messages: List[Dict[str, Any]],
        model: str,
        response_model: BaseModel,
        api_call_method,
        use_ephemeral_cache_only: bool = False,
        reasoning_effort: str = "high",
    ) -> BaseLMResponse:
        pass


class StringifiedJSONHandler(StructuredHandlerBase):
    core_client: VendorBase
    retry_client: VendorBase
    handler_params: Dict[str, Any]

    def __init__(
        self,
        core_client: VendorBase,
        retry_client: VendorBase,
        handler_params: Dict[str, Any] = {"retries": 3},
    ):
        super().__init__(
            core_client,
            retry_client,
            handler_params,
            structured_output_mode="stringified_json",
        )

    async def _process_call_async(
        self,
        messages: List[Dict[str, Any]],
        model: str,
        response_model: BaseModel,
        temperature: float,
        api_call_method: Callable,
        use_ephemeral_cache_only: bool = False,
        reasoning_effort: str = "high",
    ) -> BaseLMResponse:
        #ogger.info(f"Processing structured output call for model: {model}")
        assert callable(api_call_method), "api_call_method must be a callable"
        assert (
            response_model is not None
        ), "Don't use this handler for unstructured outputs"
        remaining_retries = self.handler_params.get("retries", 2)
        previously_failed_error_messages = []
        structured_output = None

        while remaining_retries > 0:
            messages_with_json_formatting_instructions = (
                inject_structured_output_instructions(
                    messages=messages,
                    response_model=response_model,
                    previously_failed_error_messages=previously_failed_error_messages,
                )
            )
            t0 = time.time()
            raw_text_response_or_cached_hit = await api_call_method(
                messages=messages_with_json_formatting_instructions,
                model=model,
                lm_config={"response_model": None, "temperature": temperature},
                use_ephemeral_cache_only=use_ephemeral_cache_only,
                reasoning_effort=reasoning_effort,
            )
            #logger.debug(f"Time to get response: {time.time() - t0:.2f}s")

            # Check if we got a cached BaseLMResponse
            assert (
                type(raw_text_response_or_cached_hit) in [str, BaseLMResponse]
            ), f"Expected str or BaseLMResponse, got {type(raw_text_response_or_cached_hit)}"
            if type(raw_text_response_or_cached_hit) == BaseLMResponse:
                #print("Got cached hit, returning directly")
                raw_text_response = raw_text_response_or_cached_hit.raw_response
            else:
                raw_text_response = raw_text_response_or_cached_hit
            #logger.debug(f"Raw response from model:\n{raw_text_response}")

            #print("Trying to parse structured output")
            try:
                structured_output = pull_out_structured_output(
                    raw_text_response, response_model
                )

                #print("Successfully parsed structured output on first attempt")
                break
            except Exception as e:
                logger.warning(f"Failed to parse structured output: {str(e)}")
                try:
                    print("Attempting to fix with forced JSON parser")
                    structured_output = await fix_errant_forced_async(
                        messages_with_json_formatting_instructions,
                        raw_text_response,
                        response_model,
                        "gpt-4o-mini",
                    )
                    assert isinstance(structured_output, BaseModel), "Structured output must be a Pydantic model"
                    assert not isinstance(structured_output, BaseLMResponse), "Got BaseLMResponse instead of Pydantic model"
                    print("Successfully fixed and parsed structured output")
                    break
                except Exception as e:
                    logger.error(f"Failed to fix structured output: {str(e)}")
                    previously_failed_error_messages.append(
                        f"Generated attempt and got error. Attempt:\n\n{raw_text_response}\n\nError:\n\n{e}"
                    )
                    remaining_retries -= 1
                    logger.warning(f"Retries remaining: {remaining_retries}")

        if structured_output is None:
            logger.error("Failed to get structured output after all retries")
            raise StructuredOutputCoercionFailureException(
                "Failed to get structured output"
            )
        #print("Successfully parsed structured output")
        #print(structured_output)
        assert isinstance(structured_output, BaseModel), "Structured output must be a Pydantic model"
        assert not isinstance(structured_output, BaseLMResponse),"Got BaseLMResponse instead of Pydantic model" 
        return BaseLMResponse(
            raw_response=raw_text_response,
            structured_output=structured_output,
            tool_calls=None,
        )

    def _process_call_sync(
        self,
        messages: List[Dict[str, Any]],
        model: str,
        response_model: BaseModel,
        temperature: float,
        api_call_method: Callable,
        use_ephemeral_cache_only: bool = False,
        reasoning_effort: str = "high",
    ) -> BaseLMResponse:
        #logger.info(f"Processing structured output call for model: {model}")
        assert callable(api_call_method), "api_call_method must be a callable"
        assert (
            response_model is not None
        ), "Don't use this handler for unstructured outputs"
        remaining_retries = self.handler_params.get("retries", 2)
        previously_failed_error_messages = []
        structured_output = None

        while remaining_retries > 0:
            messages_with_json_formatting_instructions = (
                inject_structured_output_instructions(
                    messages=messages,
                    response_model=response_model,
                    previously_failed_error_messages=previously_failed_error_messages,
                )
            )
            t0 = time.time()
            raw_text_response_or_cached_hit = api_call_method(
                messages=messages_with_json_formatting_instructions,
                model=model,
                lm_config={"response_model": None, "temperature": temperature},
                use_ephemeral_cache_only=use_ephemeral_cache_only,
                reasoning_effort=reasoning_effort,
            )
            logger.debug(f"Time to get response: {time.time() - t0:.2f}s")

            # Check if we got a cached BaseLMResponse
            assert (
                type(raw_text_response_or_cached_hit) in [str, BaseLMResponse]
            ), f"Expected str or BaseLMResponse, got {type(raw_text_response_or_cached_hit)}"
            if type(raw_text_response_or_cached_hit) == BaseLMResponse:
                logger.info("Got cached hit, returning directly")
                raw_text_response = raw_text_response_or_cached_hit.raw_response
            else:   
                raw_text_response = raw_text_response_or_cached_hit
            #logger.debug(f"Raw response from model:\n{raw_text_response}")

            try:
                structured_output = pull_out_structured_output(
                    raw_text_response, response_model
                )
                #print("Successfully parsed structured output on first attempt")
                break
            except Exception as e:
                logger.warning(f"Failed to parse structured output: {str(e)}")
                try:
                    print("Attempting to fix with forced JSON parser")
                    structured_output = fix_errant_forced_sync(
                        raw_text_response, response_model, "gpt-4o-mini"
                    )
                    print("Successfully fixed and parsed structured output")
                    break
                except Exception as e:
                    logger.error(f"Failed to fix structured output: {str(e)}")
                    previously_failed_error_messages.append(
                        f"Generated attempt and got error. Attempt:\n\n{raw_text_response}\n\nError:\n\n{e}"
                    )
                    remaining_retries -= 1
                    logger.warning(f"Retries remaining: {remaining_retries}")

        #print("Successfully parsed structured output")
        #print(structured_output)
        if structured_output is None:
            logger.error("Failed to get structured output after all retries")
            raise StructuredOutputCoercionFailureException(
                "Failed to get structured output"
            )
        return BaseLMResponse(
            raw_response=raw_text_response,
            structured_output=structured_output,
            tool_calls=None,
        )


class ForcedJSONHandler(StructuredHandlerBase):
    core_client: VendorBase
    retry_client: VendorBase
    handler_params: Dict[str, Any]

    def __init__(
        self,
        core_client: VendorBase,
        retry_client: VendorBase,
        handler_params: Dict[str, Any] = {},
        reasoning_effort: str = "high",
    ):
        super().__init__(
            core_client,
            retry_client,
            handler_params,
            structured_output_mode="forced_json",
        )
        self.reasoning_effort = reasoning_effort

    async def _process_call_async(
        self,
        messages: List[Dict[str, Any]],
        model: str,
        response_model: BaseModel,
        api_call_method: Callable,
        temperature: float = 0.0,
        use_ephemeral_cache_only: bool = False,
        reasoning_effort: str = "high",
    ) -> BaseLMResponse:
        # print("Forced JSON")
        assert (
            response_model is not None
        ), "Don't use this handler for unstructured outputs"
        return await api_call_method(
            messages=messages,
            model=model,
            response_model=response_model,
            temperature=temperature,
            use_ephemeral_cache_only=use_ephemeral_cache_only,
            reasoning_effort=reasoning_effort,
        )

    def _process_call_sync(
        self,
        messages: List[Dict[str, Any]],
        model: str,
        response_model: BaseModel,
        api_call_method: Callable,
        temperature: float = 0.0,
        use_ephemeral_cache_only: bool = False,
        reasoning_effort: str = "high",
    ) -> BaseLMResponse:
        assert (
            response_model is not None
        ), "Don't use this handler for unstructured outputs"
        return api_call_method(
            messages=messages,
            model=model,
            response_model=response_model,
            temperature=temperature,
            use_ephemeral_cache_only=use_ephemeral_cache_only,
            reasoning_effort=reasoning_effort,
        )


class StructuredOutputHandler:
    handler: Union[StringifiedJSONHandler, ForcedJSONHandler]
    mode: Literal["stringified_json", "forced_json"]
    handler_params: Dict[str, Any]

    def __init__(
        self,
        core_client: VendorBase,
        retry_client: VendorBase,
        mode: Literal["stringified_json", "forced_json"],
        handler_params: Dict[str, Any] = {},
    ):
        self.mode = mode
        if self.mode == "stringified_json":
            self.handler = StringifiedJSONHandler(
                core_client, retry_client, handler_params
            )
        elif self.mode == "forced_json":
            # print("Forced JSON")
            self.handler = ForcedJSONHandler(core_client, retry_client, handler_params)
        else:
            raise ValueError(f"Invalid mode: {mode}")

    async def call_async(
        self,
        messages: List[Dict[str, Any]],
        model: str,
        response_model: BaseModel,
        use_ephemeral_cache_only: bool = False,
        lm_config: Dict[str, Any] = {},
        reasoning_effort: str = "high",
    ) -> BaseLMResponse:
        # print("Output handler call async")
        return await self.handler.call_async(
            messages=messages,
            model=model,
            response_model=response_model,
            temperature=lm_config.get(
                "temperature", SPECIAL_BASE_TEMPS.get(model, 0.0)
            ),
            use_ephemeral_cache_only=use_ephemeral_cache_only,
            reasoning_effort=reasoning_effort,
        )

    def call_sync(
        self,
        messages: List[Dict[str, Any]],
        model: str,
        response_model: BaseModel,
        use_ephemeral_cache_only: bool = False,
        lm_config: Dict[str, Any] = {},
        reasoning_effort: str = "high",
    ) -> BaseLMResponse:
        return self.handler.call_sync(
            messages=messages,
            model=model,
            response_model=response_model,
            temperature=lm_config.get(
                "temperature", SPECIAL_BASE_TEMPS.get(model, 0.0)
            ),
            use_ephemeral_cache_only=use_ephemeral_cache_only,
            reasoning_effort=reasoning_effort,
        )
