import json
import logging
import os
import warnings
from typing import Any, Dict, List, Optional, Tuple, Type

import google.generativeai as genai
from google.api_core.exceptions import ResourceExhausted
from google.generativeai.types import HarmBlockThreshold, HarmCategory, Tool

from synth_ai.zyk.lms.caching.initialize import (
    get_cache_handler,
)
from synth_ai.zyk.lms.tools.base import BaseTool
from synth_ai.zyk.lms.vendors.base import BaseLMResponse, VendorBase
from synth_ai.zyk.lms.vendors.constants import SPECIAL_BASE_TEMPS
from synth_ai.zyk.lms.vendors.retries import BACKOFF_TOLERANCE, backoff

GEMINI_EXCEPTIONS_TO_RETRY: Tuple[Type[Exception], ...] = (ResourceExhausted,)
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

    def _convert_messages_to_contents(
        self, messages: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        contents = []
        system_instruction = None
        for message in messages:
            if message["role"] == "system":
                system_instruction = (
                    f"<instructions>\n{message['content']}\n</instructions>"
                )
                continue
            elif system_instruction:
                text = system_instruction + "\n" + message["content"]
            else:
                text = message["content"]
            contents.append(
                {
                    "role": message["role"],
                    "parts": [{"text": text}],
                }
            )
        return contents

    def _convert_tools_to_gemini_format(self, tools: List[BaseTool]) -> Tool:
        function_declarations = []
        for tool in tools:
            function_declarations.append(tool.to_gemini_tool())
        return Tool(function_declarations=function_declarations)

    async def _private_request_async(
        self,
        messages: List[Dict],
        temperature: float = 0,
        model_name: str = "gemini-1.5-flash",
        reasoning_effort: str = "high",
        tools: Optional[List[BaseTool]] = None,
        lm_config: Optional[Dict[str, Any]] = None,
    ) -> Tuple[str, Optional[List[Dict]]]:
        generation_config = {
            "temperature": temperature,
        }

        tools_config = None
        if tools:
            tools_config = self._convert_tools_to_gemini_format(tools)

        # Extract tool_config from lm_config if provided
        tool_config = lm_config.get("tool_config") if lm_config else {
            "function_calling_config": {
                "mode": "any"
            }
        }

        code_generation_model = genai.GenerativeModel(
            model_name=model_name,
            generation_config=generation_config,
            tools=tools_config if tools_config else None,
            tool_config=tool_config,
        )

        contents = self._convert_messages_to_contents(messages)
        result = await code_generation_model.generate_content_async(
            contents=contents,
            safety_settings=SAFETY_SETTINGS,
        )

        text = result.candidates[0].content.parts[0].text
        tool_calls = []
        for part in result.candidates[0].content.parts:
            if part.function_call:
                # Convert MapComposite args to dict
                args_dict = dict(part.function_call.args)
                tool_calls.append(
                    {
                        "id": f"call_{len(tool_calls) + 1}",  # Generate unique IDs
                        "type": "function",
                        "function": {
                            "name": part.function_call.name,
                            "arguments": json.dumps(args_dict),
                        },
                    }
                )
        return text, tool_calls if tool_calls else None

    def _private_request_sync(
        self,
        messages: List[Dict],
        temperature: float = 0,
        model_name: str = "gemini-1.5-flash",
        reasoning_effort: str = "high",
        tools: Optional[List[BaseTool]] = None,
        lm_config: Optional[Dict[str, Any]] = None,
    ) -> Tuple[str, Optional[List[Dict]]]:
        generation_config = {
            "temperature": temperature,
        }

        tools_config = None
        if tools:
            tools_config = self._convert_tools_to_gemini_format(tools)

        # Extract tool_config from lm_config if provided
        tool_config = lm_config.get("tool_config") if lm_config else {
            "function_calling_config": {
                "mode": "any"
            }
        }

        code_generation_model = genai.GenerativeModel(
            model_name=model_name,
            generation_config=generation_config,
            tools=tools_config if tools_config else None,
            tool_config=tool_config,
        )

        contents = self._convert_messages_to_contents(messages)
        result = code_generation_model.generate_content(
            contents=contents,
            safety_settings=SAFETY_SETTINGS,
        )

        text = result.candidates[0].content.parts[0].text
        tool_calls = []
        for part in result.candidates[0].content.parts:
            if part.function_call:
                # Convert MapComposite args to dict
                args_dict = dict(part.function_call.args)
                tool_calls.append(
                    {
                        "id": f"call_{len(tool_calls) + 1}",  # Generate unique IDs
                        "type": "function",
                        "function": {
                            "name": part.function_call.name,
                            "arguments": json.dumps(args_dict),
                        },
                    }
                )
        return text, tool_calls if tool_calls else None

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
        tools: Optional[List[BaseTool]] = None,
    ) -> BaseLMResponse:
        assert (
            lm_config.get("response_model", None) is None
        ), "response_model is not supported for standard calls"
        used_cache_handler = get_cache_handler(use_ephemeral_cache_only)
        cache_result = used_cache_handler.hit_managed_cache(
            model, messages, lm_config=lm_config, tools=tools
        )
        if cache_result:
            return cache_result

        raw_response, tool_calls = await self._private_request_async(
            messages,
            temperature=lm_config.get("temperature", SPECIAL_BASE_TEMPS.get(model, 0)),
            reasoning_effort=reasoning_effort,
            tools=tools,
        )

        lm_response = BaseLMResponse(
            raw_response=raw_response,
            structured_output=None,
            tool_calls=tool_calls,
        )

        used_cache_handler.add_to_managed_cache(
            model, messages, lm_config=lm_config, output=lm_response, tools=tools
        )
        return lm_response

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
        tools: Optional[List[BaseTool]] = None,
    ) -> BaseLMResponse:
        assert (
            lm_config.get("response_model", None) is None
        ), "response_model is not supported for standard calls"
        used_cache_handler = get_cache_handler(
            use_ephemeral_cache_only=use_ephemeral_cache_only
        )
        cache_result = used_cache_handler.hit_managed_cache(
            model, messages, lm_config=lm_config, tools=tools
        )
        if cache_result:
            return cache_result

        raw_response, tool_calls = self._private_request_sync(
            messages,
            temperature=lm_config.get("temperature", SPECIAL_BASE_TEMPS.get(model, 0)),
            reasoning_effort=reasoning_effort,
            tools=tools,
        )

        lm_response = BaseLMResponse(
            raw_response=raw_response,
            structured_output=None,
            tool_calls=tool_calls,
        )

        used_cache_handler.add_to_managed_cache(
            model, messages, lm_config=lm_config, output=lm_response, tools=tools
        )
        return lm_response
