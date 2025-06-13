import json
import logging
import os
import warnings
from typing import Any, Dict, List, Optional, Tuple, Type

from google import genai
from google.api_core.exceptions import ResourceExhausted
from google.genai import types
from synth_ai.zyk.lms.caching.initialize import get_cache_handler
from synth_ai.zyk.lms.tools.base import BaseTool
from synth_ai.zyk.lms.vendors.base import BaseLMResponse, VendorBase
from synth_ai.zyk.lms.constants import (
    SPECIAL_BASE_TEMPS,
    GEMINI_REASONING_MODELS,
    GEMINI_THINKING_BUDGETS,
)
from synth_ai.zyk.lms.vendors.retries import BACKOFF_TOLERANCE, backoff
import logging


ALIASES = {
    "gemini-2.5-flash": "gemini-2.5-flash-preview-04-17",
}

logger = logging.getLogger(__name__)
_CLIENT = None  # Initialize lazily when needed
GEMINI_EXCEPTIONS_TO_RETRY: Tuple[Type[Exception], ...] = (ResourceExhausted,)
logging.getLogger("google.genai").setLevel(logging.ERROR)
os.environ["GRPC_VERBOSITY"] = "ERROR"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
warnings.filterwarnings("ignore")

SAFETY_SETTINGS = {
    types.HarmCategory.HARM_CATEGORY_HATE_SPEECH: types.HarmBlockThreshold.BLOCK_NONE,
    types.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: types.HarmBlockThreshold.BLOCK_NONE,
    types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: types.HarmBlockThreshold.BLOCK_NONE,
    types.HarmCategory.HARM_CATEGORY_HARASSMENT: types.HarmBlockThreshold.BLOCK_NONE,
}


def _get_client():
    """Get or create the Gemini client lazily."""
    global _CLIENT
    if _CLIENT is None:
        _CLIENT = genai.Client()
    return _CLIENT


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


    def get_aliased_model_name(self, model_name: str) -> str:
        if model_name in ALIASES:
            return ALIASES[model_name]
        return model_name

    @staticmethod
    def _msg_to_contents(messages: List[Dict[str, Any]]) -> List[types.Content]:
        # contents, sys_instr = [], None
        contents = []
        for m in messages:
            # if m["role"] == "system":
            #     sys_instr = f"<instructions>\n{m['content']}\n</instructions>"
            #     continue
            # text = (sys_instr + "\n" + m["content"]) if sys_instr else m["content"]
            if m["role"].lower() not in ["user", "assistant"]:
                continue
            role = "user" if m["role"] == "user" else "assistant"
            contents.append(types.Content(role=role, parts=[types.Part.from_text(text=m["content"])]))
        return contents

    @staticmethod
    def _tools_to_genai(tools: List[BaseTool]) -> List[types.Tool]:
        """Convert internal BaseTool → genai Tool."""
        out: List[types.Tool] = []
        for t in tools:
            # Assume t.to_gemini_tool() now correctly returns a FunctionDeclaration
            #func_decl = t.to_gemini_tool()
            if isinstance(t, dict):
                func_decl = t
            else:
                func_decl = t.to_gemini_tool()
            if not isinstance(func_decl, types.FunctionDeclaration):
                 # Or fetch schema parts if to_gemini_tool still returns dict
                 # This depends on BaseTool.to_gemini_tool implementation
                tool_dict = func_decl # Assuming it's a dict for now
                func_decl = types.FunctionDeclaration(
                    name=tool_dict['name'],
                    description=tool_dict['description'],
                    parameters=tool_dict['parameters'], # Expects OpenAPI-style dict
                )
            out.append(types.Tool(function_declarations=[func_decl]))
        return out

    async def _gen_content_async(
        self,
        messages: List[Dict],
        temperature: float,
        model_name: str,
        reasoning_effort: str,
        tools: Optional[List[BaseTool]],
        lm_config: Optional[Dict[str, Any]],
    ) -> Tuple[str, Optional[List[Dict]]]:
        model_name = self.get_aliased_model_name(model_name)
        cfg_kwargs: Dict[str, Any] = {"temperature": temperature}
        if model_name in GEMINI_REASONING_MODELS and reasoning_effort in GEMINI_THINKING_BUDGETS:
            cfg_kwargs["thinking_config"] = types.ThinkingConfig(
                thinking_budget=GEMINI_THINKING_BUDGETS[reasoning_effort]
            )
        
        if any(m["role"] == "system" for m in messages):
            cfg_kwargs["system_instruction"] = next(m["content"] for m in messages if m["role"] == "system")
        
        generation_config = types.GenerateContentConfig(
            **cfg_kwargs,
            tool_config=lm_config.get("tool_config") if lm_config else None,
            tools=self._tools_to_genai(tools) if tools else None
        )
        client = _get_client()
        resp = await client.aio.models.generate_content(
            model=model_name,
            contents=self._msg_to_contents(messages),
            config=generation_config,
            #safety_settings=SAFETY_SETTINGS,
        )
        return self._extract(resp)

    def _gen_content_sync(
        self,
        messages: List[Dict],
        temperature: float,
        model_name: str,
        reasoning_effort: str,
        tools: Optional[List[BaseTool]],
        lm_config: Optional[Dict[str, Any]],
    ) -> Tuple[str, Optional[List[Dict]]]:
        model_name = self.get_aliased_model_name(model_name)
        cfg_kwargs: Dict[str, Any] = {"temperature": temperature}
        if model_name in GEMINI_REASONING_MODELS and reasoning_effort in GEMINI_THINKING_BUDGETS:
            cfg_kwargs["thinking_config"] = types.ThinkingConfig(
                thinking_budget=GEMINI_THINKING_BUDGETS[reasoning_effort]
            )
        if any(m["role"] == "system" for m in messages):
            cfg_kwargs["system_instruction"] = next(m["content"] for m in messages if m["role"] == "system")
        generation_config = types.GenerateContentConfig(
            **cfg_kwargs,
            tool_config=lm_config.get("tool_config") if lm_config else None,
            tools=self._tools_to_genai(tools) if tools else None
        )

        client = _get_client()
        resp = client.models.generate_content(
            model=model_name,
            contents=self._msg_to_contents(messages),
            safety_settings=SAFETY_SETTINGS,
            config=generation_config,
        )
        return self._extract(resp)

    @staticmethod
    def _extract(response) -> Tuple[str, Optional[List[Dict]]]:
        # Extract text, handling cases where it might be missing
        try:
            text = response.text
        except ValueError: # Handle cases where only non-text parts exist
            text = "" 

        calls = []
        # Access parts through candidates[0].content
        if response.candidates and response.candidates[0].content:
            for part in response.candidates[0].content.parts:
                if part.function_call:
                    calls.append(
                        {
                            "id": f"call_{len(calls) + 1}",
                            "type": "function",
                            "function": {
                                "name": part.function_call.name,
                                "arguments": json.dumps(dict(part.function_call.args)),
                            },
                        }
                    )
        return text, calls or None

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
        lm_config["reasoning_effort"] = reasoning_effort
        cache_result = used_cache_handler.hit_managed_cache(
            model, messages, lm_config=lm_config, tools=tools
        )
        if cache_result:
            return cache_result

        raw_response, tool_calls = await self._gen_content_async(
            messages,
            temperature=lm_config.get("temperature", SPECIAL_BASE_TEMPS.get(model, 0)),
            reasoning_effort=reasoning_effort,
            tools=tools,
            lm_config=lm_config,
            model_name=model,
        )
        if not raw_response:
            raw_response = ""
        lm_response = BaseLMResponse(
            raw_response=raw_response,
            structured_output=None,
            tool_calls=tool_calls,
        )

        lm_config["reasoning_effort"] = reasoning_effort
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
        lm_config["reasoning_effort"] = reasoning_effort
        cache_result = used_cache_handler.hit_managed_cache(
            model, messages, lm_config=lm_config, tools=tools
        )
        if cache_result:
            return cache_result

        raw_response, tool_calls = self._gen_content_sync(
            messages,
            temperature=lm_config.get("temperature", SPECIAL_BASE_TEMPS.get(model, 0)),
            reasoning_effort=reasoning_effort,
            tools=tools,
            lm_config=lm_config,
            model_name=model,
        )
        if not raw_response:
            raw_response = ""
        lm_response = BaseLMResponse(
            raw_response=raw_response,
            structured_output=None,
            tool_calls=tool_calls,
        )

        lm_config["reasoning_effort"] = reasoning_effort
        used_cache_handler.add_to_managed_cache(
            model, messages, lm_config=lm_config, output=lm_response, tools=tools
        )
        return lm_response
