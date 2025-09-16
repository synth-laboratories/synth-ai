import asyncio
import os
import time
from typing import Any

import backoff
import groq
import openai
import pydantic_core
from pydantic import BaseModel

from synth_ai.lm.caching.initialize import (
    get_cache_handler,
)
from synth_ai.lm.constants import SPECIAL_BASE_TEMPS
from synth_ai.lm.injection import apply_injection
from synth_ai.lm.overrides import (
    apply_param_overrides,
    apply_tool_overrides,
    use_overrides_for_messages,
)
from synth_ai.lm.tools.base import BaseTool
from synth_ai.lm.vendors.base import BaseLMResponse, VendorBase
from synth_ai.lm.vendors.openai_standard_responses import OpenAIResponsesAPIMixin
from synth_ai.lm.vendors.retries import MAX_BACKOFF

DEFAULT_EXCEPTIONS_TO_RETRY = (
    pydantic_core._pydantic_core.ValidationError,
    openai.APIConnectionError,
    openai.APITimeoutError,
    groq.InternalServerError,
    groq.APITimeoutError,
    groq.APIConnectionError,
)


def special_orion_transform(model: str, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Transform messages for O1 series models which don't support system messages.

    Args:
        model: Model name to check
        messages: Original messages list

    Returns:
        Transformed messages list with system content merged into user message
    """
    if "o1-" in model:
        messages = [
            {
                "role": "user",
                "content": f"<instructions>{messages[0]['content']}</instructions><information>{messages[1]}</information>",
            }
        ]
    return messages


def _silent_backoff_handler(_details):
    """No-op handler to keep stdout clean while still allowing visibility via logging if desired."""
    pass


class OpenAIStandard(VendorBase, OpenAIResponsesAPIMixin):
    """
    Standard OpenAI-compatible vendor implementation.

    This class provides a standard implementation for OpenAI-compatible APIs,
    including proper retry logic, caching, and support for various model features.

    Attributes:
        used_for_structured_outputs: Whether this client supports structured outputs
        exceptions_to_retry: List of exceptions that trigger automatic retries
        sync_client: Synchronous API client
        async_client: Asynchronous API client
    """

    used_for_structured_outputs: bool = True
    exceptions_to_retry: list = DEFAULT_EXCEPTIONS_TO_RETRY
    sync_client: Any
    async_client: Any

    def __init__(
        self,
        sync_client: Any,
        async_client: Any,
        exceptions_to_retry: list[Exception] = DEFAULT_EXCEPTIONS_TO_RETRY,
        used_for_structured_outputs: bool = False,
    ):
        self.sync_client = sync_client
        self.async_client = async_client
        self.used_for_structured_outputs = used_for_structured_outputs
        self.exceptions_to_retry = exceptions_to_retry

        # Initialize Harmony support for OSS models
        self.harmony_available = False
        self.harmony_enc = None
        try:
            from openai_harmony import HarmonyEncodingName, load_harmony_encoding

            self.harmony_available = True
            self.harmony_enc = load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)
        except ImportError:
            pass

    @backoff.on_exception(
        backoff.expo,
        DEFAULT_EXCEPTIONS_TO_RETRY,
        max_time=MAX_BACKOFF,
        jitter=backoff.full_jitter,
        on_backoff=_silent_backoff_handler,
    )
    async def _hit_api_async(
        self,
        model: str,
        messages: list[dict[str, Any]],
        lm_config: dict[str, Any],
        use_ephemeral_cache_only: bool = False,
        reasoning_effort: str = "high",
        tools: list[BaseTool] | None = None,
    ) -> BaseLMResponse:
        assert lm_config.get("response_model") is None, (
            "response_model is not supported for standard calls"
        )

        debug = os.getenv("SYNTH_OPENAI_DEBUG") == "1"
        if debug:
            print("🔍 OPENAI DEBUG: _hit_api_async called with:")
            print(f"   Model: {model}")
            print(f"   Messages: {len(messages)} messages")
            print(f"   Tools: {len(tools) if tools else 0} tools")
            print(f"   LM config: {lm_config}")

        messages = special_orion_transform(model, messages)
        # Apply context-scoped overrides and prompt injection just before building API params
        with use_overrides_for_messages(messages):
            messages = apply_injection(messages)
        used_cache_handler = get_cache_handler(use_ephemeral_cache_only)
        lm_config["reasoning_effort"] = reasoning_effort
        cache_result = used_cache_handler.hit_managed_cache(
            model, messages, lm_config=lm_config, tools=tools
        )
        if cache_result and debug:
            print("🔍 OPENAI DEBUG: Cache hit! Returning cached result")
            print(f"   Cache result type: {type(cache_result)}")
            print("🔍 OPENAI DEBUG: DISABLING CACHE FOR DEBUGGING - forcing API call")
            # return cache_result  # Commented out intentionally when debug is on

        if debug:
            print("🔍 OPENAI DEBUG: Cache miss, making actual API call")

        # Common API call params
        api_params = {
            "model": model,
            "messages": messages,
        }
        with use_overrides_for_messages(messages):
            api_params = apply_param_overrides(api_params)

        # Add tools if provided
        if tools and all(isinstance(tool, BaseTool) for tool in tools):
            api_params["tools"] = [tool.to_openai_tool() for tool in tools]
        elif tools:
            api_params["tools"] = tools

        # Only add temperature for non o1/o3 models, and do not override if already set via overrides
        if (
            not any(prefix in model for prefix in ["o1-", "o3-"])
            and "temperature" not in api_params
        ):
            api_params["temperature"] = lm_config.get(
                "temperature", SPECIAL_BASE_TEMPS.get(model, 0)
            )

        # Forward additional sampling / control params if provided
        if lm_config.get("max_tokens") is not None:
            api_params["max_tokens"] = lm_config["max_tokens"]
        if lm_config.get("top_p") is not None:
            api_params["top_p"] = lm_config["top_p"]
        if lm_config.get("frequency_penalty") is not None:
            api_params["frequency_penalty"] = lm_config["frequency_penalty"]
        if lm_config.get("presence_penalty") is not None:
            api_params["presence_penalty"] = lm_config["presence_penalty"]
        if lm_config.get("stop") is not None:
            api_params["stop"] = lm_config["stop"]
        if lm_config.get("tool_choice") is not None:
            api_params["tool_choice"] = lm_config["tool_choice"]
        # Forward GPU preference to backend (body + header)
        if lm_config.get("gpu_preference") is not None:
            api_params["gpu_preference"] = lm_config["gpu_preference"]
            # Also set header so proxies that read headers can honor it
            hdrs = api_params.get("extra_headers", {})
            hdrs["X-GPU-Preference"] = lm_config["gpu_preference"]
            api_params["extra_headers"] = hdrs
        # Also mirror stop_after_tool_calls into a header for robustness
        try:
            satc_val = None
            if isinstance(lm_config.get("extra_body"), dict):
                satc_val = lm_config["extra_body"].get("stop_after_tool_calls")
            if satc_val is not None:
                hdrs = api_params.get("extra_headers", {})
                hdrs["X-Stop-After-Tool-Calls"] = str(satc_val)
                api_params["extra_headers"] = hdrs
        except Exception:
            pass
        # Apply overrides (tools and params) from context after building baseline params
        with use_overrides_for_messages(messages):
            api_params = apply_tool_overrides(api_params)
            api_params = apply_param_overrides(api_params)

        # Thinking controls: route via extra_body.chat_template_kwargs for compatibility
        thinking_mode_val = lm_config.get("thinking_mode")
        thinking_budget_val = lm_config.get("thinking_budget")
        if thinking_mode_val is not None or thinking_budget_val is not None:
            api_params["extra_body"] = api_params.get("extra_body", {})
            ctk = api_params["extra_body"].get("chat_template_kwargs", {})
            if thinking_mode_val is not None:
                ctk["thinking_mode"] = thinking_mode_val
            if thinking_budget_val is not None:
                try:
                    ctk["thinking_budget"] = int(thinking_budget_val)
                except Exception:
                    ctk["thinking_budget"] = thinking_budget_val
            api_params["extra_body"]["chat_template_kwargs"] = ctk

        # Backward-compatible: forward legacy enable_thinking only via extra_body for callers still using it
        if lm_config.get("enable_thinking") is not None:
            api_params["extra_body"] = api_params.get("extra_body", {})
            ctk = api_params["extra_body"].get("chat_template_kwargs", {})
            ctk["enable_thinking"] = lm_config["enable_thinking"]
            api_params["extra_body"]["chat_template_kwargs"] = ctk
        # Forward arbitrary extra_body from lm_config if provided (merge)
        if lm_config.get("extra_body") is not None:
            # Shallow-merge top-level keys; nested keys (like chat_template_kwargs) should be provided whole
            api_params["extra_body"] = {
                **api_params.get("extra_body", {}),
                **(lm_config.get("extra_body") or {}),
            }
        # Ensure legacy extra_body flag remains merged (do not override top-level fields)
        if lm_config.get("enable_thinking") is not None:
            api_params["extra_body"] = api_params.get("extra_body", {})
            ctk = api_params["extra_body"].get("chat_template_kwargs", {})
            ctk["enable_thinking"] = lm_config["enable_thinking"]
            api_params["extra_body"]["chat_template_kwargs"] = ctk

        # Add reasoning_effort only for o3-mini
        if model in ["o3-mini"]:
            print("Reasoning effort:", reasoning_effort)
            api_params["reasoning_effort"] = reasoning_effort

        # Filter Synth-only params when calling external OpenAI-compatible providers
        # External providers (e.g., OpenAI, Groq) reject unknown fields like
        # extra_body.chat_template_kwargs or stop_after_tool_calls.
        try:
            base_url_obj = getattr(self.async_client, "base_url", None)
            base_url_str = str(base_url_obj) if base_url_obj is not None else ""
        except Exception:
            base_url_str = ""

        is_external_provider = "openai.com" in base_url_str or "api.groq.com" in base_url_str

        if is_external_provider:
            # Remove extra_body entirely; this is Synth-specific plumbing
            if "extra_body" in api_params:
                api_params.pop("extra_body", None)

            # Also ensure we don't pass stray vendor-specific fields if present
            # (defensive in case upstream added them at top-level later)
            for k in ["chat_template_kwargs", "stop_after_tool_calls"]:
                api_params.pop(k, None)

            # GPT-5 models: parameter normalization
            if model.startswith("gpt-5"):
                # Require max_completion_tokens instead of max_tokens
                if "max_tokens" in api_params:
                    api_params["max_completion_tokens"] = api_params.pop("max_tokens")
                # Only default temperature=1 supported; omit custom temperature
                if "temperature" in api_params:
                    api_params.pop("temperature", None)

        # Call API with better auth error reporting
        # try:
        if debug:
            print("🔍 OPENAI DEBUG: Making request with params:")
            print(f"   Model: {api_params.get('model')}")
            print(f"   Messages: {len(api_params.get('messages', []))} messages")
            print(f"   Tools: {len(api_params.get('tools', []))} tools")
            print(f"   Max tokens: {api_params.get('max_tokens', 'NOT SET')}")
            print(f"   Temperature: {api_params.get('temperature', 'NOT SET')}")
            if "tools" in api_params:
                print(f"   First tool: {api_params['tools'][0]}")
            print(f"   FULL API PARAMS: {api_params}")

        # Quiet targeted retry for OpenAI 400 tool_use_failed during tool-calling
        try:
            max_attempts_for_tool_use = int(os.getenv("SYNTH_TOOL_USE_RETRIES", "5"))
        except Exception:
            max_attempts_for_tool_use = 5
        try:
            backoff_seconds = float(os.getenv("SYNTH_TOOL_USE_BACKOFF_INITIAL", "0.5"))
        except Exception:
            backoff_seconds = 0.5

        attempt_index = 0
        while True:
            try:
                output = await self.async_client.chat.completions.create(**api_params)
                break
            except openai.BadRequestError as err:
                # Detect tool-use failure from various SDK surfaces
                should_retry = False
                # 1) Body dict
                body = getattr(err, "body", None)
                if isinstance(body, dict):
                    try:
                        err_obj = body.get("error") if isinstance(body.get("error"), dict) else {}
                        code_val = err_obj.get("code")
                        msg_val = err_obj.get("message")
                        if code_val == "tool_use_failed" or (
                            isinstance(msg_val, str) and "Failed to call a function" in msg_val
                        ):
                            should_retry = True
                    except Exception:
                        pass
                # 2) Response JSON
                if not should_retry:
                    try:
                        resp = getattr(err, "response", None)
                        if resp is not None:
                            j = resp.json()
                            if isinstance(j, dict):
                                err_obj = j.get("error") if isinstance(j.get("error"), dict) else {}
                                code_val = err_obj.get("code")
                                msg_val = err_obj.get("message")
                                if code_val == "tool_use_failed" or (
                                    isinstance(msg_val, str)
                                    and "Failed to call a function" in msg_val
                                ):
                                    should_retry = True
                    except Exception:
                        pass
                # 3) Fallback to string match
                if not should_retry:
                    err_text = str(err)
                    if "tool_use_failed" in err_text or "Failed to call a function" in err_text:
                        should_retry = True
                if should_retry and attempt_index + 1 < max_attempts_for_tool_use:
                    await asyncio.sleep(backoff_seconds)
                    backoff_seconds = min(backoff_seconds * 2.0, 2.0)
                    attempt_index += 1
                    continue
                raise

        if debug:
            print("🔍 OPENAI DEBUG: Response received:")
            print(f"   Type: {type(output)}")
            print(f"   Choices: {len(output.choices) if hasattr(output, 'choices') else 'N/A'}")
            if hasattr(output, "choices") and output.choices:
                choice = output.choices[0]
                print(f"   Choice type: {type(choice)}")
                if hasattr(choice, "message"):
                    message = choice.message
                    print(f"   Message type: {type(message)}")
                    print(f"   Has tool_calls: {hasattr(message, 'tool_calls')}")
                    if hasattr(message, "tool_calls"):
                        print(f"   Tool calls: {message.tool_calls}")
                    print(
                        f"   Content: {message.content[:200] if hasattr(message, 'content') and message.content else 'None'}..."
                    )
                # Show finish_reason and usage if available
                try:
                    print(f"   finish_reason: {getattr(choice, 'finish_reason', None)}")
                    usage = getattr(output, "usage", None)
                    if usage:
                        print(
                            f"   usage: prompt_tokens={getattr(usage, 'prompt_tokens', None)}, completion_tokens={getattr(usage, 'completion_tokens', None)}, total_tokens={getattr(usage, 'total_tokens', None)}"
                        )
                except Exception:
                    pass

        if debug:
            print("🔍 OPENAI DEBUG: FULL RAW RESPONSE:")
            if hasattr(output.choices[0].message, "content") and output.choices[0].message.content:
                print(f"   FULL CONTENT:\n{output.choices[0].message.content}")
            print(f"   Raw choice: {choice}")
            print(f"   Raw message: {message}")
        # except Exception as e:
        #     try:
        #         from openai import AuthenticationError as _OpenAIAuthErr  # type: ignore
        #     except ModuleNotFoundError:
        #         _OpenAIAuthErr = type(e)
        #     if isinstance(e, _OpenAIAuthErr):
        #         key_preview = (os.getenv("OPENAI_API_KEY") or "")[:8]
        #         # Create a more informative error message but preserve the original exception
        #         enhanced_msg = f"Invalid API key format. Expected prefix 'sk-' or 'sk_live_'. Provided key begins with '{key_preview}'. Original error: {str(e)}"
        #         # Re-raise the original exception with enhanced message if possible
        #         if hasattr(e, 'response') and hasattr(e, 'body'):
        #             raise _OpenAIAuthErr(enhanced_msg, response=e.response, body=e.body) from None
        #         else:
        #             # Fallback: just re-raise the original with a print for debugging
        #             print(f"🔑 API Key Debug: {enhanced_msg}")
        #             raise e from None
        #     raise
        message = output.choices[0].message

        # Convert tool calls to dict format, preferring dict-shaped entries first
        tool_calls = None
        if message.tool_calls:
            converted: list[dict] = []
            for tc in message.tool_calls:
                if isinstance(tc, dict):
                    fn = tc.get("function") or {}
                    converted.append(
                        {
                            "id": tc.get("id"),
                            "type": tc.get("type", "function"),
                            "function": {
                                "name": fn.get("name") or tc.get("name"),
                                "arguments": fn.get("arguments") or tc.get("arguments"),
                            },
                        }
                    )
                else:
                    # SDK object path
                    converted.append(
                        {
                            "id": getattr(tc, "id", None),
                            "type": getattr(tc, "type", "function"),
                            "function": {
                                "name": getattr(getattr(tc, "function", None), "name", None),
                                "arguments": getattr(getattr(tc, "function", None), "arguments", None),
                            },
                        }
                    )
            tool_calls = converted or None

        # Attach basic usage if available
        usage_dict = None
        try:
            usage_obj = getattr(output, "usage", None)
            if usage_obj is not None:
                usage_dict = {
                    "prompt_tokens": getattr(usage_obj, "prompt_tokens", None),
                    "completion_tokens": getattr(usage_obj, "completion_tokens", None),
                    "total_tokens": getattr(usage_obj, "total_tokens", None),
                }
        except Exception:
            usage_dict = None

        lm_response = BaseLMResponse(
            raw_response=message.content or "",  # Use empty string if no content
            structured_output=None,
            tool_calls=tool_calls,
            usage=usage_dict,
        )
        lm_config["reasoning_effort"] = reasoning_effort
        used_cache_handler.add_to_managed_cache(
            model, messages, lm_config=lm_config, output=lm_response, tools=tools
        )
        return lm_response

    @backoff.on_exception(
        backoff.expo,
        DEFAULT_EXCEPTIONS_TO_RETRY,
        max_time=MAX_BACKOFF,
        jitter=backoff.full_jitter,
        on_backoff=_silent_backoff_handler,
    )
    def _hit_api_sync(
        self,
        model: str,
        messages: list[dict[str, Any]],
        lm_config: dict[str, Any],
        use_ephemeral_cache_only: bool = False,
        reasoning_effort: str = "high",
        tools: list[BaseTool] | None = None,
    ) -> BaseLMResponse:
        assert lm_config.get("response_model") is None, (
            "response_model is not supported for standard calls"
        )
        messages = special_orion_transform(model, messages)
        with use_overrides_for_messages(messages):
            # Apply context-scoped prompt injection just before building API params
            messages = apply_injection(messages)
        used_cache_handler = get_cache_handler(use_ephemeral_cache_only=use_ephemeral_cache_only)
        lm_config["reasoning_effort"] = reasoning_effort
        cache_result = used_cache_handler.hit_managed_cache(
            model, messages, lm_config=lm_config, tools=tools
        )
        # During pytest runs, bypass returning cache to allow tests to inspect outgoing params
        in_pytest = os.getenv("PYTEST_CURRENT_TEST") is not None
        if cache_result and not in_pytest:
            return cache_result

        # Common API call params
        api_params = {
            "model": model,
            "messages": messages,
        }
        with use_overrides_for_messages(messages):
            api_params = apply_param_overrides(api_params)

        # Add tools if provided
        if tools and all(isinstance(tool, BaseTool) for tool in tools):
            api_params["tools"] = [tool.to_openai_tool() for tool in tools]
        elif tools:
            api_params["tools"] = tools

        # Apply overrides (tools and params) using module-level imports
        with use_overrides_for_messages(messages):
            api_params = apply_tool_overrides(api_params)
            api_params = apply_param_overrides(api_params)

        # Only add temperature for non o1/o3 models, and do not override if already set via overrides
        if (
            not any(prefix in model for prefix in ["o1-", "o3-"])
            and "temperature" not in api_params
        ):
            api_params["temperature"] = lm_config.get(
                "temperature", SPECIAL_BASE_TEMPS.get(model, 0)
            )

        # Forward additional sampling / control params if provided
        if lm_config.get("max_tokens") is not None:
            api_params["max_tokens"] = lm_config["max_tokens"]
        if lm_config.get("top_p") is not None:
            api_params["top_p"] = lm_config["top_p"]
        if lm_config.get("frequency_penalty") is not None:
            api_params["frequency_penalty"] = lm_config["frequency_penalty"]
        if lm_config.get("presence_penalty") is not None:
            api_params["presence_penalty"] = lm_config["presence_penalty"]
        if lm_config.get("stop") is not None:
            api_params["stop"] = lm_config["stop"]
        if lm_config.get("tool_choice") is not None:
            api_params["tool_choice"] = lm_config["tool_choice"]

        # Add reasoning_effort only for o3-mini
        if model in ["o3-mini"]:
            api_params["reasoning_effort"] = reasoning_effort

        # Sync path: apply the same targeted retry
        try:
            max_attempts_for_tool_use = int(os.getenv("SYNTH_TOOL_USE_RETRIES", "5"))
        except Exception:
            max_attempts_for_tool_use = 5
        try:
            backoff_seconds = float(os.getenv("SYNTH_TOOL_USE_BACKOFF_INITIAL", "0.5"))
        except Exception:
            backoff_seconds = 0.5

        attempt_index = 0
        while True:
            try:
                output = self.sync_client.chat.completions.create(**api_params)
                break
            except openai.BadRequestError as err:
                should_retry = False
                body = getattr(err, "body", None)
                if isinstance(body, dict):
                    try:
                        err_obj = body.get("error") if isinstance(body.get("error"), dict) else {}
                        code_val = err_obj.get("code")
                        msg_val = err_obj.get("message")
                        if code_val == "tool_use_failed" or (
                            isinstance(msg_val, str) and "Failed to call a function" in msg_val
                        ):
                            should_retry = True
                    except Exception:
                        pass
                if not should_retry:
                    try:
                        resp = getattr(err, "response", None)
                        if resp is not None:
                            j = resp.json()
                            if isinstance(j, dict):
                                err_obj = j.get("error") if isinstance(j.get("error"), dict) else {}
                                code_val = err_obj.get("code")
                                msg_val = err_obj.get("message")
                                if code_val == "tool_use_failed" or (
                                    isinstance(msg_val, str)
                                    and "Failed to call a function" in msg_val
                                ):
                                    should_retry = True
                    except Exception:
                        pass
                if not should_retry:
                    err_text = str(err)
                    if "tool_use_failed" in err_text or "Failed to call a function" in err_text:
                        should_retry = True
                if should_retry and attempt_index + 1 < max_attempts_for_tool_use:
                    time.sleep(backoff_seconds)
                    backoff_seconds = min(backoff_seconds * 2.0, 2.0)
                    attempt_index += 1
                    continue
                raise
        message = output.choices[0].message
        debug_sync = os.getenv("SYNTH_OPENAI_DEBUG") == "1"
        if debug_sync:
            try:
                print(
                    f"🔍 OPENAI DEBUG (sync): finish_reason={getattr(output.choices[0], 'finish_reason', None)}"
                )
                usage = getattr(output, "usage", None)
                if usage:
                    print(
                        f"🔍 OPENAI DEBUG (sync): usage prompt_tokens={getattr(usage, 'prompt_tokens', None)}, completion_tokens={getattr(usage, 'completion_tokens', None)}, total_tokens={getattr(usage, 'total_tokens', None)}"
                    )
            except Exception:
                pass

        # Convert tool calls to dict format
        tool_calls = None
        if message.tool_calls:
            tool_calls = [
                {
                    "id": tc.id,
                    "type": tc.type,
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    },
                }
                for tc in message.tool_calls
            ]

        # Attach basic usage if available
        usage_dict = None
        try:
            usage_obj = getattr(output, "usage", None)
            if usage_obj is not None:
                usage_dict = {
                    "prompt_tokens": getattr(usage_obj, "prompt_tokens", None),
                    "completion_tokens": getattr(usage_obj, "completion_tokens", None),
                    "total_tokens": getattr(usage_obj, "total_tokens", None),
                }
        except Exception:
            usage_dict = None

        lm_response = BaseLMResponse(
            raw_response=message.content or "",  # Use empty string if no content
            structured_output=None,
            tool_calls=tool_calls,
            usage=usage_dict,
        )
        lm_config["reasoning_effort"] = reasoning_effort
        used_cache_handler.add_to_managed_cache(
            model, messages, lm_config=lm_config, output=lm_response, tools=tools
        )
        return lm_response

    async def _hit_api_async_structured_output(
        self,
        model: str,
        messages: list[dict[str, Any]],
        response_model: BaseModel,
        temperature: float,
        use_ephemeral_cache_only: bool = False,
        reasoning_effort: str = "high",
        tools: list[BaseTool] | None = None,
    ) -> BaseLMResponse:
        lm_config = {
            "temperature": temperature,
            "response_model": response_model,
            "reasoning_effort": reasoning_effort,
        }
        used_cache_handler = get_cache_handler(use_ephemeral_cache_only)
        cache_result: BaseLMResponse | None = used_cache_handler.hit_managed_cache(
            model, messages, lm_config=lm_config, tools=tools
        )
        if cache_result is not None:
            return cache_result

        # Common API call params
        api_params = {
            "model": model,
            "messages": messages,
        }

        # Add tools if provided
        if tools and all(isinstance(tool, BaseTool) for tool in tools):
            api_params["tools"] = [tool.to_openai_tool() for tool in tools]
        elif tools:
            api_params["tools"] = tools

        # Only add temperature for non o1/o3 models
        if not any(prefix in model for prefix in ["o1-", "o3-"]):
            api_params["temperature"] = lm_config.get(
                "temperature", SPECIAL_BASE_TEMPS.get(model, 0)
            )

        # Add reasoning_effort only for o3-mini
        if model in ["o3-mini"]:
            api_params["reasoning_effort"] = reasoning_effort

        output = await self.async_client.chat.completions.create(**api_params)

        structured_output_api_result = response_model(**output.choices[0].message.content)
        tool_calls = output.choices[0].message.tool_calls
        lm_response = BaseLMResponse(
            raw_response=output.choices[0].message.content,
            structured_output=structured_output_api_result,
            tool_calls=tool_calls,
        )
        lm_config["reasoning_effort"] = reasoning_effort
        used_cache_handler.add_to_managed_cache(
            model, messages, lm_config=lm_config, output=lm_response, tools=tools
        )
        return lm_response

    def _hit_api_sync_structured_output(
        self,
        model: str,
        messages: list[dict[str, Any]],
        response_model: BaseModel,
        temperature: float,
        use_ephemeral_cache_only: bool = False,
        reasoning_effort: str = "high",
        tools: list[BaseTool] | None = None,
    ) -> BaseLMResponse:
        lm_config = {
            "temperature": temperature,
            "response_model": response_model,
            "reasoning_effort": reasoning_effort,
        }
        used_cache_handler = get_cache_handler(use_ephemeral_cache_only)
        cache_result: BaseLMResponse | None = used_cache_handler.hit_managed_cache(
            model, messages, lm_config=lm_config, tools=tools
        )
        if cache_result is not None:
            return cache_result

        # Common API call params
        api_params = {
            "model": model,
            "messages": messages,
        }

        # Add tools if provided
        if tools and all(isinstance(tool, BaseTool) for tool in tools):
            api_params["tools"] = [tool.to_openai_tool() for tool in tools]
        elif tools:
            api_params["tools"] = tools

        # Only add temperature for non o1/o3 models
        if not any(prefix in model for prefix in ["o1-", "o3-"]):
            api_params["temperature"] = lm_config.get(
                "temperature", SPECIAL_BASE_TEMPS.get(model, 0)
            )

        # Add reasoning_effort only for o3-mini
        if model in ["o3-mini"]:
            api_params["reasoning_effort"] = reasoning_effort

        # Normalize for external OpenAI as well in sync path
        try:
            base_url_obj = getattr(self.sync_client, "base_url", None)
            base_url_str_sync = str(base_url_obj) if base_url_obj is not None else ""
        except Exception:
            base_url_str_sync = ""
        if (
            "openai.com" in base_url_str_sync or "api.groq.com" in base_url_str_sync
        ) and model.startswith("gpt-5"):
            if "max_tokens" in api_params:
                api_params["max_completion_tokens"] = api_params.pop("max_tokens")
            if "temperature" in api_params:
                api_params.pop("temperature", None)

        output = self.sync_client.chat.completions.create(**api_params)

        structured_output_api_result = response_model(**output.choices[0].message.content)
        tool_calls = output.choices[0].message.tool_calls
        lm_response = BaseLMResponse(
            raw_response=output.choices[0].message.content,
            structured_output=structured_output_api_result,
            tool_calls=tool_calls,
        )
        lm_config["reasoning_effort"] = reasoning_effort
        used_cache_handler.add_to_managed_cache(
            model, messages, lm_config=lm_config, output=lm_response, tools=tools
        )
        return lm_response
