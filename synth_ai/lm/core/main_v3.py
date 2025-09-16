"""
Enhanced LM class with native v3 tracing support.

This module provides the LM class with async v3 tracing support,
replacing the v2 DuckDB-based implementation.
"""

import asyncio
import time
from typing import Any, Literal

from pydantic import BaseModel

from synth_ai.lm.config import reasoning_models
from synth_ai.lm.core.vendor_clients import (
    anthropic_naming_regexes,
    get_client,
    openai_naming_regexes,
)
from synth_ai.lm.structured_outputs.handler import StructuredOutputHandler
from synth_ai.lm.tools.base import BaseTool
from synth_ai.lm.vendors.base import BaseLMResponse, VendorBase

# V3 tracing imports
from synth_ai.tracing_v3.abstractions import LMCAISEvent, TimeRecord
from synth_ai.tracing_v3.decorators import set_turn_number
from synth_ai.tracing_v3.llm_call_record_helpers import (
    compute_aggregates_from_call_records,
    create_llm_call_record_from_response,
)
from synth_ai.tracing_v3.session_tracer import SessionTracer


def build_messages(
    sys_msg: str,
    user_msg: str,
    images_bytes: list | None = None,
    model_name: str | None = None,
) -> list[dict]:
    images_bytes = images_bytes or []
    if len(images_bytes) > 0 and any(regex.match(model_name) for regex in openai_naming_regexes):
        return [
            {"role": "system", "content": sys_msg},
            {
                "role": "user",
                "content": [{"type": "text", "text": user_msg}]
                + [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{image_bytes}"},
                    }
                    for image_bytes in images_bytes
                ],
            },
        ]
    elif len(images_bytes) > 0 and any(
        regex.match(model_name) for regex in anthropic_naming_regexes
    ):
        return [
            {"role": "system", "content": sys_msg},
            {
                "role": "user",
                "content": [{"type": "text", "text": user_msg}]
                + [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/jpeg",
                            "data": image_bytes,
                        },
                    }
                    for image_bytes in images_bytes
                ],
            },
        ]
    else:
        return [
            {"role": "system", "content": sys_msg},
            {"role": "user", "content": user_msg},
        ]


class LM:
    """Language Model interface with v3 tracing support."""

    def __init__(
        self,
        vendor: str | None = None,
        model: str | None = None,
        # v2 compatibility parameters
        model_name: str | None = None,  # Alias for model
        formatting_model_name: str | None = None,  # For structured outputs
        provider: str | None = None,  # Alias for vendor
        synth_logging: bool = True,  # v2 compatibility
        max_retries: Literal["None", "Few", "Many"] = "Few",  # v2 compatibility
        # v3 parameters
        is_structured: bool | None = None,
        structured_outputs_vendor: str | None = None,
        response_format: type[BaseModel] | dict[str, Any] | None = None,
        json_mode: bool = False,
        temperature: float = 0.8,
        session_tracer: SessionTracer | None = None,
        system_id: str | None = None,
        enable_v3_tracing: bool = True,
        enable_v2_tracing: bool | None = None,  # v2 compatibility
        # Responses API parameters
        auto_store_responses: bool = True,
        use_responses_api: bool | None = None,
        **additional_params,
    ):
        # Handle v2 compatibility parameters
        if model_name and not model:
            model = model_name
        if provider and not vendor:
            vendor = provider
        if enable_v2_tracing is not None:
            enable_v3_tracing = enable_v2_tracing

        # Debug logging
        print(f"🔍 LM __init__: provider={provider}, vendor={vendor}, model={model}")

        # If vendor not provided, infer from model name
        # But only if no explicit provider was given
        if vendor is None and model is not None:
            # Import vendor detection logic
            from synth_ai.lm.core.vendor_clients import (
                anthropic_naming_regexes,
                custom_endpoint_naming_regexes,
                deepseek_naming_regexes,
                gemini_naming_regexes,
                grok_naming_regexes,
                groq_naming_regexes,
                openai_naming_regexes,
                openrouter_naming_regexes,
                together_naming_regexes,
            )

            # Check model name against patterns
            if any(regex.match(model) for regex in openai_naming_regexes):
                vendor = "openai"
            elif any(regex.match(model) for regex in anthropic_naming_regexes):
                vendor = "anthropic"
            elif any(regex.match(model) for regex in gemini_naming_regexes):
                vendor = "gemini"
            elif any(regex.match(model) for regex in deepseek_naming_regexes):
                vendor = "deepseek"
            elif any(regex.match(model) for regex in groq_naming_regexes):
                vendor = "groq"
            elif any(regex.match(model) for regex in grok_naming_regexes):
                vendor = "grok"
            elif any(regex.match(model) for regex in openrouter_naming_regexes):
                vendor = "openrouter"
            elif any(regex.match(model) for regex in custom_endpoint_naming_regexes):
                vendor = "custom_endpoint"
            elif any(regex.match(model) for regex in together_naming_regexes):
                vendor = "together"
            else:
                raise ValueError(f"Could not infer vendor from model name: {model}")

        self.vendor = vendor
        self.model = model
        print(f"🔍 LM final: vendor={self.vendor}, model={self.model}")
        self.is_structured = is_structured
        self.structured_outputs_vendor = structured_outputs_vendor
        self.response_format = response_format
        self.json_mode = json_mode
        self.temperature = temperature
        self.session_tracer = session_tracer
        self.system_id = system_id or f"lm_{self.vendor or 'unknown'}_{self.model or 'unknown'}"
        self.enable_v3_tracing = enable_v3_tracing
        self.additional_params = additional_params

        # Initialize vendor wrapper early, before any potential usage
        # (e.g., within StructuredOutputHandler initialization below)
        self._vendor_wrapper = None

        # Responses API thread management
        self.auto_store_responses = auto_store_responses
        self.use_responses_api = use_responses_api
        self._last_response_id: str | None = None

        # Set structured output handler if needed
        if self.response_format:
            self.is_structured = True
            # Choose mode automatically: prefer forced_json for OpenAI/reasoning models
            forced_json_preferred = (self.vendor == "openai") or (
                self.model in reasoning_models if self.model else False
            )
            structured_output_mode = "forced_json" if forced_json_preferred else "stringified_json"

            # Build core and formatting clients
            core_client = get_client(
                self.model,
                with_formatting=(structured_output_mode == "forced_json"),
                provider=self.vendor,
            )
            formatting_model = formatting_model_name or self.model
            formatting_client = get_client(
                formatting_model,
                with_formatting=True,
                provider=self.vendor if self.vendor != "custom_endpoint" else None,
            )

            # Map retries
            max_retries_dict = {"None": 0, "Few": 2, "Many": 5}
            handler_params = {"max_retries": max_retries_dict.get(max_retries, 2)}

            self.structured_output_handler = StructuredOutputHandler(
                core_client,
                formatting_client,
                structured_output_mode,
                handler_params,
            )
        else:
            self.structured_output_handler = None

        # Vendor wrapper lazy-instantiated via get_vendor_wrapper()

    def get_vendor_wrapper(self) -> VendorBase:
        """Get or create the vendor wrapper."""
        if self._vendor_wrapper is None:
            # For now, just use the vendor client directly as it implements the needed interface
            self._vendor_wrapper = get_client(self.model, provider=self.vendor)
        return self._vendor_wrapper

    def _should_use_responses_api(self) -> bool:
        """Determine if Responses API should be used."""
        if self.use_responses_api is not None:
            return self.use_responses_api

        # Auto-detect based on model
        responses_models = {
            "o4-mini",
            "o3",
            "o3-mini",  # Supported Synth-hosted models
            "gpt-oss-120b",
            "gpt-oss-20b",  # OSS models via Synth
        }
        return self.model in responses_models or (self.model and self.model in reasoning_models)

    def _should_use_harmony(self) -> bool:
        """Determine if Harmony encoding should be used for OSS models."""
        # Only use Harmony for OSS models when NOT using OpenAI vendor
        # OpenAI hosts these models directly via Responses API
        harmony_models = {"gpt-oss-120b", "gpt-oss-20b"}
        return self.model in harmony_models and self.vendor != "openai"

    async def respond_async(
        self,
        system_message: str | None = None,
        user_message: str | None = None,
        messages: list[dict] | None = None,  # v2 compatibility
        images_bytes: list[bytes] | None = None,
        images_as_bytes: list[bytes] | None = None,  # v2 compatibility
        response_model: type[BaseModel] | None = None,  # v2 compatibility
        tools: list[BaseTool] | None = None,
        turn_number: int | None = None,
        previous_response_id: str | None = None,  # Responses API thread management
        **kwargs,
    ) -> BaseLMResponse:
        """Async method to get LM response with v3 tracing."""
        start_time = time.time()

        # Handle v2 compatibility
        images_bytes = images_as_bytes if images_as_bytes is not None else (images_bytes or [])

        # Handle response_model for structured outputs (runtime-provided)
        if response_model and not self.response_format:
            self.response_format = response_model
            self.is_structured = True
            # Mirror initialization logic from __init__
            forced_json_preferred = (self.vendor == "openai") or (
                self.model in reasoning_models if self.model else False
            )
            structured_output_mode = "forced_json" if forced_json_preferred else "stringified_json"
            core_client = get_client(
                self.model,
                with_formatting=(structured_output_mode == "forced_json"),
                provider=self.vendor,
            )
            formatting_client = get_client(
                self.model,
                with_formatting=True,
                provider=self.vendor if self.vendor != "custom_endpoint" else None,
            )
            self.structured_output_handler = StructuredOutputHandler(
                core_client,
                formatting_client,
                structured_output_mode,
                {"max_retries": 2},
            )

        # Set turn number if provided
        if turn_number is not None:
            set_turn_number(turn_number)

        # Handle messages parameter (v2 compatibility)
        if messages is not None:
            # Use provided messages directly
            if system_message or user_message:
                raise ValueError(
                    "Cannot specify both 'messages' and 'system_message'/'user_message'"
                )
            messages_to_use = messages
        else:
            # Build messages from system and user messages
            if not system_message or not user_message:
                raise ValueError(
                    "Must provide either 'messages' or both 'system_message' and 'user_message'"
                )
            messages_to_use = build_messages(system_message, user_message, images_bytes, self.model)

        # If using structured outputs, route through the handler
        if self.structured_output_handler and self.response_format:
            if tools:
                raise ValueError("Tools are not supported with structured output mode")
            response = await self.structured_output_handler.call_async(
                messages=messages_to_use,
                model=self.model,
                response_model=self.response_format,
                use_ephemeral_cache_only=False,
                lm_config={"temperature": self.temperature, **self.additional_params, **kwargs},
                reasoning_effort="high",
            )
        else:
            # Get vendor wrapper
            vendor_wrapper = self.get_vendor_wrapper()

            # Determine API type to use
            use_responses = self._should_use_responses_api()
            use_harmony = self._should_use_harmony()

            # Decide response ID to use for thread management
            response_id_to_use = None
            if previous_response_id:
                response_id_to_use = previous_response_id  # Manual override
            elif self.auto_store_responses and self._last_response_id:
                response_id_to_use = self._last_response_id  # Auto-chain

            # Prepare parameters based on vendor type
            if hasattr(vendor_wrapper, "_hit_api_async"):
                # OpenAIStandard expects lm_config
                lm_config = {"temperature": self.temperature, **self.additional_params, **kwargs}
                # Map convenience enable_thinking => thinking_mode unless explicitly set
                if "enable_thinking" in lm_config and "thinking_mode" not in lm_config:
                    try:
                        et = lm_config.get("enable_thinking")
                        if isinstance(et, bool):
                            lm_config["thinking_mode"] = "think" if et else "no_think"
                    except Exception:
                        pass
                if self.json_mode:
                    lm_config["response_format"] = {"type": "json_object"}

                params = {"model": self.model, "messages": messages_to_use, "lm_config": lm_config}
                if tools:
                    params["tools"] = tools
            else:
                # Other vendors use flat params
                params = {
                    "model": self.model,
                    "messages": messages_to_use,
                    "temperature": self.temperature,
                    **self.additional_params,
                    **kwargs,
                }

                if tools:
                    params["tools"] = [tool.to_dict() for tool in tools]

                if self.json_mode:
                    params["response_format"] = {"type": "json_object"}

            # Call vendor with appropriate API type
            try:
                # Route to appropriate API
                if use_harmony and hasattr(vendor_wrapper, "_hit_api_async_harmony"):
                    params["previous_response_id"] = response_id_to_use
                    response = await vendor_wrapper._hit_api_async_harmony(**params)
                elif use_responses and hasattr(vendor_wrapper, "_hit_api_async_responses"):
                    params["previous_response_id"] = response_id_to_use
                    response = await vendor_wrapper._hit_api_async_responses(**params)
                else:
                    # Standard chat completions API
                    if hasattr(vendor_wrapper, "_hit_api_async"):
                        response = await vendor_wrapper._hit_api_async(**params)
                    elif hasattr(vendor_wrapper, "respond_async"):
                        response = await vendor_wrapper.respond_async(**params)
                    elif hasattr(vendor_wrapper, "respond"):
                        # Fallback to sync in executor
                        loop = asyncio.get_event_loop()
                        response = await loop.run_in_executor(None, vendor_wrapper.respond, params)
                    else:
                        raise AttributeError(
                            f"Vendor wrapper {type(vendor_wrapper).__name__} has no suitable response method"
                        )
                    if not hasattr(response, "api_type"):
                        response.api_type = "chat"

                # Update stored response ID if auto-storing
                if (
                    self.auto_store_responses
                    and hasattr(response, "response_id")
                    and response.response_id
                ):
                    self._last_response_id = response.response_id

            except Exception as e:
                print(f"Error calling vendor: {e}")
                raise

        # No additional post-processing needed for structured outputs here

        # Record tracing event if enabled
        if (
            self.enable_v3_tracing
            and self.session_tracer
            and hasattr(self.session_tracer, "current_session")
        ):
            latency_ms = int((time.time() - start_time) * 1000)

            # Create LLMCallRecord from the response
            from datetime import datetime

            started_at = datetime.utcnow()
            completed_at = datetime.utcnow()

            call_record = create_llm_call_record_from_response(
                response=response,
                model_name=self.model or self.vendor,
                provider=self.vendor,
                messages=messages_to_use,
                temperature=self.temperature,
                request_params={**self.additional_params, **kwargs},
                tools=tools,
                started_at=started_at,
                completed_at=completed_at,
                latency_ms=latency_ms,
            )

            # Compute aggregates from the call record
            aggregates = compute_aggregates_from_call_records([call_record])

            # Create LM event with call_records
            lm_event = LMCAISEvent(
                system_instance_id=self.system_id,
                time_record=TimeRecord(event_time=time.time(), message_time=turn_number),
                # Aggregates at event level
                input_tokens=aggregates["input_tokens"],
                output_tokens=aggregates["output_tokens"],
                total_tokens=aggregates["total_tokens"],
                cost_usd=aggregates["cost_usd"],
                latency_ms=aggregates["latency_ms"],
                # Store the call record
                call_records=[call_record],
                metadata={
                    "temperature": self.temperature,
                    "json_mode": self.json_mode,
                    "has_tools": tools is not None,
                    "is_structured": self.is_structured,
                },
            )

            await self.session_tracer.record_event(lm_event)

            # Also record messages
            if user_message:
                await self.session_tracer.record_message(
                    content=user_message,
                    message_type="user",
                    metadata={"system_id": self.system_id},
                )
            elif messages:
                # Record the last user message from messages array
                for msg in reversed(messages_to_use):
                    if msg.get("role") == "user":
                        content = msg.get("content", "")
                        if isinstance(content, list):
                            # Extract text from multi-modal content
                            text_parts = [
                                part.get("text", "")
                                for part in content
                                if part.get("type") == "text"
                            ]
                            content = " ".join(text_parts)
                        await self.session_tracer.record_message(
                            content=content,
                            message_type="user",
                            metadata={"system_id": self.system_id},
                        )
                        break

            await self.session_tracer.record_message(
                content=response.raw_response,
                message_type="assistant",
                metadata={"system_id": self.system_id},
            )

        return response

    def respond(
        self,
        system_message: str | None = None,
        user_message: str | None = None,
        messages: list[dict] | None = None,  # v2 compatibility
        images_bytes: list[bytes] | None = None,
        images_as_bytes: list[bytes] | None = None,  # v2 compatibility
        response_model: type[BaseModel] | None = None,  # v2 compatibility
        tools: list[BaseTool] | None = None,
        previous_response_id: str | None = None,  # Responses API thread management
        turn_number: int | None = None,
        **kwargs,
    ) -> BaseLMResponse:
        """Synchronous wrapper for respond_async."""
        # For backward compatibility, run async in new event loop
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # If loop is already running, schedule coroutine
                future = asyncio.ensure_future(
                    self.respond_async(
                        system_message=system_message,
                        user_message=user_message,
                        messages=messages,
                        images_bytes=images_bytes,
                        images_as_bytes=images_as_bytes,
                        response_model=response_model,
                        tools=tools,
                        turn_number=turn_number,
                        **kwargs,
                    )
                )
                return loop.run_until_complete(future)
            else:
                # Create new loop
                return asyncio.run(
                    self.respond_async(
                        system_message=system_message,
                        user_message=user_message,
                        messages=messages,
                        images_bytes=images_bytes,
                        images_as_bytes=images_as_bytes,
                        response_model=response_model,
                        tools=tools,
                        turn_number=turn_number,
                        **kwargs,
                    )
                )
        except RuntimeError:
            # Fallback: create new event loop
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                return loop.run_until_complete(
                    self.respond_async(
                        system_message=system_message,
                        user_message=user_message,
                        messages=messages,
                        images_bytes=images_bytes,
                        images_as_bytes=images_as_bytes,
                        response_model=response_model,
                        tools=tools,
                        turn_number=turn_number,
                        **kwargs,
                    )
                )
            finally:
                loop.close()

    async def __aenter__(self):
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        pass

    def __enter__(self):
        """Sync context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Sync context manager exit."""
        pass

    # v2 compatibility methods
    def respond_sync(self, **kwargs) -> BaseLMResponse:
        """Alias for respond() for v2 compatibility."""
        return self.respond(**kwargs)

    async def respond_async_v2(self, **kwargs) -> BaseLMResponse:
        """Alias for respond_async() for v2 compatibility."""
        return await self.respond_async(**kwargs)
