"""
Enhanced LM class with native v3 tracing support.

This module provides the LM class with async v3 tracing support,
replacing the v2 DuckDB-based implementation.
"""

from typing import Any, Dict, List, Literal, Optional, Union
import os
import functools
import asyncio
import time

from pydantic import BaseModel, Field

from synth_ai.lm.core.exceptions import StructuredOutputCoercionFailureException
from synth_ai.lm.core.vendor_clients import (
    anthropic_naming_regexes,
    get_client,
    openai_naming_regexes,
)
from synth_ai.lm.structured_outputs.handler import StructuredOutputHandler
from synth_ai.lm.vendors.base import VendorBase, BaseLMResponse
from synth_ai.lm.tools.base import BaseTool
from synth_ai.lm.config import reasoning_models

# V3 tracing imports
from synth_ai.tracing_v3.session_tracer import SessionTracer
from synth_ai.tracing_v3.decorators import set_session_id, set_turn_number, set_session_tracer
from synth_ai.tracing_v3.abstractions import LMCAISEvent, TimeRecord


def build_messages(
    sys_msg: str,
    user_msg: str,
    images_bytes: List = [],
    model_name: Optional[str] = None,
) -> List[Dict]:
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
        vendor: Optional[str] = None,
        model: Optional[str] = None,
        # v2 compatibility parameters
        model_name: Optional[str] = None,  # Alias for model
        formatting_model_name: Optional[str] = None,  # For structured outputs
        provider: Optional[str] = None,  # Alias for vendor
        synth_logging: bool = True,  # v2 compatibility
        max_retries: Literal["None", "Few", "Many"] = "Few",  # v2 compatibility
        # v3 parameters
        is_structured: Optional[bool] = None,
        structured_outputs_vendor: Optional[str] = None,
        response_format: Union[BaseModel, Dict[str, Any], None] = None,
        json_mode: bool = False,
        temperature: float = 0.8,
        session_tracer: Optional[SessionTracer] = None,
        system_id: Optional[str] = None,
        enable_v3_tracing: bool = True,
        enable_v2_tracing: Optional[bool] = None,  # v2 compatibility
        **additional_params,
    ):
        # Handle v2 compatibility parameters
        if model_name and not model:
            model = model_name
        if provider and not vendor:
            vendor = provider
        if enable_v2_tracing is not None:
            enable_v3_tracing = enable_v2_tracing

        # If vendor not provided, infer from model name
        if vendor is None and model is not None:
            # Import vendor detection logic
            from synth_ai.lm.core.vendor_clients import (
                openai_naming_regexes,
                anthropic_naming_regexes,
                gemini_naming_regexes,
                deepseek_naming_regexes,
                groq_naming_regexes,
                grok_naming_regexes,
                openrouter_naming_regexes,
                custom_endpoint_naming_regexes,
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
        self.is_structured = is_structured
        self.structured_outputs_vendor = structured_outputs_vendor
        self.response_format = response_format
        self.json_mode = json_mode
        self.temperature = temperature
        self.session_tracer = session_tracer
        self.system_id = system_id or f"lm_{self.vendor or 'unknown'}_{self.model or 'unknown'}"
        self.enable_v3_tracing = enable_v3_tracing
        self.additional_params = additional_params

        # Set structured output handler if needed
        if self.response_format:
            self.is_structured = True
            self.structured_output_handler = StructuredOutputHandler(
                response_format=self.response_format, vendor_wrapper=self.get_vendor_wrapper()
            )
        else:
            self.structured_output_handler = None

        # Initialize vendor wrapper
        self._vendor_wrapper = None

    def get_vendor_wrapper(self) -> VendorBase:
        """Get or create the vendor wrapper."""
        if self._vendor_wrapper is None:
            # For now, just use the vendor client directly as it implements the needed interface
            self._vendor_wrapper = get_client(self.model, provider=self.vendor)
        return self._vendor_wrapper

    async def respond_async(
        self,
        system_message: Optional[str] = None,
        user_message: Optional[str] = None,
        messages: Optional[List[Dict]] = None,  # v2 compatibility
        images_bytes: List[bytes] = [],
        images_as_bytes: Optional[List[bytes]] = None,  # v2 compatibility
        response_model: Optional[BaseModel] = None,  # v2 compatibility
        tools: Optional[List[BaseTool]] = None,
        turn_number: Optional[int] = None,
        **kwargs,
    ) -> BaseLMResponse:
        """Async method to get LM response with v3 tracing."""
        start_time = time.time()

        # Handle v2 compatibility
        if images_as_bytes is not None:
            images_bytes = images_as_bytes

        # Handle response_model for structured outputs
        if response_model and not self.response_format:
            self.response_format = response_model
            self.is_structured = True
            self.structured_output_handler = StructuredOutputHandler(
                response_format=self.response_format, vendor_wrapper=self.get_vendor_wrapper()
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

        # Get vendor wrapper
        vendor_wrapper = self.get_vendor_wrapper()

        # Prepare parameters based on vendor type
        if hasattr(vendor_wrapper, "_hit_api_async"):
            # OpenAIStandard expects lm_config
            lm_config = {"temperature": self.temperature, **self.additional_params, **kwargs}
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

        # Call vendor
        try:
            # Try the standard method names
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
        except Exception as e:
            print(f"Error calling vendor: {e}")
            raise

        # Handle structured output
        if self.structured_output_handler:
            response = self.structured_output_handler.process_response(response)

        # Record tracing event if enabled
        if (
            self.enable_v3_tracing
            and self.session_tracer
            and hasattr(self.session_tracer, "current_session")
        ):
            latency_ms = int((time.time() - start_time) * 1000)

            # Extract usage info if available
            usage_info = {}
            if hasattr(response, "usage") and response.usage:
                usage_info = {
                    "input_tokens": response.usage.get("input_tokens", 0),
                    "output_tokens": response.usage.get("output_tokens", 0),
                    "total_tokens": response.usage.get("total_tokens", 0),
                    "cost_usd": response.usage.get("cost_usd", 0.0),
                }
            else:
                # Default values when usage is not available
                usage_info = {
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "total_tokens": 0,
                    "cost_usd": 0.0,
                }

            # Create LM event
            lm_event = LMCAISEvent(
                system_instance_id=self.system_id,
                time_record=TimeRecord(event_time=time.time(), message_time=turn_number),
                model_name=self.model or self.vendor,
                provider=self.vendor,
                input_tokens=usage_info["input_tokens"],
                output_tokens=usage_info["output_tokens"],
                total_tokens=usage_info["total_tokens"],
                cost_usd=usage_info["cost_usd"],
                latency_ms=latency_ms,
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
        system_message: Optional[str] = None,
        user_message: Optional[str] = None,
        messages: Optional[List[Dict]] = None,  # v2 compatibility
        images_bytes: List[bytes] = [],
        images_as_bytes: Optional[List[bytes]] = None,  # v2 compatibility
        response_model: Optional[BaseModel] = None,  # v2 compatibility
        tools: Optional[List[BaseTool]] = None,
        turn_number: Optional[int] = None,
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
