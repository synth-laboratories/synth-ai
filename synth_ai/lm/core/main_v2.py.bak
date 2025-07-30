"""
Enhanced LM class with native v2 tracing support.

This module extends the LM class to support v2 tracing through decorators,
enabling clean integration without modifying provider wrappers.
"""

from typing import Any, Dict, List, Literal, Optional, Union
import os
import functools

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

# V2 tracing imports
from synth_ai.tracing_v2.session_tracer import SessionTracer
from synth_ai.tracing_v2.decorators import (
    trace_span, set_active_session_tracer, 
    set_system_id, set_turn_number, get_config
)
from synth_ai.tracing_v2.abstractions import CAISEvent
from opentelemetry.trace import SpanKind


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
        system_info = {"role": "system", "content": sys_msg}
        user_info = {
            "role": "user",
            "content": [{"type": "text", "text": user_msg}]
            + [
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/png",
                        "data": image_bytes,
                    },
                }
                for image_bytes in images_bytes
            ],
        }
        return [system_info, user_info]
    elif len(images_bytes) > 0:
        raise ValueError("Images are not yet supported for this model")
    else:
        return [
            {"role": "system", "content": sys_msg},
            {"role": "user", "content": user_msg},
        ]


def extract_lm_attributes(args, kwargs, result=None, error=None) -> Dict[str, Any]:
    """Custom attribute extraction for LM calls following OTel GenAI conventions."""
    attrs = {}
    
    # Extract input attributes
    messages = kwargs.get('messages')
    if not messages and 'system_message' in kwargs and 'user_message' in kwargs:
        messages = [
            {"role": "system", "content": kwargs['system_message']},
            {"role": "user", "content": kwargs['user_message']}
        ]
    
    if messages:
        attrs['gen_ai.request.messages'] = messages
    
    # Get model from LM instance (first arg is self)
    if args and hasattr(args[0], 'model_name'):
        attrs['gen_ai.request.model'] = args[0].model_name
    
    # Temperature from LM config
    if args and hasattr(args[0], 'lm_config'):
        attrs['gen_ai.request.temperature'] = args[0].lm_config.get('temperature')
    
    # Other request attributes
    if 'response_model' in kwargs and kwargs['response_model']:
        attrs['gen_ai.request.response_format'] = kwargs['response_model'].__name__
    if 'tools' in kwargs and kwargs['tools']:
        attrs['gen_ai.request.tools_count'] = len(kwargs['tools'])
    if 'reasoning_effort' in kwargs:
        attrs['gen_ai.request.reasoning_effort'] = kwargs['reasoning_effort']
    
    # Extract output attributes
    if result and isinstance(result, BaseLMResponse):
        if result.raw_response:
            attrs['gen_ai.response.content'] = result.raw_response[:1000]  # Truncate
            
        if result.structured_output:
            attrs['gen_ai.response.has_structured_output'] = True
            attrs['gen_ai.response.structured_output_type'] = type(result.structured_output).__name__
            
        if result.tool_calls:
            attrs['gen_ai.response.tool_calls_count'] = len(result.tool_calls)
            
        # Try to extract usage if available (provider-specific)
        # This would need to be enhanced based on specific provider response formats
        if hasattr(result, '_raw_response'):
            raw = result._raw_response
            if hasattr(raw, 'usage'):
                attrs['gen_ai.response.usage.prompt_tokens'] = getattr(raw.usage, 'prompt_tokens', 0)
                attrs['gen_ai.response.usage.completion_tokens'] = getattr(raw.usage, 'completion_tokens', 0)
                attrs['gen_ai.response.usage.total_tokens'] = getattr(raw.usage, 'total_tokens', 0)
    
    # Error handling
    if error:
        attrs['gen_ai.error'] = str(error)
        attrs['gen_ai.error_type'] = type(error).__name__
    
    return attrs


# Create the trace decorator for LM calls
trace_lm_call = functools.partial(
    trace_span,
    kind=SpanKind.CLIENT,
    event_type=CAISEvent,
    attrs_fn=extract_lm_attributes
)


class LM:
    """
    Enhanced Language Model interface with native v2 tracing support.

    Args:
        model_name: The name of the model to use.
        formatting_model_name: The model to use for formatting structured outputs.
        temperature: The temperature setting for the model (0.0 to 1.0).
        max_retries: Number of retries for API calls ("None", "Few", or "Many").
        structured_output_mode: Mode for structured outputs ("stringified_json" or "forced_json").
        synth_logging: Whether to enable Synth v1 logging (for backwards compatibility).
        provider: Optional provider override.
        session_tracer: Optional v2 SessionTracer instance for tracing.
        system_id: Optional system ID for v2 tracing (defaults to "lm_{model_name}").
        enable_v2_tracing: Whether to enable v2 tracing (defaults to True).
    """

    model_name: str
    client: VendorBase
    lm_config: Dict[str, Any]
    structured_output_handler: StructuredOutputHandler
    
    # V2 tracing attributes
    session_tracer: Optional[SessionTracer]
    system_id: str
    enable_v2_tracing: bool

    def __init__(
        self,
        model_name: str,
        formatting_model_name: str,
        temperature: float,
        max_retries: Literal["None", "Few", "Many"] = "Few",
        structured_output_mode: Literal["stringified_json", "forced_json"] = "stringified_json",
        synth_logging: bool = True,
        provider: Optional[
            Union[
                Literal[
                    "openai",
                    "anthropic",
                    "groq",
                    "gemini",
                    "deepseek",
                    "grok",
                    "mistral",
                    "openrouter",
                    "together",
                    "synth",
                ],
                str,
            ]
        ] = None,
        # New v2 tracing parameters
        session_tracer: Optional[SessionTracer] = None,
        system_id: Optional[str] = None,
        enable_v2_tracing: bool = True,
    ):
        # Check for environment variable if provider is not specified
        effective_provider = provider or os.environ.get("SYNTH_AI_DEFAULT_PROVIDER")

        self.client = get_client(
            model_name,
            with_formatting=structured_output_mode == "forced_json",
            synth_logging=synth_logging,
            provider=effective_provider,
        )

        formatting_client = get_client(
            formatting_model_name, with_formatting=True, synth_logging=synth_logging, provider=effective_provider
        )

        max_retries_dict = {"None": 0, "Few": 2, "Many": 5}
        self.structured_output_handler = StructuredOutputHandler(
            self.client,
            formatting_client,
            structured_output_mode,
            {"max_retries": max_retries_dict.get(max_retries, 2)},
        )
        self.backup_structured_output_handler = StructuredOutputHandler(
            self.client,
            formatting_client,
            "forced_json",
            {"max_retries": max_retries_dict.get(max_retries, 2)},
        )
        # Override temperature to 1 for reasoning models
        effective_temperature = 1.0 if model_name in reasoning_models else temperature
        self.lm_config = {"temperature": effective_temperature}
        self.model_name = model_name
        
        # V2 tracing setup
        self.session_tracer = session_tracer
        self.system_id = system_id or f"lm_{model_name}"
        self.enable_v2_tracing = enable_v2_tracing and get_config().is_tracing_enabled()
        
        # Set context if provided
        if self.session_tracer and self.enable_v2_tracing:
            set_active_session_tracer(self.session_tracer)
            set_system_id(self.system_id)

    @trace_lm_call(name="lm.respond")
    def respond_sync(
        self,
        system_message: Optional[str] = None,
        user_message: Optional[str] = None,
        messages: Optional[List[Dict]] = None,
        images_as_bytes: List[Any] = [],
        response_model: Optional[BaseModel] = None,
        use_ephemeral_cache_only: bool = False,
        tools: Optional[List[BaseTool]] = None,
        reasoning_effort: str = "low",
        # New v2 parameter
        turn_number: Optional[int] = None,
    ):
        # Update turn context if provided
        if turn_number is not None and self.enable_v2_tracing:
            set_turn_number(turn_number)
            
        assert (system_message is None) == (user_message is None), (
            "Must provide both system_message and user_message or neither"
        )
        assert (messages is None) != (system_message is None), (
            "Must provide either messages or system_message/user_message pair, but not both"
        )
        assert not (response_model and tools), "Cannot provide both response_model and tools"
        if messages is None:
            messages = build_messages(
                system_message, user_message, images_as_bytes, self.model_name
            )
        result = None
        if response_model:
            try:
                result = self.structured_output_handler.call_sync(
                    messages,
                    model=self.model_name,
                    lm_config=self.lm_config,
                    response_model=response_model,
                    use_ephemeral_cache_only=use_ephemeral_cache_only,
                    reasoning_effort=reasoning_effort,
                )
            except StructuredOutputCoercionFailureException:
                result = self.backup_structured_output_handler.call_sync(
                    messages,
                    model=self.model_name,
                    lm_config=self.lm_config,
                    response_model=response_model,
                    use_ephemeral_cache_only=use_ephemeral_cache_only,
                    reasoning_effort=reasoning_effort,
                )
        else:
            result = self.client._hit_api_sync(
                messages=messages,
                model=self.model_name,
                lm_config=self.lm_config,
                use_ephemeral_cache_only=use_ephemeral_cache_only,
                tools=tools,
                reasoning_effort=reasoning_effort,
            )
        assert isinstance(result.raw_response, str), "Raw response must be a string"
        assert (
            isinstance(result.structured_output, BaseModel) or result.structured_output is None
        ), "Structured output must be a Pydantic model or None"
        assert isinstance(result.tool_calls, list) or result.tool_calls is None, (
            "Tool calls must be a list or None"
        )
        return result

    @trace_lm_call(name="lm.respond")
    async def respond_async(
        self,
        system_message: Optional[str] = None,
        user_message: Optional[str] = None,
        messages: Optional[List[Dict]] = None,
        images_as_bytes: List[Any] = [],
        response_model: Optional[BaseModel] = None,
        use_ephemeral_cache_only: bool = False,
        tools: Optional[List[BaseTool]] = None,
        reasoning_effort: str = "low",
        # New v2 parameter
        turn_number: Optional[int] = None,
    ):
        # Update turn context if provided
        if turn_number is not None and self.enable_v2_tracing:
            set_turn_number(turn_number)
            
        assert (system_message is None) == (user_message is None), (
            "Must provide both system_message and user_message or neither"
        )
        assert (messages is None) != (system_message is None), (
            "Must provide either messages or system_message/user_message pair, but not both"
        )

        assert not (response_model and tools), "Cannot provide both response_model and tools"
        if messages is None:
            messages = build_messages(
                system_message, user_message, images_as_bytes, self.model_name
            )
        result = None
        if response_model:
            try:
                result = await self.structured_output_handler.call_async(
                    messages,
                    model=self.model_name,
                    lm_config=self.lm_config,
                    response_model=response_model,
                    use_ephemeral_cache_only=use_ephemeral_cache_only,
                    reasoning_effort=reasoning_effort,
                )
            except StructuredOutputCoercionFailureException:
                result = await self.backup_structured_output_handler.call_async(
                    messages,
                    model=self.model_name,
                    lm_config=self.lm_config,
                    response_model=response_model,
                    use_ephemeral_cache_only=use_ephemeral_cache_only,
                    reasoning_effort=reasoning_effort,
                )
        else:
            result = await self.client._hit_api_async(
                messages=messages,
                model=self.model_name,
                lm_config=self.lm_config,
                use_ephemeral_cache_only=use_ephemeral_cache_only,
                tools=tools,
                reasoning_effort=reasoning_effort,
            )
        assert isinstance(result.raw_response, str), "Raw response must be a string"
        assert (
            isinstance(result.structured_output, BaseModel) or result.structured_output is None
        ), "Structured output must be a Pydantic model or None"
        assert isinstance(result.tool_calls, list) or result.tool_calls is None, (
            "Tool calls must be a list or None"
        )
        return result


class LMTracingContext:
    """Context manager for LM with v2 tracing."""
    
    def __init__(self, lm: LM, session_tracer: SessionTracer):
        self.lm = lm
        self.session_tracer = session_tracer
        self.original_tracer = lm.session_tracer
        
    def __enter__(self):
        self.lm.session_tracer = self.session_tracer
        set_active_session_tracer(self.session_tracer)
        set_system_id(self.lm.system_id)
        return self.lm
        
    def __exit__(self, *args):
        self.lm.session_tracer = self.original_tracer
        if self.original_tracer:
            set_active_session_tracer(self.original_tracer)