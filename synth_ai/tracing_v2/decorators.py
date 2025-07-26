"""
V3 Decorator Implementation for Synth-AI Tracing (Improved)

This module provides decorators that emit OpenTelemetry spans while maintaining
compatibility with v2 SessionTracer patterns. Designed for < 5% overhead and
proper context propagation across async/thread boundaries.

Key improvements:
- Relies on OTel SDK sampling instead of double sampling
- Uses OTel events for prompt/response bodies
- Adds PII masking capability
- Includes cost tracking for AI calls
- Better resource attributes
- Early exit when tracing disabled
"""

import asyncio
import functools
import time
import os
import re
import json
import multiprocessing
import contextvars
from typing import Any, Callable, Optional, Dict, TypeVar, Union, Tuple, List, Pattern
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor

from opentelemetry import trace, context
from opentelemetry.trace import SpanKind, Status, StatusCode
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.sdk.resources import Resource
from opentelemetry.semconv.trace import SpanAttributes
from opentelemetry.semconv.resource import ResourceAttributes

from synth_ai.tracing_v2.session_tracer import SessionTracer, TimeRecord
from synth_ai.tracing_v2.abstractions import (
    CAISEvent, EnvironmentEvent, RuntimeEvent, SessionEvent
)

# Type definitions
T = TypeVar('T')
F = TypeVar('F', bound=Callable[..., Any])

# Context variables for active tracers
_session_tracer_ctx: contextvars.ContextVar[Optional[SessionTracer]] = \
    contextvars.ContextVar("session_tracer", default=None)
_system_id_ctx: contextvars.ContextVar[Optional[str]] = \
    contextvars.ContextVar("system_id", default=None)
_turn_number_ctx: contextvars.ContextVar[Optional[int]] = \
    contextvars.ContextVar("turn_number", default=None)

# Import configuration
from synth_ai.tracing_v2.config import get_config

# Model pricing (per 1M tokens) - add more as needed
MODEL_PRICING = {
    "gpt-4": {"input": 30.0, "output": 60.0},
    "gpt-4-turbo": {"input": 10.0, "output": 30.0},
    "gpt-3.5-turbo": {"input": 0.5, "output": 1.5},
    "claude-3-opus": {"input": 15.0, "output": 75.0},
    "claude-3-sonnet": {"input": 3.0, "output": 15.0},
    "claude-3-haiku": {"input": 0.25, "output": 1.25},
}

# PII masking patterns
DEFAULT_PII_PATTERNS = [
    (r'\b\d{3}-\d{2}-\d{4}\b', '[SSN]'),  # SSN
    (r'\b\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}\b', '[CARD]'),  # Credit card
    (r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '[EMAIL]'),  # Email
    (r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b', '[PHONE]'),  # Phone
]

# Initialize OTel with resource attributes
def setup_otel_tracer():
    """Set up OTel tracer with proper resource attributes and batching."""
    config = get_config()
    
    # Create resource with service info
    resource = Resource.create({
        ResourceAttributes.SERVICE_NAME: config.otel_service_name or "synth-ai",
        ResourceAttributes.SERVICE_VERSION: "1.0.0",
        ResourceAttributes.DEPLOYMENT_ENVIRONMENT: os.getenv("DEPLOYMENT_ENV", "development"),
        ResourceAttributes.SERVICE_INSTANCE_ID: os.getenv("HOSTNAME", "local"),
    })
    
    # Create tracer provider with resource
    provider = TracerProvider(resource=resource)
    
    # Add batch processor if exporter configured
    if config.otel_exporter:
        processor = BatchSpanProcessor(
            config.otel_exporter,
            max_queue_size=config.max_queue_size,
            max_export_batch_size=config.batch_size,
            schedule_delay_millis=config.flush_interval_ms,
        )
        provider.add_span_processor(processor)
    
    # Set as global provider
    trace.set_tracer_provider(provider)
    
    return trace.get_tracer("synth-ai", "1.0.0")

# Global tracer instance
tracer = setup_otel_tracer()


def set_active_session_tracer(tracer: SessionTracer) -> None:
    """Set the active v2 session tracer for the current context."""
    _session_tracer_ctx.set(tracer)


def get_active_session_tracer() -> Optional[SessionTracer]:
    """Get the active v2 session tracer from context."""
    return _session_tracer_ctx.get()


def set_system_id(system_id: str) -> None:
    """Set the system ID for the current context."""
    _system_id_ctx.set(system_id)


def get_system_id() -> Optional[str]:
    """Get the system ID from context."""
    return _system_id_ctx.get()


def set_turn_number(turn: int) -> None:
    """Set the current turn number for the context."""
    _turn_number_ctx.set(turn)


def get_turn_number() -> Optional[int]:
    """Get the current turn number from context."""
    return _turn_number_ctx.get()


def mask_pii(text: str, patterns: Optional[List[Tuple[str, str]]] = None) -> str:
    """Mask PII in text using regex patterns."""
    if not isinstance(text, str):
        return text
    
    patterns = patterns or DEFAULT_PII_PATTERNS
    masked = text
    for pattern, replacement in patterns:
        masked = re.sub(pattern, replacement, masked, flags=re.IGNORECASE)
    
    return masked


def truncate_and_mask(data: Any, max_bytes: Optional[int] = None) -> Any:
    """Truncate large payloads and mask PII."""
    config = get_config()
    if not config.truncate_enabled:
        return data
    
    if max_bytes is None:
        max_bytes = config.max_payload_bytes
    
    # Mask PII in strings
    if isinstance(data, str):
        masked = mask_pii(data) if config.mask_pii else data
        if len(masked) > max_bytes:
            return masked[:max_bytes] + "... [truncated]"
        return masked
    elif isinstance(data, dict):
        return {k: truncate_and_mask(v, max_bytes) for k, v in data.items()}
    elif isinstance(data, list):
        # For large lists, hash them instead
        if len(data) > 100:
            import hashlib
            list_hash = hashlib.sha256(json.dumps(data, default=str).encode()).hexdigest()[:8]
            return f"[list of {len(data)} items, hash: {list_hash}]"
        return [truncate_and_mask(item, max_bytes) for item in data[:100]]
    return data


def calculate_cost(model: str, prompt_tokens: int, completion_tokens: int) -> Optional[float]:
    """Calculate cost in USD for AI model usage."""
    if model not in MODEL_PRICING:
        # Try to match partial model names
        for key in MODEL_PRICING:
            if key in model:
                model = key
                break
        else:
            return None
    
    pricing = MODEL_PRICING[model]
    input_cost = (prompt_tokens / 1_000_000) * pricing["input"]
    output_cost = (completion_tokens / 1_000_000) * pricing["output"]
    return round(input_cost + output_cost, 6)


def create_v2_event(
    session_tracer: SessionTracer,
    event_type: type,
    before_state: Dict[str, Any],
    after_state: Dict[str, Any],
    duration_ns: int,
    metadata: Optional[Dict[str, Any]] = None
) -> None:
    """Create a v2 event for compatibility."""
    if not session_tracer or not session_tracer.current_session:
        return
    
    # Get context values
    system_id = get_system_id() or "unknown"
    turn = get_turn_number()
    
    # Create event instance
    event = event_type()
    event.time_record = TimeRecord(
        event_time=datetime.now().isoformat(),
        message_time=turn if turn is not None else 0
    )
    event.system_instance_id = system_id
    event.system_state_before = before_state
    event.system_state_after = after_state
    
    # Add metadata
    if not hasattr(event, 'metadata'):
        event.metadata = {}
    if metadata:
        event.metadata.update(metadata)
    event.metadata['duration_ms'] = duration_ns / 1_000_000
    
    # Add to session
    try:
        session_tracer.current_session.add_event(event)
    except Exception as e:
        # Log but don't fail the traced function
        import logging
        logging.warning(f"Failed to add v2 event: {e}")


def add_otel_events(
    span: trace.Span,
    messages: Optional[List[Dict[str, str]]] = None,
    completion: Optional[str] = None
) -> None:
    """Add OTel events for prompts and completions per GenAI spec."""
    if messages:
        for msg in messages:
            role = msg.get("role", "unknown")
            content = truncate_and_mask(msg.get("content", ""))
            if role == "user":
                span.add_event("gen_ai.user.message", {"body": content})
            elif role == "assistant":
                span.add_event("gen_ai.assistant.message", {"body": content})
            elif role == "system":
                span.add_event("gen_ai.system.message", {"body": content})
    
    if completion:
        span.add_event("gen_ai.assistant.message", {"body": truncate_and_mask(completion)})


def capture_state(
    args: tuple,
    kwargs: dict,
    attrs_fn: Optional[Callable] = None,
    result: Any = None,
    error: Optional[Exception] = None,
    is_before: bool = True
) -> Dict[str, Any]:
    """Capture state for tracing, with optional custom attribute extraction."""
    state = {}
    
    if attrs_fn:
        # Use custom attribute extraction
        try:
            custom_attrs = attrs_fn(args, kwargs, result, error)
            if custom_attrs:
                # Filter attributes based on whether this is before or after state
                if is_before:
                    # Only include request/input attributes
                    state.update({k: v for k, v in custom_attrs.items() 
                                if not k.startswith('gen_ai.response') and not k.startswith('env.reward') 
                                and not k.startswith('env.done') and not k.startswith('runtime.success')
                                and not k.startswith('gen_ai.usage')})
                else:
                    # Only include response/output attributes
                    state.update({k: v for k, v in custom_attrs.items() 
                                if k.startswith('gen_ai.response') or k.startswith('env.reward') 
                                or k.startswith('env.done') or k.startswith('runtime.success')
                                or k.startswith('gen_ai.usage')
                                or k == 'gen_ai.error' or k == 'error'})
        except Exception as e:
            state["attrs_error"] = str(e)
            
        # If custom extraction didn't produce anything, fall back to default
        if not state:
            if is_before and (args or kwargs):
                state["input"] = {
                    "args": truncate_and_mask(list(args)),
                    "kwargs": truncate_and_mask(kwargs)
                }
            elif not is_before:
                if result is not None:
                    state["output"] = truncate_and_mask(result)
                if error:
                    state["error"] = str(error)
    else:
        # Default state capture
        if is_before and (args or kwargs):
            state["input"] = {
                "args": truncate_and_mask(list(args)),
                "kwargs": truncate_and_mask(kwargs)
            }
        elif not is_before:
            if result is not None:
                state["output"] = truncate_and_mask(result)
            if error:
                state["error"] = str(error)
    
    return state


class ContextPropagatingExecutor(ThreadPoolExecutor):
    """ThreadPoolExecutor that automatically propagates OTel context."""
    
    def submit(self, fn, *args, **kwargs):
        """Submit with context propagation."""
        # Capture current context
        ctx = context.get_current()
        session_tracer = get_active_session_tracer()
        system_id = get_system_id()
        turn = get_turn_number()
        
        def wrapped_fn(*args, **kwargs):
            # Restore context in executor thread
            token = context.attach(ctx)
            if session_tracer:
                set_active_session_tracer(session_tracer)
            if system_id:
                set_system_id(system_id)
            if turn is not None:
                set_turn_number(turn)
            try:
                return fn(*args, **kwargs)
            finally:
                context.detach(token)
        
        return super().submit(wrapped_fn, *args, **kwargs)


def trace_span(
    name: Union[str, Callable[..., str]],
    kind: SpanKind = SpanKind.INTERNAL,
    attrs_fn: Optional[Callable[[tuple, dict, Any, Optional[Exception]], Dict[str, Any]]] = None,
    event_type: Optional[type] = None,
    v2_only: bool = False,
    otel_only: bool = False,
) -> Callable[[F], F]:
    """
    Unified decorator for OTel spans + v2 events.
    
    Args:
        name: Span name or callable to generate it
        kind: OTel span kind  
        attrs_fn: Function to extract attributes
        event_type: V2 event class (CAISEvent, etc.)
        v2_only: Only emit v2 events (no OTel)
        otel_only: Only emit OTel spans (no v2)
    """
    def decorator(func: F) -> F:
        is_async = asyncio.iscoroutinefunction(func)
        
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            # Early exit if tracing disabled
            config = get_config()
            if not config.is_tracing_enabled():
                return await func(*args, **kwargs)
            
            # Get tracers based on mode
            session_tracer = get_active_session_tracer() if not otel_only else None
            use_otel = (config.mode in ["dual", "otel"]) and not v2_only
            
            # Generate span name
            span_name = name(*args, **kwargs) if callable(name) else name
            
            # Start timing
            start_ns = time.perf_counter_ns()
            
            # Capture before state
            before_state = capture_state(args, kwargs, attrs_fn, is_before=True)
            
            # Execute with OTel span if enabled
            if use_otel:
                with tracer.start_as_current_span(
                    span_name,
                    kind=kind,
                    record_exception=True,
                    set_status_on_exception=True
                ) as span:
                    result = None
                    error = None
                    try:
                        result = await func(*args, **kwargs)
                        span.set_status(Status(StatusCode.OK))
                    except Exception as e:
                        error = e
                        span.record_exception(e)
                        span.set_status(Status(StatusCode.ERROR, str(e)))
                        raise
                    finally:
                        # Capture after state
                        duration_ns = time.perf_counter_ns() - start_ns
                        after_state = capture_state(args, kwargs, attrs_fn, result, error, is_before=False)
                        
                        # Set OTel attributes (flattened)
                        span.set_attribute("duration_ms", duration_ns / 1_000_000)
                        
                        # Add all captured attributes
                        all_attrs = {**before_state, **after_state}
                        for key, value in all_attrs.items():
                            if isinstance(value, (str, int, float, bool)):
                                span.set_attribute(key, value)
                            elif isinstance(value, list) and all(isinstance(v, (str, int, float, bool)) for v in value):
                                span.set_attribute(key, value)
                        
                        # Add context info
                        if system_id := get_system_id():
                            span.set_attribute("system_id", system_id)
                        if turn := get_turn_number():
                            span.set_attribute("turn_number", turn)
                        
                        # Add OTel events for messages/completions
                        if 'gen_ai.request.messages' in all_attrs:
                            add_otel_events(span, messages=all_attrs['gen_ai.request.messages'])
                        if 'gen_ai.response.content' in all_attrs:
                            add_otel_events(span, completion=all_attrs['gen_ai.response.content'])
                        
                        # Add cost if available
                        if all(k in all_attrs for k in ['gen_ai.response.usage.prompt_tokens', 
                                                         'gen_ai.response.usage.completion_tokens',
                                                         'gen_ai.request.model']):
                            cost = calculate_cost(
                                all_attrs['gen_ai.request.model'],
                                all_attrs['gen_ai.response.usage.prompt_tokens'],
                                all_attrs['gen_ai.response.usage.completion_tokens']
                            )
                            if cost is not None:
                                span.set_attribute("gen_ai.usage.cost_usd", cost)
                        
                        # Create v2 event if needed
                        if event_type and session_tracer and config.mode in ["dual", "v2"]:
                            create_v2_event(
                                session_tracer,
                                event_type,
                                before_state,
                                after_state,
                                duration_ns
                            )
            else:
                # No OTel span, just execute
                result = None
                error = None
                try:
                    result = await func(*args, **kwargs)
                except Exception as e:
                    error = e
                    raise
                finally:
                    # Still create v2 event if needed
                    if event_type and session_tracer:
                        duration_ns = time.perf_counter_ns() - start_ns
                        after_state = capture_state(args, kwargs, attrs_fn, result, error, is_before=False)
                        create_v2_event(
                            session_tracer,
                            event_type,
                            before_state,
                            after_state,
                            duration_ns
                        )
            
            return result
        
        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            # Early exit if tracing disabled
            config = get_config()
            if not config.is_tracing_enabled():
                return func(*args, **kwargs)
            
            # Similar implementation for sync functions
            session_tracer = get_active_session_tracer() if not otel_only else None
            use_otel = (config.mode in ["dual", "otel"]) and not v2_only
            
            span_name = name(*args, **kwargs) if callable(name) else name
            start_ns = time.perf_counter_ns()
            before_state = capture_state(args, kwargs, attrs_fn, is_before=True)
            
            if use_otel:
                with tracer.start_as_current_span(
                    span_name,
                    kind=kind,
                    record_exception=True,
                    set_status_on_exception=True
                ) as span:
                    result = None
                    error = None
                    try:
                        result = func(*args, **kwargs)
                        span.set_status(Status(StatusCode.OK))
                    except Exception as e:
                        error = e
                        span.record_exception(e)
                        span.set_status(Status(StatusCode.ERROR, str(e)))
                        raise
                    finally:
                        duration_ns = time.perf_counter_ns() - start_ns
                        after_state = capture_state(args, kwargs, attrs_fn, result, error, is_before=False)
                        
                        # Set attributes and events (same as async)
                        span.set_attribute("duration_ms", duration_ns / 1_000_000)
                        
                        all_attrs = {**before_state, **after_state}
                        for key, value in all_attrs.items():
                            if isinstance(value, (str, int, float, bool)):
                                span.set_attribute(key, value)
                            elif isinstance(value, list) and all(isinstance(v, (str, int, float, bool)) for v in value):
                                span.set_attribute(key, value)
                        
                        if system_id := get_system_id():
                            span.set_attribute("system_id", system_id)
                        if turn := get_turn_number():
                            span.set_attribute("turn_number", turn)
                        
                        # Add OTel events
                        if 'gen_ai.request.messages' in all_attrs:
                            add_otel_events(span, messages=all_attrs['gen_ai.request.messages'])
                        if 'gen_ai.response.content' in all_attrs:
                            add_otel_events(span, completion=all_attrs['gen_ai.response.content'])
                        
                        # Add cost
                        if all(k in all_attrs for k in ['gen_ai.response.usage.prompt_tokens', 
                                                         'gen_ai.response.usage.completion_tokens',
                                                         'gen_ai.request.model']):
                            cost = calculate_cost(
                                all_attrs['gen_ai.request.model'],
                                all_attrs['gen_ai.response.usage.prompt_tokens'],
                                all_attrs['gen_ai.response.usage.completion_tokens']
                            )
                            if cost is not None:
                                span.set_attribute("gen_ai.usage.cost_usd", cost)
                        
                        if event_type and session_tracer and config.mode in ["dual", "v2"]:
                            create_v2_event(
                                session_tracer,
                                event_type,
                                before_state,
                                after_state,
                                duration_ns
                            )
            else:
                result = None
                error = None
                try:
                    result = func(*args, **kwargs)
                except Exception as e:
                    error = e
                    raise
                finally:
                    if event_type and session_tracer:
                        duration_ns = time.perf_counter_ns() - start_ns
                        after_state = capture_state(args, kwargs, attrs_fn, result, error, is_before=False)
                        create_v2_event(
                            session_tracer,
                            event_type,
                            before_state,
                            after_state,
                            duration_ns
                        )
            
            return result
        
        return async_wrapper if is_async else sync_wrapper  # type: ignore
    
    return decorator


# Attribute extraction functions following OTel GenAI conventions
def extract_ai_attributes(args, kwargs, result=None, error=None) -> Dict[str, Any]:
    """Extract attributes for AI/LLM calls following OTel GenAI conventions."""
    attrs = {}
    
    # From before state (args/kwargs)
    if 'messages' in kwargs:
        attrs['gen_ai.request.messages'] = kwargs['messages']
    # Check both positional and keyword args for model
    if 'model' in kwargs:
        attrs['gen_ai.request.model'] = kwargs['model']
    elif len(args) > 1 and isinstance(args[1], str):
        # Assuming first arg is self/messages, second might be model
        attrs['gen_ai.request.model'] = args[1]
    if 'temperature' in kwargs:
        attrs['gen_ai.request.temperature'] = kwargs['temperature']
    if 'max_tokens' in kwargs:
        attrs['gen_ai.request.max_tokens'] = kwargs['max_tokens']
    
    # From after state (result)
    if result:
        if hasattr(result, 'usage'):
            attrs['gen_ai.response.usage.prompt_tokens'] = result.usage.prompt_tokens
            attrs['gen_ai.response.usage.completion_tokens'] = result.usage.completion_tokens
            attrs['gen_ai.response.usage.total_tokens'] = result.usage.total_tokens
        if hasattr(result, 'model'):
            attrs['gen_ai.response.model'] = result.model
        if hasattr(result, 'choices') and result.choices:
            try:
                content = result.choices[0].message.content
                attrs['gen_ai.response.content'] = content
                attrs['gen_ai.response.finish_reason'] = result.choices[0].finish_reason
            except (IndexError, AttributeError):
                pass
    
    # Error handling
    if error:
        attrs['gen_ai.error'] = str(error)
    
    return attrs


def extract_env_attributes(args, kwargs, result=None, error=None) -> Dict[str, Any]:
    """Extract attributes for environment operations."""
    attrs = {}
    
    # From before state
    if 'actions' in kwargs:
        actions = kwargs['actions']
    elif len(args) > 0:
        # Check if this is a method (first arg is self) or a function
        if hasattr(args[0], '__class__') and not isinstance(args[0], (list, dict, str, int, float)):
            # Likely a method, actions is second arg
            actions = args[1] if len(args) > 1 else None
        else:
            # Likely a function, actions is first arg
            actions = args[0]
    else:
        actions = None
    
    # Handle different action formats
    if actions is not None:
        if isinstance(actions, (list, tuple)):
            # Convert list/tuple to JSON string for OTel compatibility
            attrs['env.actions'] = json.dumps(actions)
        else:
            attrs['env.actions'] = str(actions)
    
    # From after state
    if result and isinstance(result, dict):
        if 'reward' in result:
            attrs['env.reward'] = result['reward']
        if 'done' in result:
            attrs['env.done'] = result['done']
        if 'info' in result and isinstance(result['info'], dict):
            for key, value in result['info'].items():
                if isinstance(value, (str, int, float, bool)):
                    attrs[f'env.info.{key}'] = value
    
    return attrs


def extract_runtime_attributes(args, kwargs, result=None, error=None) -> Dict[str, Any]:
    """Extract attributes for runtime operations."""
    attrs = {}
    
    # Generic runtime attributes
    if 'operation' in kwargs:
        attrs['runtime.operation'] = kwargs['operation']
    if 'duration' in kwargs:
        attrs['runtime.duration'] = kwargs['duration']
    
    # From result
    if result and isinstance(result, dict):
        if 'success' in result:
            attrs['runtime.success'] = result['success']
        if 'error' in result:
            attrs['runtime.error'] = result['error']
    
    return attrs


# Convenience decorators
trace_ai_call = functools.partial(
    trace_span,
    kind=SpanKind.CLIENT,
    event_type=CAISEvent,
    attrs_fn=extract_ai_attributes
)

trace_env_step = functools.partial(
    trace_span,
    kind=SpanKind.INTERNAL,
    event_type=EnvironmentEvent,
    attrs_fn=extract_env_attributes
)

trace_runtime_op = functools.partial(
    trace_span,
    kind=SpanKind.INTERNAL,
    event_type=RuntimeEvent,
    attrs_fn=extract_runtime_attributes
)


# Flush handlers for graceful shutdown
def setup_flush_handlers():
    """Ensure traces are flushed on exit."""
    import atexit
    import signal
    
    def flush_all():
        try:
            # Flush OTel
            if provider := trace.get_tracer_provider():
                provider.force_flush()
            
            # Save v2 sessions
            session_tracer = get_active_session_tracer()
            if session_tracer and hasattr(session_tracer, 'save_all_sessions'):
                session_tracer.save_all_sessions()
        except Exception:
            pass  # Don't fail on shutdown
    
    atexit.register(flush_all)
    
    # Handle SIGTERM
    def handle_sigterm(*args):
        flush_all()
        exit(0)
    
    signal.signal(signal.SIGTERM, handle_sigterm)
    
    # Handle child processes (forked workers)
    if multiprocessing.get_start_method() == "fork":
        # In forked processes, re-register handlers
        def init_worker():
            setup_flush_handlers()
        
        # Store reference for multiprocessing pools
        global _worker_init
        _worker_init = init_worker


# Auto-setup flush handlers on import
setup_flush_handlers()