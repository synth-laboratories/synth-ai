"""Async-aware decorators for tracing v3."""
import contextvars
import functools
import time
from typing import Callable, Any, Optional, TypeVar, Union
import asyncio
import inspect

from .abstractions import LMCAISEvent, TimeRecord
from .utils import detect_provider, calculate_cost


# Context variables for session and turn tracking
_session_id_ctx: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar("session_id", default=None)
_turn_number_ctx: contextvars.ContextVar[Optional[int]] = contextvars.ContextVar("turn_number", default=None)
_session_tracer_ctx: contextvars.ContextVar[Optional[Any]] = contextvars.ContextVar("session_tracer", default=None)


def set_session_id(session_id: str):
    """Set the current session ID in context."""
    _session_id_ctx.set(session_id)


def get_session_id() -> Optional[str]:
    """Get the current session ID from context."""
    return _session_id_ctx.get()


def set_turn_number(turn: int):
    """Set the current turn number in context."""
    _turn_number_ctx.set(turn)


def get_turn_number() -> Optional[int]:
    """Get the current turn number from context."""
    return _turn_number_ctx.get()


def set_session_tracer(tracer):
    """Set the current session tracer in context."""
    _session_tracer_ctx.set(tracer)


def get_session_tracer():
    """Get the current session tracer from context."""
    return _session_tracer_ctx.get()


T = TypeVar('T')


def with_session(require: bool = True):
    """Decorator that ensures a session is active."""
    def decorator(fn: Callable[..., T]) -> Callable[..., T]:
        if asyncio.iscoroutinefunction(fn):
            @functools.wraps(fn)
            async def async_wrapper(*args, **kwargs):
                session_id = get_session_id()
                if require and session_id is None:
                    raise RuntimeError(f"No active session for {fn.__name__}")
                return await fn(*args, **kwargs)
            return async_wrapper
        else:
            @functools.wraps(fn)
            def sync_wrapper(*args, **kwargs):
                session_id = get_session_id()
                if require and session_id is None:
                    raise RuntimeError(f"No active session for {fn.__name__}")
                return fn(*args, **kwargs)
            return sync_wrapper
    return decorator


def trace_llm_call(
    model_name: str = None,
    system_id: str = "llm",
    extract_tokens: bool = True,
    extract_cost: bool = True,
):
    """Decorator to trace LLM API calls."""
    def decorator(fn: Callable[..., T]) -> Callable[..., T]:
        if asyncio.iscoroutinefunction(fn):
            @functools.wraps(fn)
            async def async_wrapper(*args, **kwargs):
                tracer = get_session_tracer()
                if not tracer:
                    return await fn(*args, **kwargs)
                
                start_time = time.time()
                system_state_before = kwargs.get("state_before", {})
                
                try:
                    result = await fn(*args, **kwargs)
                    
                    # Extract metrics from result
                    if extract_tokens and isinstance(result, dict):
                        input_tokens = result.get("usage", {}).get("prompt_tokens")
                        output_tokens = result.get("usage", {}).get("completion_tokens")
                        total_tokens = result.get("usage", {}).get("total_tokens")
                        actual_model = result.get("model", model_name)
                    else:
                        input_tokens = output_tokens = total_tokens = None
                        actual_model = model_name
                    
                    latency_ms = int((time.time() - start_time) * 1000)
                    
                    # Create event
                    event = LMCAISEvent(
                        system_instance_id=system_id,
                        time_record=TimeRecord(event_time=time.time()),
                        model_name=actual_model or "unknown",
                        provider=detect_provider(actual_model),
                        input_tokens=input_tokens,
                        output_tokens=output_tokens,
                        total_tokens=total_tokens,
                        cost_usd=calculate_cost(actual_model, input_tokens or 0, output_tokens or 0) if extract_cost else None,
                        latency_ms=latency_ms,
                        system_state_before=system_state_before,
                        system_state_after=kwargs.get("state_after", {}),
                        metadata={
                            "function": fn.__name__,
                            "step_id": kwargs.get("step_id"),
                        }
                    )
                    
                    await tracer.record_event(event)
                    return result
                    
                except Exception as e:
                    # Record error event
                    if tracer:
                        event = LMCAISEvent(
                            system_instance_id=system_id,
                            time_record=TimeRecord(event_time=time.time()),
                            model_name=model_name or "unknown",
                            provider=detect_provider(model_name),
                            latency_ms=int((time.time() - start_time) * 1000),
                            metadata={
                                "function": fn.__name__,
                                "error": str(e),
                                "error_type": type(e).__name__,
                            }
                        )
                        await tracer.record_event(event)
                    raise
                    
            return async_wrapper
        else:
            raise ValueError("trace_llm_call only supports async functions")
    return decorator


def trace_method(event_type: str = "runtime", system_id: str = None):
    """Generic method tracing decorator."""
    def decorator(fn: Callable[..., T]) -> Callable[..., T]:
        if asyncio.iscoroutinefunction(fn):
            @functools.wraps(fn)
            async def async_wrapper(self, *args, **kwargs):
                tracer = get_session_tracer()
                if not tracer:
                    return await fn(self, *args, **kwargs)
                    
                from .abstractions import BaseEvent, RuntimeEvent
                
                # Use class name as system_id if not provided
                actual_system_id = system_id or self.__class__.__name__
                
                event = RuntimeEvent(
                    system_instance_id=actual_system_id,
                    time_record=TimeRecord(event_time=time.time()),
                    actions=[],  # Can be overridden in metadata
                    metadata={
                        "method": fn.__name__,
                        "args": str(args)[:100],  # Truncate for safety
                        "step_id": kwargs.get("step_id"),
                    }
                )
                
                await tracer.record_event(event)
                return await fn(self, *args, **kwargs)
                
            return async_wrapper
        else:
            @functools.wraps(fn)
            def sync_wrapper(self, *args, **kwargs):
                # For sync methods, we can't easily trace without blocking
                # Consider converting to async or using a different approach
                return fn(self, *args, **kwargs)
            return sync_wrapper
    return decorator


class SessionContext:
    """Context manager for session tracking."""
    
    def __init__(self, session_id: str, tracer=None):
        self.session_id = session_id
        self.tracer = tracer
        self._token = None
        self._tracer_token = None
        
    def __enter__(self):
        self._token = _session_id_ctx.set(self.session_id)
        if self.tracer:
            self._tracer_token = _session_tracer_ctx.set(self.tracer)
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        _session_id_ctx.reset(self._token)
        if self._tracer_token:
            _session_tracer_ctx.reset(self._tracer_token)
            
    async def __aenter__(self):
        self._token = _session_id_ctx.set(self.session_id)
        if self.tracer:
            self._tracer_token = _session_tracer_ctx.set(self.tracer)
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        _session_id_ctx.reset(self._token)
        if self._tracer_token:
            _session_tracer_ctx.reset(self._tracer_token)