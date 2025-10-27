"""Async-aware decorators for tracing v3.

This module provides decorators and context management utilities for the tracing
system. The key innovation is the use of asyncio's ContextVar for propagating
tracing context across async boundaries without explicit parameter passing.

Context Variables:
-----------------
We use ContextVar instead of threading.local because:
1. ContextVars work correctly with asyncio tasks and coroutines
2. They automatically propagate across async boundaries
3. They're isolated between concurrent tasks
4. They work with sync code too (unlike pure async solutions)

Key Decorators:
--------------
- @with_session: Ensures a session is active before function execution
- @trace_llm_call: Automatically traces LLM API calls with metrics
- @trace_method: Generic method tracing for any class method

The decorators support both sync and async functions where appropriate,
though async is preferred for consistency with the rest of the system.
"""

from __future__ import annotations

import asyncio
import contextvars
import functools
import time
from collections.abc import Awaitable, Callable, Mapping
from contextvars import Token
from typing import Any, TypeVar, cast, overload

from .abstractions import LMCAISEvent, TimeRecord
from .utils import calculate_cost, detect_provider

# Context variables for session and turn tracking
# These variables automatically propagate across async call boundaries,
# allowing deeply nested code to access tracing context without explicit passing
_session_id_ctx: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "session_id"
)
_turn_number_ctx: contextvars.ContextVar[int | None] = contextvars.ContextVar(
    "turn_number"
)
_session_tracer_ctx: contextvars.ContextVar[Any | None] = contextvars.ContextVar(
    "session_tracer"
)


def set_session_id(session_id: str | None) -> None:
    """Set the current session ID in context.

    This ID will be available to all async tasks spawned from the current context.
    Setting to None clears the session context.

    Args:
        session_id: The session ID to set, or None to clear
    """
    _session_id_ctx.set(session_id)


def get_session_id() -> str | None:
    """Get the current session ID from context.

    Returns:
        The current session ID if one is set, None otherwise
    """
    return _session_id_ctx.get()


def set_turn_number(turn: int | None) -> None:
    """Set the current turn number in context."""
    _turn_number_ctx.set(turn)


def get_turn_number() -> int | None:
    """Get the current turn number from context."""
    return _turn_number_ctx.get()


def set_session_tracer(tracer: Any) -> None:
    """Set the current session tracer in context."""
    _session_tracer_ctx.set(tracer)


def get_session_tracer() -> Any:
    """Get the current session tracer from context."""
    return _session_tracer_ctx.get()


T = TypeVar("T")


@overload
def with_session(require: bool = True) -> Callable[[Callable[..., Awaitable[T]]], Callable[..., Awaitable[T]]]:
    ...


@overload
def with_session(require: bool = True) -> Callable[[Callable[..., T]], Callable[..., T]]:
    ...


def with_session(require: bool = True):
    """Decorator that ensures a session is active.

    This decorator checks if a session is active before allowing the decorated
    function to execute. It supports both sync and async functions.

    Args:
        require: If True, raises RuntimeError when no session is active.
                If False, allows execution without a session (useful for
                optional tracing).

    Example:
        ```python
        @with_session()
        async def process_message(content: str):
            # This will only run if a session is active
            tracer = get_session_tracer()
            await tracer.record_message(content, "user")
        ```
    """

    def decorator(fn: Callable[..., Awaitable[T]] | Callable[..., T]) -> Callable[..., Awaitable[T]] | Callable[..., T]:
        if asyncio.iscoroutinefunction(fn):

            @functools.wraps(fn)
            async def async_wrapper(*args: Any, **kwargs: Any) -> T:
                session_id = get_session_id()
                if require and session_id is None:
                    raise RuntimeError(
                        f"No active session for {getattr(fn, '__name__', 'unknown')}"
                    )
                async_fn = cast(Callable[..., Awaitable[T]], fn)
                return await async_fn(*args, **kwargs)

            return async_wrapper
        else:

            @functools.wraps(fn)
            def sync_wrapper(*args: Any, **kwargs: Any) -> T:
                session_id = get_session_id()
                if require and session_id is None:
                    raise RuntimeError(
                        f"No active session for {getattr(fn, '__name__', 'unknown')}"
                    )
                sync_fn = cast(Callable[..., T], fn)
                return sync_fn(*args, **kwargs)

            return sync_wrapper

    return decorator


def trace_llm_call(
    model_name: str | None = None,
    system_id: str = "llm",
    extract_tokens: bool = True,
    extract_cost: bool = True,
):
    """Decorator to trace LLM API calls.

    Automatically records LLM API calls as LMCAISEvent instances. Extracts token
    counts, calculates costs, and measures latency. Only works with async functions.

    Args:
        model_name: Model name to record (can be overridden by actual response)
        system_id: System identifier for the event (default: "llm")
        extract_tokens: Whether to extract token counts from response
        extract_cost: Whether to calculate USD cost from token counts

    Expected Response Format:
        The decorated function should return a dict with:
        - 'usage': dict with 'prompt_tokens', 'completion_tokens', 'total_tokens'
        - 'model': actual model name (optional, falls back to model_name param)

    Example:
        ```python
        @trace_llm_call(model_name="gpt-4")
        async def call_openai(prompt: str) -> dict:
            response = await openai_client.completions.create(
                model="gpt-4",
                messages=[{"role": "user", "content": prompt}]
            )
            return response.model_dump()
        ```
    """

    def decorator(fn: Callable[..., Awaitable[T]]) -> Callable[..., Awaitable[T]]:
        if asyncio.iscoroutinefunction(fn):
            async_fn: Callable[..., Awaitable[T]] = fn

            @functools.wraps(fn)
            async def async_wrapper(*args: Any, **kwargs: Any) -> T:
                tracer = get_session_tracer()
                if not tracer:
                    return await async_fn(*args, **kwargs)

                start_time = time.time()
                system_state_before = kwargs.get("state_before", {})

                try:
                    result = await async_fn(*args, **kwargs)

                    # Extract metrics from result - this assumes the result follows
                    # common LLM API response formats (OpenAI, Anthropic, etc.)
                    input_tokens = output_tokens = total_tokens = None
                    actual_model = model_name
                    if extract_tokens and isinstance(result, Mapping):
                        result_mapping = cast(Mapping[str, Any], result)
                        usage = result_mapping.get("usage")
                        if isinstance(usage, Mapping):
                            input_tokens = usage.get("prompt_tokens")
                            output_tokens = usage.get("completion_tokens")
                            total_tokens = usage.get("total_tokens")
                        value = result_mapping.get("model")
                        if isinstance(value, str):
                            actual_model = value

                    latency_ms = int((time.time() - start_time) * 1000)

                    # Create event with all extracted metrics
                    event = LMCAISEvent(
                        system_instance_id=system_id,
                        time_record=TimeRecord(event_time=time.time()),
                        model_name=actual_model or "unknown",
                        provider=detect_provider(actual_model),
                        input_tokens=input_tokens,
                        output_tokens=output_tokens,
                        total_tokens=total_tokens,
                        cost_usd=calculate_cost(
                            actual_model or "unknown", input_tokens or 0, output_tokens or 0
                        )
                        if extract_cost
                        else None,
                        latency_ms=latency_ms,
                        system_state_before=system_state_before,
                        system_state_after=kwargs.get("state_after", {}),
                        metadata={
                            "function": getattr(fn, "__name__", "unknown"),
                            "step_id": kwargs.get("step_id"),
                        },
                    )

                    await tracer.record_event(event)
                    return result

                except Exception as e:
                    # Record error event - we still want to track failed LLM calls
                    # for debugging and cost analysis
                    if tracer:
                        event = LMCAISEvent(
                            system_instance_id=system_id,
                            time_record=TimeRecord(event_time=time.time()),
                            model_name=model_name or "unknown",
                            provider=detect_provider(model_name),
                            latency_ms=int((time.time() - start_time) * 1000),
                            metadata={
                                "function": getattr(fn, "__name__", "unknown"),
                                "error": str(e),
                                "error_type": type(e).__name__,
                            },
                        )
                        await tracer.record_event(event)
                    raise

            return async_wrapper
        else:
            raise ValueError("trace_llm_call only supports async functions")

    return decorator


def trace_method(event_type: str = "runtime", system_id: str | None = None):
    """Generic method tracing decorator.

    Traces any method call by recording it as a RuntimeEvent. Supports both
    sync and async methods, though async is preferred.

    Args:
        event_type: Type of event to create (default: "runtime")
        system_id: System identifier (defaults to class name)

    Example:
        ```python
        class Agent:
            @trace_method(system_id="my_agent")
            async def take_action(self, observation):
                # Method execution is automatically traced
                return self.policy(observation)
        ```
    """

    def decorator(
        fn: Callable[..., Awaitable[T]] | Callable[..., T]
    ) -> Callable[..., Awaitable[T]] | Callable[..., T]:
        if asyncio.iscoroutinefunction(fn):
            async_fn = cast(Callable[..., Awaitable[T]], fn)

            @functools.wraps(fn)
            async def async_wrapper(*args: Any, **kwargs: Any) -> T:
                tracer = get_session_tracer()
                if not tracer:
                    return await async_fn(*args, **kwargs)

                from .abstractions import RuntimeEvent

                # Use class name as system_id if not provided
                self_obj = args[0] if args else None
                inferred_system_id = (
                    self_obj.__class__.__name__ if self_obj is not None else "unknown"
                )
                actual_system_id = system_id or inferred_system_id

                event = RuntimeEvent(
                    system_instance_id=actual_system_id,
                    time_record=TimeRecord(event_time=time.time()),
                    actions=[],  # Can be overridden in metadata
                    metadata={
                        "method": getattr(fn, "__name__", "unknown"),
                        "args": str(args)[:100],  # Truncate for safety
                        "step_id": kwargs.get("step_id"),
                    },
                )

                await tracer.record_event(event)
                return await async_fn(*args, **kwargs)

            return async_wrapper
        else:

            @functools.wraps(fn)
            def sync_wrapper(*args: Any, **kwargs: Any) -> T:
                # For sync methods, we can't easily trace without blocking
                # the event loop. This is a limitation of the async-first design.
                # Consider converting to async or using a different approach
                sync_fn = cast(Callable[..., T], fn)
                return sync_fn(*args, **kwargs)

            return sync_wrapper

    return decorator


class SessionContext:
    """Context manager for session tracking.

    Provides a way to temporarily set session context, useful for testing
    or when you need to manually manage context outside of SessionTracer.

    This context manager properly handles both sync and async contexts,
    and ensures the previous context is restored on exit.

    Example:
        ```python
        # Sync usage
        with SessionContext("test_session_123", tracer):
            # Code here sees the test session
            process_data()

        # Async usage
        async with SessionContext("test_session_123", tracer):
            # Async code here sees the test session
            await process_data_async()
        ```
    """

    def __init__(self, session_id: str, tracer: Any | None = None):
        self.session_id = session_id
        self.tracer = tracer
        self._token: Token[str | None] | None = None
        self._tracer_token: Token[Any] | None = None

    def __enter__(self):
        # Store tokens to restore previous context on exit
        self._token = _session_id_ctx.set(self.session_id)
        if self.tracer:
            self._tracer_token = _session_tracer_ctx.set(self.tracer)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # Restore previous context - this is crucial for proper isolation
        if self._token is not None:
            _session_id_ctx.reset(self._token)
        if self._tracer_token is not None:
            _session_tracer_ctx.reset(self._tracer_token)

    async def __aenter__(self):
        self._token = _session_id_ctx.set(self.session_id)
        if self.tracer:
            self._tracer_token = _session_tracer_ctx.set(self.tracer)
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self._token is not None:
            _session_id_ctx.reset(self._token)
        if self._tracer_token is not None:
            _session_tracer_ctx.reset(self._tracer_token)
