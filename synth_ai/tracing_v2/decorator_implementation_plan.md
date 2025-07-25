# Tracing v2 Decorator Implementation Plan

## Executive Summary

This document outlines a comprehensive plan to implement a decorator-based tracing system for Synth-AI v2, running in parallel with existing manual tracing to ensure zero data loss and complete backwards compatibility.

## Table of Contents

1. [Objectives](#objectives)
2. [Architecture](#architecture)
3. [Implementation Phases](#implementation-phases)
4. [Detailed Design](#detailed-design)
5. [Validation Strategy](#validation-strategy)
6. [Migration Path](#migration-path)
7. [Risk Mitigation](#risk-mitigation)
8. [Timeline](#timeline)

## Objectives

### Primary Goals
1. **Zero Data Loss**: Ensure decorator-based tracing captures all data currently captured by manual tracing
2. **Clean API**: Reduce boilerplate code and improve developer experience
3. **Backwards Compatibility**: Existing code continues to work unchanged
4. **Validation**: Mathematically prove equivalence between manual and decorator approaches
5. **Performance**: Maintain or improve current performance characteristics

### Secondary Goals
1. **Extensibility**: Easy to add new event types and metadata
2. **Type Safety**: Full type hints and runtime validation
3. **Debugging**: Better error messages and trace inspection tools
4. **Documentation**: Comprehensive guides and examples

## Architecture

### Core Components

```python
# synth_ai/tracing_v2/decorators.py

import asyncio
import functools
import contextvars
import inspect
from typing import Any, Callable, Optional, Dict, List, Union, TypeVar, Type
from datetime import datetime
import time

from synth_ai.tracing_v2.session_tracer import (
    SessionTracer, TimeRecord, SessionEventMessage,
    CAISEvent, RuntimeEvent, EnvironmentEvent, SessionEvent
)
from synth_ai.tracing_v2.abstractions import SessionMessage

# Type variables for better type hints
T = TypeVar('T')
EventType = TypeVar('EventType', bound=SessionEvent)

# Context variables for thread-safe tracer access
_tracer_ctx: contextvars.ContextVar[Optional[SessionTracer]] = contextvars.ContextVar(
    "session_tracer", default=None
)
_turn_ctx: contextvars.ContextVar[Optional[int]] = contextvars.ContextVar(
    "turn_number", default=None
)

class TracerContext:
    """Context manager for setting active tracer"""
    
    def __init__(self, tracer: SessionTracer):
        self.tracer = tracer
        self.token = None
        
    def __enter__(self):
        self.token = _tracer_ctx.set(self.tracer)
        return self.tracer
        
    def __exit__(self, *args):
        _tracer_ctx.reset(self.token)

def set_active_tracer(tracer: SessionTracer) -> None:
    """Set the active tracer for the current context"""
    _tracer_ctx.set(tracer)

def get_active_tracer() -> Optional[SessionTracer]:
    """Get the active tracer, returns None if not set"""
    return _tracer_ctx.get()

def set_turn_number(turn: int) -> None:
    """Set the current turn number"""
    _turn_ctx.set(turn)

def get_turn_number() -> Optional[int]:
    """Get the current turn number"""
    return _turn_ctx.get()

# Decorator configuration classes
class DecoratorConfig:
    """Configuration for tracing decorators"""
    
    def __init__(
        self,
        event_class: Type[EventType],
        system_id: str,
        capture_args: bool = True,
        capture_result: bool = True,
        capture_self: bool = False,
        arg_filter: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None,
        result_filter: Optional[Callable[[Any], Any]] = None,
        metadata_fn: Optional[Callable[..., Dict[str, Any]]] = None,
        message_fn: Optional[Callable[..., Optional[SessionMessage]]] = None,
        auto_timestep: bool = False,
        timestep_name: Optional[str] = None,
        include_timing: bool = True,
        include_memory: bool = False,
        error_handler: Optional[Callable[[Exception], None]] = None,
    ):
        self.event_class = event_class
        self.system_id = system_id
        self.capture_args = capture_args
        self.capture_result = capture_result
        self.capture_self = capture_self
        self.arg_filter = arg_filter
        self.result_filter = result_filter
        self.metadata_fn = metadata_fn
        self.message_fn = message_fn
        self.auto_timestep = auto_timestep
        self.timestep_name = timestep_name
        self.include_timing = include_timing
        self.include_memory = include_memory
        self.error_handler = error_handler

# Main decorator factory
def trace_event(config: DecoratorConfig) -> Callable[[T], T]:
    """
    Main decorator factory for tracing events.
    
    Args:
        config: DecoratorConfig instance with all settings
        
    Returns:
        Decorator function
    """
    def decorator(fn: T) -> T:
        # Determine if function is async
        is_async = asyncio.iscoroutinefunction(fn)
        is_method = inspect.ismethod(fn) or (
            len(inspect.signature(fn).parameters) > 0 and
            list(inspect.signature(fn).parameters.keys())[0] == 'self'
        )
        
        @functools.wraps(fn)
        async def async_wrapper(*args, **kwargs):
            tracer = get_active_tracer()
            if not tracer:
                # No active tracer, execute normally
                return await fn(*args, **kwargs)
            
            # Start timing
            start_time = time.time()
            
            # Auto-create timestep if needed
            timestep_context = None
            if config.auto_timestep and not tracer.in_timestep:
                timestep_name = config.timestep_name or f"{config.system_id}:{fn.__name__}"
                turn = get_turn_number()
                timestep_context = tracer.timestep(turn_number=turn or 0)
                await timestep_context.__aenter__()
            
            try:
                # Capture before state
                before_state = _capture_before_state(
                    args, kwargs, is_method, config
                )
                
                # Execute function
                result = await fn(*args, **kwargs)
                error = None
                
            except Exception as e:
                result = None
                error = e
                if config.error_handler:
                    config.error_handler(e)
                raise
                
            finally:
                # Capture after state
                after_state = _capture_after_state(
                    result, error, config
                )
                
                # Calculate timing
                end_time = time.time()
                duration = end_time - start_time
                
                # Generate metadata
                metadata = _generate_metadata(
                    args, kwargs, result, error, duration, config
                )
                
                # Create event
                event = _create_event(
                    before_state, after_state, metadata, config
                )
                
                # Record event
                if hasattr(tracer, 'current_session') and tracer.current_session:
                    tracer.current_session.add_event(event)
                
                # Generate and record message if needed
                if config.message_fn:
                    message = config.message_fn(args, kwargs, result, error)
                    if message:
                        await tracer.add_message(message)
                
                # Clean up timestep if we created it
                if timestep_context:
                    await timestep_context.__aexit__(None, None, None)
            
            return result
        
        @functools.wraps(fn)
        def sync_wrapper(*args, **kwargs):
            tracer = get_active_tracer()
            if not tracer:
                return fn(*args, **kwargs)
            
            # Similar implementation for sync functions
            # (Convert to async and run in event loop if needed)
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # We're in an async context, use sync implementation
                return _sync_trace_implementation(
                    fn, args, kwargs, is_method, config
                )
            else:
                # No event loop, create one
                return loop.run_until_complete(
                    _async_trace_implementation(
                        fn, args, kwargs, is_method, config
                    )
                )
        
        return async_wrapper if is_async else sync_wrapper
    
    return decorator

# Helper functions
def _capture_before_state(
    args: tuple,
    kwargs: dict,
    is_method: bool,
    config: DecoratorConfig
) -> Dict[str, Any]:
    """Capture state before function execution"""
    if not config.capture_args:
        return {}
    
    state = {}
    
    # Handle self separately if it's a method
    if is_method and args:
        if config.capture_self:
            state['self'] = args[0]
        args = args[1:]  # Remove self from args
    
    # Apply filters
    if config.arg_filter:
        filtered = config.arg_filter({
            'args': args,
            'kwargs': kwargs
        })
        state.update(filtered)
    else:
        if args:
            state['args'] = list(args)
        if kwargs:
            state['kwargs'] = dict(kwargs)
    
    return state

def _capture_after_state(
    result: Any,
    error: Optional[Exception],
    config: DecoratorConfig
) -> Dict[str, Any]:
    """Capture state after function execution"""
    state = {}
    
    if error:
        state['error'] = {
            'type': type(error).__name__,
            'message': str(error),
            'traceback': _get_traceback(error)
        }
    elif config.capture_result and result is not None:
        if config.result_filter:
            state['result'] = config.result_filter(result)
        else:
            state['result'] = result
    
    return state

def _generate_metadata(
    args: tuple,
    kwargs: dict,
    result: Any,
    error: Optional[Exception],
    duration: float,
    config: DecoratorConfig
) -> Dict[str, Any]:
    """Generate metadata for the event"""
    metadata = {}
    
    # Add timing information
    if config.include_timing:
        metadata['duration_seconds'] = duration
    
    # Add memory information if requested
    if config.include_memory:
        metadata['memory_usage'] = _get_memory_usage()
    
    # Add custom metadata
    if config.metadata_fn:
        custom_metadata = config.metadata_fn(args, kwargs, result, error)
        if custom_metadata:
            metadata.update(custom_metadata)
    
    # Add turn number if available
    turn = get_turn_number()
    if turn is not None:
        metadata['turn_number'] = turn
    
    return metadata

def _create_event(
    before_state: Dict[str, Any],
    after_state: Dict[str, Any],
    metadata: Dict[str, Any],
    config: DecoratorConfig
) -> SessionEvent:
    """Create the appropriate event instance"""
    return config.event_class(
        system_instance_id=config.system_id,
        system_state_before=before_state,
        system_state_after=after_state,
        metadata=metadata,
        time_record=TimeRecord(
            event_time=time.time(),
            message_time=None
        )
    )

# Convenience decorator factories
def trace_agent(
    system_id: str = "agent",
    **kwargs
) -> Callable[[T], T]:
    """Decorator for tracing agent functions"""
    config = DecoratorConfig(
        event_class=CAISEvent,
        system_id=system_id,
        **kwargs
    )
    return trace_event(config)

def trace_env(
    system_id: str = "environment",
    **kwargs
) -> Callable[[T], T]:
    """Decorator for tracing environment functions"""
    config = DecoratorConfig(
        event_class=EnvironmentEvent,
        system_id=system_id,
        **kwargs
    )
    return trace_event(config)

def trace_runtime(
    system_id: str = "runtime",
    **kwargs
) -> Callable[[T], T]:
    """Decorator for tracing runtime functions"""
    config = DecoratorConfig(
        event_class=RuntimeEvent,
        system_id=system_id,
        **kwargs
    )
    return trace_event(config)

# Specialized decorators for common patterns
def trace_llm_call(
    system_id: str = "agent",
    extract_messages: Callable[..., List[Dict[str, str]]] = None,
    extract_response: Callable[..., str] = None,
    **kwargs
) -> Callable[[T], T]:
    """Specialized decorator for LLM calls"""
    def metadata_fn(args, kwargs, result, error):
        metadata = {}
        if extract_messages:
            metadata['messages'] = extract_messages(args, kwargs)
        if extract_response and result:
            metadata['response'] = extract_response(result)
        return metadata
    
    return trace_agent(
        system_id=system_id,
        metadata_fn=metadata_fn,
        **kwargs
    )

def trace_tool_call(
    system_id: str = "agent",
    extract_tool_name: Callable[..., str] = None,
    extract_tool_args: Callable[..., Dict] = None,
    **kwargs
) -> Callable[[T], T]:
    """Specialized decorator for tool calls"""
    def metadata_fn(args, kwargs, result, error):
        metadata = {}
        if extract_tool_name:
            metadata['tool_name'] = extract_tool_name(args, kwargs)
        if extract_tool_args:
            metadata['tool_args'] = extract_tool_args(args, kwargs)
        return metadata
    
    def message_fn(args, kwargs, result, error):
        if error or not result:
            return None
        
        tool_name = extract_tool_name(args, kwargs) if extract_tool_name else "unknown"
        tool_args = extract_tool_args(args, kwargs) if extract_tool_args else {}
        
        return SessionEventMessage(
            source_id=system_id,
            target_id="environment",
            message_type="tool_call",
            content={
                "tool": tool_name,
                "args": tool_args,
                "result": result
            }
        )
    
    return trace_agent(
        system_id=system_id,
        metadata_fn=metadata_fn,
        message_fn=message_fn,
        **kwargs
    )

# Utility functions
def _get_traceback(error: Exception) -> str:
    """Get formatted traceback from exception"""
    import traceback
    return ''.join(traceback.format_exception(
        type(error), error, error.__traceback__
    ))

def _get_memory_usage() -> Dict[str, float]:
    """Get current memory usage"""
    import psutil
    import os
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    return {
        'rss_mb': memory_info.rss / 1024 / 1024,
        'vms_mb': memory_info.vms / 1024 / 1024,
        'percent': process.memory_percent()
    }

# Testing utilities
class TraceRecorder:
    """Utility class for recording traces during testing"""
    
    def __init__(self):
        self.events: List[SessionEvent] = []
        self.messages: List[SessionMessage] = []
        
    def record_event(self, event: SessionEvent):
        self.events.append(event)
        
    def record_message(self, message: SessionMessage):
        self.messages.append(message)
        
    def clear(self):
        self.events.clear()
        self.messages.clear()
        
    def get_events_by_type(self, event_type: Type[EventType]) -> List[EventType]:
        return [e for e in self.events if isinstance(e, event_type)]
```

## Implementation Phases

### Phase 1: Core Infrastructure (Week 1-2)

1. **Implement Core Decorators**
   - [ ] Create `decorators.py` in tracing_v2
   - [ ] Implement `trace_event` factory
   - [ ] Implement convenience decorators (`trace_agent`, `trace_env`, `trace_runtime`)
   - [ ] Add context variable management
   - [ ] Write unit tests

2. **Integration Points**
   - [ ] Update SessionTracer to work with decorators
   - [ ] Ensure thread safety with context variables
   - [ ] Test async/sync compatibility

### Phase 2: Parallel Implementation (Week 3-4)

1. **Agent Integration**
   ```python
   # crafter_react_agent.py
   class ReActAgent:
       @trace_agent(
           metadata_fn=lambda self, *a, **k: {
               'agent_type': 'react',
               'system_name': self.system_name
           }
       )
       @trace_event_async(event_type="react_agent_decide")  # Keep existing
       async def decide(self, obs_str: str, current_raw_obs: Dict[str, Any]) -> List[int]:
           # Existing implementation unchanged
   ```

2. **Environment Integration**
   ```python
   # environment.py
   class CrafterClassicEnvironment:
       @trace_env(
           arg_filter=lambda args: {
               'tool_calls': _safe_serialize_tool_calls(args['args'][1])
           },
           auto_timestep=False  # Managed by test script
       )
       async def step(self, tool_calls):
           # Existing implementation unchanged
   ```

3. **Runtime Integration**
   ```python
   # In test script
   with TracerContext(session_tracer):
       set_turn_number(turn)
       # All decorated functions will now trace automatically
   ```

### Phase 3: Validation Framework (Week 5-6)

1. **Create Validation Tools**
   ```python
   # validation/trace_validator.py
   class TraceValidator:
       def __init__(self):
           self.manual_events = []
           self.decorator_events = []
           
       def validate_equivalence(self) -> ValidationReport:
           """Compare manual and decorator traces"""
           return ValidationReport(
               event_count_match=self._check_event_counts(),
               state_capture_match=self._check_state_capture(),
               timing_match=self._check_timing(),
               metadata_match=self._check_metadata(),
               message_match=self._check_messages()
           )
   ```

2. **Test Harness**
   ```python
   # validation/test_harness.py
   async def run_validation_test():
       # Run with manual only
       manual_trace = await run_episode(decorators_enabled=False)
       
       # Run with both
       dual_trace = await run_episode(decorators_enabled=True)
       
       # Validate
       report = validator.validate_equivalence()
       assert report.is_valid(), report.get_differences()
   ```

### Phase 4: Gap Analysis and Refinement (Week 7-8)

1. **Identify Gaps**
   - Complex state transformations
   - High-frequency events
   - Large data structures
   - Edge cases

2. **Add Specialized Decorators**
   ```python
   @trace_batch_operation(
       batch_size_fn=lambda args: len(args[0]),
       summarize=True
   )
   async def process_batch(items: List[Any]):
       pass
   ```

### Phase 5: Performance Optimization (Week 9-10)

1. **Benchmarking**
   ```python
   # benchmarks/decorator_performance.py
   async def benchmark_decorators():
       # Measure overhead
       # Compare with manual tracing
       # Identify bottlenecks
   ```

2. **Optimizations**
   - Lazy state capture
   - Conditional tracing
   - Batched event recording
   - Memory pooling

### Phase 6: Documentation and Migration Guide (Week 11-12)

1. **Documentation**
   - API reference
   - Migration guide
   - Best practices
   - Troubleshooting

2. **Migration Tools**
   - Automated code transformation scripts
   - Compatibility checkers
   - Rollback procedures

## Detailed Design

### State Capture Strategies

1. **Selective Capture**
   ```python
   @trace_agent(
       arg_filter=lambda args: {
           'action': args['args'][1].action if len(args['args']) > 1 else None,
           'reasoning': args['kwargs'].get('reasoning')
       }
   )
   ```

2. **Deep vs Shallow Copy**
   ```python
   @trace_env(
       capture_args=True,
       arg_filter=lambda args: {
           'state_summary': _summarize_state(args['args'][0])
       }
   )
   ```

3. **Lazy Evaluation**
   ```python
   @trace_runtime(
       metadata_fn=lambda *a, **k: {
           'large_data': LazySerializer(k.get('data'))
       }
   )
   ```

### Message Generation Patterns

1. **Conditional Messages**
   ```python
   @trace_agent(
       message_fn=lambda a, k, r, e: SessionEventMessage(
           source_id="agent",
           target_id="environment",
           content=r
       ) if r and not e else None
   )
   ```

2. **Message Transformation**
   ```python
   @trace_tool_call(
       extract_tool_name=lambda a, k: a[1].tool,
       extract_tool_args=lambda a, k: a[1].args
   )
   ```

### Error Handling Patterns

1. **Graceful Degradation**
   ```python
   @trace_agent(
       error_handler=lambda e: logger.warning(f"Trace error: {e}"),
       capture_args=False  # Don't capture on error
   )
   ```

2. **Error Context**
   ```python
   @trace_env(
       metadata_fn=lambda a, k, r, e: {
           'error_context': _get_error_context() if e else None
       }
   )
   ```

## Validation Strategy

### Validation Levels

1. **Level 1: Event Count**
   - Same number of events
   - Same event types
   - Same temporal ordering

2. **Level 2: State Equivalence**
   - Before/after states match
   - Metadata present
   - No data loss

3. **Level 3: Semantic Equivalence**
   - Business logic unchanged
   - Same decisions made
   - Same outcomes

### Validation Tools

```python
# validation/assertions.py
def assert_trace_equivalence(
    manual: SessionTrace,
    decorated: SessionTrace,
    strict: bool = True
):
    """Assert two traces are equivalent"""
    # Event counts
    assert len(manual.events) == len(decorated.events)
    
    # Event types
    manual_types = [type(e).__name__ for e in manual.events]
    decorated_types = [type(e).__name__ for e in decorated.events]
    assert manual_types == decorated_types
    
    # State capture
    for m, d in zip(manual.events, decorated.events):
        if strict:
            assert m.system_state_before == d.system_state_before
            assert m.system_state_after == d.system_state_after
        else:
            assert_equivalent_state(m.system_state_before, d.system_state_before)
            assert_equivalent_state(m.system_state_after, d.system_state_after)
```

### Continuous Validation

```python
# ci/validate_traces.py
@pytest.mark.parametrize("episode_config", test_episodes)
async def test_decorator_parity(episode_config):
    """Run on every commit"""
    async with TraceValidation() as validator:
        await validator.run_episode(episode_config)
        validator.assert_parity()
```

## Migration Path

### Stage 1: Opt-in (Months 1-2)
- Decorators available but not required
- Documentation and examples
- Early adopters provide feedback

### Stage 2: Recommended (Months 3-4)
- New code uses decorators
- Migration guide published
- Tooling support added

### Stage 3: Default (Months 5-6)
- Decorators become the default
- Manual tracing still supported
- Deprecation warnings added

### Stage 4: Legacy (Months 7+)
- Manual tracing moved to legacy module
- Full decorator adoption
- Simplified codebase

## Risk Mitigation

### Technical Risks

1. **Performance Regression**
   - Mitigation: Comprehensive benchmarking
   - Fallback: Conditional decorator enabling

2. **Data Loss**
   - Mitigation: Parallel running with validation
   - Fallback: Manual tracing remains available

3. **Breaking Changes**
   - Mitigation: Careful API design
   - Fallback: Compatibility layer

### Process Risks

1. **Adoption Resistance**
   - Mitigation: Clear benefits demonstration
   - Fallback: Gradual migration

2. **Complexity Increase**
   - Mitigation: Simple API, good docs
   - Fallback: Support both approaches

## Timeline

### Month 1
- Week 1-2: Core infrastructure
- Week 3-4: Integration implementation

### Month 2
- Week 5-6: Validation framework
- Week 7-8: Gap analysis

### Month 3
- Week 9-10: Performance optimization
- Week 11-12: Documentation

### Ongoing
- Continuous validation
- Community feedback
- Incremental improvements

## Success Criteria

1. **Technical Success**
   - 100% trace parity proven
   - <5% performance overhead
   - Zero breaking changes

2. **Adoption Success**
   - 50% adoption in 3 months
   - 90% adoption in 6 months
   - Positive developer feedback

3. **Quality Success**
   - Reduced bug reports
   - Cleaner codebase
   - Better debugging tools

## Conclusion

This decorator implementation plan provides a safe, validated path to cleaner tracing code while maintaining complete backwards compatibility. The parallel implementation and validation approach ensures zero risk while delivering significant benefits in code maintainability and developer experience.