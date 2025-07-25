# Decorators and Trackers in Synth-AI Tracing v2

## Overview

The Synth-AI framework uses a hybrid approach combining decorators from the v1 tracing system with the new v2 session-based tracing. This document explains how decorators and trackers work together in the Crafter environment example.

## Architecture

### 1. Decorator System (from v1)

The decorator system is imported from `synth_sdk.tracing.decorators`:

```python
from synth_sdk.tracing.decorators import trace_event_async
```

#### Key Decorator: `@trace_event_async`

This decorator is used to trace async function executions. In the Crafter agent:

```python
@trace_event_async(event_type="react_agent_decide")
async def decide(self, obs_str: str, current_raw_obs: Dict[str, Any]) -> List[int]:
```

The decorator:
- Captures function entry/exit
- Records timing information
- Associates events with a specific event type
- Integrates with the SynthTracker system

### 2. Tracker System (from v1)

The tracker system uses `SynthTracker` from `synth_sdk.tracing.trackers`:

```python
from synth_sdk.tracing.trackers import SynthTracker
```

Key usage patterns:
- `SynthTracker.track_lm()` - Tracks LLM interactions with full message context
- `SynthTracker.track_reward_signal()` - Tracks rewards and achievements
- `SynthTracker.finish()` - Completes tracking session

### 3. Session Tracer (v2)

The v2 system introduces `SessionTracer` for comprehensive session management:

```python
from synth_ai.tracing_v2.session_tracer import (
    SessionTracer, SessionEventMessage, TimeRecord,
    RuntimeEvent, EnvironmentEvent, CAISEvent
)
```

Key components:
- **SessionTracer**: Main class managing sessions and timesteps
- **Event Types**:
  - `CAISEvent`: For LLM/AI system events
  - `EnvironmentEvent`: For environment state changes
  - `RuntimeEvent`: For runtime system events
  - `SessionEventMessage`: For messages between components

### 4. Hook System (v2)

The v2 system includes a hook mechanism for analyzing state transitions:

```python
# From trace_hooks.py
class EasyAchievementHook(TraceStateHook):
    """Fires when easy achievements are unlocked"""
    
class InvalidActionHook(TraceStateHook):
    """Detects when actions don't have expected effects"""
```

Hooks analyze state transitions and can:
- Detect specific conditions (achievements, invalid actions)
- Add metadata to events
- Fire custom events based on state changes

## Integration Patterns

### 1. Agent Integration

In the ReAct agent (`crafter_react_agent.py`):

```python
class ReActAgent:
    def __init__(self, llm, max_turns: int = 50):
        self.system_id = get_system_id(self.system_name)
        self.system_instance_id = str(uuid.uuid4())
        
    @trace_event_async(event_type="react_agent_decide")
    async def decide(self, obs_str: str, current_raw_obs: Dict[str, Any]) -> List[int]:
        # Track LLM interaction
        SynthTracker.track_lm(
            system_id=self.system_id,
            system_instance_id=self.system_instance_id,
            turn_number=turn_number,
            messages=[{"role": "system", "content": system_message}, 
                     {"role": "user", "content": prompt}]
        )
```

### 2. Environment Integration

In the Crafter environment (`environment.py`):

```python
class CrafterClassicEnvironment:
    def __init__(self, ..., session_tracer: Optional[Any] = None):
        self.session_tracer = session_tracer
        
    def validate_tool_calls(self, tool_calls):
        # Record runtime event
        if self.session_tracer:
            runtime_event = RuntimeEvent()
            runtime_event.time_record = TimeRecord()
            runtime_event.system_state_before = state_before
            runtime_event.system_state_after = {"validated_call": agent_call}
            self.session_tracer.current_session.add_event(runtime_event)
```

### 3. Test Script Integration

In the test script (`test_crafter_react_agent_openai.py`):

```python
# Create session tracer
session_tracer = SessionTracer()

# Start session
session_context = await session_tracer.start_session(
    session_id=episode_id,
    metadata={
        "episode": episode + 1,
        "task_instance": task_instance.metadata.model_dump()
    }
)

# During episode execution
async with session_tracer.timestep(turn_number=turn):
    # Add messages
    await session_tracer.add_message(message)
    
    # Agent decides
    actions = await agent.decide(obs_str, raw_obs_dict)
    
    # Environment steps
    raw_obs = await env.step(agent_actions)
```

## Data Flow

1. **Function Decoration**: `@trace_event_async` captures function execution
2. **LLM Tracking**: `SynthTracker.track_lm()` records LLM interactions
3. **Session Management**: `SessionTracer` organizes events into sessions/timesteps
4. **Message Recording**: Messages between agent/environment are captured
5. **Hook Execution**: Hooks analyze state transitions and fire events
6. **Storage**: Sessions saved to DuckDB or cloud storage

## Key Design Decisions

1. **Hybrid Approach**: Combines v1 decorators with v2 session management
2. **Event Types**: Separates CAIS (AI), Environment, and Runtime events
3. **Temporal Organization**: Events organized by session � timestep � event
4. **Hook System**: Extensible analysis of state transitions
5. **Storage Flexibility**: Local DuckDB or future cloud storage

## Best Practices

1. **Use Decorators** for function-level tracing:
   ```python
   @trace_event_async(event_type="meaningful_name")
   async def my_function():
   ```

2. **Track LLM Calls** with full context:
   ```python
   SynthTracker.track_lm(messages=[...], turn_number=n)
   ```

3. **Manage Sessions** properly:
   ```python
   async with session_tracer.timestep(turn_number=n):
       # All events within this timestep
   ```

4. **Record State Transitions** in environments:
   ```python
   runtime_event.system_state_before = {...}
   runtime_event.system_state_after = {...}
   ```

5. **Create Custom Hooks** for domain-specific analysis:
   ```python
   class MyCustomHook(TraceStateHook):
       def analyze_transition(self, before, after):
           # Custom logic
   ```

## Future Considerations

- The empty `decorators.py` in v2 suggests future decorator implementations
- Cloud storage integration is planned but not yet implemented
- The hook system can be extended for more sophisticated analysis
- Performance optimizations may be needed for high-frequency events

## Proposed v2 Decorator System

### Design Goals

1. **Maintain Manual Approach**: Keep all existing manual tracing intact
2. **Add Decorators in Parallel**: Implement new decorator system alongside manual tracing
3. **Validate 1:1 Match**: Assert that decorator-based tracing produces identical output to manual tracing
4. **Gradual Migration**: Once validated, optionally migrate to decorator-only approach

### Proposed Implementation

```python
import asyncio, functools, contextvars
from datetime import datetime
from synth_ai.tracing_v2.session_tracer import (
    SessionTracer, TimeRecord,
    CAISEvent, RuntimeEvent, EnvironmentEvent
)

# Context variable to store active tracer
_tracer_ctx: contextvars.ContextVar[SessionTracer] = contextvars.ContextVar("session_tracer")

def set_active_tracer(tracer: SessionTracer):
    """Set the active tracer for the current context"""
    _tracer_ctx.set(tracer)

def get_active_tracer() -> Optional[SessionTracer]:
    """Get the active tracer, returns None if not set"""
    return _tracer_ctx.get(None)

def _trace_factory(
    event_cls,
    system_id: str,
    capture_args: bool = True,
    capture_result: bool = True,
    metadata_fn: Optional[Callable] = None,
    message_fn: Optional[Callable] = None,
    auto_timestep: bool = False,
    state_filter: Optional[Callable] = None
):
    """
    Factory for creating tracing decorators.
    
    Args:
        event_cls: Event class to instantiate (CAISEvent, RuntimeEvent, etc.)
        system_id: System identifier
        capture_args: Whether to capture function arguments
        capture_result: Whether to capture function result
        metadata_fn: Function to generate custom metadata
        message_fn: Function to generate messages
        auto_timestep: Whether to auto-create timestep if not in one
        state_filter: Function to filter/transform state before capture
    """
    def decorator(fn):
        is_async = asyncio.iscoroutinefunction(fn)

        @functools.wraps(fn)
        async def async_wrapper(*args, **kwargs):
            tracer = get_active_tracer()
            if not tracer:
                # No active tracer, execute normally
                return await fn(*args, **kwargs)
            
            # Optionally start timestep
            if auto_timestep and not tracer.in_timestep:
                await tracer.start_timestep(f"{system_id}:{fn.__name__}")
            
            # Capture before state
            before = {}
            if capture_args:
                raw_state = {"args": args, "kwargs": kwargs}
                before = state_filter(raw_state) if state_filter else raw_state
            
            # Execute function
            try:
                result = await fn(*args, **kwargs)
                error = None
            except Exception as e:
                result = None
                error = str(e)
                raise
            finally:
                # Capture after state
                after = {}
                if capture_result and result is not None:
                    raw_after = {"result": result}
                    after = state_filter(raw_after) if state_filter else raw_after
                if error:
                    after["error"] = error
                
                # Generate metadata
                metadata = {}
                if metadata_fn:
                    metadata = metadata_fn(args, kwargs, result, error)
                
                # Create and record event
                event = event_cls(
                    system_instance_id=system_id,
                    system_state_before=before,
                    system_state_after=after,
                    metadata=metadata,
                    time_record=TimeRecord(
                        event_time=datetime.now().timestamp(),
                        message_time=None
                    )
                )
                tracer.record_event(event)
                
                # Optionally record message
                if message_fn:
                    message = message_fn(args, kwargs, result, error)
                    if message:
                        await tracer.add_message(message)
            
            return result

        @functools.wraps(fn)
        def sync_wrapper(*args, **kwargs):
            # Similar implementation for sync functions
            tracer = get_active_tracer()
            if not tracer:
                return fn(*args, **kwargs)
            # ... (similar logic as async_wrapper)

        return async_wrapper if is_async else sync_wrapper
    return decorator

# Convenience decorators
trace_agent = functools.partial(_trace_factory, CAISEvent, "agent")
trace_runtime = functools.partial(_trace_factory, RuntimeEvent, "runtime")
trace_env = functools.partial(_trace_factory, EnvironmentEvent, "environment")
```

### Migration Strategy

#### Phase 1: Parallel Implementation
1. Keep all existing manual tracing code
2. Add decorators to the same functions
3. Both systems run simultaneously

Example:
```python
@trace_agent(
    metadata_fn=lambda a, k, r, e: {"function": "decide", "turn": get_turn()}
)
@trace_event_async(event_type="react_agent_decide")  # Existing v1 decorator
async def decide(self, obs_str: str, current_raw_obs: Dict[str, Any]) -> List[int]:
    # Existing manual tracing
    SynthTracker.track_lm(...)
    
    # Rest of implementation
```

#### Phase 2: Validation
Create a validation harness that:
1. Runs the same test scenario twice (with/without decorators)
2. Compares the generated traces
3. Asserts structural equivalence

```python
# validation_harness.py
async def validate_decorator_parity():
    # Run with manual tracing only
    manual_trace = await run_episode(use_decorators=False)
    
    # Run with both manual and decorator tracing
    combined_trace = await run_episode(use_decorators=True)
    
    # Extract decorator-only events
    decorator_events = extract_decorator_events(combined_trace)
    
    # Compare
    assert_trace_equivalence(manual_trace, decorator_events)
```

#### Phase 3: Analysis and Refinement
1. Identify any gaps in decorator coverage
2. Add configuration options for edge cases
3. Document any intentional differences

#### Phase 4: Optional Migration
Once validated:
1. Remove manual tracing code
2. Rely solely on decorators
3. Keep manual tracing available for complex scenarios

### Benefits of This Approach

1. **Zero Risk**: No changes to existing behavior
2. **Validation**: Mathematically prove equivalence
3. **Flexibility**: Can use both approaches where appropriate
4. **Clean Code**: Decorators reduce boilerplate
5. **Backwards Compatible**: Old code continues to work

### Example Usage After Migration

```python
# Clean agent code
class ReActAgent:
    @trace_agent(
        message_fn=lambda a, k, r, e: SessionEventMessage(
            source_id="agent",
            target_id="runtime",
            message_type="tool_call",
            content=r
        ) if r else None
    )
    async def decide(self, obs_str: str, current_raw_obs: Dict[str, Any]) -> List[int]:
        # Just the business logic, no manual tracing needed
        prompt = self._build_prompt(obs_str)
        response = await self.llm.generate(prompt)
        return self._parse_actions(response)

# Clean environment code  
class CrafterClassicEnvironment:
    @trace_env(
        state_filter=lambda s: {
            "action": s["args"][1] if len(s["args"]) > 1 else None
        }
    )
    async def step(self, tool_calls):
        # Just the step logic, tracing handled by decorator
        agent_call = self.validate_tool_calls(tool_calls)
        return await self._interact_tool(agent_call)
```

### Validation Checklist

- [ ] All manual trace events have decorator equivalents
- [ ] Event timestamps match (within tolerance)
- [ ] State capture is identical
- [ ] Messages are recorded correctly
- [ ] Metadata is preserved
- [ ] Turn advancement matches
- [ ] Error cases are handled
- [ ] Performance is acceptable