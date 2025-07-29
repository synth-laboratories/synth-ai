# LM Class with Native V2 Tracing

This document explains how to use the enhanced LM class with native v2 tracing support.

## Overview

The enhanced LM class (`main_v2.py`) provides seamless integration with v2 tracing through decorators, without requiring modifications to provider wrappers. This enables:

- Clean v2 event capture for all LM calls
- Automatic cost tracking with OTel attributes
- PII masking and payload truncation
- Dual-mode support (v1 + v2 tracing)
- Provider-agnostic implementation

## Quick Start

### 1. Basic Usage

```python
from synth_ai.lm.core.main_v2 import LM
from synth_ai.tracing_v2.session_tracer import SessionTracer

# Create session tracer
tracer = SessionTracer()

# Create LM with v2 tracing enabled
lm = LM(
    model_name="gpt-4o-mini",
    formatting_model_name="gpt-4o-mini",
    temperature=0.7,
    synth_logging=False,     # Disable v1 tracing
    session_tracer=tracer,    # Enable v2 tracing
    system_id="my_assistant", # Custom system ID
    enable_v2_tracing=True    # Explicitly enable v2
)

# Use within a session
async with tracer.start_session("conversation-123"):
    # Each call is automatically traced
    response = await lm.respond_async(
        system_message="You are a helpful assistant",
        user_message="What is the capital of France?",
        turn_number=0  # Optional: track conversation turns
    )
    print(response.raw_response)
```

### 2. With Structured Outputs

```python
from pydantic import BaseModel

class CityInfo(BaseModel):
    name: str
    country: str
    population: int

# V2 tracing captures structured output metadata
response = await lm.respond_async(
    system_message="Extract city information",
    user_message="Paris is the capital of France with 2.2 million people",
    response_model=CityInfo,
    turn_number=0
)
# Traced attributes include model, messages, response_model type, and structured output
```

### 3. Using Context Manager

```python
from synth_ai.lm.core.main_v2 import LMTracingContext

# Switch tracing context dynamically
lm = LM(model_name="gpt-4", ...)  # Created without tracer

with LMTracingContext(lm, session_tracer):
    # All calls within this context are traced
    response = await lm.respond_async(...)
```

## Configuration

### Environment Variables

```bash
# V2 Tracing Mode
export SYNTH_TRACING_MODE=dual       # Options: dual, v2, otel, disabled

# PII Masking
export SYNTH_MASK_PII=true           # Enable PII masking
export SYNTH_MAX_PAYLOAD_BYTES=10240 # Truncate large payloads

# OTel Configuration (for dual/otel modes)
export OTEL_SERVICE_NAME=my-service
export OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4317
export OTEL_TRACES_SAMPLER=traceidratio
export OTEL_TRACES_SAMPLER_ARG=0.1  # 10% sampling
```

### Migration Strategies

#### Phase 1: Dual Mode (Recommended Start)
```python
# Both v1 and v2 tracing active
lm = LM(
    model_name="gpt-4o-mini",
    synth_logging=True,      # Keep v1 for compatibility
    enable_v2_tracing=True,  # Add v2 tracing
    session_tracer=tracer
)
```

#### Phase 2: V2-Only Mode
```python
# Only v2 tracing (disable v1 wrappers)
lm = LM(
    model_name="gpt-4o-mini",
    synth_logging=False,     # Disable v1
    enable_v2_tracing=True,  # Use only v2
    session_tracer=tracer
)
```

#### Phase 3: Multi-Provider Support
```python
# V2 tracing works with any provider
lm = LM(
    model_name="claude-3-sonnet",
    provider="anthropic",
    synth_logging=False,
    enable_v2_tracing=True,
    session_tracer=tracer
)
```

## Captured Attributes

The v2 tracing automatically captures:

### Request Attributes (gen_ai.request.*)
- `gen_ai.request.model` - Model name
- `gen_ai.request.messages` - Input messages
- `gen_ai.request.temperature` - Temperature setting
- `gen_ai.request.max_tokens` - Max tokens (if set)
- `gen_ai.request.response_format` - Response model name (if structured)
- `gen_ai.request.tools_count` - Number of tools provided
- `gen_ai.request.reasoning_effort` - Reasoning effort level

### Response Attributes (gen_ai.response.*)
- `gen_ai.response.content` - Response text (truncated)
- `gen_ai.response.usage.prompt_tokens` - Input token count
- `gen_ai.response.usage.completion_tokens` - Output token count
- `gen_ai.response.usage.total_tokens` - Total tokens
- `gen_ai.response.model` - Actual model used
- `gen_ai.response.has_structured_output` - Boolean flag
- `gen_ai.response.structured_output_type` - Pydantic model name
- `gen_ai.response.tool_calls_count` - Number of tool calls

### Cost Tracking (gen_ai.usage.*)
- `gen_ai.usage.cost_usd` - Calculated cost in USD

## Example: Agent Integration

```python
class MyAgent:
    def __init__(self, session_tracer: SessionTracer):
        self.lm = LM(
            model_name="gpt-4",
            session_tracer=session_tracer,
            system_id="my_agent_brain",
            enable_v2_tracing=True
        )
        self.turn = 0
    
    async def think(self, observation: str) -> str:
        # Turn number automatically tracked
        response = await self.lm.respond_async(
            system_message="You are a strategic agent",
            user_message=f"Observation: {observation}",
            turn_number=self.turn
        )
        self.turn += 1
        return response.raw_response

# Usage
tracer = SessionTracer()
agent = MyAgent(tracer)

async with tracer.start_session("game-session"):
    thought = await agent.think("I see a tree")
    # Automatically creates CAISEvent with proper turn tracking
```

## Testing with Crafter

Run the Crafter demo with LM v2 tracing:

```bash
# Using config file
python test_crafter_react_agent_lm.py --config crafter_lm_config.toml

# Or with command line args
python test_crafter_react_agent_lm.py \
    --model gpt-4o-mini \
    --episodes 5 \
    --max-turns 10 \
    --verbose
```

## Performance

- Decorator overhead: ~200μs per call (negligible for LLM calls)
- Zero overhead when v2 tracing disabled
- Automatic batching for efficient export
- PII masking adds ~50μs for typical payloads

## Debugging

Enable debug logging:
```bash
export SYNTH_TRACE_DEBUG=true
export SYNTH_TRACE_LOG_LEVEL=DEBUG
```

Check if tracing is active:
```python
if lm.enable_v2_tracing and lm.session_tracer:
    print("V2 tracing is active")
```

## Best Practices

1. **Always use turn numbers** for conversation tracking:
   ```python
   response = await lm.respond_async(..., turn_number=turn)
   ```

2. **Set meaningful system IDs**:
   ```python
   system_id = f"{agent_type}_{model}_{instance_id}"
   ```

3. **Use session context managers**:
   ```python
   async with tracer.start_session(session_id):
       # All LM calls within are grouped
   ```

4. **Handle errors gracefully**:
   ```python
   try:
       response = await lm.respond_async(...)
   except Exception as e:
       # V2 tracing captures errors automatically
       logger.error(f"LM call failed: {e}")
   ```

## Limitations

1. Tool calls currently require manual parsing (pending BaseTool integration)
2. Provider-specific usage data extraction may vary
3. Some providers may not expose token counts immediately

## Future Enhancements

1. **Automatic Turn Tracking**:
   ```python
   lm = LM(..., auto_increment_turns=True)
   ```

2. **Conversation Memory**:
   ```python
   lm = LM(..., track_conversation=True)
   ```

3. **Native Tool Support**:
   ```python
   response = await lm.respond_async(..., tools=[MyTool()])
   ```