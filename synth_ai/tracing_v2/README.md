# Synth-AI V2 Tracing System

This directory contains the V2 tracing implementation for Synth-AI, providing comprehensive observability for AI agents, environments, and runtime systems.

## Overview

The V2 tracing system captures detailed execution traces with:
- **Session-based organization** - All traces belong to a session with metadata
- **Timestep grouping** - Events and messages organized by logical turns
- **Three event types** - CAISEvent (AI/LLM), EnvironmentEvent, RuntimeEvent  
- **Message passing** - Cross-system communication with origin tracking
- **Dual timestamps** - Wall clock time and logical time (turn number)

## Quick Start

### Basic Usage

```python
from synth_ai.tracing_v2.session_tracer import SessionTracer
from synth_ai.lm.core.main_v2 import LM

# Create tracer
tracer = SessionTracer()

# Use with LM class (recommended)
lm = LM(
    model="gpt-4",
    session_tracer=tracer,
    system_id="my_agent",
    enable_v2_tracing=True
)

# Start session
async with tracer.start_session("session-123", {"experiment": "test"}):
    # Use timesteps to organize turns
    async with tracer.timestep(0):
        response = await lm.agenerate("Hello!")
```

### Manual Tracing Pattern

```python
from synth_ai.tracing_v2.abstractions import CAISEvent
from synth_ai.tracing_v2.session_tracer import SessionEventMessage, TimeRecord

# Create event for LLM call
event = CAISEvent(
    system_instance_id="agent_123",
    model_name="gpt-4",
    prompt_tokens=50,
    completion_tokens=100,
    time_record=TimeRecord(
        event_time=datetime.now().isoformat(),
        message_time=turn_number
    )
)
tracer.add_event(event)

# Create message
message = SessionEventMessage(
    content={
        "origin_system_id": "agent_123",
        "payload": {"action": "move_north"}
    },
    message_type="action",
    time_record=TimeRecord(...)
)
await tracer.add_message(message)
```

## Core Components

### SessionTracer
Main orchestrator that manages sessions, timesteps, events, and messages.

### Event Types
- **CAISEvent** - AI/LLM system events with token usage and model info
- **EnvironmentEvent** - Environment state changes with rewards
- **RuntimeEvent** - Runtime operations like tool validation

### Decorators (decorators_v3_improved.py)
Production-ready decorators with OpenTelemetry integration:
- `@ai_call` - Decorate LLM/AI functions
- `@environment_step` - Decorate environment step functions
- `@runtime_operation` - Decorate runtime operations
- `@trace_span` - Generic span creation

Features:
- <5% performance overhead
- PII masking (SSN, credit cards, emails, phones)
- Cost tracking for AI models
- Dual-mode support (V2 events + OTel spans)
- Context propagation across async/thread boundaries

### Configuration (tracing_config.py)
Flexible configuration via code, environment variables, or files:

```python
from synth_ai.tracing_v2.tracing_config import TracingConfig

config = TracingConfig(
    enabled=True,
    otel_enabled=True,
    otel_service_name="my-service",
    mask_pii=True,
    track_costs=True
)
```

Environment variables:
- `SYNTH_TRACING_ENABLED`
- `SYNTH_TRACING_OTEL_ENABLED`
- `SYNTH_TRACING_OTEL_SERVICE_NAME`
- etc.

### DuckDB Integration
Store and query traces in DuckDB for analysis:

```python
from synth_ai.tracing_v2.duckdb.manager import DuckDBTraceManager

manager = DuckDBTraceManager("traces.db")
manager.insert_trace(session_trace)

# Query traces
df = manager.query_messages(
    session_ids=["session-123"],
    message_types=["observation", "action"]
)
```

## LM Class Integration

The LM class in `synth_ai.lm.core.main_v2` has native V2 tracing support:

```python
# Enable V2 tracing
lm = LM(
    model="gpt-4",
    session_tracer=tracer,
    system_id="my_agent",
    enable_v2_tracing=True,
    synth_logging=False  # Disable V1
)

# Automatic tracing
response = await lm.agenerate("What is 2+2?")
# Creates CAISEvent with token usage, cost, latency
```

Benefits:
- No provider wrapper modifications
- Works with all providers (OpenAI, Anthropic, Groq, etc.)
- Automatic cost calculation
- Turn tracking for conversations

## Migration from V1

### Dual Mode (V1 + V2)
```python
lm = LM(..., synth_logging=True, enable_v2_tracing=True)
```

### V2 Only Mode
```python
lm = LM(..., synth_logging=False, enable_v2_tracing=True)
```

## Testing

Comprehensive test coverage in `public_tests/tracing/`:
- `test_v2_basic_functionality.py` - Core components
- `test_v2_decorators.py` - Decorator system
- `test_v2_lm_integration.py` - LM class integration
- `test_v2_end_to_end.py` - Complete scenarios
- `test_v2_otel_integration.py` - OpenTelemetry
- `test_v2_configuration.py` - Configuration
- `test_v2_serialization.py` - Serialization
- `test_v2_duckdb_integration.py` - DuckDB storage

Run tests:
```bash
pytest public_tests/tracing/test_v2_*.py -v
```

## Examples

See `examples/duckdb_integration_example.py` for a complete example of:
- Creating sessions and timesteps
- Recording events and messages
- Storing traces in DuckDB
- Querying and analyzing traces

## Architecture

```
SessionTracer (orchestrator)
├── Sessions (with metadata)
│   ├── Timesteps (logical turns)
│   │   ├── Messages (cross-system communication)
│   │   └── Events (within timestep)
│   └── Event History (all events)
│
├── Decorators (automatic instrumentation)
│   ├── @ai_call → CAISEvent
│   ├── @environment_step → EnvironmentEvent
│   └── @runtime_operation → RuntimeEvent
│
└── Storage
    ├── JSON serialization
    └── DuckDB integration
```

## Key Patterns

### Message Structure
```python
{
    "origin_system_id": "agent_123",  # Always embedded
    "payload": {...}                  # Actual content
}
```

### TimeRecord
```python
TimeRecord(
    event_time="2024-01-01T12:00:00",  # Wall clock
    message_time=5                      # Turn number
)
```

### Event State Tracking
```python
event.system_state_before = {...}  # State before operation
event.system_state_after = {...}   # State after operation
event.metadata = {...}             # Additional context
```

## Performance

- Decorator overhead: <5% (validated by benchmarks)
- Zero overhead when disabled
- Efficient batching with OpenTelemetry
- PII masking adds <1ms per operation
- Context propagation: ~200μs per call

## Production Features

- **PII Masking** - Automatic removal of sensitive data
- **Cost Tracking** - Track AI model costs per call
- **Error Handling** - Graceful degradation on failures
- **Sampling** - Reduce volume with configurable sampling
- **Batching** - Efficient export with BatchSpanProcessor
- **Resource Attributes** - Service metadata for filtering

## V2 vs V1 Comparison

After comprehensive testing, **V2 tracing captures everything V1 did and more**.

### ✅ V2 Successfully Captures All V1 Data
- **Session Structure**: Identical top-level keys
- **Event Data**: System states, time records, metadata
- **Messages**: All cross-system communication

### 🚀 V2 Enhancements Over V1

1. **Model Information in DuckDB**
   - V1: Model info NOT saved to DuckDB
   - V2: Model info properly extracted and saved

2. **Direct Model Fields**
   - V2 uses `LMCAISEvent` with direct fields
   - Auto-detects provider from model name
   - Tracks tokens and costs when available

3. **Cleaner Architecture**
   - V1: Required patching OpenAI client
   - V2: Native integration in LM class
   - Proper class hierarchy

### Technical Implementation
- **Base**: `CAISEvent` in `abstractions.py`
- **Extended**: `LMCAISEvent` in `session_tracer.py`
- **DuckDB**: Handles both types, extracts model info

## Cleanup History

This directory was cleaned up to contain only production code:
- Tests moved to `public_tests/tracing/`
- Planning documents removed
- Temporary files deleted
- Documentation consolidated into this README

For implementation details and design decisions, see the git history.