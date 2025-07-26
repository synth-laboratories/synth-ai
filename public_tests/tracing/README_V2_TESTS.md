# V2 Tracing System Tests

This directory contains comprehensive tests for the v2 tracing system in synth-ai.

## Test Organization

### Core Functionality Tests

- **`test_v2_basic_functionality.py`**: Tests for basic v2 tracing components
  - SessionTracer functionality
  - Event creation and management (CAISEvent, EnvironmentEvent, RuntimeEvent)
  - Message handling and TimeRecord
  - Session and timestep management
  - Trace hooks integration

- **`test_v2_serialization.py`**: Tests for serialization functionality
  - `to_dict()` methods for all dataclasses
  - JSON compatibility
  - UUID and datetime handling
  - Complex nested structure serialization
  - Round-trip serialization tests

### Decorator System Tests

- **`test_v2_decorators.py`**: Tests for the v3 decorator system
  - Basic decorator functionality (@ai_call, @environment_step, @runtime_operation)
  - Context propagation across async/sync boundaries
  - Thread pool context preservation
  - Decorator parity with manual tracing
  - Performance overhead validation (<5% requirement)
  - Error handling in decorated functions

### Integration Tests

- **`test_v2_lm_integration.py`**: Tests for LM class v2 tracing integration
  - Native v2 tracing support in LM class
  - Provider-agnostic tracing (OpenAI, Anthropic, etc.)
  - System state capture
  - Streaming support
  - Error handling with tracing
  - Context propagation through LM calls

- **`test_v2_end_to_end.py`**: End-to-end integration tests
  - Complete agent-environment interaction loops
  - Trace serialization/deserialization
  - Hook integration in real scenarios
  - Performance characteristics
  - Error recovery
  - Multi-turn conversations with LM

### OpenTelemetry Tests

- **`test_v2_otel_integration.py`**: OpenTelemetry integration tests
  - OTel span creation and attributes
  - Semantic conventions compliance
  - BatchSpanProcessor integration
  - Cost metrics tracking
  - PII masking in spans
  - Dual-mode tracing (v2 + OTel)

### Configuration Tests

- **`test_v2_configuration.py`**: Configuration system tests
  - TracingConfig loading and validation
  - Environment variable handling
  - Configuration file support (JSON/YAML)
  - Configuration precedence (args > env > file > defaults)
  - PII pattern configuration
  - Caching behavior

## Key Test Scenarios

### 1. Basic Tracing Flow
```python
# From test_v2_basic_functionality.py
async with tracer.start_session(session_id):
    async with tracer.timestep(1):
        await tracer.add_message(message)
        await tracer.add_event(event)
```

### 2. Decorator Usage
```python
# From test_v2_decorators.py
@ai_call
async def llm_function(prompt: str) -> Dict[str, Any]:
    return {"response": "...", "model": "gpt-4", "usage": {...}}
```

### 3. LM Integration
```python
# From test_v2_lm_integration.py
lm = LM(
    provider="openai",
    model="gpt-4",
    session_tracer=tracer,
    system_id="agent",
    enable_v2_tracing=True
)
response = await lm.agenerate("Hello!")
```

### 4. End-to-End Flow
```python
# From test_v2_end_to_end.py
async with tracer.start_session(session_id):
    for turn in range(max_turns):
        async with tracer.timestep(turn):
            # Environment observation
            await tracer.add_message(obs_message)
            # Agent thinks
            thought = await agent.think(observation)
            # Agent action
            await tracer.add_message(action_message)
            # Environment step
            step_result = env.step(action)
```

## Performance Requirements

All tests validate that:
- Decorator overhead is <5% compared to undecorated functions
- Tracing overhead in end-to-end scenarios is <5%
- Batch processing reduces export frequency
- PII masking doesn't significantly impact performance

## Configuration Testing

Tests cover all configuration sources:
- Default values
- Environment variables (SYNTH_TRACING_*)
- Configuration files (JSON/YAML)
- Runtime arguments
- Precedence rules

## Error Handling

Tests ensure graceful handling of:
- API errors during LM calls
- Agent failures during execution
- Invalid configuration values
- Missing or malformed trace data
- Context propagation failures

## Running the Tests

```bash
# Run all v2 tracing tests
pytest public_tests/tracing/test_v2_*.py -v

# Run specific test categories
pytest public_tests/tracing/test_v2_basic_functionality.py -v
pytest public_tests/tracing/test_v2_decorators.py -v
pytest public_tests/tracing/test_v2_lm_integration.py -v

# Run with coverage
pytest public_tests/tracing/test_v2_*.py --cov=synth_ai.tracing_v2 --cov-report=html
```

## Test Coverage

The tests aim for comprehensive coverage of:
- All v2 tracing components
- Decorator functionality and parity
- LM class integration points
- Configuration system
- Serialization logic
- OpenTelemetry integration
- Error handling paths
- Performance characteristics