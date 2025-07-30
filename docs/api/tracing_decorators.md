---
title: 'Tracing Decorators'
description: 'Decorators for tracing functionality'
---

# Tracing Decorators

Decorators for adding tracing functionality to functions and methods.

## Functions

### set_active_session_tracer

Decorator to set the active session tracer for a function.

**Signature:** `set_active_session_tracer(tracer: SessionTracer)`

**Example:**
```python
from synth_ai.tracing.decorators import set_active_session_tracer

@set_active_session_tracer(my_tracer)
def my_function():
    # Function will use the specified tracer
    pass
```

### set_system_id

Decorator to set the system ID for tracing.

**Signature:** `set_system_id(system_id: str)`

### set_turn_number

Decorator to set the turn number for tracing.

**Signature:** `set_turn_number(turn: int)`

