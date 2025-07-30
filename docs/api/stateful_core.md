---
title: 'Stateful Environment Core'
description: 'Stateful environment base classes'
---

# Stateful Environment Core

The stateful environment core provides base classes for environments that maintain state.

## Classes

### StatefulEnvironment

Abstract base class for stateful environments.

**Signature:** `StatefulEnvironment()`

**Abstract Methods:**
- `reset()` - Reset the environment state
- `step(action)` - Take an action and update state
- `get_state()` - Get current environment state
- `set_state(state)` - Set environment state

**Example:**
```python
from synth_ai.environments.stateful.core import StatefulEnvironment

class MyStatefulEnvironment(StatefulEnvironment):
    def reset(self):
        # Reset state to initial values
        pass
    
    def step(self, action):
        # Process action and update state
        pass
```

