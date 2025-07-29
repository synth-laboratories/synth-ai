---
title: 'Environment Core'
description: 'Core environment functionality'
---

# Environment Core

The core environment module provides the base classes and functionality for creating AI environments.

## Classes

### Environment

Base class for all environments in the Synth AI framework.

**Signature:** `Environment()`

**Methods:**
- `reset()` - Reset the environment to initial state
- `step(action)` - Take an action and return observation, reward, done, info
- `render()` - Render the current state of the environment

**Example:**
```python
from synth_ai.environments.environment.core import Environment

class MyEnvironment(Environment):
    def reset(self):
        # Reset to initial state
        pass
    
    def step(self, action):
        # Process action and return results
        pass
```

