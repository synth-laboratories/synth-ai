# Environments

This package provides environment implementations for various tasks and games.

## Registering Custom Environments

To register a custom environment for use with the Environments daemon:

### 1. Install & Import

```bash
pip install synth-env
```

```python
from synth_env.stateful.core import StatefulEnvironment
from synth_env.environment.registry import register_environment
```

### 2. Subclass StatefulEnvironment

```python
class MyCounterEnv(StatefulEnvironment):
    async def initialize(self):
        self.counter = 0
        return {"obs": self.counter}

    async def step(self, tool_calls):
        self.counter += 1
        return {"obs": self.counter}
```

### 3. Register Your Environment

```python
register_environment("MyCounter-v0", MyCounterEnv)
```

See `manual_registry.md` in `old/` for detailed registration patterns and examples.
