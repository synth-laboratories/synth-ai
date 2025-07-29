---
title: 'Language Model Core'
description: 'Core language model functionality'
---

# Language Model Core

The core language model module provides unified interfaces for various LLM providers.

## Classes

### LM

Main language model class for interacting with various providers.

**Signature:** `LM(model_name: str, temperature: float = 0.7, session_tracer: SessionTracer = None)`

**Methods:**
- `respond(message: str)` - Generate a response to a message
- `respond_async(message: str)` - Generate a response asynchronously
- `chat(messages: List[dict])` - Chat with multiple messages

**Example:**
```python
from synth_ai.lm.core.main import LM

# Initialize language model
lm = LM(model_name="gpt-4o-mini", temperature=0.7)

# Generate response
response = lm.respond("Hello, world!")
print(response.raw_response)
```

