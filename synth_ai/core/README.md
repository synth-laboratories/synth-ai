# Synth AI Core

Internal infrastructure and shared utilities for Synth AI. Not user-facing.

## Structure

```
core/
├── __init__.py      # Internal exports
├── auth/            # Authentication and API key management
├── config/          # Configuration management
├── errors.py        # Error types and handling
├── logging.py       # Logging configuration
├── streaming/       # SSE and streaming infrastructure
├── tracing_v3/      # Trace storage, serialization, helpers
├── tunnels/         # Tunnel management for local development
└── utils/           # General utilities
```

## Guidelines for Additions

### DO add to `core/` if:

1. **It's shared infrastructure** - Used by multiple modules (sdk, cli)
2. **It's internal implementation** - Not meant for direct user consumption
3. **It's system configuration** - Environment, settings, feature flags
4. **It's a utility** - Helpers, converters, formatters

### DO NOT add to `core/` if:

1. **It's a user-facing API** - Put it in `sdk/`
2. **It's a pure data type** - Put it in `data/`
3. **It's CLI-specific** - Put it in `cli/`
4. **It's a public client** - Put it in `sdk/`

## Module Descriptions

### `auth/`
API key management, token handling, and authentication utilities.

### `config/`
Configuration loading, environment variable handling, and settings management.

### `errors.py`
Custom exception types and error handling utilities.

### `logging.py`
Logging configuration and structured logging helpers.

### `streaming/`
Server-sent events (SSE) infrastructure for job streaming and real-time updates.

### `tracing_v3/`
Trace storage, serialization, and helper functions. Note: The data structures (`SessionTrace`, `LLMCallRecord`, etc.) have moved to `data/` - this module now re-exports them for backward compatibility and contains implementation logic.

### `tunnels/`
Local development tunnel management (ngrok, cloudflare, etc.) for connecting local task apps to Synth backend.

### `utils/`
General-purpose utilities: hashing, encoding, file operations, etc.

## Import Rules

- `core/` can import from `data/`
- `core/` should NOT import from `sdk/` or `cli/`
- Internal modules should not be imported directly by users

```python
# Internal usage (within synth_ai)
from synth_ai.core.auth import get_api_key
from synth_ai.core.config import load_config

# Users should NOT do this - use sdk/ instead
# from synth_ai.core.streaming import StreamClient  # Avoid
```

## Backward Compatibility

Some modules in `core/` re-export types that have moved to `data/` for backward compatibility:

```python
# These work but prefer importing from data/
from synth_ai.core.tracing_v3.abstractions import SessionTrace  # Legacy
from synth_ai.data.traces import SessionTrace  # Preferred
```

## Relationship to Other Modules

| Module | Relationship |
|--------|--------------|
| `data/` | `core/` imports from `data/` for data types |
| `sdk/` | `sdk/` imports from `core/` for infrastructure |
| `cli/` | `cli/` imports from `core/` for infrastructure |
