# Synth AI Core

Shared runtime helpers for the live containers, tunnels, and pools SDK.

## Structure

```
core/
├── __init__.py      # Internal exports
├── errors.py        # Error types and handling
└── utils/           # General utilities
```

## Guidelines for Additions

### DO add to `core/` if:

1. **It's shared infrastructure** - Used by multiple live modules
2. **It's internal implementation** - Not meant for direct user consumption
3. **It's runtime plumbing** - Errors, URL resolution, env helpers
4. **It's a utility** - Helpers used by the narrowed SDK

### DO NOT add to `core/` if:

1. **It's a user-facing API** - Put it in `sdk/`
2. **It's CLI-specific** - Put it in `cli/`
3. **It's a public client** - Put it in `sdk/` or a top-level public module
4. **It's outside containers/tunnels/pools** - Archive it under `../research/old/synth_ai` unless it is required by live flows

## Module Descriptions

### `errors.py`
Custom exception types and error handling utilities.

### `utils/`
General-purpose utilities for environment lookup, URLs, JSON helpers, and secure file handling.

## Import Rules

- `core/` should NOT import from `sdk/` or `cli/`
- Internal modules should not be imported directly by users unless explicitly documented

```python
# Supported internal usage
from synth_ai.core.errors import SynthError
from synth_ai.core.utils.env import get_api_key
```

Legacy auth/config/streaming/tracing helpers have been archived under `../research/old/synth_ai` and are not part of the shipped package.

## Relationship to Other Modules

| Module | Relationship |
|--------|--------------|
| `sdk/` | `sdk/` imports from `core/` for runtime helpers |
| `cli/` | `cli/` imports from `core/` for shared errors and env utilities |
| `../research/old/synth_ai` | Archived legacy infra lives there for reference only |
