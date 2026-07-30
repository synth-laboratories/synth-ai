# Core Modules

Shared runtime plumbing for the Synth AI SDK — and nothing else.

`core/` is the bottom of the layering DAG (see `../README.md` and
`unify_sdk_layering.md`). It may import only from `core/`. In particular it must
not import `sdk/`: plumbing cannot depend on the clients built on top of it.
`check_sdk_layering.py` in the sibling `testing` repo enforces this.

## Structure

```
core/
├── __init__.py      # Internal exports
├── auth/            # Credentials and auth context
├── http/            # Transport, request, retry
├── errors.py        # Error types and handling
├── contracts/       # Generic JSON/error contracts only
├── utils/           # General utilities
└── research/        # DEPRECATED alias -> synth_ai.sdk.research
```

`core/research/` is no longer an implementation tree. Research moved to
`sdk/research/` so that `core/` means one thing; what remains is an import alias
that forwards and warns, scheduled for deletion once sibling repos are clean.
Do not add modules there.

## Guidelines for Additions

### DO add to `core/` if:

1. **It's shared infrastructure** - Used by multiple live modules
2. **It's internal implementation** - Not meant for direct user consumption
3. **It's runtime plumbing** - Errors, URL resolution, env helpers
4. **It's a utility** - Helpers used by the narrowed live SDK

### DO NOT add to `core/` if:

1. **It's a user-facing API** - Put it in `sdk/`
2. **It's CLI-specific** - Put it in `cli/`
3. **It's a public client** - Put it in `sdk/` or `client.py`
4. **It's outside the public SDK runtime** - keep it out of `core/` unless it is required by supported live flows

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

## Relationship to Other Modules

| Module | Relationship |
|--------|--------------|
| `sdk/` | `sdk/` imports from `core/` for runtime helpers |
| `cli/` | `cli/` imports from `core/` for shared errors and env utilities |
