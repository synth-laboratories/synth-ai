# Synth AI SDK

User-facing programmatic API for Synth AI.

## CRITICAL: Container Auth Payload Rule

For policy-optimization job submission, never embed container auth credentials
in payload/config/overrides. Container auth is server-resolved from org storage.

## Structure

```
sdk/
├── optimization/   # Policy optimization (PolicyOptimizationOfflineJob, OnlineSession)
├── container/       # Container for running containers (ContainerClient, InProcessContainer)
├── artifacts/      # Artifact management (models, prompts)
├── managed_research.py  # Managed Research project/run control (SmrControlClient)
├── shared/         # Internal utilities (NOT public API)
│   ├── auth.py     # API key helpers
│   ├── streaming/  # SSE job streaming
│   └── tunnels/    # Tunnel management
└── __init__.py     # Public exports
```

## Guidelines for Additions

### DO add to `sdk/` if:

1. **It's a user-facing API** - Something users import and use directly
2. **It has a clear domain** - Fits into one of: jobs, clients, or services
3. **It's stable** - API surface is designed for external consumption

### DO NOT add to `sdk/` if:

1. **It's internal implementation** - Put it in `_impl/` subdirectory of the relevant module
2. **It's shared infrastructure** - Put it in `shared/` (streaming, tunnels, auth)
3. **It's business logic** - Put it in `core/`
4. **It's CLI-specific** - Put it in `cli/`
5. **It's configuration/constants** - Put it in `core/`

### Module Structure Pattern

Each public API module should follow this pattern:

```
module/
├── __init__.py      # Public exports only
├── _impl/           # Internal implementation (not imported directly)
│   └── ...
└── job.py           # Or client.py, etc. (public classes)
```

### Import Rules

- `sdk/` can import from `core/` and `data/`
- `sdk/` should NOT import from `cli/`
- Public API should be importable from the module root where available.

## Public Exports

All user-facing classes are re-exported from `sdk/__init__.py`:

```python
from synth_ai.sdk import (
    # Optimization
    OfflineJob,
    OnlineSession,
    
    # Container
    ContainerClient,
    InProcessContainer,
    
    # Other
    PoolsClient,
    System,
    create_container,
)
```
