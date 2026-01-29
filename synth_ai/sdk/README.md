# Synth AI SDK

User-facing programmatic API for Synth AI.

## Structure

```
sdk/
├── eval/           # Evaluation jobs (EvalJob, EvalJobConfig)
├── optimization/   # Policy & Graph optimization (PolicyOptimizationJob, GraphOptimizationJob)
├── localapi/       # LocalAPI for running task apps (LocalAPIClient, InProcessTaskApp)
├── artifacts/      # Artifact management (models, prompts)
├── graphs/         # Graph completions & verifiers (VerifierClient, GraphCompletionsClient)
├── inference/      # Model inference proxy (InferenceClient)
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
- Public API should be importable from the module root:
  ```python
  from synth_ai.sdk.eval import EvalJob  # Good
  from synth_ai.sdk.eval.job import EvalJob  # Also fine
  from synth_ai.sdk.eval._impl.internals import X  # Avoid - internal
  ```

## Public Exports

All user-facing classes are re-exported from `sdk/__init__.py`:

```python
from synth_ai.sdk import (
    # Optimization
    PolicyOptimizationJob,
    GraphOptimizationJob,
    
    # Evaluation
    EvalJob,
    
    # LocalAPI
    LocalAPIClient,
    InProcessTaskApp,
    
    # Clients
    VerifierClient,
    GraphCompletionsClient,
    InferenceClient,
)
```
