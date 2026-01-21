# Synth AI Python Package

Root package for Synth AI SDK, CLI, and supporting infrastructure.

## Quick Start

```python
# SDK classes are available directly from synth_ai
from synth_ai import (
    PolicyOptimizationJob,
    EvalJob,
    InProcessTaskApp,
    VerifierClient,
)

# Data types are available from synth_ai.data
from synth_ai.data import SessionTrace, Rubric, Criterion
```

## Package Structure

```
synth_ai/
├── data/           # Pure data types (traces, rewards, rubrics, enums)
├── core/           # Internal infrastructure (auth, config, streaming, tunnels)
├── sdk/            # User-facing SDK (jobs, clients, optimization)
├── cli/            # Command-line interface
├── __init__.py     # Package version and top-level exports
└── __main__.py     # Entry point for `python -m synth_ai`
```

## Module Hierarchy

The modules follow a strict dependency hierarchy:

```
┌─────────┐
│  data/  │  ← Pure data types, no internal dependencies
└────┬────┘
     │
┌────▼────┐
│  core/  │  ← Internal infrastructure, imports from data/
└────┬────┘
     │
┌────▼────┐
│  sdk/   │  ← User-facing SDK, imports from data/ and core/
└────┬────┘
     │
┌────▼────┐
│  cli/   │  ← CLI layer, imports from data/, core/, and sdk/
└─────────┘
```

### Dependency Rules

| Module | Can Import From | Cannot Import From |
|--------|-----------------|-------------------|
| `data/` | standard library only | `core/`, `sdk/`, `cli/` |
| `core/` | `data/` | `sdk/`, `cli/` |
| `sdk/` | `data/`, `core/` | `cli/` |
| `cli/` | `data/`, `core/`, `sdk/` | — |

## Module Purposes

### `data/` - Data Layer
Pure data types and structures. The canonical source for all user-facing data format classes.

- Traces, events, LLM call records
- Rubrics and judgements
- Rewards and objectives
- Domain enums

**Design principle**: Use `@dataclass`, not Pydantic. No business logic.

### `core/` - Infrastructure Layer
Internal shared utilities. Not user-facing.

- Authentication and API keys
- Configuration management
- Logging and errors
- Streaming infrastructure
- Tunnel management

**Design principle**: Shared infrastructure that both SDK and CLI need.

### `sdk/` - SDK Layer
User-facing programmatic API.

- Optimization jobs (PolicyOptimizationJob, GraphOptimizationJob)
- Evaluation jobs (EvalJob)
- Clients (VerifierClient, InferenceClient)
- LocalAPI for task apps

**Design principle**: Clean, stable API for external consumption.

### `cli/` - CLI Layer
Command-line interface. Thin wrapper around SDK.

- Authentication commands
- Job management
- Local development tools

**Design principle**: Minimal logic; delegate to SDK.

## Top-Level Exports

The package exports verifier API contracts for backward compatibility:

```python
from synth_ai import (
    VerifierScoreRequest,
    VerifierScoreResponse,
    VerifierOptions,
    # ...
)
```

For most use cases, import from the specific module:

```python
from synth_ai.sdk import EvalJob, PolicyOptimizationJob
from synth_ai.data import SessionTrace, Rubric
```

## Entry Points

- `python -m synth_ai` → CLI (via `__main__.py`)
- `synth` command → CLI (installed by package)

## Guidelines for New Code

1. **Data types** → `data/`
2. **Internal utilities** → `core/`
3. **User-facing APIs** → `sdk/`
4. **CLI commands** → `cli/`

When in doubt, check the README in each module for specific guidelines.
