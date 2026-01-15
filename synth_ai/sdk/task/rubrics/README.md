# Rubrics Module

This module provides rubric schema, loading, evaluation, and validation utilities for Task Apps.

## Organization

The rubrics module is organized into focused submodules:

- **`models.py`** - Core `Criterion` and `Rubric` Pydantic models with flexible validation
- **`loaders.py`** - Loading from JSON/YAML/HTTP sources and rubric blending utilities
- **`evaluation.py`** - Evaluation utilities for events and outcomes against rubrics
- **`strict.py`** - Strict validators (`StrictCriterion`, `StrictRubric`) for step-wise verifiers

## Quick Start

### Basic Usage

```python
from synth_ai.task.rubrics import Criterion, Rubric, load_rubric, evaluate_events_against_rubric

# Load from file
rubric = load_rubric("path/to/rubric.json")

# Or create programmatically
rubric = Rubric(
    version="1.0",
    goal_text="Evaluate code quality",
    criteria=[
        Criterion(id="correctness", description="Code works correctly", weight=2.0),
        Criterion(id="style", description="Follows style guide", weight=1.0),
    ],
    aggregation="weighted_sum"
)

# Evaluate events
events = [
    {"criterion_id": "correctness", "reward": 0.9},
    {"criterion_id": "style", "reward": 0.8},
]
result = evaluate_events_against_rubric(events, rubric)
print(result["reward"])  # Weighted average
```

### Strict Validation (for Verifier Configs)

```python
from synth_ai.task.rubrics import StrictRubric, validate_rubric_file, ValidationError

try:
    # Strict rubrics require:
    # - Weights sum to exactly 1.0
    # - All weights ≤ 1.0
    # - Only weighted_sum aggregation
    rubric = validate_rubric_file("verifier_rubric.json")
except ValidationError as e:
    print(f"Invalid rubric: {e}")
```

## Flexible vs Strict Models

### Flexible Models (`Criterion`, `Rubric`)

Used by task apps for general reward computation:
- ✅ Weights can be > 1.0
- ✅ Weights don't need to sum to 1.0
- ✅ Supports multiple aggregation types: `sum`, `weighted_sum`, `custom`, `inherit`
- ✅ Optional `goal_text` and criteria
- ✅ Rich blending and loading features

### Strict Models (`StrictCriterion`, `StrictRubric`)

Used for step-wise verifier validation:
- ✅ Weights must be ≤ 1.0 and sum to exactly 1.0
- ✅ Only `weighted_sum` aggregation
- ✅ Required `goal_text` and at least one criterion
- ✅ Stricter validation for production verifier configs

## API Reference

### Models

#### `Criterion`
```python
class Criterion(BaseModel):
    id: str                    # Unique criterion identifier
    description: str           # Human-readable description
    weight: float = 1.0        # Criterion weight (must be > 0)
    required: bool = False     # Whether criterion is required
```

#### `Rubric`
```python
class Rubric(BaseModel):
    version: str                         # Rubric version
    goal_text: str | None = None         # Optional goal description
    criteria: list[Criterion] = []       # List of criteria
    aggregation: str = "weighted_sum"    # Aggregation method
```

### Loading

#### `load_rubric(source)`
Load rubric from file path, dict, URL, or existing Rubric.

**Supported formats:**
- JSON files (`.json`)
- YAML files (`.yaml`, `.yml`)
- HTTP URLs (returns JSON)
- Raw JSON/YAML strings
- Python dicts
- Existing `Rubric` objects (pass-through)

```python
# From file
rubric = load_rubric("rubric.json")

# From dict
rubric = load_rubric({"version": "1.0", "criteria": [...]})

# From URL
rubric = load_rubric("https://example.com/rubric.json")
```

#### `blend_rubrics(base, override)`
Merge two rubrics, with override taking precedence.

```python
base = load_rubric("base_rubric.json")
override = load_rubric("custom_tweaks.json")
merged = blend_rubrics(base, override)
```

### Evaluation

#### `evaluate_events_against_rubric(events, rubric)`
Evaluate events (list of criterion rewards).

```python
events = [
    {"criterion_id": "quality", "reward": 0.9},
    {"id": "performance", "reward": 0.8},
]
result = evaluate_events_against_rubric(events, rubric)
# Returns: {"aggregation": "weighted_sum", "reward": 0.85, "per_criterion": {...}}
```

#### `evaluate_outcome_against_rubric(outcome, rubric)`
Evaluate a rollout outcome (dict of criterion rewards).

```python
outcome = {
    "quality": 0.9,
    "performance": 0.8,
}
result = evaluate_outcome_against_rubric(outcome, rubric)
```

### Strict Validation

#### `validate_rubric_dict(payload)`
Validate dict with strict rules (weights sum to 1.0, etc.).

```python
try:
    rubric = validate_rubric_dict({
        "version": "1.0",
        "goal_text": "Evaluate X",
        "aggregation": "weighted_sum",
        "criteria": [
            {"id": "a", "description": "A", "weight": 0.6},
            {"id": "b", "description": "B", "weight": 0.4},
        ]
    })
except ValidationError as e:
    print(f"Validation failed: {e}")
```

#### `validate_rubric_file(path)`
Load and validate JSON rubric file with strict rules.

#### `validate_rubric_files(paths)`
Bulk validate multiple rubric files.

## Migration Guide

### From `synth_ai.rubrics`

The old `synth_ai/rubrics/` directory has been consolidated here:

```python
# Old imports (DEPRECATED)
from synth_ai.rubrics import RubricSpec, RubricCriterion, validate_rubric_file

# New imports
from synth_ai.task.rubrics import StrictRubric, StrictCriterion, validate_rubric_file

# Backwards compatibility aliases (will be removed in future)
from synth_ai.task.rubrics import RubricSpec, RubricCriterion
```

### From single `rubrics.py` file

The old single-file rubric helpers have been split into submodules:

```python
# Old (still works via __init__.py)
from synth_ai.task.rubrics import Criterion, Rubric, load_rubric

# New (explicit submodules)
from synth_ai.task.rubrics.models import Criterion, Rubric
from synth_ai.task.rubrics.loaders import load_rubric
from synth_ai.task.rubrics.evaluation import evaluate_events_against_rubric
```

**Recommended:** Use the consolidated imports from `__init__.py`:
```python
from synth_ai.task.rubrics import (
    Criterion,
    Rubric,
    load_rubric,
    blend_rubrics,
    evaluate_events_against_rubric,
    evaluate_outcome_against_rubric,
    StrictRubric,
    validate_rubric_file,
)
```

## Examples

### Task Apps

See these task apps for real-world usage:
- `examples/task_apps/crafter/task_app/grpo_crafter.py`
- `examples/task_apps/verilog/task_app/grpo_verilog.py`
- `examples/task_apps/enron/task_app/grpo_enron.py`

### Example Rubrics

JSON rubrics demonstrating strict validation:
- `examples/multi_step/rubrics/crafter_events_rubric.json`
- `examples/multi_step/rubrics/crafter_outcome_rubric.json`

## Testing

```bash
# Run rubric tests
pytest tests/unit/rubrics/ -v

# Test specific validator
pytest tests/unit/rubrics/test_rubric_validation.py -v
```

## Related

- **Trace Utils**: Moved to `synth_ai/tracing_v3/trace_utils.py` (was incorrectly in old rubrics dir)
- **Verifier Schemas**: See `synth_ai/sdk/graphs/verifier_schemas.py` for verifier API contracts
- **Task Server**: See `synth_ai/task/server.py` for rubric integration in task apps


