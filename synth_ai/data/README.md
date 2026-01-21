# Synth AI Data Layer

Pure data types and structures for Synth AI. This is the canonical source for all user-facing data format classes.

## Structure

```
data/
├── __init__.py              # Public exports
├── enums.py                 # Domain enums (JobType, RewardType, OutputMode, etc.)
├── traces.py                # Session trace structures (SessionTrace, BaseEvent, etc.)
├── llm_calls.py             # LLM call records (LLMCallRecord, LLMUsage, etc.)
├── rubrics.py               # Rubric definitions (Criterion, Rubric)
├── rewards.py               # Reward structures (OutcomeRewardRecord, EventRewardRecord)
├── objectives.py            # Objective specifications and assignments
├── judgements.py            # Judgement and rubric score results
├── artifacts.py             # Rollout artifacts
└── coding_agent_context.py  # Context overrides for coding agents
```

## Guidelines for Additions

### DO add to `data/` if:

1. **It's a pure data structure** - No business logic, just fields
2. **Users assemble/submit it** - Traces, rewards, rubrics that users create
3. **It's a canonical format** - The authoritative definition for persisted data
4. **It's a domain enum** - Enums defining domain concepts (JobType, RewardType, etc.)

### DO NOT add to `data/` if:

1. **It's an API contract** - Request/response schemas belong in `sdk/`
2. **It's an API output** - Verifier responses, scores belong in `sdk/`
3. **It's internal state** - Progress tracking, streaming state belong in `core/` or `sdk/`
4. **It has business logic** - Validation, computation belong elsewhere
5. **It's configuration** - Config classes belong in `core/config/`

## Design Principles

### Use `@dataclass` (not Pydantic)

All data structures in `data/` should use Python's `@dataclass`:

```python
from dataclasses import dataclass, field

@dataclass
class MyDataType:
    id: str
    value: float = 1.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.value <= 0:
            raise ValueError("value must be positive")
```

**Why dataclass over Pydantic?**
- Lighter weight, fewer dependencies
- Sufficient for pure data (no runtime validation needed)
- Pydantic stays in `sdk/` for API contracts that need validation

### Minimal Dependencies

`data/` should only import from:
- Standard library (`typing`, `dataclasses`, `datetime`, `enum`)
- Other `data/` modules

`data/` should NOT import from:
- `synth_ai.core` (except in `TYPE_CHECKING` blocks)
- `synth_ai.sdk`
- `synth_ai.cli`

### Immutability Preferred

Use frozen dataclasses where possible:

```python
@dataclass(frozen=True)
class SessionMessageContent:
    text: str | None = None
    json_payload: str | None = None
```

## Import Rules

All data types are exported from the package root:

```python
# Preferred - import from package root
from synth_ai.data import SessionTrace, LLMCallRecord, Criterion, Rubric

# Also fine - import from specific module
from synth_ai.data.traces import SessionTrace
from synth_ai.data.rubrics import Criterion, Rubric
```

## Relationship to Other Modules

| Module | Relationship |
|--------|--------------|
| `core/` | `data/` is independent; `core/` can import from `data/` |
| `sdk/` | `sdk/` imports from `data/` for data types |
| `cli/` | `cli/` imports from `data/` for data types |

## Data Type Categories

### User Inputs (users create these)
- `Criterion`, `Rubric` - Evaluation criteria definitions
- `SessionTrace`, events - Execution traces
- `ContextOverride` - Agent context modifications

### User Outputs (users receive these)
- `RubricAssignment`, `CriterionScoreData` - Evaluation results
- `Judgement` - Verifier judgements

### Domain Enums
- `JobType`, `JobStatus` - Job lifecycle
- `RewardType`, `RewardScope` - Reward classification
- `OutputMode` - Policy output format
