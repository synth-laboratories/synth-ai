"""Synth AI Data Layer.

This module provides pure data types with no IO dependencies.
Contains actual data schemas for traces, rewards, rubrics, and LLM calls.

Data vs SDK distinction:
- data/: Pure data records (traces, rewards, rubrics, llm_calls) - actual data
- sdk/: API abstractions (jobs, training, graphs) - SDK interfaces

Dependency rule: data/ imports nothing from synth_ai except typing helpers.

Rust-backed types:
    For performance-critical code, equivalent Rust types are available via
    `synth_ai_py` (the PyO3 bindings). These provide the same data structures
    with Rust-backed serialization/deserialization:

    ```python
    # Python dataclasses (this module) - user-friendly, documented
    from synth_ai.data import SessionTrace, LLMCallRecord

    # Rust-backed types (synth_ai_py) - for performance
    from synth_ai_py import SessionTrace, LLMCallRecord
    ```

    The Rust types support `from_dict()` and `to_dict()` methods for
    interoperability with the Python dataclasses.
"""

from __future__ import annotations

# Trace data types
import inspect
from typing import Any

# Artifact data types
from synth_ai.data.artifacts import Artifact

# Context override types (coding agent context)
from synth_ai.data.coding_agent_context import (
    ApplicationErrorType,
    ApplicationStatus,
    ContextOverride,
    ContextOverrideStatus,
    FolderMode,
    OverrideApplicationError,
    OverrideOperation,
    OverrideType,
)

# Enums - pure types
from synth_ai.data.enums import (
    AdaptiveBatchLevel,
    AdaptiveCurriculumLevel,
    GraphType,
    InferenceMode,
    JobStatus,
    JobType,
    ObjectiveDirection,
    ObjectiveKey,
    OptimizationMode,
    OutputMode,
    ProviderName,
    RewardScope,
    RewardSource,
    RewardType,
    SuccessStatus,
    TrainingType,
    VerifierMode,
)

# Judgements
from synth_ai.data.judgements import (
    CriterionScoreData,
    Judgement,
    RubricAssignment,
)

# LLM call record data types
from synth_ai.data.llm_calls import (
    LLMCallRecord,
    LLMChunk,
    LLMContentPart,
    LLMMessage,
    LLMRequestParams,
    LLMUsage,
    ToolCallResult,
    ToolCallSpec,
)

# Objective definitions
from synth_ai.data.objectives import (
    OBJECTIVE_REGISTRY,
    EventObjectiveAssignment,
    InstanceObjectiveAssignment,
    ObjectiveSpec,
    OutcomeObjectiveAssignment,
    RewardObservation,
)

# Reward data types
from synth_ai.data.rewards import (
    CalibrationExample,
    EventRewardRecord,
    GoldExample,
    OutcomeRewardRecord,
    RewardAggregates,
)

# Rubric definitions (user input structures)
from synth_ai.data.rubrics import (
    Criterion,
    CriterionExample,
    Rubric,
)
from synth_ai.data.traces import (
    BaseEvent,
    EnvironmentEvent,
    LMCAISEvent,
    RuntimeEvent,
    SessionEventMarkovBlanketMessage,
    SessionMessageContent,
    SessionTimeStep,
    SessionTrace,
    TimeRecord,
)

try:  # Require Rust-backed data models
    import synth_ai_py as _rust_data  # type: ignore
except Exception as exc:  # pragma: no cover - rust bindings required
    raise RuntimeError("synth_ai_py is required for synth_ai.data.") from exc

_RUST_EXPORTS = [
    "CriterionExample",
    "Criterion",
    "Rubric",
    "CriterionScoreData",
    "RubricAssignment",
    "Judgement",
    "ObjectiveSpec",
    "RewardObservation",
    "OutcomeObjectiveAssignment",
    "EventObjectiveAssignment",
    "InstanceObjectiveAssignment",
    "OutcomeRewardRecord",
    "EventRewardRecord",
    "RewardAggregates",
    "CalibrationExample",
    "GoldExample",
    "Artifact",
    "ContextOverride",
    "ContextOverrideStatus",
    "SessionTrace",
    "SessionTimeStep",
    "TracingEvent",
    "RuntimeEvent",
    "EnvironmentEvent",
    "LMCAISEvent",
    "SessionEventMarkovBlanketMessage",
    "SessionMessageContent",
    "TimeRecord",
    "LLMUsage",
    "LLMRequestParams",
    "LLMContentPart",
    "LLMMessage",
    "ToolCallSpec",
    "ToolCallResult",
    "LLMChunk",
    "LLMCallRecord",
]


def _is_constructible(cls: Any) -> bool:
    try:
        sig = inspect.signature(cls)
    except Exception:
        return False
    return bool(sig.parameters)


for _name in _RUST_EXPORTS:
    if not hasattr(_rust_data, _name):
        continue
    _cls = getattr(_rust_data, _name)
    if _is_constructible(_cls):
        globals()[_name if _name != "TracingEvent" else "BaseEvent"] = _cls

__all__ = [
    # Enums
    "JobType",
    "JobStatus",
    "SuccessStatus",
    "InferenceMode",
    "ProviderName",
    "RewardSource",
    "RewardType",
    "RewardScope",
    "ObjectiveKey",
    "ObjectiveDirection",
    "GraphType",
    "OptimizationMode",
    "VerifierMode",
    "TrainingType",
    "AdaptiveCurriculumLevel",
    "AdaptiveBatchLevel",
    "OutputMode",
    # Objectives
    "ObjectiveSpec",
    "OBJECTIVE_REGISTRY",
    "RewardObservation",
    "OutcomeObjectiveAssignment",
    "EventObjectiveAssignment",
    "InstanceObjectiveAssignment",
    # Judgements
    "CriterionScoreData",
    "RubricAssignment",
    "Judgement",
    # Reward data
    "OutcomeRewardRecord",
    "EventRewardRecord",
    "RewardAggregates",
    "CalibrationExample",
    "GoldExample",
    # Artifacts
    "Artifact",
    # Context override types
    "ContextOverride",
    "ContextOverrideStatus",
    "OverrideApplicationError",
    "OverrideType",
    "OverrideOperation",
    "FolderMode",
    "ApplicationStatus",
    "ApplicationErrorType",
    # Trace data
    "SessionTrace",
    "SessionTimeStep",
    "BaseEvent",
    "RuntimeEvent",
    "EnvironmentEvent",
    "LMCAISEvent",
    "SessionEventMarkovBlanketMessage",
    "SessionMessageContent",
    "TimeRecord",
    # LLM call records
    "LLMUsage",
    "LLMRequestParams",
    "LLMContentPart",
    "LLMMessage",
    "ToolCallSpec",
    "ToolCallResult",
    "LLMChunk",
    "LLMCallRecord",
    # Rubrics
    "CriterionExample",
    "Criterion",
    "Rubric",
    "CriterionExample",
]
