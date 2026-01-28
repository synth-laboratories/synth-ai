"""Synth AI Data Layer.

This module provides pure data types with no IO dependencies.
Contains actual data schemas for traces, rewards, rubrics, and LLM calls.

Data vs SDK distinction:
- data/: Pure data records (traces, rewards, rubrics, llm_calls) - actual data
- sdk/: API abstractions (jobs, training, graphs) - SDK interfaces

Dependency rule: data/ imports nothing from synth_ai except typing helpers.
"""

from __future__ import annotations

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

# Trace data types
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

CriterionExample = _rust_data.CriterionExample  # noqa: F811
Criterion = _rust_data.Criterion  # noqa: F811
Rubric = _rust_data.Rubric  # noqa: F811
CriterionScoreData = _rust_data.CriterionScoreData  # noqa: F811
RubricAssignment = _rust_data.RubricAssignment  # noqa: F811
Judgement = _rust_data.Judgement  # noqa: F811
ObjectiveSpec = _rust_data.ObjectiveSpec  # noqa: F811
RewardObservation = _rust_data.RewardObservation  # noqa: F811
OutcomeObjectiveAssignment = _rust_data.OutcomeObjectiveAssignment  # noqa: F811
EventObjectiveAssignment = _rust_data.EventObjectiveAssignment  # noqa: F811
InstanceObjectiveAssignment = _rust_data.InstanceObjectiveAssignment  # noqa: F811
OutcomeRewardRecord = _rust_data.OutcomeRewardRecord  # noqa: F811
EventRewardRecord = _rust_data.EventRewardRecord  # noqa: F811
RewardAggregates = _rust_data.RewardAggregates  # noqa: F811
CalibrationExample = _rust_data.CalibrationExample  # noqa: F811
GoldExample = _rust_data.GoldExample  # noqa: F811
Artifact = _rust_data.Artifact  # noqa: F811
ContextOverride = _rust_data.ContextOverride  # noqa: F811
ContextOverrideStatus = _rust_data.ContextOverrideStatus  # noqa: F811
SessionTrace = _rust_data.SessionTrace  # noqa: F811
SessionTimeStep = _rust_data.SessionTimeStep  # noqa: F811
BaseEvent = _rust_data.TracingEvent  # noqa: F811
RuntimeEvent = _rust_data.RuntimeEvent  # noqa: F811
EnvironmentEvent = _rust_data.EnvironmentEvent  # noqa: F811
LMCAISEvent = _rust_data.LMCAISEvent  # noqa: F811
SessionEventMarkovBlanketMessage = _rust_data.SessionEventMarkovBlanketMessage  # noqa: F811
SessionMessageContent = _rust_data.SessionMessageContent  # noqa: F811
TimeRecord = _rust_data.TimeRecord  # noqa: F811
LLMUsage = _rust_data.LLMUsage  # noqa: F811
LLMRequestParams = _rust_data.LLMRequestParams  # noqa: F811
LLMContentPart = _rust_data.LLMContentPart  # noqa: F811
LLMMessage = _rust_data.LLMMessage  # noqa: F811
ToolCallSpec = _rust_data.ToolCallSpec  # noqa: F811
ToolCallResult = _rust_data.ToolCallResult  # noqa: F811
LLMChunk = _rust_data.LLMChunk  # noqa: F811
LLMCallRecord = _rust_data.LLMCallRecord  # noqa: F811

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
]
