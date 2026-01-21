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
    "Criterion",
    "Rubric",
]
