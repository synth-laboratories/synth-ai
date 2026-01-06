"""Synth AI Data Layer.

This module provides pure data types with no IO dependencies.
Contains actual data schemas for traces and rewards.

Data vs SDK distinction:
- data/: Pure data records (traces, rewards) - actual data
- sdk/: API abstractions (jobs, training, graphs) - SDK interfaces

Dependency rule: data/ imports nothing from synth_ai except typing helpers.
"""

from __future__ import annotations

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
    PromptLearningMethod,
    ProviderName,
    RewardScope,
    RewardSource,
    RewardType,
    RLMethod,
    SFTMethod,
    TrainingType,
    VerifierMode,
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

# Judgements
from synth_ai.data.judgements import (
    CriterionScoreData,
    Judgement,
    RubricAssignment,
)

# Objective compatibility helpers
from synth_ai.data.objectives_compat import (
    extract_instance_rewards,
    extract_outcome_reward,
    normalize_to_event_objectives,
    normalize_to_outcome_objectives,
    to_legacy_format,
)
# Reward data types
from synth_ai.data.rewards import (
    CalibrationExample,
    EventRewardRecord,
    GoldExample,
    OutcomeRewardRecord,
    RewardAggregates,
)

# Trace data types (re-exports from tracing_v3)
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
    "PromptLearningMethod",
    "RLMethod",
    "SFTMethod",
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
    "ObjectiveSpec",
    "OBJECTIVE_REGISTRY",
    "RewardObservation",
    "OutcomeObjectiveAssignment",
    "EventObjectiveAssignment",
    "InstanceObjectiveAssignment",
    "CriterionScoreData",
    "RubricAssignment",
    "Judgement",
    "extract_outcome_reward",
    "extract_instance_rewards",
    "normalize_to_outcome_objectives",
    "normalize_to_event_objectives",
    "to_legacy_format",
    # Reward data
    "OutcomeRewardRecord",
    "EventRewardRecord",
    "RewardAggregates",
    "CalibrationExample",
    "GoldExample",
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
]
