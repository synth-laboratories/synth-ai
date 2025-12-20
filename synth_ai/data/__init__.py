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
    InferenceMode,
    JobStatus,
    JobType,
    PromptLearningMethod,
    ProviderName,
    RewardSource,
    RLMethod,
    SFTMethod,
)

# Reward data types
from synth_ai.data.rewards import (
    CalibrationExample,
    EventRewardRecord,
    GoldExample,
    OutcomeRewardRecord,
    RewardAggregates,
    RewardRecord,
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
    "AdaptiveCurriculumLevel",
    "AdaptiveBatchLevel",
    # Reward data
    "RewardRecord",
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


