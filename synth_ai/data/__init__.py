"""Synth AI Data Layer.

This module provides pure data types with no IO dependencies.
Contains actual data schemas for traces, rewards, and specs.

Data vs SDK distinction:
- data/: Pure data records (traces, rewards, specs) - actual data
- sdk/: API abstractions (jobs, training, judging) - SDK interfaces

Dependency rule: data/ imports nothing from synth_ai except typing helpers.
"""

from __future__ import annotations

# Enums - pure types
from synth_ai.data.enums import (
    AdaptiveBatchLevel,
    AdaptiveCurriculumLevel,
    ContainerBackend,
    InferenceMode,
    JobStatus,
    JobType,
    PromptLearningMethod,
    ProviderName,
    ResearchAgentAlgorithm,
    RewardSource,
    RLMethod,
    SFTMethod,
)

# Reward data types
from synth_ai.data.rewards import (
    EventRewardRecord,
    OutcomeRewardRecord,
    RewardAggregates,
    RewardRecord,
)

# Spec data types (re-exports)
from synth_ai.data.specs import (
    Constraints,
    Example,
    GlossaryItem,
    Interfaces,
    Metadata,
    Principle,
    Rule,
    Spec,
    TestCase,
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
    "ResearchAgentAlgorithm",
    "ContainerBackend",
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
    # Spec data
    "Spec",
    "Metadata",
    "Principle",
    "Rule",
    "Constraints",
    "Example",
    "TestCase",
    "Interfaces",
    "GlossaryItem",
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


