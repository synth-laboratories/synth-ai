"""Optimization SDK namespace (canonical policy v1 + graph).

# See: specifications/daily/feb24_2026/tinker_synth_final.md
"""

from __future__ import annotations

from synth_ai.sdk.shared.orchestration.events import (
    BaseJobEvent,
    CandidateEvent,
    JobEvent,
    JobEventType,
    is_failure_event,
    is_success_event,
    is_terminal_event,
    parse_event,
)

from .graph import (
    GraphOptimizationJob,
    GraphOptimizationJobConfig,
    GraphOptimizationResult,
)
from .job import JobLifecycle, JobStatus
from .policy import (
    GepaOnlineSession,
    MiproOnlineSession,
    PolicyOptimizationOfflineJob,
    PolicyOptimizationOnlineSession,
    PolicyOptimizationSystem,
)

__all__ = [
    # Policy optimization (canonical v1)
    "PolicyOptimizationOfflineJob",
    "PolicyOptimizationOnlineSession",
    "PolicyOptimizationSystem",
    "GepaOnlineSession",
    "MiproOnlineSession",
    # Graph optimization
    "GraphOptimizationJob",
    "GraphOptimizationJobConfig",
    "GraphOptimizationResult",
    # Event types
    "BaseJobEvent",
    "CandidateEvent",
    "JobEvent",
    "JobEventType",
    # Parser utilities
    "parse_event",
    "is_terminal_event",
    "is_success_event",
    "is_failure_event",
    # Job lifecycle
    "JobLifecycle",
    "JobStatus",
]
