"""Optimization SDK namespace (canonical policy v1 + graph).

# See: specs/sdk_logic.md
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
    PolicyOptimizationOfflineJob,
    PolicyOptimizationOnlineSession,
    PolicyOptimizationSystem,
)

__all__ = [
    # Policy optimization (canonical v1)
    "PolicyOptimizationOfflineJob",
    "PolicyOptimizationOnlineSession",
    "PolicyOptimizationSystem",
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
