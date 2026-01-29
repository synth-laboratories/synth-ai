"""Optimization SDK namespace (policy + graph).

This package provides the unified optimization surface for both policy optimization
(GEPA, MIPRO) and graph optimization (Graph-GEPA). Key features:

- **Policy Optimization**: `PolicyOptimizationJob` for prompt/instruction optimization
- **Typed Events**: OpenResponses-aligned event schemas for job/candidate lifecycle
- **Event Parsing**: Utilities to convert raw backend events to typed events
- **Job Lifecycle**: Utilities for tracking and emitting job lifecycle events
- **Stream Handlers**: Base classes for building typed event handlers

Example usage:
    >>> from synth_ai.sdk.optimization import PolicyOptimizationJob
    >>>
    >>> # Create and run a policy optimization job
    >>> job = PolicyOptimizationJob.from_config("config.toml")
    >>> job.submit()
    >>> result = job.stream_until_complete()
    >>> print(f"Best score: {result.best_score}")
    >>>
    >>> # Or use the event system directly
    >>> from synth_ai.sdk.optimization import parse_event, is_terminal_event
    >>> raw = {"type": "job.completed", "job_id": "abc", "data": {}}
    >>> event = parse_event(raw)
    >>> print(is_terminal_event(event))  # True
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
    MiproOnlineSession,
    PolicyOptimizationJob,
    PolicyOptimizationJobConfig,
    PolicyOptimizationResult,
)

__all__ = [
    # Policy optimization
    "PolicyOptimizationJob",
    "PolicyOptimizationJobConfig",
    "PolicyOptimizationResult",
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
