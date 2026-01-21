"""OpenResponses-aligned event schema scaffolding.

DEPRECATED: This module has moved to synth_ai.sdk.shared.orchestration.events.base.
This re-export is provided for backwards compatibility.
"""

from __future__ import annotations

# Re-export everything from the new canonical location
from synth_ai.sdk.shared.orchestration.events.base import (
    BASE_EVENT_SCHEMAS,
    BASE_JOB_EVENT_SCHEMA,
    CANDIDATE_ADDED_EVENT_SCHEMA,
    CANDIDATE_COMPLETED_EVENT_SCHEMA,
    CANDIDATE_EVALUATED_EVENT_SCHEMA,
    JOB_COMPLETED_EVENT_SCHEMA,
    JOB_FAILED_EVENT_SCHEMA,
    JOB_STARTED_EVENT_SCHEMA,
    BaseJobEvent,
    CandidateEvent,
    CandidateStatus,
    EventLevel,
    JobEvent,
    JobEventType,
    JobStatus,
)

__all__ = [
    "EventLevel",
    "JobEventType",
    "CandidateStatus",
    "JobStatus",
    "BaseJobEvent",
    "JobEvent",
    "CandidateEvent",
    "BASE_JOB_EVENT_SCHEMA",
    "JOB_STARTED_EVENT_SCHEMA",
    "JOB_COMPLETED_EVENT_SCHEMA",
    "JOB_FAILED_EVENT_SCHEMA",
    "CANDIDATE_ADDED_EVENT_SCHEMA",
    "CANDIDATE_EVALUATED_EVENT_SCHEMA",
    "CANDIDATE_COMPLETED_EVENT_SCHEMA",
    "BASE_EVENT_SCHEMAS",
]
