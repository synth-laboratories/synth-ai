"""Shared optimization event schemas (OpenResponses-aligned).

DEPRECATED: This module has moved to synth_ai.sdk.shared.orchestration.events.
This re-export is provided for backwards compatibility.

New imports should use:
    from synth_ai.sdk.shared.orchestration.events import ...
"""

from __future__ import annotations

# Re-export everything from the new canonical location
from synth_ai.sdk.shared.orchestration.events import (
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
    EventSchemaRegistry,
    JobEvent,
    JobEventType,
    JobStatus,
    ValidationError,
    ValidationResult,
    get_event_type,
    get_registry,
    is_failure_event,
    is_success_event,
    is_terminal_event,
    parse_event,
    register_algorithm_schema,
    validate_base_event,
    validate_event,
    validate_event_strict,
    validate_typed_event,
)

__all__ = [
    # Enums
    "EventLevel",
    "JobEventType",
    "CandidateStatus",
    "JobStatus",
    # Event classes
    "BaseJobEvent",
    "JobEvent",
    "CandidateEvent",
    # JSON Schemas
    "BASE_JOB_EVENT_SCHEMA",
    "JOB_STARTED_EVENT_SCHEMA",
    "JOB_COMPLETED_EVENT_SCHEMA",
    "JOB_FAILED_EVENT_SCHEMA",
    "CANDIDATE_ADDED_EVENT_SCHEMA",
    "CANDIDATE_EVALUATED_EVENT_SCHEMA",
    "CANDIDATE_COMPLETED_EVENT_SCHEMA",
    "BASE_EVENT_SCHEMAS",
    # Parser utilities
    "parse_event",
    "is_terminal_event",
    "is_success_event",
    "is_failure_event",
    "get_event_type",
    # Schema registry
    "EventSchemaRegistry",
    "get_registry",
    "register_algorithm_schema",
    # Validation
    "ValidationError",
    "ValidationResult",
    "validate_base_event",
    "validate_event",
    "validate_event_strict",
    "validate_typed_event",
]
