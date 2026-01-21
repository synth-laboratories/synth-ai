"""OpenResponses-aligned event schemas and utilities.

This package provides the single source of truth for event schemas used across
Python (synth-ai SDK) and Rust (rust_backend) codebases. It includes:

- **Base event structures**: Typed dataclasses for job and candidate lifecycle events
- **JSON Schema definitions**: For code generation and validation
- **Parser utilities**: Convert raw backend events to typed event objects
- **Schema registry**: Manage algorithm-specific schema extensions
- **Validation**: Lightweight and strict JSON Schema validation

Event Structure (OpenResponses-aligned):
    - job_id: str - Job identifier
    - seq: int - Monotonically increasing sequence number within a job
    - ts: str - ISO 8601 timestamp
    - type: str - Event type (e.g., "job.started", "gepa:candidate.evaluated")
    - level: str - Log level ("info", "warn", "error")
    - message: str - Human-readable event description
    - data: dict - Event-specific payload
    - run_id: str | None - Optional run identifier for grouping related events

Example:
    >>> from synth_ai.sdk.shared.orchestration.events import parse_event, is_terminal_event
    >>> raw = {"type": "job.completed", "job_id": "abc", "seq": 5, "ts": "2024-01-01T00:00:00Z", ...}
    >>> event = parse_event(raw)
    >>> is_terminal_event(event)
    True
"""

from __future__ import annotations

# Base event types and dataclasses
from .base import (
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

# Parser utilities
from .parser import (
    get_event_type,
    is_failure_event,
    is_success_event,
    is_terminal_event,
    parse_event,
)

# Schema registry
from .registry import (
    EventSchemaRegistry,
    get_registry,
    register_algorithm_schema,
)

# Validation
from .validation import (
    ValidationError,
    ValidationResult,
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
