"""OpenResponses-aligned event schemas and utilities.

This package provides the single source of truth for event schemas used across
Python (synth-ai SDK, backend) and Rust (rust_backend) codebases. It includes:

- **Base event structures**: Typed dataclasses for job and candidate lifecycle events
- **JSON Schema definitions**: For code generation and validation
- **Event types**: Comprehensive taxonomy of all event types
- **Data schemas**: ProgramCandidate, StageInfo, TokenUsage, etc.
- **Parser utilities**: Convert raw backend events to typed event objects
- **Schema registry**: Manage algorithm-specific schema extensions
- **Validation**: Lightweight and strict JSON Schema validation
- **Extraction**: Stage and content extraction from candidate objects

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

# Base event types and dataclasses (OpenResponses core)
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
    EventLevel,
    JobEvent,
    JobEventType,
)

# Extraction utilities (from backend)
from .extraction import (
    StageExtractionError,
    build_program_candidate,
    extract_program_candidate_content,
    extract_prompt_text_from_candidate,
    extract_stages_from_candidate,
    extract_stages_required,
    normalize_transformation,
    seed_reward_entry,
    seed_score_entry,
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

# Data schemas (from backend)
from .schemas import (
    MAX_INSTRUCTION_LENGTH,
    MAX_ROLLOUT_SAMPLES,
    MAX_SEED_INFO_COUNT,
    BaseCandidateEventData,
    MutationSummary,
    MutationTypeStats,
    PhaseSummary,
    ProgramCandidate,
    SeedAnalysis,
    SeedInfo,
    StageInfo,
    TokenUsage,
)

# Extended type definitions (from backend)
from .types import (
    CandidateStatus,
    ErrorType,
    EventType,
    JobStatus,
    MutationType,
    Phase,
    TerminationReason,
    is_valid_event_type,
    validate_event_type,
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
    # ==========================================================================
    # Enums (OpenResponses core)
    # ==========================================================================
    "EventLevel",
    "JobEventType",
    # ==========================================================================
    # Enums (Extended from backend)
    # ==========================================================================
    "JobStatus",
    "Phase",
    "CandidateStatus",
    "MutationType",
    "TerminationReason",
    "ErrorType",
    "EventType",
    # ==========================================================================
    # Event classes (OpenResponses core)
    # ==========================================================================
    "BaseJobEvent",
    "JobEvent",
    "CandidateEvent",
    # ==========================================================================
    # JSON Schemas
    # ==========================================================================
    "BASE_JOB_EVENT_SCHEMA",
    "JOB_STARTED_EVENT_SCHEMA",
    "JOB_COMPLETED_EVENT_SCHEMA",
    "JOB_FAILED_EVENT_SCHEMA",
    "CANDIDATE_ADDED_EVENT_SCHEMA",
    "CANDIDATE_EVALUATED_EVENT_SCHEMA",
    "CANDIDATE_COMPLETED_EVENT_SCHEMA",
    "BASE_EVENT_SCHEMAS",
    # ==========================================================================
    # Data Schemas (from backend)
    # ==========================================================================
    # Constants
    "MAX_INSTRUCTION_LENGTH",
    "MAX_ROLLOUT_SAMPLES",
    "MAX_SEED_INFO_COUNT",
    # Summary classes
    "MutationTypeStats",
    "MutationSummary",
    "SeedAnalysis",
    "PhaseSummary",
    # Stage/seed classes
    "StageInfo",
    "SeedInfo",
    "TokenUsage",
    # Program candidate
    "ProgramCandidate",
    # Base event data (for subclassing)
    "BaseCandidateEventData",
    # ==========================================================================
    # Extraction utilities (from backend)
    # ==========================================================================
    "StageExtractionError",
    "seed_reward_entry",
    "seed_score_entry",
    "extract_stages_from_candidate",
    "extract_stages_required",
    "extract_program_candidate_content",
    "extract_prompt_text_from_candidate",
    "normalize_transformation",
    "build_program_candidate",
    # ==========================================================================
    # Parser utilities
    # ==========================================================================
    "parse_event",
    "is_terminal_event",
    "is_success_event",
    "is_failure_event",
    "get_event_type",
    # ==========================================================================
    # Event type validation
    # ==========================================================================
    "is_valid_event_type",
    "validate_event_type",
    # ==========================================================================
    # Schema registry
    # ==========================================================================
    "EventSchemaRegistry",
    "get_registry",
    "register_algorithm_schema",
    # ==========================================================================
    # Validation
    # ==========================================================================
    "ValidationError",
    "ValidationResult",
    "validate_base_event",
    "validate_event",
    "validate_event_strict",
    "validate_typed_event",
]
