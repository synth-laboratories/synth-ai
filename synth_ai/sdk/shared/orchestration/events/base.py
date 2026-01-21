"""OpenResponses-aligned event schema definitions.

This module defines the base event structure and JSON Schema definitions for
job and candidate lifecycle events. These schemas are the single source of truth
for event structures across Python (backend) and Rust (rust_backend) codebases.

Event Structure (OpenResponses-aligned):
- job_id: str - Job identifier
- seq: int - Monotonically increasing sequence number within a job
- ts: str - ISO 8601 timestamp
- type: str - Event type (e.g., "job.started", "gepa:candidate.evaluated")
- level: str - Log level ("info", "warn", "error")
- message: str - Human-readable event description
- data: dict - Event-specific payload
- run_id: str | None - Optional run identifier for grouping related events
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from typing import Any, Dict, Optional

# Import comprehensive enums from types module
from .types import CandidateStatus, JobStatus


class EventLevel(str, Enum):
    """Event severity level."""

    INFO = "info"
    WARN = "warn"
    ERROR = "error"


class JobEventType(str, Enum):
    """Canonical lifecycle events for optimization jobs.

    First-class entities (Job, Candidate) have explicit lifecycle events.
    Algorithm-specific concepts (Generation, Phase, Trial) remain implicit in event data.
    """

    # Job lifecycle (first-class, stable across algorithms)
    JOB_STARTED = "job.started"
    JOB_IN_PROGRESS = "job.in_progress"
    JOB_COMPLETED = "job.completed"
    JOB_FAILED = "job.failed"
    JOB_CANCELLED = "job.cancelled"

    # Candidate lifecycle (first-class, stable across algorithms)
    CANDIDATE_ADDED = "candidate.added"
    CANDIDATE_EVALUATED = "candidate.evaluated"
    CANDIDATE_COMPLETED = "candidate.completed"


@dataclass
class BaseJobEvent:
    """Base event structure with all OpenResponses-aligned fields.

    All job events inherit from this base class which provides:
    - Core identification (job_id, seq, ts)
    - Event classification (type, level, message)
    - Flexible payload (data)
    - Optional correlation (run_id)
    """

    job_id: str
    seq: int
    ts: str  # ISO 8601 timestamp
    type: str  # Event type string (e.g., "job.started", "gepa:candidate.evaluated")
    level: EventLevel
    message: str
    data: Dict[str, Any] = field(default_factory=dict)
    run_id: Optional[str] = None

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> BaseJobEvent:
        """Parse a raw event dictionary into a BaseJobEvent."""
        return cls(
            job_id=payload.get("job_id", ""),
            seq=int(payload.get("seq", 0)),
            ts=payload.get("ts", datetime.now(UTC).isoformat()),
            type=payload.get("type", ""),
            level=EventLevel(payload.get("level", "info")),
            message=payload.get("message", ""),
            data=dict(payload.get("data") or {}),
            run_id=payload.get("run_id"),
        )

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary for SSE/JSON transmission."""
        result = {
            "job_id": self.job_id,
            "seq": self.seq,
            "ts": self.ts,
            "type": self.type,
            "level": self.level.value,
            "message": self.message,
            "data": self.data,
        }
        if self.run_id is not None:
            result["run_id"] = self.run_id
        return result


@dataclass
class JobEvent(BaseJobEvent):
    """Job lifecycle event with status tracking."""

    status: Optional[JobStatus] = None

    def to_dict(self) -> Dict[str, Any]:
        result = super().to_dict()
        if self.status is not None:
            result["data"]["status"] = self.status.value
        return result


@dataclass
class CandidateEvent(BaseJobEvent):
    """Candidate lifecycle event with candidate identification and status."""

    candidate_id: Optional[str] = None
    status: Optional[CandidateStatus] = None

    def to_dict(self) -> Dict[str, Any]:
        result = super().to_dict()
        if self.candidate_id is not None:
            result["data"]["candidate_id"] = self.candidate_id
        if self.status is not None:
            result["data"]["status"] = self.status.value
        return result


# =============================================================================
# JSON Schema Definitions (for code generation and validation)
# =============================================================================

BASE_JOB_EVENT_SCHEMA: Dict[str, Any] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "$id": "https://synth.ai/schemas/events/base-job-event.json",
    "title": "BaseJobEvent",
    "description": "Base event structure for all job-related events",
    "type": "object",
    "required": ["job_id", "seq", "ts", "type", "level", "message"],
    "properties": {
        "job_id": {
            "type": "string",
            "description": "Unique identifier for the job",
        },
        "seq": {
            "type": "integer",
            "minimum": 0,
            "description": "Monotonically increasing sequence number within a job",
        },
        "ts": {
            "type": "string",
            "format": "date-time",
            "description": "ISO 8601 timestamp of when the event occurred",
        },
        "type": {
            "type": "string",
            "description": "Event type identifier (e.g., 'job.started', 'gepa:candidate.evaluated')",
        },
        "level": {
            "type": "string",
            "enum": ["info", "warn", "error"],
            "description": "Event severity level",
        },
        "message": {
            "type": "string",
            "description": "Human-readable event description",
        },
        "data": {
            "type": "object",
            "description": "Event-specific payload data",
            "additionalProperties": True,
        },
        "run_id": {
            "type": ["string", "null"],
            "description": "Optional run identifier for grouping related events",
        },
    },
    "additionalProperties": False,
}

JOB_STARTED_EVENT_SCHEMA: Dict[str, Any] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "$id": "https://synth.ai/schemas/events/job-started.json",
    "title": "JobStartedEvent",
    "description": "Event emitted when a job starts execution",
    "allOf": [
        {"$ref": "base-job-event.json"},
        {
            "properties": {
                "type": {"const": "job.started"},
                "data": {
                    "type": "object",
                    "properties": {
                        "job_type": {"type": "string"},
                        "algorithm": {"type": "string"},
                        "status": {"type": "string", "enum": ["in_progress"]},
                    },
                },
            },
        },
    ],
}

JOB_COMPLETED_EVENT_SCHEMA: Dict[str, Any] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "$id": "https://synth.ai/schemas/events/job-completed.json",
    "title": "JobCompletedEvent",
    "description": "Event emitted when a job completes successfully",
    "allOf": [
        {"$ref": "base-job-event.json"},
        {
            "properties": {
                "type": {"const": "job.completed"},
                "data": {
                    "type": "object",
                    "properties": {
                        "status": {"type": "string", "enum": ["completed"]},
                        "best_candidate_id": {"type": ["string", "null"]},
                        "best_score": {"type": ["number", "null"]},
                        "total_candidates": {"type": "integer"},
                        "total_cost_usd": {"type": ["number", "null"]},
                        "duration_s": {"type": ["number", "null"]},
                    },
                },
            },
        },
    ],
}

JOB_FAILED_EVENT_SCHEMA: Dict[str, Any] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "$id": "https://synth.ai/schemas/events/job-failed.json",
    "title": "JobFailedEvent",
    "description": "Event emitted when a job fails",
    "allOf": [
        {"$ref": "base-job-event.json"},
        {
            "properties": {
                "type": {"const": "job.failed"},
                "level": {"const": "error"},
                "data": {
                    "type": "object",
                    "properties": {
                        "status": {"type": "string", "enum": ["failed"]},
                        "error": {"type": "string"},
                        "error_type": {"type": "string"},
                    },
                },
            },
        },
    ],
}

CANDIDATE_ADDED_EVENT_SCHEMA: Dict[str, Any] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "$id": "https://synth.ai/schemas/events/candidate-added.json",
    "title": "CandidateAddedEvent",
    "description": "Event emitted when a new candidate is created",
    "allOf": [
        {"$ref": "base-job-event.json"},
        {
            "properties": {
                "type": {"const": "candidate.added"},
                "data": {
                    "type": "object",
                    "required": ["candidate_id", "status"],
                    "properties": {
                        "candidate_id": {"type": "string"},
                        "status": {"type": "string", "enum": ["in_progress"]},
                        "version_id": {"type": "string"},
                        "parent_id": {"type": ["string", "null"]},
                        "generation": {"type": "integer"},
                        "mutation_type": {"type": "string"},
                    },
                },
            },
        },
    ],
}

CANDIDATE_EVALUATED_EVENT_SCHEMA: Dict[str, Any] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "$id": "https://synth.ai/schemas/events/candidate-evaluated.json",
    "title": "CandidateEvaluatedEvent",
    "description": "Event emitted when a candidate has been evaluated",
    "allOf": [
        {"$ref": "base-job-event.json"},
        {
            "properties": {
                "type": {"const": "candidate.evaluated"},
                "data": {
                    "type": "object",
                    "required": ["candidate_id", "reward"],
                    "properties": {
                        "candidate_id": {"type": "string"},
                        "reward": {"type": "number"},
                        "status": {"type": "string", "enum": ["completed", "failed"]},
                        "version_id": {"type": "string"},
                        "parent_id": {"type": ["string", "null"]},
                        "generation": {"type": "integer"},
                        "mutation_type": {"type": "string"},
                        "is_pareto": {"type": "boolean"},
                    },
                },
            },
        },
    ],
}

CANDIDATE_COMPLETED_EVENT_SCHEMA: Dict[str, Any] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "$id": "https://synth.ai/schemas/events/candidate-completed.json",
    "title": "CandidateCompletedEvent",
    "description": "Event emitted when a candidate lifecycle is complete",
    "allOf": [
        {"$ref": "base-job-event.json"},
        {
            "properties": {
                "type": {"const": "candidate.completed"},
                "data": {
                    "type": "object",
                    "required": ["candidate_id", "status"],
                    "properties": {
                        "candidate_id": {"type": "string"},
                        "status": {
                            "type": "string",
                            "enum": ["completed", "rejected", "accepted"],
                        },
                    },
                },
            },
        },
    ],
}

# Collection of all base schemas
BASE_EVENT_SCHEMAS: Dict[str, Dict[str, Any]] = {
    "base": BASE_JOB_EVENT_SCHEMA,
    "job.started": JOB_STARTED_EVENT_SCHEMA,
    "job.completed": JOB_COMPLETED_EVENT_SCHEMA,
    "job.failed": JOB_FAILED_EVENT_SCHEMA,
    "candidate.added": CANDIDATE_ADDED_EVENT_SCHEMA,
    "candidate.evaluated": CANDIDATE_EVALUATED_EVENT_SCHEMA,
    "candidate.completed": CANDIDATE_COMPLETED_EVENT_SCHEMA,
}


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
]
