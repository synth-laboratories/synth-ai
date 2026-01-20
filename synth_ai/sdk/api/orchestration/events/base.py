"""Shared JSON Schema definitions for job events."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

JSON_SCHEMA_VERSION = "https://json-schema.org/draft/2020-12/schema"


BASE_JOB_EVENT_SCHEMA: Dict[str, Any] = {
    "$schema": JSON_SCHEMA_VERSION,
    "type": "object",
    "required": ["job_id", "seq", "ts", "type", "level", "message"],
    "properties": {
        "job_id": {"type": "string"},
        "seq": {"type": "integer", "minimum": 0},
        "ts": {"type": "string", "format": "date-time"},
        "type": {"type": "string"},
        "level": {"type": "string", "enum": ["info", "warn", "error"]},
        "message": {"type": "string"},
        "data": {"type": "object"},
        "run_id": {"type": ["string", "null"]},
    },
}


def _data_schema(
    properties: Optional[Dict[str, Any]] = None,
    required: Optional[List[str]] = None,
) -> Dict[str, Any]:
    schema: Dict[str, Any] = {
        "type": "object",
        "properties": properties or {},
        "additionalProperties": True,
    }
    if required:
        schema["required"] = required
    return schema


def make_event_schema(
    event_type: str,
    *,
    data_properties: Optional[Dict[str, Any]] = None,
    data_required: Optional[List[str]] = None,
    require_data: bool = False,
) -> Dict[str, Any]:
    """Build an event schema that composes the base schema."""
    event_schema: Dict[str, Any] = {
        "$schema": JSON_SCHEMA_VERSION,
        "allOf": [
            BASE_JOB_EVENT_SCHEMA,
            {"type": "object", "properties": {"type": {"const": event_type}}},
        ],
    }

    if data_properties is not None or data_required is not None or require_data:
        data_schema = _data_schema(data_properties, data_required)
        event_schema["allOf"][1]["properties"]["data"] = data_schema
        event_schema["allOf"][1]["required"] = ["data"]

    return event_schema


JOB_STARTED_SCHEMA = make_event_schema(
    "job.started",
    data_properties={
        "job_type": {"type": "string"},
        "algorithm": {"type": "string"},
    },
)

JOB_COMPLETED_SCHEMA = make_event_schema(
    "job.completed",
    data_properties={
        "status": {"type": "string"},
        "best_candidate_id": {"type": "string"},
        "best_score": {"type": "number"},
        "total_candidates": {"type": "integer"},
        "total_cost_usd": {"type": "number"},
        "duration_s": {"type": "number"},
    },
)

JOB_FAILED_SCHEMA = make_event_schema(
    "job.failed",
    data_properties={
        "status": {"type": "string"},
        "error": {"type": "string"},
        "error_type": {"type": "string"},
    },
)

CANDIDATE_ADDED_SCHEMA = make_event_schema(
    "candidate.added",
    data_properties={
        "candidate_id": {"type": "string"},
        "status": {"type": "string", "enum": ["in_progress"]},
        "version_id": {"type": "string"},
        "parent_id": {"type": ["string", "null"]},
        "generation": {"type": "integer"},
        "mutation_type": {"type": "string"},
    },
    data_required=["candidate_id", "status"],
)

CANDIDATE_COMPLETED_SCHEMA = make_event_schema(
    "candidate.completed",
    data_properties={
        "candidate_id": {"type": "string"},
        "status": {"type": "string", "enum": ["completed", "rejected", "accepted"]},
        "reward": {"type": "number"},
    },
    data_required=["candidate_id", "status"],
)


BASE_EVENT_SCHEMAS: Dict[str, Dict[str, Any]] = {
    "job.started": JOB_STARTED_SCHEMA,
    "job.completed": JOB_COMPLETED_SCHEMA,
    "job.failed": JOB_FAILED_SCHEMA,
    "candidate.added": CANDIDATE_ADDED_SCHEMA,
    "candidate.completed": CANDIDATE_COMPLETED_SCHEMA,
}
