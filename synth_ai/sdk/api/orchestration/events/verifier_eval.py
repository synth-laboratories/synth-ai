"""Verifier evaluation job event schemas."""

from __future__ import annotations

from typing import Any, Dict, Optional

from .base import BASE_EVENT_SCHEMAS, make_event_schema
from .registry import merge_event_schemas
from .validation import SchemaValidationError
from .validation import validate_event as _validate_schema

VERIFIER_EVAL_STARTED_SCHEMA = make_event_schema(
    "eval:verifier.started",
    data_properties={
        "verifier_id": {"type": "string"},
        "status": {"type": "string"},
    },
)

VERIFIER_EVAL_COMPLETED_SCHEMA = make_event_schema(
    "eval:verifier.completed",
    data_properties={
        "verifier_id": {"type": "string"},
        "status": {"type": "string"},
        "score": {"type": "number"},
    },
)

VERIFIER_EVAL_FAILED_SCHEMA = make_event_schema(
    "eval:verifier.failed",
    data_properties={
        "verifier_id": {"type": "string"},
        "status": {"type": "string"},
        "error": {"type": "string"},
    },
)


VERIFIER_EVAL_EVENT_SCHEMAS = merge_event_schemas(
    BASE_EVENT_SCHEMAS,
    {
        "eval:verifier.started": VERIFIER_EVAL_STARTED_SCHEMA,
        "eval:verifier.completed": VERIFIER_EVAL_COMPLETED_SCHEMA,
        "eval:verifier.failed": VERIFIER_EVAL_FAILED_SCHEMA,
    },
)


def get_event_schema(event_type: str) -> Optional[Dict[str, Any]]:
    """Get schema for a specific event type."""
    return VERIFIER_EVAL_EVENT_SCHEMAS.get(event_type)


def validate_event(event: Dict[str, Any], event_type: str) -> bool:
    """Validate an event payload against the verifier eval schemas."""
    schema = get_event_schema(event_type)
    if not schema:
        return False
    try:
        _validate_schema(event, schema)
    except SchemaValidationError:
        return False
    return True
