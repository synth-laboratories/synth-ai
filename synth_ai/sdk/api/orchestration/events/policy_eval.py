"""Policy evaluation job event schemas."""

from __future__ import annotations

from typing import Any, Dict, Optional

from .base import BASE_EVENT_SCHEMAS, make_event_schema
from .registry import merge_event_schemas
from .validation import SchemaValidationError
from .validation import validate_event as _validate_schema

POLICY_EVAL_STARTED_SCHEMA = make_event_schema(
    "eval:policy.started",
    data_properties={
        "policy_id": {"type": "string"},
        "status": {"type": "string"},
    },
)

POLICY_EVAL_COMPLETED_SCHEMA = make_event_schema(
    "eval:policy.completed",
    data_properties={
        "policy_id": {"type": "string"},
        "status": {"type": "string"},
        "score": {"type": "number"},
    },
)

POLICY_EVAL_FAILED_SCHEMA = make_event_schema(
    "eval:policy.failed",
    data_properties={
        "policy_id": {"type": "string"},
        "status": {"type": "string"},
        "error": {"type": "string"},
    },
)


POLICY_EVAL_EVENT_SCHEMAS = merge_event_schemas(
    BASE_EVENT_SCHEMAS,
    {
        "eval:policy.started": POLICY_EVAL_STARTED_SCHEMA,
        "eval:policy.completed": POLICY_EVAL_COMPLETED_SCHEMA,
        "eval:policy.failed": POLICY_EVAL_FAILED_SCHEMA,
    },
)


def get_event_schema(event_type: str) -> Optional[Dict[str, Any]]:
    """Get schema for a specific event type."""
    return POLICY_EVAL_EVENT_SCHEMAS.get(event_type)


def validate_event(event: Dict[str, Any], event_type: str) -> bool:
    """Validate an event payload against the policy eval schemas."""
    schema = get_event_schema(event_type)
    if not schema:
        return False
    try:
        _validate_schema(event, schema)
    except SchemaValidationError:
        return False
    return True
