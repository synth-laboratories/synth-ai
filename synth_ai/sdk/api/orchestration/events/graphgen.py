"""GraphGen job event schemas (graph completions + RLM variants)."""

from __future__ import annotations

from typing import Any, Dict, Optional

from .base import BASE_EVENT_SCHEMAS, make_event_schema
from .registry import apply_aliases, merge_event_schemas
from .validation import SchemaValidationError
from .validation import validate_event as _validate_schema

GRAPH_COMPLETION_STARTED_SCHEMA = make_event_schema(
    "graph:completion.started",
    data_properties={
        "graph_id": {"type": "string"},
        "status": {"type": "string"},
    },
)

GRAPH_COMPLETION_COMPLETED_SCHEMA = make_event_schema(
    "graph:completion.completed",
    data_properties={
        "graph_id": {"type": "string"},
        "status": {"type": "string"},
        "best_score": {"type": "number"},
    },
)

GRAPH_COMPLETION_ROLLOUT_COMPLETED_SCHEMA = make_event_schema(
    "graph:completion.rollout.completed",
    data_properties={
        "rollout_id": {"type": "string"},
        "reward": {"type": "number"},
        "status": {"type": "string"},
    },
)


GRAPHGEN_EVENT_SCHEMAS = merge_event_schemas(
    BASE_EVENT_SCHEMAS,
    {
        "graph:completion.started": GRAPH_COMPLETION_STARTED_SCHEMA,
        "graph:completion.completed": GRAPH_COMPLETION_COMPLETED_SCHEMA,
        "graph:completion.rollout.completed": GRAPH_COMPLETION_ROLLOUT_COMPLETED_SCHEMA,
    },
)

GRAPHGEN_ALIASES = {
    "graph:completion.rlm_v1:rollout.completed": "graph:completion.rollout.completed",
    "graph:completion.rlm_v2:rollout.completed": "graph:completion.rollout.completed",
}

EVENT_SCHEMAS = apply_aliases(GRAPHGEN_EVENT_SCHEMAS, GRAPHGEN_ALIASES)


def get_event_schema(event_type: str) -> Optional[Dict[str, Any]]:
    """Get schema for a specific event type."""
    return EVENT_SCHEMAS.get(event_type)


def validate_event(event: Dict[str, Any], event_type: str) -> bool:
    """Validate an event payload against the graphgen schemas."""
    schema = get_event_schema(event_type)
    if not schema:
        return False
    try:
        _validate_schema(event, schema)
    except SchemaValidationError:
        return False
    return True
