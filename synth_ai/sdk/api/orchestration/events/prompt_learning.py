"""Prompt learning job event schemas (GEPA/MIPRO)."""

from __future__ import annotations

from typing import Any, Dict, Optional

from .base import BASE_EVENT_SCHEMAS, make_event_schema
from .registry import apply_aliases, merge_event_schemas
from .validation import SchemaValidationError
from .validation import validate_event as _validate_schema

CANDIDATE_EVALUATED_SCHEMA = make_event_schema(
    "candidate.evaluated",
    data_properties={
        "candidate_id": {"type": "string"},
        "reward": {"type": "number"},
        "status": {"type": "string", "enum": ["completed", "failed"]},
        "version_id": {"type": "string"},
        "parent_id": {"type": ["string", "null"]},
        "generation": {"type": "integer"},
        "mutation_type": {"type": "string"},
        "is_pareto": {"type": "boolean"},
    },
    data_required=["candidate_id", "reward", "status"],
)

GEPA_GENERATION_STARTED_SCHEMA = make_event_schema(
    "gepa:generation.started",
    data_properties={
        "generation": {"type": "integer"},
        "candidates_proposed": {"type": "integer"},
    },
    data_required=["generation"],
)

GEPA_GENERATION_COMPLETED_SCHEMA = make_event_schema(
    "gepa:generation.completed",
    data_properties={
        "generation": {"type": "integer"},
        "candidates_proposed": {"type": "integer"},
        "candidates_accepted": {"type": "integer"},
        "best_score": {"type": "number"},
    },
    data_required=["generation"],
)

GEPA_FRONTIER_UPDATED_SCHEMA = make_event_schema(
    "gepa:frontier.updated",
    data_properties={
        "frontier_size": {"type": "integer"},
        "best_score": {"type": "number"},
        "baseline_score": {"type": "number"},
        "added": {"type": "array"},
        "removed": {"type": "array"},
    },
)

MIPRO_TRIAL_STARTED_SCHEMA = make_event_schema(
    "mipro:trial.started",
    data_properties={
        "trial_num": {"type": "integer"},
        "iteration": {"type": "integer"},
    },
    data_required=["trial_num", "iteration"],
)

MIPRO_TRIAL_COMPLETED_SCHEMA = make_event_schema(
    "mipro:trial.completed",
    data_properties={
        "trial_num": {"type": "integer"},
        "iteration": {"type": "integer"},
        "candidate_id": {"type": "string"},
        "reward": {"type": "number"},
        "rank": {"type": ["integer", "null"]},
    },
    data_required=["trial_num", "iteration"],
)

MIPRO_ITERATION_STARTED_SCHEMA = make_event_schema(
    "mipro:iteration.started",
    data_properties={"iteration": {"type": "integer"}},
    data_required=["iteration"],
)

MIPRO_ITERATION_COMPLETED_SCHEMA = make_event_schema(
    "mipro:iteration.completed",
    data_properties={
        "iteration": {"type": "integer"},
        "best_score": {"type": "number"},
    },
    data_required=["iteration"],
)


PROMPT_LEARNING_EVENT_SCHEMAS = merge_event_schemas(
    BASE_EVENT_SCHEMAS,
    {
        "candidate.evaluated": CANDIDATE_EVALUATED_SCHEMA,
        "gepa:generation.started": GEPA_GENERATION_STARTED_SCHEMA,
        "gepa:generation.completed": GEPA_GENERATION_COMPLETED_SCHEMA,
        "gepa:frontier.updated": GEPA_FRONTIER_UPDATED_SCHEMA,
        "mipro:trial.started": MIPRO_TRIAL_STARTED_SCHEMA,
        "mipro:trial.completed": MIPRO_TRIAL_COMPLETED_SCHEMA,
        "mipro:iteration.started": MIPRO_ITERATION_STARTED_SCHEMA,
        "mipro:iteration.completed": MIPRO_ITERATION_COMPLETED_SCHEMA,
    },
)

PROMPT_LEARNING_ALIASES = {
    "gepa:candidate.evaluated": "candidate.evaluated",
}

EVENT_SCHEMAS = apply_aliases(PROMPT_LEARNING_EVENT_SCHEMAS, PROMPT_LEARNING_ALIASES)


def get_event_schema(event_type: str) -> Optional[Dict[str, Any]]:
    """Get schema for a specific event type."""
    return EVENT_SCHEMAS.get(event_type)


def validate_event(event: Dict[str, Any], event_type: str) -> bool:
    """Validate an event payload against the prompt learning schemas."""
    schema = get_event_schema(event_type)
    if not schema:
        return False
    try:
        _validate_schema(event, schema)
    except SchemaValidationError:
        return False
    return True
