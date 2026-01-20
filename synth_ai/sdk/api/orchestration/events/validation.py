"""JSON Schema validation helpers for job events."""

from __future__ import annotations

from typing import Any, Mapping

try:
    import jsonschema
except ImportError:  # pragma: no cover - optional dependency
    jsonschema = None


class SchemaValidationError(ValueError):
    """Raised when an event fails schema validation."""


def validate_event(event: Mapping[str, Any], schema: Mapping[str, Any]) -> None:
    """Validate an event payload against a JSON Schema."""
    if jsonschema is None:
        raise RuntimeError("jsonschema is not installed; cannot validate events")
    try:
        jsonschema.validate(event, schema)
    except jsonschema.ValidationError as exc:
        raise SchemaValidationError(str(exc)) from exc
