"""Event validation using JSON Schema.

This module provides validation utilities for event payloads against JSON Schemas
defined in base.py and registered in the schema registry. Validation is optional
at runtime but can be enabled for debugging or strict mode.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from .base import BASE_JOB_EVENT_SCHEMA, BaseJobEvent
from .registry import get_registry


@dataclass
class ValidationError:
    """Represents a validation error."""

    path: str
    message: str
    schema_path: Optional[str] = None

    def __str__(self) -> str:
        if self.path:
            return f"{self.path}: {self.message}"
        return self.message


@dataclass
class ValidationResult:
    """Result of event validation."""

    valid: bool
    errors: List[ValidationError]
    event_type: Optional[str] = None

    @classmethod
    def success(cls, event_type: Optional[str] = None) -> ValidationResult:
        return cls(valid=True, errors=[], event_type=event_type)

    @classmethod
    def failure(
        cls, errors: List[ValidationError], event_type: Optional[str] = None
    ) -> ValidationResult:
        return cls(valid=False, errors=errors, event_type=event_type)


def _validate_required_fields(
    data: Dict[str, Any], required: List[str], path: str = ""
) -> List[ValidationError]:
    """Validate that required fields are present."""
    errors = []
    for field in required:
        if field not in data:
            errors.append(
                ValidationError(
                    path=f"{path}.{field}" if path else field,
                    message=f"Required field '{field}' is missing",
                )
            )
    return errors


def _validate_field_type(value: Any, expected_type: str, path: str) -> Optional[ValidationError]:
    """Validate a field's type against JSON Schema type."""
    type_checks = {
        "string": lambda v: isinstance(v, str),
        "integer": lambda v: isinstance(v, int) and not isinstance(v, bool),
        "number": lambda v: isinstance(v, (int, float)) and not isinstance(v, bool),
        "boolean": lambda v: isinstance(v, bool),
        "object": lambda v: isinstance(v, dict),
        "array": lambda v: isinstance(v, list),
        "null": lambda v: v is None,
    }

    # Handle union types like ["string", "null"]
    if isinstance(expected_type, list):
        for t in expected_type:
            check = type_checks.get(t)
            if check and check(value):
                return None
        return ValidationError(
            path=path, message=f"Expected one of {expected_type}, got {type(value).__name__}"
        )

    check = type_checks.get(expected_type)
    if check and not check(value):
        return ValidationError(
            path=path, message=f"Expected {expected_type}, got {type(value).__name__}"
        )
    return None


def _validate_enum(value: Any, enum_values: List[Any], path: str) -> Optional[ValidationError]:
    """Validate a value against an enum constraint."""
    if value not in enum_values:
        return ValidationError(
            path=path, message=f"Value '{value}' not in allowed values: {enum_values}"
        )
    return None


def validate_base_event(event: Dict[str, Any]) -> ValidationResult:
    """Validate an event against the base job event schema.

    This performs lightweight validation without requiring jsonschema library.
    For full JSON Schema validation, use validate_event_strict().

    Args:
        event: Raw event dictionary

    Returns:
        ValidationResult with valid=True if event passes validation
    """
    errors: List[ValidationError] = []
    event_type = event.get("type")

    # Check required fields
    required = ["job_id", "seq", "ts", "type", "level", "message"]
    errors.extend(_validate_required_fields(event, required))

    if errors:
        return ValidationResult.failure(errors, event_type)

    # Type checks for present fields
    type_checks = [
        ("job_id", "string"),
        ("seq", "integer"),
        ("ts", "string"),
        ("type", "string"),
        ("level", "string"),
        ("message", "string"),
    ]

    for field, expected_type in type_checks:
        if field in event:
            error = _validate_field_type(event[field], expected_type, field)
            if error:
                errors.append(error)

    # Validate level enum
    if "level" in event:
        error = _validate_enum(event["level"], ["info", "warn", "error"], "level")
        if error:
            errors.append(error)

    # Validate data is object if present
    if "data" in event and event["data"] is not None:
        error = _validate_field_type(event["data"], "object", "data")
        if error:
            errors.append(error)

    # Validate run_id is string or null if present
    if "run_id" in event and event["run_id"] is not None:
        error = _validate_field_type(event["run_id"], "string", "run_id")
        if error:
            errors.append(error)

    if errors:
        return ValidationResult.failure(errors, event_type)

    return ValidationResult.success(event_type)


def validate_event(event: Dict[str, Any], event_type: Optional[str] = None) -> ValidationResult:
    """Validate an event against its schema (base + algorithm if applicable).

    Args:
        event: Raw event dictionary
        event_type: Optional event type override (defaults to event["type"])

    Returns:
        ValidationResult with validation status and any errors
    """
    # First, validate against base schema
    result = validate_base_event(event)
    if not result.valid:
        return result

    # Get event type
    full_type = event_type or event.get("type", "")
    if not full_type:
        return ValidationResult.failure(
            [ValidationError(path="type", message="Event type is required")]
        )

    # Check for custom validator in registry
    registry = get_registry()
    validator = registry.get_validator(full_type)
    if validator is not None:
        try:
            if not validator(event):
                return ValidationResult.failure(
                    [ValidationError(path="", message=f"Custom validation failed for {full_type}")],
                    full_type,
                )
        except Exception as e:
            return ValidationResult.failure(
                [ValidationError(path="", message=f"Validation error: {e}")], full_type
            )

    return ValidationResult.success(full_type)


def validate_event_strict(
    event: Dict[str, Any], event_type: Optional[str] = None
) -> ValidationResult:
    """Validate an event using full JSON Schema validation.

    This requires the jsonschema library to be installed. If not available,
    falls back to validate_event().

    Args:
        event: Raw event dictionary
        event_type: Optional event type override

    Returns:
        ValidationResult with validation status and any errors
    """
    try:
        import jsonschema
    except ImportError:
        # Fall back to lightweight validation
        return validate_event(event, event_type)

    full_type = event_type or event.get("type", "")
    registry = get_registry()
    schema = registry.get_merged_schema(full_type)

    if schema is None:
        # Fall back to base schema
        schema = BASE_JOB_EVENT_SCHEMA

    errors: List[ValidationError] = []

    try:
        jsonschema.validate(event, schema)
    except jsonschema.ValidationError as e:
        errors.append(
            ValidationError(
                path=".".join(str(p) for p in e.absolute_path),
                message=e.message,
                schema_path=".".join(str(p) for p in e.absolute_schema_path),
            )
        )
    except jsonschema.SchemaError as e:
        errors.append(ValidationError(path="", message=f"Schema error: {e.message}"))

    if errors:
        return ValidationResult.failure(errors, full_type)

    return ValidationResult.success(full_type)


def validate_typed_event(event: BaseJobEvent) -> ValidationResult:
    """Validate a typed event object.

    Args:
        event: A BaseJobEvent (or subclass) instance

    Returns:
        ValidationResult with validation status
    """
    return validate_event(event.to_dict(), event.type)


__all__ = [
    "ValidationError",
    "ValidationResult",
    "validate_base_event",
    "validate_event",
    "validate_event_strict",
    "validate_typed_event",
]
