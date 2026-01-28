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

try:
    import synth_ai_py  # type: ignore
except Exception as exc:  # pragma: no cover
    raise RuntimeError("synth_ai_py is required for orchestration.events.validation.") from exc


def _require_rust() -> Any:
    if synth_ai_py is None or not hasattr(synth_ai_py, "validate_base_event"):
        raise RuntimeError("Rust core event validation required; synth_ai_py is unavailable.")
    return synth_ai_py


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


def validate_base_event(event: Dict[str, Any]) -> ValidationResult:
    """Validate an event against the base job event schema.

    This performs lightweight validation without requiring jsonschema library.
    For full JSON Schema validation, use validate_event_strict().

    Args:
        event: Raw event dictionary

    Returns:
        ValidationResult with valid=True if event passes validation
    """
    rust = _require_rust()
    result = rust.validate_base_event(event)
    errors = []
    if isinstance(result, dict):
        for err in result.get("errors", []) or []:
            if isinstance(err, dict):
                errors.append(
                    ValidationError(
                        path=err.get("path", ""),
                        message=err.get("message", ""),
                        schema_path=err.get("schema_path"),
                    )
                )
            else:
                errors.append(ValidationError(path="", message=str(err)))
        return ValidationResult(
            valid=bool(result.get("valid")),
            errors=errors,
            event_type=result.get("event_type"),
        )
    return ValidationResult.failure(
        [ValidationError(path="", message="invalid validation response")],
        event.get("type"),
    )


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
