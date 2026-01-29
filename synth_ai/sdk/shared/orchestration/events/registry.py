"""Event schema registry for managing and merging event schemas.

This module provides a registry for algorithm-specific event schemas that can be
merged with base schemas for validation and code generation. The registry supports:

- Registering algorithm-prefixed event schemas (e.g., "gepa:candidate.evaluated")
- Merging base schemas with algorithm extensions
- Generating complete schema bundles for code generation tools
"""

from __future__ import annotations

import copy
from typing import Any, Callable, Dict, List, Optional

from .base import BASE_EVENT_SCHEMAS

try:
    import synth_ai_py
except Exception as exc:  # pragma: no cover
    raise RuntimeError("synth_ai_py is required for orchestration.events.registry.") from exc


def _require_rust() -> Any:
    if synth_ai_py is None or not hasattr(synth_ai_py, "orchestration_merge_event_schema"):
        raise RuntimeError("Rust core schema merging required; synth_ai_py is unavailable.")
    return synth_ai_py


class EventSchemaRegistry:
    """Registry for managing algorithm-specific event schemas.

    The registry maintains:
    - Base schemas (from base.py) as the foundation
    - Algorithm-specific schemas that extend base events
    - Prefix mappings for algorithm namespacing (e.g., "gepa:", "mipro:")

    Example:
        >>> registry = EventSchemaRegistry()
        >>> registry.register_algorithm_schema("gepa", "candidate.evaluated", gepa_eval_schema)
        >>> full_schema = registry.get_merged_schema("gepa:candidate.evaluated")
    """

    def __init__(self) -> None:
        """Initialize registry with base schemas."""
        self._base_schemas: Dict[str, Dict[str, Any]] = dict(BASE_EVENT_SCHEMAS)
        self._algorithm_schemas: Dict[str, Dict[str, Dict[str, Any]]] = {}
        self._validators: Dict[str, Callable[[Dict[str, Any]], bool]] = {}

    def register_algorithm_schema(
        self,
        algorithm: str,
        event_type: str,
        schema: Dict[str, Any],
        *,
        validator: Optional[Callable[[Dict[str, Any]], bool]] = None,
    ) -> None:
        """Register an algorithm-specific event schema.

        Args:
            algorithm: Algorithm identifier (e.g., "gepa", "mipro")
            event_type: Base event type (e.g., "candidate.evaluated")
            schema: JSON Schema extension for algorithm-specific fields
            validator: Optional custom validator function for this event type
        """
        if algorithm not in self._algorithm_schemas:
            self._algorithm_schemas[algorithm] = {}

        full_type = f"{algorithm}:{event_type}"
        self._algorithm_schemas[algorithm][event_type] = schema

        if validator is not None:
            self._validators[full_type] = validator

    def get_base_schema(self, event_type: str) -> Optional[Dict[str, Any]]:
        """Get a base schema by event type.

        Args:
            event_type: Event type (e.g., "job.completed", "candidate.evaluated")

        Returns:
            The base JSON Schema, or None if not found
        """
        return self._base_schemas.get(event_type)

    def get_algorithm_schema(self, algorithm: str, event_type: str) -> Optional[Dict[str, Any]]:
        """Get an algorithm-specific schema.

        Args:
            algorithm: Algorithm identifier
            event_type: Base event type

        Returns:
            The algorithm-specific JSON Schema extension, or None if not found
        """
        alg_schemas = self._algorithm_schemas.get(algorithm, {})
        return alg_schemas.get(event_type)

    def get_merged_schema(self, full_type: str) -> Optional[Dict[str, Any]]:
        """Get a merged schema combining base and algorithm-specific extensions.

        Args:
            full_type: Full event type with optional prefix (e.g., "gepa:candidate.evaluated")

        Returns:
            Merged JSON Schema, or base schema if no algorithm extension exists
        """
        # Parse algorithm prefix if present
        if ":" in full_type:
            algorithm, event_type = full_type.split(":", 1)
        else:
            algorithm = None
            event_type = full_type

        base_schema = self.get_base_schema(event_type)
        if base_schema is None:
            return None

        if algorithm is None:
            return copy.deepcopy(base_schema)

        alg_schema = self.get_algorithm_schema(algorithm, event_type)
        if alg_schema is None:
            return copy.deepcopy(base_schema)

        # Merge schemas: base + algorithm extension
        return self._merge_schemas(base_schema, alg_schema, algorithm, event_type)

    def _merge_schemas(
        self,
        base: Dict[str, Any],
        extension: Dict[str, Any],
        algorithm: str,
        event_type: str,
    ) -> Dict[str, Any]:
        """Merge a base schema with an algorithm extension (Rust-backed)."""
        rust = _require_rust()
        return rust.orchestration_merge_event_schema(base, extension, algorithm, event_type)

    def list_event_types(self, include_algorithm_prefixed: bool = True) -> List[str]:
        """List all registered event types.

        Args:
            include_algorithm_prefixed: Whether to include algorithm-prefixed types

        Returns:
            List of event type strings
        """
        types = list(self._base_schemas.keys())

        if include_algorithm_prefixed:
            for algorithm, schemas in self._algorithm_schemas.items():
                for event_type in schemas:
                    types.append(f"{algorithm}:{event_type}")

        return sorted(types)

    def list_algorithms(self) -> List[str]:
        """List all registered algorithms."""
        return sorted(self._algorithm_schemas.keys())

    def get_validator(self, full_type: str) -> Optional[Callable[[Dict[str, Any]], bool]]:
        """Get a custom validator for an event type, if registered."""
        return self._validators.get(full_type)

    def export_all_schemas(self) -> Dict[str, Dict[str, Any]]:
        """Export all schemas (base + merged algorithm schemas) as a dictionary.

        Returns:
            Dictionary mapping event type to JSON Schema
        """
        schemas: Dict[str, Dict[str, Any]] = {}

        # Add base schemas
        for event_type, schema in self._base_schemas.items():
            schemas[event_type] = copy.deepcopy(schema)

        # Add merged algorithm schemas
        for algorithm, alg_schemas in self._algorithm_schemas.items():
            for event_type in alg_schemas:
                full_type = f"{algorithm}:{event_type}"
                merged = self.get_merged_schema(full_type)
                if merged:
                    schemas[full_type] = merged

        return schemas


# Global registry instance (singleton pattern)
_global_registry: Optional[EventSchemaRegistry] = None


def get_registry() -> EventSchemaRegistry:
    """Get the global event schema registry.

    Returns:
        The singleton EventSchemaRegistry instance
    """
    global _global_registry
    if _global_registry is None:
        _global_registry = EventSchemaRegistry()
    return _global_registry


def register_algorithm_schema(
    algorithm: str,
    event_type: str,
    schema: Dict[str, Any],
    *,
    validator: Optional[Callable[[Dict[str, Any]], bool]] = None,
) -> None:
    """Register an algorithm-specific event schema in the global registry.

    Convenience function that delegates to get_registry().register_algorithm_schema().
    """
    get_registry().register_algorithm_schema(algorithm, event_type, schema, validator=validator)


__all__ = [
    "EventSchemaRegistry",
    "get_registry",
    "register_algorithm_schema",
]
