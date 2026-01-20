"""Event schema registry helpers."""

from __future__ import annotations

from typing import Any, Dict, Mapping


def merge_event_schemas(*schema_maps: Mapping[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """Merge multiple event schema registries with conflict detection."""
    merged: Dict[str, Dict[str, Any]] = {}
    for schema_map in schema_maps:
        for event_type, schema in schema_map.items():
            if event_type in merged and merged[event_type] != schema:
                raise ValueError(f"Schema conflict for event type: {event_type}")
            merged[event_type] = schema
    return merged


def apply_aliases(
    schemas: Mapping[str, Dict[str, Any]],
    aliases: Mapping[str, str],
) -> Dict[str, Dict[str, Any]]:
    """Return a new registry with alias event types mapped to canonical schemas."""
    expanded = dict(schemas)
    for alias, canonical in aliases.items():
        schema = schemas.get(canonical)
        if schema is None:
            raise KeyError(f"Alias targets unknown schema: {canonical}")
        expanded[alias] = schema
    return expanded
