"""Strict JSON response boundary decoders."""

from __future__ import annotations

from synth_ai.core.contracts.json_value import JsonObject, JsonValue


def require_json_object(value: JsonValue, *, operation_id: str) -> JsonObject:
    if not isinstance(value, dict):
        raise ValueError(f"{operation_id} response must be a JSON object")
    return value


def require_json_array(value: JsonValue, *, operation_id: str) -> list[JsonValue]:
    if not isinstance(value, list):
        raise ValueError(f"{operation_id} response must be a JSON array")
    return value


__all__ = ["require_json_array", "require_json_object"]
