"""Private strict JSON decoding helpers for Research contracts."""

from __future__ import annotations

from datetime import datetime

from synth_ai.core.contracts.json_value import JsonObject, JsonValue


def object_value(value: JsonValue, *, operation_id: str) -> JsonObject:
    if not isinstance(value, dict):
        raise ValueError(f"{operation_id} response must be an object")
    return value


def array_value(value: JsonValue, *, operation_id: str) -> list[JsonValue]:
    if not isinstance(value, list):
        raise ValueError(f"{operation_id} response must be an array")
    return value


def required_text(payload: JsonObject, name: str) -> str:
    value = payload.get(name)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value.strip()


def optional_text(payload: JsonObject, name: str) -> str | None:
    value = payload.get(name)
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string when provided")
    return value.strip()


def required_bool(payload: JsonObject, name: str) -> bool:
    value = payload.get(name)
    if not isinstance(value, bool):
        raise ValueError(f"{name} must be a boolean")
    return value


def optional_bool(payload: JsonObject, name: str, *, default: bool = False) -> bool:
    value = payload.get(name, default)
    if not isinstance(value, bool):
        raise ValueError(f"{name} must be a boolean")
    return value


def required_datetime(payload: JsonObject, name: str) -> datetime:
    value = required_text(payload, name)
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as error:
        raise ValueError(f"{name} must be an ISO-8601 datetime") from error
    if parsed.tzinfo is None:
        raise ValueError(f"{name} must include a timezone")
    return parsed


def optional_datetime(payload: JsonObject, name: str) -> datetime | None:
    if payload.get(name) is None:
        return None
    return required_datetime(payload, name)


__all__ = [
    "array_value",
    "object_value",
    "optional_bool",
    "optional_datetime",
    "optional_text",
    "required_bool",
    "required_datetime",
    "required_text",
]
