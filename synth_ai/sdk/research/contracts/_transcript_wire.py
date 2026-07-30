"""Strict JSON decoding helpers for transcript and runtime-stream contracts."""

from __future__ import annotations

from datetime import datetime

from synth_ai.core.contracts.json_value import JsonObject, JsonValue


def exact_object(
    value: JsonValue,
    *,
    label: str,
    fields: frozenset[str],
) -> JsonObject:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be an object")
    missing = fields - value.keys()
    extra = value.keys() - fields
    if missing or extra:
        raise ValueError(
            f"{label} fields drifted: missing={sorted(missing)!r} extra={sorted(extra)!r}"
        )
    return value


def required_string(payload: JsonObject, name: str) -> str:
    value = payload.get(name)
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a non-empty string")
    return value


def optional_string(payload: JsonObject, name: str) -> str | None:
    value = payload.get(name)
    if value is not None and not isinstance(value, str):
        raise ValueError(f"{name} must be a string or null")
    return value


def required_datetime(payload: JsonObject, name: str) -> datetime:
    value = required_string(payload, name)
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


def required_bool(payload: JsonObject, name: str) -> bool:
    value = payload.get(name)
    if not isinstance(value, bool):
        raise ValueError(f"{name} must be a boolean")
    return value


def non_negative_int(payload: JsonObject, name: str) -> int:
    value = payload.get(name)
    if type(value) is not int or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value


def optional_non_negative_int(payload: JsonObject, name: str) -> int | None:
    if payload.get(name) is None:
        return None
    return non_negative_int(payload, name)


def object_field(payload: JsonObject, name: str) -> JsonObject:
    value = payload.get(name)
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be an object")
    return dict(value)


def object_tuple(payload: JsonObject, name: str) -> tuple[JsonObject, ...]:
    value = payload.get(name)
    if not isinstance(value, list):
        raise ValueError(f"{name} must be an array")
    objects: list[JsonObject] = []
    for index, item in enumerate(value):
        if not isinstance(item, dict):
            raise ValueError(f"{name}[{index}] must be an object")
        objects.append(dict(item))
    return tuple(objects)


def string_tuple(payload: JsonObject, name: str) -> tuple[str, ...]:
    value = payload.get(name)
    if not isinstance(value, list):
        raise ValueError(f"{name} must be an array")
    items: list[str] = []
    for index, item in enumerate(value):
        if not isinstance(item, str):
            raise ValueError(f"{name}[{index}] must be a string")
        items.append(item)
    return tuple(items)


def string_mapping(payload: JsonObject, name: str) -> dict[str, str]:
    value = payload.get(name)
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be an object")
    result: dict[str, str] = {}
    for key, item in value.items():
        if not isinstance(item, str):
            raise ValueError(f"{name}.{key} must be a string")
        result[key] = item
    return result


def non_negative_int_mapping(payload: JsonObject, name: str) -> dict[str, int]:
    value = payload.get(name)
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be an object")
    result: dict[str, int] = {}
    for key, item in value.items():
        if type(item) is not int or item < 0:
            raise ValueError(f"{name}.{key} must be a non-negative integer")
        result[key] = item
    return result


__all__ = [
    "exact_object",
    "non_negative_int",
    "non_negative_int_mapping",
    "object_field",
    "object_tuple",
    "optional_datetime",
    "optional_non_negative_int",
    "optional_string",
    "required_bool",
    "required_datetime",
    "required_string",
    "string_mapping",
    "string_tuple",
]
