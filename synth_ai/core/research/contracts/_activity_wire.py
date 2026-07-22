"""Strict wire helpers shared by bounded swarm-activity contracts."""

from __future__ import annotations

from datetime import datetime

from synth_ai.core.contracts.json_value import JsonObject, JsonValue
from synth_ai.core.research.contracts._wire import (
    array_value,
    object_value,
    required_datetime,
    required_text,
)


def exact_object(
    value: JsonValue,
    *,
    label: str,
    fields: frozenset[str],
) -> JsonObject:
    payload = object_value(value, operation_id=label)
    missing = fields - payload.keys()
    extra = payload.keys() - fields
    if missing or extra:
        raise ValueError(
            f"{label} fields drifted: missing={sorted(missing)!r} "
            f"extra={sorted(extra)!r}"
        )
    return payload


def optional_datetime(payload: JsonObject, name: str) -> datetime | None:
    if payload.get(name) is None:
        return None
    return required_datetime(payload, name)


def non_negative_int(payload: JsonObject, name: str) -> int:
    value = payload.get(name)
    if type(value) is not int or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value


def optional_non_negative_int(payload: JsonObject, name: str) -> int | None:
    if payload.get(name) is None:
        return None
    return non_negative_int(payload, name)


def text_tuple(payload: JsonObject, name: str) -> tuple[str, ...]:
    return tuple(
        required_text({name: item}, name)
        for item in array_value(payload[name], operation_id=name)
    )


__all__ = [
    "exact_object",
    "non_negative_int",
    "optional_datetime",
    "optional_non_negative_int",
    "text_tuple",
]
