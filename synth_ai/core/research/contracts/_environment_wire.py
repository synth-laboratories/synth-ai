"""Private exact decoders for Environment catalog contracts."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from math import isfinite
from types import MappingProxyType
from typing import TypeAlias

from synth_ai.core.contracts.json_value import JsonObject, JsonValue
from synth_ai.core.research.contracts._wire import object_value


JsonScalar: TypeAlias = str | int | float | bool | None
JsonInput: TypeAlias = JsonScalar | Sequence["JsonInput"] | Mapping[str, "JsonInput"]
FrozenJson: TypeAlias = JsonScalar | tuple["FrozenJson", ...] | Mapping[str, "FrozenJson"]

ENVIRONMENT_SCHEMA_VERSION = "2026-05-14-environment-v1"
SHA256_PATTERN = re.compile(r"^sha256:[0-9a-f]{64}$")
IMAGE_REF_PATTERN = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9._:/@+-]*$")


def freeze_json(value: object, *, field: str) -> FrozenJson:
    if value is None or isinstance(value, (str, bool)):
        return value
    if type(value) is int:
        return value
    if type(value) is float:
        if not isfinite(value):
            raise ValueError(f"{field} must contain finite numbers")
        return value
    if isinstance(value, Mapping):
        frozen: dict[str, FrozenJson] = {}
        for key, child in value.items():
            if not isinstance(key, str) or not key:
                raise ValueError(f"{field} keys must be non-empty strings")
            frozen[key] = freeze_json(child, field=f"{field}.{key}")
        return MappingProxyType(frozen)
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return tuple(freeze_json(child, field=f"{field}[{index}]") for index, child in enumerate(value))
    raise ValueError(f"{field} must contain only JSON values")


def thaw_json(value: FrozenJson) -> JsonValue:
    if isinstance(value, Mapping):
        return {key: thaw_json(child) for key, child in value.items()}
    if isinstance(value, tuple):
        return [thaw_json(child) for child in value]
    return value


def exact_object(value: JsonValue, *, label: str, fields: frozenset[str]) -> JsonObject:
    payload = object_value(value, operation_id=label)
    actual = frozenset(payload)
    if actual != fields:
        raise ValueError(
            f"{label} fields drifted: missing={sorted(fields - actual)!r} "
            f"extra={sorted(actual - fields)!r}"
        )
    return payload


def input_object(
    value: JsonValue,
    *,
    label: str,
    allowed: frozenset[str],
    required: frozenset[str],
) -> JsonObject:
    payload = object_value(value, operation_id=label)
    actual = frozenset(payload)
    if required - actual or actual - allowed:
        raise ValueError(
            f"{label} fields invalid: missing={sorted(required - actual)!r} "
            f"extra={sorted(actual - allowed)!r}"
        )
    return payload


def text(value: object, *, field: str, maximum: int) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty string")
    normalized = value.strip()
    if len(normalized) > maximum:
        raise ValueError(f"{field} must be at most {maximum} characters")
    return normalized


def optional_text(value: object, *, field: str, maximum: int) -> str | None:
    return None if value is None else text(value, field=field, maximum=maximum)


def digest(value: object, *, field: str) -> str:
    if not isinstance(value, str) or SHA256_PATTERN.fullmatch(value.strip()) is None:
        raise ValueError(f"{field} must be sha256:<64 lowercase hex>")
    return value.strip()


def optional_digest(value: object, *, field: str) -> str | None:
    return None if value is None else digest(value, field=field)


def boolean(value: object, *, field: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{field} must be a boolean")
    return value


def integer(value: object, *, field: str, minimum: int, maximum: int) -> int:
    if type(value) is not int or not minimum <= value <= maximum:
        raise ValueError(f"{field} must be an integer from {minimum} through {maximum}")
    return value


__all__ = [
    "ENVIRONMENT_SCHEMA_VERSION",
    "FrozenJson",
    "IMAGE_REF_PATTERN",
    "JsonInput",
    "SHA256_PATTERN",
    "boolean",
    "digest",
    "exact_object",
    "freeze_json",
    "input_object",
    "integer",
    "optional_digest",
    "optional_text",
    "text",
    "thaw_json",
]
