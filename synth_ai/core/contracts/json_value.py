"""Recursive JSON value boundary types."""

from __future__ import annotations

from typing import TypeAlias

JsonScalar: TypeAlias = str | int | float | bool | None
JsonArray: TypeAlias = list["JsonValue"]
JsonObject: TypeAlias = dict[str, "JsonValue"]
JsonValue: TypeAlias = JsonScalar | JsonArray | JsonObject


__all__ = ["JsonArray", "JsonObject", "JsonScalar", "JsonValue"]
