"""Strict MCP serialization arguments; omission has its declared optional meaning."""

# ruff: noqa: UP045 -- Named optionals follow the boundary vocabulary.
from collections.abc import Mapping
from typing import Optional, TypeVar

from pydantic import JsonValue, TypeAdapter

from synth_ai._limits import MCP_PAGE_LIMIT_DEFAULT

__all__ = ["optional", "page_limit", "required"]

T = TypeVar("T")


def required(arguments: Mapping[str, JsonValue], name: str, model: type[T]) -> T:
    return TypeAdapter(model).validate_python(arguments[name], strict=True)


def optional(arguments: Mapping[str, JsonValue], name: str, model: type[T]) -> Optional[T]:
    if name not in arguments:
        return None
    return TypeAdapter(model).validate_python(arguments[name], strict=True)


def page_limit(arguments: Mapping[str, JsonValue]) -> int:
    # The generated public page schema defines 20 as its omitted limit.
    return required(arguments, "limit", int) if "limit" in arguments else MCP_PAGE_LIMIT_DEFAULT
