"""Strict MCP serialization arguments; omission has its declared optional meaning."""

# ruff: noqa: UP045 -- Named optionals follow the boundary vocabulary.
from collections.abc import Mapping
from typing import Optional, TypeVar

from pydantic import JsonValue, TypeAdapter, ValidationError

from synth_ai._limits import MCP_PAGE_LIMIT_DEFAULT
from synth_ai._validation import safe_validation_detail
from synth_ai.errors import InvalidRequestError, local_problem

__all__ = ["optional", "page_limit", "required", "validate_with"]

T = TypeVar("T")


def _invalid_argument(error: ValidationError) -> InvalidRequestError:
    return InvalidRequestError(
        local_problem(
            "tool_arguments_invalid",
            "mcp_input",
            status=422,
            detail=safe_validation_detail(error),
        )
    )


def validate_with(adapter: TypeAdapter[T], value: JsonValue) -> T:
    try:
        return adapter.validate_python(value, strict=True)
    except ValidationError as error:
        raise _invalid_argument(error) from None


def required(arguments: Mapping[str, JsonValue], name: str, model: type[T]) -> T:
    return validate_with(TypeAdapter(model), arguments[name])


def optional(
    arguments: Mapping[str, JsonValue], name: str, model: type[T], *, allow_none: bool = False
) -> Optional[T]:
    if name not in arguments:
        return None
    if allow_none and arguments[name] is None:
        return None
    return validate_with(TypeAdapter(model), arguments[name])


def page_limit(arguments: Mapping[str, JsonValue]) -> int:
    # The generated public page schema defines 20 as its omitted limit.
    return required(arguments, "limit", int) if "limit" in arguments else MCP_PAGE_LIMIT_DEFAULT
