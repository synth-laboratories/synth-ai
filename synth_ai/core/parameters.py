"""Bounded opaque pagination parameters. See SDK index_contract.md."""

from __future__ import annotations

from typing import TypeVar

from pydantic import TypeAdapter

from synth_ai.errors import ContractError
from synth_ai.types.base import Cursor, Identifier

T = TypeVar("T")


def identifier(value: str) -> str:
    return TypeAdapter(Identifier).validate_python(value)


def page_parameters(limit: int, cursor: str | None) -> dict[str, str | int]:
    if type(limit) is not int or not 1 <= limit <= 200:
        raise ValueError("page limit must be an integer from 1 to 200")
    values: dict[str, str | int] = {"limit": limit}
    if cursor is not None:
        values["cursor"] = TypeAdapter(Cursor).validate_python(cursor)
    return values


def verified(result: T, condition: bool, message: str) -> T:
    """Semantic response checks run inside the mutation uncertainty boundary."""
    if not condition:
        raise ContractError(message)
    return result
