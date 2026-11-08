"""Boundary validators generated schemas can reuse without duplicating policy."""

from collections.abc import Callable, Mapping
from enum import StrEnum
from typing import TypeVar

from pydantic import BaseModel, ValidationInfo

E = TypeVar("E", bound=StrEnum)


def enum_value(enum: type[E]) -> Callable[[object], E]:
    def validate(value: object) -> E:
        if not isinstance(value, str):
            raise ValueError("expected a string enum value")
        return enum(value)

    return validate


def utf8_max_bytes(maximum: int) -> Callable[[str], str]:
    def validate(value: str) -> str:
        if len(value.encode("utf-8")) > maximum:
            raise ValueError(f"text exceeds {maximum} UTF-8 bytes")
        return value

    return validate


def message_text(value: str) -> str:
    # Unicode White_Space agrees with Core's unicode.IsSpace. Python additionally
    # treats four information separators as whitespace; they are not blank here.
    blank = all(c.isspace() and c not in "\x1c\x1d\x1e\x1f" for c in value)
    if blank or "\0" in value:
        raise ValueError("message must be nonblank and contain no NUL")
    return value


def unique_items(value: list[str]) -> list[str]:
    if len(value) != len(set(value)):
        raise ValueError("choices must be unique")
    return value


def budget_amount(value: str) -> str:
    from decimal import Decimal

    amount = Decimal(value)
    if not amount.is_finite() or amount <= 0:
        raise ValueError("budget must be positive and finite")
    if amount * 1_000_000 != (amount * 1_000_000).to_integral_value():
        raise ValueError("budget must be exactly representable in millionths of USD")
    if amount * 100_000_000 > 2**63 - 1:
        raise ValueError("budget is out of range")
    return value


def no_nul(value: str) -> str:
    if "\0" in value:
        raise ValueError("text must contain no NUL")
    return value


def event_payload(
    models: Mapping[str, type[BaseModel]],
) -> Callable[[object, ValidationInfo], object]:
    """Select semantic event data by its envelope, never by overlapping field shape."""

    def validate(value: object, info: ValidationInfo) -> object:
        if value is None:
            return None
        model = models.get(info.data.get("kind", ""))
        if model is None:
            raise ValueError("event kind does not declare a payload")
        return model.model_validate(value)

    return validate


def strict_int(value: object) -> int:
    if type(value) is not int:
        raise ValueError("expected an integer")
    return value
