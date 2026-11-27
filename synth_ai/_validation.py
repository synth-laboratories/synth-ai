"""Boundary validators generated schemas can reuse without duplicating policy."""

from __future__ import annotations

import hashlib
from typing import TYPE_CHECKING, TypeVar
from urllib.parse import urlsplit

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping
    from enum import StrEnum

    from pydantic import BaseModel, ValidationError, ValidationInfo

E = TypeVar("E", bound="StrEnum")
_CONTROL_CHARS = ("\r", "\n", "\0")
_LOOPBACK_HOSTS = frozenset({"localhost", "127.0.0.1", "::1"})


def enum_value(enum: type[E]) -> Callable[[object], E]:
    def validate(value: object) -> E:
        if not isinstance(value, str):
            raise ValueError("expected a string enum value")
        return enum(value)

    return validate


def open_enum_value(enum: type[E]) -> Callable[[object], E | str]:
    """Decode known StrEnum members while accepting forward-compatible server catalog strings."""

    def validate(value: object) -> E | str:
        if not isinstance(value, str):
            raise ValueError("expected a string enum value")
        if not value.strip() or any(c in value for c in _CONTROL_CHARS):
            raise ValueError("expected a non-empty string without control characters")
        try:
            return enum(value)
        except ValueError:
            return value

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


def require_non_empty(name: str, value: str, *, allow_dot_segments: bool = False) -> str:
    """Require a non-blank string without control characters or path dot segments."""
    if not isinstance(value, str) or not value.strip() or any(c in value for c in _CONTROL_CHARS):
        raise ValueError(f"{name} must be a non-empty string without control characters")
    if not allow_dot_segments and value.strip() in {".", ".."}:
        raise ValueError(f"{name} must not be '.' or '..'")
    return value


def validate_idempotency_key(value: str | None, *, max_bytes: int = 128) -> str | None:
    """Require None or a bounded, non-blank idempotency key without control characters."""
    if value is None:
        return None
    if (
        not isinstance(value, str)
        or not value.strip()
        or any(c in value for c in _CONTROL_CHARS)
        or len(value.encode("utf-8")) > max_bytes
    ):
        raise ValueError(
            f"idempotency_key must be 1..{max_bytes} UTF-8 bytes without blank text or control characters"
        )
    return value


def verify_sha256(content: bytes, expected: str, *, label: str) -> None:
    """Raise DataIntegrityError when received bytes differ from the pinned SHA-256 hex digest."""
    if hashlib.sha256(content).hexdigest() != expected:
        from .errors import DataIntegrityError

        raise DataIntegrityError(f"{label} digest does not match the pinned reference")


def validate_http_origin(value: str, *, allow_http_loopback: bool = True) -> str:
    """Validate an explicit HTTP(S) origin with no credentials, path, query, or fragment."""
    if not isinstance(value, str) or not value or any(c in value for c in _CONTROL_CHARS):
        raise ValueError("base_url must be an HTTP origin")
    stripped = value.strip()
    if stripped != value:
        raise ValueError("base_url must be an HTTP origin")
    origin = stripped.rstrip("/")
    parsed = urlsplit(origin)
    host = parsed.hostname
    if (
        parsed.scheme not in {"https", "http"}
        or host is None
        or host == ""
        or parsed.username is not None
        or parsed.password is not None
        or parsed.path not in {"", "/"}
        or parsed.query != ""
        or parsed.fragment != ""
    ):
        raise ValueError("base_url must be an HTTP origin")
    if parsed.scheme == "http" and not (allow_http_loopback and host in _LOOPBACK_HOSTS):
        raise ValueError("credentials require HTTPS outside loopback")
    return origin


def safe_validation_detail(error: ValidationError) -> str:
    """Format validation error locations without echoing caller-supplied input values."""
    parts = [
        f"{'.'.join(str(p) for p in item['loc']) or 'body'}: {item['msg']}"
        for item in error.errors(include_input=False)[:3]
    ]
    return "; ".join(parts) if parts else "validation failed"
