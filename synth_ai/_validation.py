"""Boundary validators generated schemas can reuse without duplicating policy."""

from collections.abc import Callable


def utf8_max_bytes(maximum: int) -> Callable[[str], str]:
    def validate(value: str) -> str:
        if len(value.encode("utf-8")) > maximum:
            raise ValueError(f"text exceeds {maximum} UTF-8 bytes")
        return value

    return validate
