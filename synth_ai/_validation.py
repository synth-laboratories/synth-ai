"""Boundary validators generated schemas can reuse without duplicating policy."""

from collections.abc import Callable


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
