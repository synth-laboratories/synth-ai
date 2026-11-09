"""Lazy cursor traversal with bounded cycle detection and no prefetch."""

from collections.abc import AsyncGenerator, Awaitable, Callable, Sequence
from typing import Protocol, TypeVar

from .errors import PaginationError

T = TypeVar("T", covariant=True)


class Page(Protocol[T]):
    @property
    def items(self) -> Sequence[T]: ...

    @property
    def has_more(self) -> bool: ...

    @property
    def next_cursor(self) -> str | None: ...


async def paginate(
    fetch: Callable[[str | None], Awaitable[Page[T]]], *, cursor: str | None
) -> AsyncGenerator[T, None]:
    # A cap also bounds the cursor ledger. Callers needing a larger traversal
    # can continue explicitly using the page-returning APIs.
    seen: set[str] = set()
    for _ in range(10_000):
        if cursor is not None:
            if not cursor or len(cursor) > 1024 or cursor in seen:
                raise PaginationError("invalid or repeated pagination cursor", cursor=cursor)
            seen.add(cursor)
        page = await fetch(cursor)
        next_cursor = page.next_cursor
        if page.has_more and (
            next_cursor is None
            or next_cursor == ""
            or len(next_cursor) > 1024
            or next_cursor in seen
        ):
            raise PaginationError("invalid or repeated pagination cursor", cursor=next_cursor)
        for item in page.items:
            yield item
        if not page.has_more:
            return
        cursor = next_cursor
    raise PaginationError("pagination exceeded 10000 pages; resume explicitly", cursor=cursor)
