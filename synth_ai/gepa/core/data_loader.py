"""Minimal data loader protocol for GEPA compatibility."""
# See: specifications/tanha/master_specification.md

from __future__ import annotations

from typing import Any, Hashable, Protocol, Sequence, TypeVar, cast, runtime_checkable

from .adapter import DataInst


class ComparableHashable(Hashable, Protocol):
    """Protocol requiring hashing and rich comparison support."""

    def __lt__(self, other: Any, /) -> bool: ...

    def __gt__(self, other: Any, /) -> bool: ...

    def __le__(self, other: Any, /) -> bool: ...

    def __ge__(self, other: Any, /) -> bool: ...


DataId = TypeVar("DataId", bound=ComparableHashable)
"""Generic for the identifier for data examples."""


@runtime_checkable
class DataLoader(Protocol[DataId, DataInst]):
    """Minimal interface for retrieving validation examples keyed by opaque ids."""

    def all_ids(self) -> Sequence[DataId]:
        """Return the ordered universe of ids currently available."""
        ...

    def fetch(self, ids: Sequence[DataId]) -> list[DataInst]:
        """Materialise the payloads corresponding to ids, preserving order."""
        ...

    def __len__(self) -> int:
        """Return current number of items in the loader."""
        ...


class ListDataLoader(DataLoader[int, DataInst]):
    """In-memory reference implementation backed by a list."""

    def __init__(self, items: Sequence[DataInst]):
        self.items = list(items)

    def all_ids(self) -> Sequence[int]:
        return list(range(len(self.items)))

    def fetch(self, ids: Sequence[int]) -> list[DataInst]:
        return [self.items[data_id] for data_id in ids]

    def __len__(self) -> int:
        return len(self.items)


def ensure_loader(
    data_or_loader: Sequence[DataInst] | DataLoader[DataId, DataInst],
) -> DataLoader[DataId, DataInst]:
    """Normalize dataset inputs to a DataLoader instance."""
    if isinstance(data_or_loader, DataLoader):
        return data_or_loader
    if isinstance(data_or_loader, Sequence):
        return cast(DataLoader[DataId, DataInst], ListDataLoader(data_or_loader))
    raise TypeError(f"Unable to cast to a DataLoader type: {type(data_or_loader)}")
