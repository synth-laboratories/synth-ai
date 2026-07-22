"""General opaque resource identity contracts."""

from __future__ import annotations

from dataclasses import dataclass


class ResourceId(str):
    """Opaque backend-owned resource identifier."""


@dataclass(frozen=True, slots=True)
class ResourceRef:
    """Stable resource kind and identifier pair."""

    kind: str
    resource_id: ResourceId

    def __post_init__(self) -> None:
        if not self.kind.strip():
            raise ValueError("resource kind must not be empty")
        if not self.resource_id.strip():
            raise ValueError("resource_id must not be empty")


__all__ = ["ResourceId", "ResourceRef"]
