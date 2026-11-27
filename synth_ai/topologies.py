"""Public names for the optimizer topologies Core serves (SYN-5667, SYN-6038, SYN-6059).

Machine IDs stay as Core serves them; known display names and experimental markers
come from the pinned catalog, and unknown server topologies fall back cleanly.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ._generated.topologies import EXPERIMENTAL_TOPOLOGIES, TOPOLOGY_DISPLAY_NAMES

if TYPE_CHECKING:
    from ._generated.models import TopologyCatalogEntry

__all__ = ["EXPERIMENTAL", "TOPOLOGY_NAMES", "topology_name"]

TOPOLOGY_NAMES: dict[str, str] = dict(TOPOLOGY_DISPLAY_NAMES)
EXPERIMENTAL: frozenset[str] = EXPERIMENTAL_TOPOLOGIES


def _format_topology_fallback(topology: str) -> str:
    return " ".join(part.capitalize() for part in topology.replace("-", "_").split("_") if part)


def topology_name(entry: TopologyCatalogEntry) -> str:
    """`Swarms v2 (default)`, `Fork & Flush (experimental)` and so on, from a catalog entry."""
    topology = str(entry.topology)
    label = str(entry.label)
    availability = str(entry.availability)
    tags: list[str] = []
    if topology in EXPERIMENTAL or label == "experimental":
        tags.append("experimental")
    elif entry.auto:
        tags.append("default")
    if availability != "served":
        tags.append("unavailable")
    name = TOPOLOGY_NAMES.get(topology) or _format_topology_fallback(topology) or topology
    return f"{name} ({', '.join(tags)})" if tags else name
