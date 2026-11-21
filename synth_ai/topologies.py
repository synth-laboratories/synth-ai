"""Public names for the optimizer topologies Core serves (SYN-5667).

Machine IDs stay as Core serves them; these names are for people. Swarms
(`swarms_v2`) is the default. Fork-and-Flush (`fork_flush`) is experimental:
opt-in only, never the default, whatever a catalog entry's `auto` says.
"""

from __future__ import annotations

from ._generated.models import TopologyCatalogEntry

TOPOLOGY_NAMES: dict[str, str] = {"swarms_v2": "Swarms", "fork_flush": "Fork-and-Flush"}
EXPERIMENTAL: frozenset[str] = frozenset({"fork_flush"})


def topology_name(entry: TopologyCatalogEntry) -> str:
    """`Swarms (default)`, `Fork-and-Flush (experimental)` and so on, from a catalog entry."""
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
    name = TOPOLOGY_NAMES.get(topology, topology)
    return f"{name} ({', '.join(tags)})" if tags else name
