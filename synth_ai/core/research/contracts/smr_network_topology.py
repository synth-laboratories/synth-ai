"""Typed network-topology values mirrored from the backend launch contract."""

from __future__ import annotations

from enum import StrEnum


class SmrNetworkTopology(StrEnum):
    LOCAL_NETWORK = "local_network"
    RAILWAY_NETWORK = "railway_network"
    NGROK_PUBLIC = "ngrok_public"


SMR_NETWORK_TOPOLOGY_VALUES: tuple[str, ...] = tuple(
    topology.value for topology in SmrNetworkTopology
)


def coerce_smr_network_topology(
    value: SmrNetworkTopology | str | None,
    *,
    field_name: str,
) -> SmrNetworkTopology | None:
    if value is None:
        return None
    if isinstance(value, SmrNetworkTopology):
        return value
    normalized = str(value).strip()
    if not normalized:
        return None
    try:
        return SmrNetworkTopology(normalized)
    except ValueError as exc:
        allowed = ", ".join(SMR_NETWORK_TOPOLOGY_VALUES)
        raise ValueError(f"{field_name} must be one of: {allowed}") from exc


__all__ = [
    "SMR_NETWORK_TOPOLOGY_VALUES",
    "SmrNetworkTopology",
    "coerce_smr_network_topology",
]
