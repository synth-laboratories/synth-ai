"""Public Managed Research host-kind enum."""

from __future__ import annotations

from enum import StrEnum


class SmrHostKind(StrEnum):
    DOCKER = "docker"
    DAYTONA = "daytona"


SMR_HOST_KIND_VALUES: tuple[str, ...] = tuple(kind.value for kind in SmrHostKind)


def coerce_smr_host_kind(
    value: SmrHostKind | str | None,
    *,
    field_name: str = "host_kind",
) -> SmrHostKind | None:
    if value is None:
        return None
    if isinstance(value, SmrHostKind):
        return value
    normalized = str(value).strip()
    if not normalized:
        return None
    try:
        return SmrHostKind(normalized)
    except ValueError as exc:
        raise ValueError(f"{field_name} must be one of: {', '.join(SMR_HOST_KIND_VALUES)}") from exc


__all__ = ["SMR_HOST_KIND_VALUES", "SmrHostKind", "coerce_smr_host_kind"]
