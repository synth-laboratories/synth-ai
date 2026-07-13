"""Public Managed Research environment-kind enum."""

from __future__ import annotations

from enum import StrEnum


class SmrEnvironmentKind(StrEnum):
    HARBOR = "harbor"
    DAYTONA = "daytona"
    OPENENV = "openenv"


SMR_ENVIRONMENT_KIND_VALUES: tuple[str, ...] = tuple(kind.value for kind in SmrEnvironmentKind)


def coerce_smr_environment_kind(
    value: SmrEnvironmentKind | str | None,
    *,
    field_name: str = "environment_kind",
) -> SmrEnvironmentKind | None:
    if value is None:
        return None
    if isinstance(value, SmrEnvironmentKind):
        return value
    normalized = str(value).strip().lower()
    if not normalized:
        return None
    try:
        return SmrEnvironmentKind(normalized)
    except ValueError as exc:
        raise ValueError(
            f"{field_name} must be one of: {', '.join(SMR_ENVIRONMENT_KIND_VALUES)}"
        ) from exc


__all__ = [
    "SMR_ENVIRONMENT_KIND_VALUES",
    "SmrEnvironmentKind",
    "coerce_smr_environment_kind",
]
