"""Public Managed Research resource-kind enum."""

from __future__ import annotations

from enum import StrEnum


class SmrResourceKind(StrEnum):
    POD = "pod"
    SANDBOX = "sandbox"
    APP = "app"


SMR_RESOURCE_KIND_VALUES: tuple[str, ...] = tuple(kind.value for kind in SmrResourceKind)


def coerce_smr_resource_kind(
    value: SmrResourceKind | str | None,
    *,
    field_name: str = "resource_kind",
) -> SmrResourceKind | None:
    if value is None:
        return None
    if isinstance(value, SmrResourceKind):
        return value
    normalized = str(value).strip()
    if not normalized:
        return None
    try:
        return SmrResourceKind(normalized)
    except ValueError as exc:
        raise ValueError(
            f"{field_name} must be one of: {', '.join(SMR_RESOURCE_KIND_VALUES)}"
        ) from exc


__all__ = ["SMR_RESOURCE_KIND_VALUES", "SmrResourceKind", "coerce_smr_resource_kind"]
