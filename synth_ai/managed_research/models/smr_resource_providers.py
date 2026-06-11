"""Public Managed Research resource-provider enum."""

from __future__ import annotations

from enum import StrEnum


class SmrResourceProvider(StrEnum):
    RUNPOD = "runpod"
    MODAL = "modal"


SMR_RESOURCE_PROVIDER_VALUES: tuple[str, ...] = tuple(
    provider.value for provider in SmrResourceProvider
)


def coerce_smr_resource_provider(
    value: SmrResourceProvider | str | None,
    *,
    field_name: str = "resource_provider",
) -> SmrResourceProvider | None:
    if value is None:
        return None
    if isinstance(value, SmrResourceProvider):
        return value
    normalized = str(value).strip()
    if not normalized:
        return None
    try:
        return SmrResourceProvider(normalized)
    except ValueError as exc:
        raise ValueError(
            f"{field_name} must be one of: {', '.join(SMR_RESOURCE_PROVIDER_VALUES)}"
        ) from exc


__all__ = [
    "SMR_RESOURCE_PROVIDER_VALUES",
    "SmrResourceProvider",
    "coerce_smr_resource_provider",
]
