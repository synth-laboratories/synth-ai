"""Public Managed Research tool-provider enum."""

from __future__ import annotations

from enum import StrEnum


class SmrToolProvider(StrEnum):
    TINKER = "tinker"
    SUBLINEAR = "sublinear"
    LINEAR = "linear"


SMR_TOOL_PROVIDER_VALUES: tuple[str, ...] = tuple(provider.value for provider in SmrToolProvider)


def coerce_smr_tool_provider(
    value: SmrToolProvider | str | None,
    *,
    field_name: str = "tool_provider",
) -> SmrToolProvider | None:
    if value is None:
        return None
    if isinstance(value, SmrToolProvider):
        return value
    normalized = str(value).strip()
    if not normalized:
        return None
    try:
        return SmrToolProvider(normalized)
    except ValueError as exc:
        raise ValueError(
            f"{field_name} must be one of: {', '.join(SMR_TOOL_PROVIDER_VALUES)}"
        ) from exc


__all__ = ["SMR_TOOL_PROVIDER_VALUES", "SmrToolProvider", "coerce_smr_tool_provider"]
