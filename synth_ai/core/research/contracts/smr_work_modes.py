"""Public Managed Research work-mode enum."""

from __future__ import annotations

from enum import StrEnum


class SmrWorkMode(StrEnum):
    GENERAL = "general"
    OPEN_ENDED_DISCOVERY = "open_ended_discovery"
    DIRECTED_EFFORT = "directed_effort"


SMR_WORK_MODE_VALUES: tuple[str, ...] = tuple(value.value for value in SmrWorkMode)


def coerce_smr_work_mode(
    value: SmrWorkMode | str | None,
    *,
    field_name: str = "work_mode",
) -> SmrWorkMode | None:
    if value is None:
        return None
    if isinstance(value, SmrWorkMode):
        return value
    normalized = str(value).strip()
    if not normalized:
        return None
    try:
        return SmrWorkMode(normalized)
    except ValueError as exc:
        raise ValueError(f"{field_name} must be one of: {', '.join(SMR_WORK_MODE_VALUES)}") from exc


__all__ = ["SMR_WORK_MODE_VALUES", "SmrWorkMode", "coerce_smr_work_mode"]
