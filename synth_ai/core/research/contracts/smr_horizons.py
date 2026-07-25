"""Public Managed Research intended-horizon enum."""

from __future__ import annotations

from enum import IntEnum


class SmrIntendedHorizonHours(IntEnum):
    ONE_HOUR = 1
    FOUR_HOURS = 4
    EIGHT_HOURS = 8
    TWENTY_FOUR_HOURS = 24
    ONE_WEEK = 168


SMR_INTENDED_HORIZON_HOURS_VALUES: tuple[int, ...] = tuple(
    int(value) for value in SmrIntendedHorizonHours
)


def coerce_intended_horizon_hours(
    value: SmrIntendedHorizonHours | int | None,
    *,
    field_name: str = "intended_horizon_hours",
) -> SmrIntendedHorizonHours | None:
    if value is None:
        return None
    if isinstance(value, bool):
        raise ValueError(
            f"{field_name} must be one of: {', '.join(map(str, SMR_INTENDED_HORIZON_HOURS_VALUES))}"
        )
    if isinstance(value, SmrIntendedHorizonHours):
        return value
    try:
        return SmrIntendedHorizonHours(int(value))
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"{field_name} must be one of: {', '.join(map(str, SMR_INTENDED_HORIZON_HOURS_VALUES))}"
        ) from exc


__all__ = [
    "SMR_INTENDED_HORIZON_HOURS_VALUES",
    "SmrIntendedHorizonHours",
    "coerce_intended_horizon_hours",
]
