"""Public Managed Research funding-source enum."""

from __future__ import annotations

from enum import StrEnum


class SmrFundingSource(StrEnum):
    SYNTH_MANAGED = "synth_managed"
    CUSTOMER_BYOK = "customer_byok"
    USER_CONNECTED = "user_connected"


SMR_FUNDING_SOURCE_VALUES: tuple[str, ...] = tuple(source.value for source in SmrFundingSource)


def coerce_smr_funding_source(
    value: SmrFundingSource | str | None,
    *,
    field_name: str = "funding_source",
) -> SmrFundingSource | None:
    if value is None:
        return None
    if isinstance(value, SmrFundingSource):
        return value
    normalized = str(value).strip()
    if not normalized:
        return None
    try:
        return SmrFundingSource(normalized)
    except ValueError as exc:
        raise ValueError(
            f"{field_name} must be one of: {', '.join(SMR_FUNDING_SOURCE_VALUES)}"
        ) from exc


__all__ = [
    "SMR_FUNDING_SOURCE_VALUES",
    "SmrFundingSource",
    "coerce_smr_funding_source",
]
