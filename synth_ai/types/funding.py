"""Shared paid-from evidence. See the public contract snapshots in contracts/"""

from __future__ import annotations

from enum import StrEnum

from .base import ResponseBase
from .money import Money, MoneyObservation


class FundingSource(StrEnum):
    NOT_BILLED = "not_billed"
    INCLUDED_IN_OPERATION = "included_in_operation"
    PLAN_ALLOWANCE = "plan_allowance"
    PROMOTION = "promotion"
    CREDITS = "credits"
    UNATTRIBUTED = "unattributed"
    UNSETTLED = "unsettled"


class FundingPart(ResponseBase):
    source: FundingSource
    amount: Money | None


def funding_fallback(
    charge: MoneyObservation, *, final: bool, included: bool = False
) -> tuple[FundingPart, ...]:
    """Missing settlement or provenance is explicit, never inferred as credits."""
    if included:
        source = FundingSource.INCLUDED_IN_OPERATION
    elif not final:
        source = FundingSource.UNSETTLED
    elif charge.amount is not None and charge.amount.coefficient == 0:
        source = FundingSource.NOT_BILLED
    else:
        source = FundingSource.UNATTRIBUTED
    return (FundingPart(source=source, amount=charge.amount),)


def validate_funding(charge: MoneyObservation, funding: tuple[FundingPart, ...]) -> None:
    """A unique source split must reconcile exactly, including decimal scales."""
    if len({part.source for part in funding}) != len(funding):
        raise ValueError("usage funding sources must be unique")
    if charge.amount is None:
        if any(part.amount is not None for part in funding):
            raise ValueError("unknown charge cannot carry an observed funding amount")
        return
    if any(part.amount is None for part in funding):
        raise ValueError("observed charge requires observed funding parts")
    total = sum(
        part.amount.coefficient * 10 ** (12 - part.amount.scale)
        for part in funding
        if part.amount is not None
    )
    expected = charge.amount.coefficient * 10 ** (12 - charge.amount.scale)
    if total != expected:
        raise ValueError("usage funding parts do not sum to customer charge")
