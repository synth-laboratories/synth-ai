"""Physical quantities, cost and customer charges are independent. See SDK index_contract.md."""

from __future__ import annotations

from decimal import Decimal
from typing import Literal, Self

from pydantic import Field, model_validator

from .base import ResponseBase, ResponseIdentifier, ResponseModel
from .funding import FundingPart, funding_fallback, validate_funding
from .money import MoneyObservation


class UsageMetric(ResponseBase):
    name: str = Field(min_length=1, max_length=128)
    unit: Literal[
        "tokens",
        "requests",
        "bytes",
        "seconds",
        "searches",
        "reads",
        "millicpu_seconds",
        "byte_seconds",
    ]
    quantity: Decimal | None = Field(ge=0)
    state: Literal["measured", "estimated", "pending", "unknown"]
    source: str | None = Field(max_length=128)

    @model_validator(mode="after")
    def validate_quantity(self) -> Self:
        if (self.state in {"measured", "estimated"}) != (self.quantity is not None):
            raise ValueError(
                "measured/estimated quantity requires a value; pending/unknown has none"
            )
        if self.quantity is not None and (not self.source):
            raise ValueError("observed quantity requires provenance")
        return self


class UsageSummary(ResponseModel):
    operation_id: ResponseIdentifier
    physical: tuple[UsageMetric, ...] = Field(max_length=128)
    cost: MoneyObservation
    customer_charge: MoneyObservation
    financial_state: Literal["pending", "final", "unknown"]
    execution_source: Literal["live", "replay", "unknown"]
    list_price: MoneyObservation = Field(
        default_factory=lambda: MoneyObservation(state="unknown", amount=None)
    )
    funding: tuple[FundingPart, ...] = Field(default=(), max_length=7)

    @model_validator(mode="after")
    def validate_final_charge(self) -> Self:
        if self.financial_state == "final" and self.customer_charge.state != "measured":
            raise ValueError("final customer settlement requires a measured charge")
        if not self.funding:
            object.__setattr__(
                self,
                "funding",
                funding_fallback(self.customer_charge, final=self.financial_state == "final"),
            )
        validate_funding(self.customer_charge, self.funding)
        return self


class UsageOperation(ResponseModel):
    operation_id: ResponseIdentifier
    parent_id: ResponseIdentifier | None
    attempt_id: ResponseIdentifier
    event_id: ResponseIdentifier
    component: str = Field(min_length=1, max_length=128)
    usage: UsageSummary
