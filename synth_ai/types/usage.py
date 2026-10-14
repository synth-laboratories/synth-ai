"""Physical quantities, cost and customer charges are independent. See SDK index_contract.md."""

from __future__ import annotations

from decimal import Decimal
from typing import Literal, Self

from pydantic import Field, model_validator

from .base import Identifier, Model, ResponseModel
from .money import MoneyObservation


class UsageMetric(Model):
    name: str = Field(min_length=1, max_length=128)
    unit: Literal["tokens", "requests", "bytes", "seconds", "searches", "reads", "millicpu_seconds", "byte_seconds"]
    quantity: Decimal | None = Field(ge=0)
    state: Literal["measured", "estimated", "pending", "unknown"]
    source: str | None = Field(max_length=128)

    @model_validator(mode="after")
    def validate_quantity(self) -> Self:
        if (self.state in {"measured", "estimated"}) != (self.quantity is not None):
            raise ValueError(
                "measured/estimated quantity requires a value; pending/unknown has none"
            )
        if self.quantity is not None and not self.source:
            raise ValueError("observed quantity requires provenance")
        return self


class UsageSummary(ResponseModel):
    operation_id: Identifier
    physical: tuple[UsageMetric, ...] = Field(max_length=128)
    cost: MoneyObservation
    customer_charge: MoneyObservation
    financial_state: Literal["pending", "final", "unknown"]
    execution_source: Literal["live", "replay", "unknown"]

    @model_validator(mode="after")
    def validate_final_charge(self) -> Self:
        if self.financial_state == "final" and self.customer_charge.state != "measured":
            raise ValueError("final customer settlement requires a measured charge")
        return self


class UsageOperation(ResponseModel):
    operation_id: Identifier
    parent_id: Identifier | None
    attempt_id: Identifier
    event_id: Identifier
    component: str = Field(min_length=1, max_length=128)
    usage: UsageSummary
