"""Exact customer money; no inferred price. See SDK index_contract.md."""

from __future__ import annotations

from typing import Literal, Self

from pydantic import Field, StrictInt, model_validator

from .base import Model


class Money(Model):
    coefficient: StrictInt = Field(ge=-(10**30), le=10**30)
    scale: StrictInt = Field(ge=0, le=12)
    currency: Literal["USD"] = "USD"


class MoneyObservation(Model):
    state: Literal["measured", "estimated", "pending", "unknown"]
    amount: Money | None
    source: str | None = Field(default=None, max_length=128)
    rate_version: str | None = Field(default=None, max_length=128)

    @model_validator(mode="after")
    def validate_observation(self) -> Self:
        if (self.state in {"measured", "estimated"}) != (self.amount is not None):
            raise ValueError(
                "measured/estimated money requires an amount; pending/unknown has none"
            )
        if self.amount is not None and (not self.source or not self.rate_version):
            raise ValueError("observed money requires provenance and rate version")
        return self
