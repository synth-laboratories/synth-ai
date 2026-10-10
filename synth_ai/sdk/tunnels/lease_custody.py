"""Lease ownership handoff contracts. See tanha/current/systems/tunnels/v2_catalog.md."""

from __future__ import annotations

from uuid import UUID

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, field_validator


class CustodyContract(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    @field_validator("*", mode="after")
    @classmethod
    def non_nil(cls, value):
        if isinstance(value, UUID) and value.int == 0:
            raise ValueError("nil_lease_custody_identity")
        return value


class HandoffOffer(CustodyContract):
    offer: UUID
    expected_lease_revision: int = Field(strict=True, ge=1)
    caller_binding: UUID
    job_id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_-]{0,127}$")
    deadline_at: AwareDatetime


class HandoffAcceptance(CustodyContract):
    org_id: UUID
    job_id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_-]{0,127}$")
    executor_instance: UUID


class LeaseRenewal(HandoffAcceptance):
    command_id: UUID
    expected_lease_revision: int = Field(strict=True, ge=1)
    expires_at: AwareDatetime
    deadline_at: AwareDatetime
