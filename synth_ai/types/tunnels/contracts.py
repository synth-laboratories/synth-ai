"""Backend-owned tunnel v2 contracts.

# See: the public contract snapshots in contracts/
"""

from __future__ import annotations

from typing import Literal, Optional
from uuid import UUID

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, SecretStr, field_validator


class TunnelModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid", frozen=True, hide_input_in_errors=True, allow_inf_nan=False
    )

    @field_validator("*", mode="after")
    @classmethod
    def non_nil_identity(cls, value):
        if isinstance(value, UUID) and value.int == 0:
            raise ValueError("nil tunnel identity")
        return value


class CatalogCommand(TunnelModel):
    command_id: UUID
    expected_revision: int = Field(ge=0)
    deadline_at: AwareDatetime

    @field_validator("command_id")
    @classmethod
    def non_nil(cls, value: UUID) -> UUID:
        if value.int == 0:
            raise ValueError("nil command identity")
        return value


class CatalogMutation(TunnelModel):
    command: UUID
    connector: UUID
    revision: int = Field(ge=1)
    delivery: Literal["pending", "claimed", "in_doubt", "applied", "stale", "failed"]


class ConnectorStatus(TunnelModel):
    connector: UUID
    revision: int = Field(ge=1)
    session: Optional[UUID]
    generation: Optional[int]
    applied_revision: Optional[int]
    applied_generation: Optional[int]
    policy_applied: bool


class Capability(TunnelModel):
    token: SecretStr
    expires_in_seconds: int = Field(ge=1, le=600)


class RouteRegistration(TunnelModel):
    route_id: UUID
    name: str
    target_key: str
    service_identity: str
    registration_owner: UUID
    expected_owner_generation: int = Field(ge=0)
    lifetime: Literal["configured", "application"]
    owner_expires_at: Optional[AwareDatetime] = None


class LeaseGrant(TunnelModel):
    lease_id: UUID
    route_token: SecretStr
    routes: tuple[UUID, ...] = Field(min_length=1, max_length=64)
    expires_at: AwareDatetime
    owner_binding: UUID
    owner_kind: Literal["application", "job", "borrower"]
    max_inflight: int = Field(default=64, ge=1, le=2048)

    def wire_payload(self) -> dict:
        """Encode the grant at the authenticated HTTP boundary, keeping repr redacted."""
        payload = self.model_dump(mode="json")
        payload["route_token"] = self.route_token.get_secret_value()
        return payload
