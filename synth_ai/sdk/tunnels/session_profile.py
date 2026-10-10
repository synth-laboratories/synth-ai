"""Connector checkpoint enrollment. See backend/notes/specifications/tanha/current/systems/tunnels/v2_catalog.md."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class SessionProfile(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    expected_profile_revision: int = Field(strict=True, ge=0, le=9007199254740990)
    service_identity: str
    semantics: Literal["none", "application_cold_restore", "true_environment_snapshot"]

    @field_validator("service_identity")
    @classmethod
    def bounded_identity(cls, value: str) -> str:
        if value != value.strip() or not 1 <= len(value.encode()) <= 128:
            raise ValueError("invalid checkpoint service identity")
        return value


class SessionProfileState(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    revision: int = Field(strict=True, ge=1, le=9007199254740991)
    service_identity: str
    semantics: Literal["none", "application_cold_restore", "true_environment_snapshot"]

    @field_validator("service_identity")
    @classmethod
    def bounded_identity(cls, value: str) -> str:
        return SessionProfile.bounded_identity(value)
