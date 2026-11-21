"""Typed local credentials and OAuth wire records; secrets never appear in repr."""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field


class CredentialRecord(BaseModel):
    model_config = ConfigDict(
        strict=True, frozen=True, hide_input_in_errors=True, extra="ignore", allow_inf_nan=False
    )
    access_token: str = Field(min_length=1, repr=False)
    refresh_token: str | None = Field(default=None, repr=False)
    client_id: str | None = None
    expires_at: float = 0


class OAuthToken(BaseModel):
    model_config = ConfigDict(
        strict=True, frozen=True, hide_input_in_errors=True, extra="ignore", allow_inf_nan=False
    )
    access_token: str = Field(min_length=1, repr=False)
    refresh_token: str | None = Field(default=None, repr=False)
    expires_in: float = Field(gt=0)


class OAuthMetadata(BaseModel):
    model_config = ConfigDict(strict=True, frozen=True, hide_input_in_errors=True, extra="ignore")
    issuer: str | None = None
    code_challenge_methods_supported: list[str] = Field(default_factory=list)
    token_endpoint: str | None = None
    registration_endpoint: str | None = None
    authorization_endpoint: str | None = None
    revocation_endpoint: str | None = None
    device_authorization_endpoint: str | None = None


class DeviceAuthorization(BaseModel):
    """RFC 8628 section 3.2 response. ``device_code`` is a credential: never shown."""

    model_config = ConfigDict(strict=True, frozen=True, hide_input_in_errors=True, extra="ignore")
    device_code: str = Field(min_length=1, repr=False)
    user_code: str = Field(min_length=1)
    verification_uri: str = Field(min_length=1)
    verification_uri_complete: str | None = None
    expires_in: int = Field(gt=0)
    interval: int = Field(default=5, gt=0)


class ClientRegistration(BaseModel):
    model_config = ConfigDict(strict=True, frozen=True, hide_input_in_errors=True, extra="ignore")
    client_id: str = Field(min_length=1)


class CredentialSource(StrEnum):
    ENVIRONMENT = "environment"
    OAUTH = "oauth"
    NONE = "none"


class AuthStatus(BaseModel):
    model_config = ConfigDict(strict=True, frozen=True, hide_input_in_errors=True, extra="forbid")
    base_url: str
    authenticated: bool
    source: CredentialSource
    expires_at: float | None = None
