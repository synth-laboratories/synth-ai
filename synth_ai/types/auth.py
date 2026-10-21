"""Verified principal context shared by both products."""

from typing import Annotated, Literal

from pydantic import Field

from .base import OAuthClientIdentifier, ResponseIdentifier, ResponseModel


class AuthContext(ResponseModel):
    tenant_id: ResponseIdentifier
    subject_id: ResponseIdentifier
    credential_id: ResponseIdentifier
    credential_kind: Literal["api_key", "oauth"]
    oauth_client_id: OAuthClientIdentifier | None
    subject_kind: Literal["user", "service"]
    scopes: tuple[Annotated[str, Field(pattern=r"^[a-z][a-z0-9_.:-]{0,127}$")], ...] = Field(
        max_length=128
    )
