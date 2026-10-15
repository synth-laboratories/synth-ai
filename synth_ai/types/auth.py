"""Verified principal context shared by both products."""

from typing import Annotated, Literal

from pydantic import Field

from .base import Identifier, OAuthClientIdentifier, ResponseModel


class AuthContext(ResponseModel):
    tenant_id: Identifier
    subject_id: Identifier
    credential_id: Identifier
    credential_kind: Literal["api_key", "oauth"]
    oauth_client_id: OAuthClientIdentifier | None
    subject_kind: Literal["user", "service"]
    scopes: tuple[Annotated[str, Field(pattern=r"^[a-z][a-z0-9_.:-]{0,127}$")], ...] = Field(
        max_length=128
    )
