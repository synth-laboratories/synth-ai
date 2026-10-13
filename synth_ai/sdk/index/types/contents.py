"""Immutable content selection and current-rights delivery. See SDK index_contract.md."""

from __future__ import annotations

from hashlib import sha256
from typing import Literal, Self

from pydantic import Field, StrictInt, model_validator

from synth_ai.types.base import Digest, Identifier, Model, ResponseModel


class ContentRef(Model):
    contribution_id: Identifier
    revision_id: Identifier
    asset_id: Identifier
    digest_sha256: Digest


class ContentOptions(Model):
    representation: Literal["text"] = "text"
    maximum_bytes: StrictInt = Field(ge=1, le=65536)


class ContentRequest(Model):
    reference: ContentRef
    options: ContentOptions


class ContentResponse(ResponseModel):
    reference: ContentRef
    status: Literal["available", "denied", "revoked", "missing"]
    text: str | None = Field(max_length=65536)
    media_type: Literal["text/plain"] | None
    start_byte: StrictInt | None = Field(ge=0)
    end_byte: StrictInt | None = Field(ge=0)
    asset_byte_size: StrictInt | None = Field(ge=0)
    truncated: bool

    @model_validator(mode="after")
    def validate_delivery(self) -> Self:
        fields = (self.text, self.media_type, self.start_byte, self.end_byte, self.asset_byte_size)
        if self.status == "available":
            if any(value is None for value in fields):
                raise ValueError("available content requires text, media and exact byte span")
            if (
                self.text is not None
                and self.start_byte is not None
                and self.end_byte is not None
                and self.end_byte - self.start_byte != len(self.text.encode("utf-8"))
            ):
                raise ValueError("byte span does not match delivered UTF-8 text")
            if (
                self.end_byte is not None
                and self.asset_byte_size is not None
                and self.end_byte > self.asset_byte_size
            ):
                raise ValueError("content span exceeds full asset size")
            if (
                self.start_byte is not None
                and self.end_byte is not None
                and self.asset_byte_size is not None
            ):
                partial = self.start_byte != 0 or self.end_byte != self.asset_byte_size
                if self.truncated != partial:
                    raise ValueError("truncation must agree with full asset bounds")
                if (
                    not partial
                    and self.text is not None
                    and sha256(self.text.encode("utf-8")).hexdigest()
                    != self.reference.digest_sha256
                ):
                    raise ValueError(
                        "complete delivered text does not match the selected asset digest"
                    )
        elif any(value is not None for value in fields) or self.truncated:
            raise ValueError("refused content must not disclose bytes or protected metadata")
        return self
