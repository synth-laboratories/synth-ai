# Generated from backend packages/forge_authority/uploads.py.
"""Bounded private Forge publication contract; forge-authority.v1.md."""

import base64
import hashlib
from typing import Literal

from pydantic import Field, model_validator

from synth_ai.sdk.research.contracts.forge.contracts import (
    Contract,
    Digest,
    ExactReference,
    Identifier,
    Scope,
)

MAX_UPLOAD_BYTES = 4 * 1024 * 1024


class UploadObject(Contract):
    logical_path: str = Field(min_length=1, max_length=1024)
    media_type: str = Field(min_length=1, max_length=128)
    content_base64: str = Field(max_length=5592408, repr=False)
    digest_sha256: Digest
    size_bytes: int = Field(strict=True, ge=0, le=MAX_UPLOAD_BYTES)

    @model_validator(mode="after")
    def exact_bytes(self):
        if (
            "\\" in self.logical_path
            or "\x00" in self.logical_path
            or any(part in {"", ".", ".."} for part in self.logical_path.split("/"))
        ):
            raise ValueError("normalized relative path required")
        if len(self.bytes()) != self.size_bytes:
            raise ValueError("declared bytes differ")
        return self

    def bytes(self):
        content = base64.b64decode(self.content_base64, validate=True)
        if (
            len(content) > MAX_UPLOAD_BYTES
            or hashlib.sha256(content).hexdigest() != self.digest_sha256
        ):
            raise ValueError("exact bounded byte digest required")
        return content


class UploadRequest(Contract):
    schema_version: Literal["synth.forge.upload.v1"] = "synth.forge.upload.v1"
    scope: Scope
    operation_id: Identifier
    objects: tuple[UploadObject, ...] = Field(min_length=1, max_length=16)

    @model_validator(mode="after")
    def bounded_unique_objects(self):
        if len({item.logical_path for item in self.objects}) != len(self.objects):
            raise ValueError("unique object paths required")
        if sum(item.size_bytes for item in self.objects) > MAX_UPLOAD_BYTES:
            raise ValueError("publication exceeds 4 MiB")
        if len({item.digest_sha256 for item in self.objects}) != len(self.objects):
            raise ValueError("unique object digests required for exact reference lookup")
        return self


class UploadedObject(Contract):
    logical_path: str
    media_type: str
    size_bytes: int
    reference: ExactReference


class Uploaded(Contract):
    schema_version: Literal["synth.forge.upload.v1"] = "synth.forge.upload.v1"
    result: Literal["committed"] = "committed"
    scope: Scope
    operation_id: Identifier
    request_digest_sha256: Digest
    publication_id: str
    collection_id: str
    revision: int = Field(strict=True, ge=1)
    manifest: ExactReference
    objects: tuple[UploadedObject, ...] = Field(min_length=1, max_length=16)
