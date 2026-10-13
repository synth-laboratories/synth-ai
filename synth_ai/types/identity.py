"""Retained mutation identity. See SDK index_contract.md."""

from __future__ import annotations

from hashlib import sha256
from uuid import uuid4

from pydantic import Field

from .base import Digest, Identifier, Model


class MutationIdentity(Model):
    key: Identifier
    operation: Identifier
    resource_path: str = Field(min_length=1, max_length=1024, pattern=r"^/api/v1/[A-Za-z0-9_./-]+$")
    payload_sha256: Digest

    @classmethod
    def create(
        cls, operation: str, payload: bytes, resource_path: str, *, key: str | None = None
    ) -> MutationIdentity:
        return cls(
            key=key if key is not None else uuid4().hex,
            operation=operation,
            resource_path=resource_path,
            payload_sha256=sha256(payload).hexdigest(),
        )
