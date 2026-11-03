"""Immutable authorized output metadata and local delivery receipts.

See the public contract snapshots in contracts/ Availability is independent
of run completion; metadata is not permission to deliver bytes later.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Literal, Self

from pydantic import Field, StrictInt, model_validator

from synth_ai.types.base import (
    Digest,
    Identifier,
    Model,
    Page,
    ResponseBase,
    ResponseIdentifier,
    ResponseModel,
)

MediaType = Annotated[
    str,
    Field(min_length=3, max_length=255, pattern=r"^[A-Za-z0-9!#$&^_.+-]+/[A-Za-z0-9!#$&^_.+-]+$"),
]


class OutputRef(Model):
    run_id: Identifier
    output_id: Identifier
    revision_id: Identifier
    digest_sha256: Digest
    byte_size: StrictInt = Field(ge=0, le=2**63 - 1)
    media_type: MediaType


class Output(ResponseModel):
    run_id: ResponseIdentifier
    output_id: ResponseIdentifier
    state: Literal["pending", "available", "withheld", "expired", "deleted"]
    reference: OutputRef | None
    filename: str | None = Field(default=None, min_length=1, max_length=255)

    @model_validator(mode="after")
    def validate_availability(self) -> Self:
        if self.state == "available":
            if self.reference is None:
                raise ValueError("available output requires an immutable reference")
            if self.reference.run_id != self.run_id or self.reference.output_id != self.output_id:
                raise ValueError("output descriptor and reference identities differ")
        elif self.reference is not None or self.filename is not None:
            raise ValueError("unavailable output cannot expose protected metadata")
        if self.filename is not None and any(char in self.filename for char in "/\\\x00\r\n"):
            raise ValueError("filename must be a label, not a filesystem path")
        return self


VerificationCode = Annotated[str, Field(min_length=1, max_length=128)]


class OutputVerificationError(ResponseBase):
    """Why a code output's public execution verification failed; never retryable."""

    code: VerificationCode
    refusal_code: VerificationCode | None = None
    message: str = Field(min_length=1, max_length=512)
    detail: str = Field(default="", max_length=2000)
    retryable: Literal[False] = False


class OutputVerification(ResponseBase):
    """Public execution verification of one pushed code output (SYN-4548).

    ``pending`` always names a ``reason``; ``failed`` always carries an
    ``error``; ``passed`` carries neither.
    """

    operation_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    status: Literal["pending", "passed", "failed"]
    reason: VerificationCode | None = None
    message: str | None = Field(default=None, max_length=512)
    error: OutputVerificationError | None = None
    attempts: StrictInt = Field(ge=0)

    @model_validator(mode="after")
    def validate_status(self) -> Self:
        if (self.status == "pending") != (self.reason is not None):
            raise ValueError("pending verification requires exactly one reason")
        if (self.status == "failed") != (self.error is not None):
            raise ValueError("failed verification requires exactly one error")
        return self


class OutputPage(Page[Output]):
    """Run outputs plus the public execution verification of pushed code.

    ``verifications`` is additive: it lists every code-publication verification
    of the run on the first page (``cursor`` absent) and is empty on later
    pages; older consumers ignore it.
    """

    verifications: tuple[OutputVerification, ...] = Field(default=(), max_length=1000)


class TransferTiming(ResponseBase):
    total_ms: float = Field(ge=0)
    first_byte_ms: float | None = Field(ge=0)
    source: Literal["sdk_monotonic", "server_monotonic"] = "sdk_monotonic"


class TextContent(ResponseModel):
    reference: OutputRef
    text: str = Field(max_length=65536)
    complete: Literal[True] = True
    transfer_timing: TransferTiming


@dataclass(frozen=True, slots=True)
class DownloadReceipt:
    reference: OutputRef
    destination: Path
    byte_size: int
    digest_sha256: str
    transfer_timing: TransferTiming
