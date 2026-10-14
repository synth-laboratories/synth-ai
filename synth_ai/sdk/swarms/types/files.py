"""Immutable project files and explicit upload phases. See SDK_0230_IMPLEMENTATION.md."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Literal, Self

from pydantic import AwareDatetime, Field, StrictInt, model_validator

from synth_ai.core.local_files import open_regular_file
from synth_ai.types.base import Digest, Identifier, Model, ResponseModel
from synth_ai.types.uploads import validate_upload_headers

from .launch import FileRef

FileMediaType = Literal[
    "text/plain",
    "text/markdown",
    "application/json",
    "application/pdf",
    "image/png",
    "image/jpeg",
    "image/webp",
]


class FileManifest(Model):
    filename: str = Field(min_length=1, max_length=255)
    byte_size: StrictInt = Field(ge=1, le=32 * 1024 * 1024)
    digest_sha256: Digest
    media_type: FileMediaType

    @model_validator(mode="after")
    def validate_filename(self) -> Self:
        if self.filename in {".", ".."} or any(
            character in self.filename for character in "/\\\x00\r\n"
        ):
            raise ValueError("filename must be a label, not a path")
        return self


class File(ResponseModel):
    project_id: Identifier
    file_id: Identifier
    state: Literal["scanning", "ready", "rejected", "deleted"]
    manifest: FileManifest
    reference: FileRef | None

    @model_validator(mode="after")
    def validate_reference(self) -> Self:
        if self.state == "ready":
            if (
                self.reference is None
                or self.reference.project_id != self.project_id
                or self.reference.file_id != self.file_id
                or self.reference.digest_sha256 != self.manifest.digest_sha256
            ):
                raise ValueError("ready file requires a matching immutable reference")
        elif self.reference is not None:
            raise ValueError("unavailable file cannot be selected as input")
        return self


class FileUploadStatus(ResponseModel):
    upload_id: Identifier
    project_id: Identifier
    state: Literal["prepared", "received", "finalized", "expired", "refused"]
    manifest: FileManifest
    transfer_url: str | None = Field(max_length=4096, repr=False)
    expires_at: AwareDatetime | None
    transfer_headers: dict[str, str] = Field(default_factory=dict, max_length=5, repr=False)

    @model_validator(mode="after")
    def validate_grant(self) -> Self:
        validate_upload_headers(
            self.transfer_headers,
            size=self.manifest.byte_size,
            digest=self.manifest.digest_sha256,
            media_type=self.manifest.media_type,
        )
        if self.state == "prepared" and (self.transfer_url is None or self.expires_at is None):
            raise ValueError("prepared upload requires a scoped expiring transfer grant")
        return self


class FileUploadReceipt(ResponseModel):
    upload_id: Identifier
    file: File
    state: Literal["finalized"]


@dataclass(frozen=True, slots=True)
class FileSource:
    path: Path
    manifest: FileManifest

    @classmethod
    def from_path(
        cls,
        path: Path,
        *,
        media_type: FileMediaType,
        maximum_bytes: int,
        filename: str | None = None,
    ) -> FileSource:
        """Hash bounded local bytes; MIME is explicit and backend scanning is authoritative."""
        if type(maximum_bytes) is not int or not 1 <= maximum_bytes <= 32 * 1024 * 1024:
            raise ValueError("source byte ceiling must be between 1 and 32 MiB")
        digest = sha256()
        size = 0
        with open_regular_file(path) as stream:
            for _ in range(maximum_bytes // 65536 + 2):
                chunk = stream.read(65536)
                if not chunk:
                    break
                size += len(chunk)
                if size > maximum_bytes:
                    raise ValueError("source exceeds selected byte ceiling")
                digest.update(chunk)
            else:
                raise ValueError("source exceeded bounded read count")
        return cls(
            path=path,
            manifest=FileManifest(
                filename=filename or path.name,
                byte_size=size,
                digest_sha256=digest.hexdigest(),
                media_type=media_type,
            ),
        )
