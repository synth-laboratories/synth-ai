"""Sealed contribution packages and scoped transfer grants. See SDK index_contract.md."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Literal, Self

from pydantic import AwareDatetime, Field, StrictInt, model_validator

from synth_ai.core.local_files import open_regular_file
from synth_ai.types.base import Digest, Identifier, Model, ResponseModel


def validate_upload_headers(
    headers: dict[str, str], *, size: int, digest: str, media_type: str
) -> None:
    """Only exact object-store headers may accompany a credential-isolated PUT."""
    allowed = {
        "Content-Length": str(size),
        "Content-Type": media_type,
        "If-None-Match": "*",
        "x-amz-meta-sha256": digest,
        "x-amz-meta-media-type": media_type,
    }
    if any(key not in allowed or value != allowed[key] for key, value in headers.items()):
        raise ValueError("upload headers do not match the immutable manifest")


class ContributionUploadManifest(Model):
    byte_size: StrictInt = Field(ge=1, le=32 * 1024 * 1024)
    digest_sha256: Digest
    media_type: Literal["application/zip", "application/gzip"]


class ContributionPackageRef(Model):
    contribution_id: Identifier
    revision_id: Identifier
    package_id: Identifier
    digest_sha256: Digest


class UploadStatus(ResponseModel):
    upload_id: Identifier
    contribution_id: Identifier
    state: Literal["prepared", "received", "finalized", "expired", "refused"]
    manifest: ContributionUploadManifest
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


class UploadReceipt(ResponseModel):
    upload_id: Identifier
    reference: ContributionPackageRef
    state: Literal["finalized"]


@dataclass(frozen=True, slots=True)
class UploadSource:
    path: Path
    manifest: ContributionUploadManifest

    @classmethod
    def from_path(
        cls,
        path: Path,
        *,
        media_type: Literal["application/zip", "application/gzip"],
        maximum_bytes: int,
    ) -> UploadSource:
        if type(maximum_bytes) is not int or not 1 <= maximum_bytes <= 32 * 1024 * 1024:
            raise ValueError("source byte bound must be between 1 and 32 MiB")
        digest = sha256()
        size = 0
        with open_regular_file(path) as stream:
            for _ in range(maximum_bytes // 65536 + 2):
                chunk = stream.read(65536)
                if not chunk:
                    break
                size += len(chunk)
                if size > maximum_bytes:
                    raise ValueError("source exceeds selected byte bound")
                digest.update(chunk)
            else:
                raise ValueError("source exceeded bounded read count")
        return cls(
            path=path,
            manifest=ContributionUploadManifest(
                byte_size=size, digest_sha256=digest.hexdigest(), media_type=media_type
            ),
        )


@dataclass(frozen=True, slots=True)
class LocalContributionPackage:
    source: UploadSource


class TransferReceipt(Model):
    upload_id: Identifier
    byte_size: StrictInt
    digest_sha256: Digest
    etag: str | None = Field(max_length=512)
