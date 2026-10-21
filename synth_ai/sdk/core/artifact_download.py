"""Exact artifact object downloads share the managed byte-response lifetime."""

import os
import re
import time
from pathlib import Path
from typing import cast

from pydantic import Field, StrictInt

from synth_ai.contracts.core import ARTIFACT_CONTENT
from synth_ai.core.http.response import classify_exchange
from synth_ai.errors import ContractError
from synth_ai.options import TransferOptions
from synth_ai.types.base import Model

from .artifact_types import (
    MAX_ARTIFACT_BYTES,
    Artifact,
    ArtifactDigestMismatch,
    ArtifactDownloadReceipt,
    ArtifactObject,
)
from .repo_archive import (
    ArchiveTransfer,
    AsyncArchive,
    SyncArchive,
    _download_owned,
    _download_owned_async,
    archive_destination,
)
from .types import RepositoryArchiveOptions


class ArtifactTransferOptions(TransferOptions):
    maximum_bytes: StrictInt = Field(ge=1, le=MAX_ARTIFACT_BYTES)


class ArtifactDownloadOptions(Model):
    timeout_seconds: float = Field(default=900, gt=0, le=900)
    maximum_bytes: StrictInt = Field(default=MAX_ARTIFACT_BYTES, ge=1, le=MAX_ARTIFACT_BYTES)


def invalid(message):
    return ContractError(
        message,
        code="artifact_delivery_invalid",
        stage="artifact_download",
        operation=ARTIFACT_CONTENT.id,
    )


class BlobPlan(ArchiveTransfer):
    operation = ARTIFACT_CONTENT
    artifact: Artifact
    selected: ArtifactObject

    def query(self) -> dict[str, str | int]:
        return {
            "path": self.selected.logical_path,
            "digest_sha256": self.artifact.ref.digest_sha256,
        }

    def headers(self, response):
        self.remaining()
        disposition = classify_exchange(
            response,
            None,
            contract="core.v1",
            mutation=False,
            identity=None,
            attempt=1,
            attempts=1,
            operation=self.operation.id,
        )
        if disposition.error is not None:
            raise disposition.error
        if response.headers.get("Content-Encoding", "identity").lower() != "identity":
            raise invalid("Compressed artifacts are outside the byte contract")
        if response.status_code != 200:
            return
        expected = {
            "X-Synth-Contract": "core.v1",
            "X-Synth-Artifact-Id": self.artifact.artifact_id,
            "X-Synth-Artifact-Revision": self.artifact.ref.revision,
            "X-Synth-Artifact-Manifest-SHA256": self.artifact.ref.digest_sha256,
            "X-Synth-Content-SHA256": self.selected.digest_sha256,
            "ETag": '"' + self.selected.digest_sha256 + '"',
        }
        if any(response.headers.get(key) != value for key, value in expected.items()):
            raise invalid("Artifact response changed selected identity")
        if (
            response.headers.get("Content-Type", "").split(";", 1)[0].strip().lower()
            != self.selected.media_type.lower()
        ):
            raise invalid("Artifact media type changed")
        length = response.headers.get("Content-Length")
        if length is not None and (
            re.fullmatch(r"[0-9]{1,20}", length) is None or int(length) != self.selected.size_bytes
        ):
            raise invalid("Artifact content length changed")
        if self.selected.size_bytes > self.options.maximum_bytes:
            raise invalid("Artifact exceeds the selected byte bound")
        self.size, self.digest = self.selected.size_bytes, self.selected.digest_sha256

    def consume(self, chunk):
        self.remaining()
        self.received += len(chunk)
        if self.received > self.size:
            raise invalid("Artifact exceeded its immutable size")
        self.hasher.update(chunk)

    def finish_bytes(self):
        self.remaining()
        if self.received != self.size:
            raise invalid("Artifact ended before its immutable size")
        if self.hasher.hexdigest() != self.digest:
            raise ArtifactDigestMismatch(self.digest, self.hasher.hexdigest())

    def extract(self, archive):
        self.remaining()
        return archive

    def publish(self, tree) -> ArtifactDownloadReceipt:
        with self.publication_lock:
            self.remaining()
            receipt = ArtifactDownloadReceipt(
                ref=self.reference,
                path=self.selected.logical_path,
                destination=self.destination,
                byte_size=self.received,
                digest_sha256=self.digest,
            )
            os.link(tree, self.destination)
            return receipt


class SyncBlob(BlobPlan, SyncArchive):
    pass


class AsyncBlob(BlobPlan, AsyncArchive):
    pass


def selected_object(artifact, path):
    if path is None:
        return artifact.manifest.objects[0]
    for item in artifact.manifest.objects:
        if item.logical_path == path:
            return item
    raise ValueError("Selected path is absent from the exact artifact manifest")


def download_object(
    transport, artifact: Artifact, destination: Path, *, path=None, options=None, deadline=None
):
    options = options or ArtifactDownloadOptions()
    selected = selected_object(artifact, path)
    state = SyncBlob(
        transport,
        artifact.ref,
        archive_destination(destination),
        RepositoryArchiveOptions(
            timeout_seconds=options.timeout_seconds,
            maximum_bytes=options.maximum_bytes,
            maximum_files=1,
        ),
        deadline if deadline is not None else time.monotonic() + options.timeout_seconds,
    )
    state.artifact, state.selected = artifact, selected
    state.accept, state.request_digest = selected.media_type, selected.digest_sha256
    target = f"/projects/{artifact.project_id}/artifacts/{artifact.artifact_id}/revisions/{artifact.ref.revision}/content"
    return cast(ArtifactDownloadReceipt, _download_owned(transport, target, state))


async def download_object_async(
    transport, artifact: Artifact, destination: Path, *, path=None, options=None, deadline=None
):
    options = options or ArtifactDownloadOptions()
    selected = selected_object(artifact, path)
    state = AsyncBlob(
        artifact.ref,
        archive_destination(destination),
        RepositoryArchiveOptions(
            timeout_seconds=options.timeout_seconds,
            maximum_bytes=options.maximum_bytes,
            maximum_files=1,
        ),
        deadline if deadline is not None else time.monotonic() + options.timeout_seconds,
    )
    state.artifact, state.selected = artifact, selected
    state.accept, state.request_digest = selected.media_type, selected.digest_sha256
    target = f"/projects/{artifact.project_id}/artifacts/{artifact.artifact_id}/revisions/{artifact.ref.revision}/content"
    return cast(ArtifactDownloadReceipt, await _download_owned_async(transport, target, state))
