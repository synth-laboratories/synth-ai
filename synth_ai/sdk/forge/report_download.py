"""Download one artifact a report version pins, as the caller (SYN-5158 gap 3).

``forge.report_reader.download`` serves the exact bytes of one artifact the
opened version's manifest pins, to a project reader or a live share grantee of
that version. The SDK opens (and re-verifies) the version first, so the
expected digest, size and media type come from the verified manifest, never
from the response. The body is written to a scratch file beside the
destination, its size and sha256 are checked, and only then is it linked into
place. The server withholds the final chunk until it has re-verified the bytes
and the reader's authority, so a late refusal ends the body short.
"""

import base64
import os
import re
import tempfile
import time
from pathlib import Path
from typing import cast

from pydantic import Field, StrictInt

from synth_ai.contracts.forge import REPORT_READER_DOWNLOAD
from synth_ai.core.http.response import classify_exchange
from synth_ai.errors import ContractError
from synth_ai.types.base import Model

from ..core.repo_archive import (
    ArchiveTransfer,
    AsyncArchive,
    SyncArchive,
    _download_owned,
    _download_owned_async,
    archive_destination,
    remove_owned_tree,
)
from .contracts.contracts import Digest, Identifier, Scope
from .records import project_identity
from .types import (
    REPORT_ARTIFACT_INLINE_MAX_BYTES,
    ReportArtifactContent,
    ReportArtifactContentResponse,
)

# Same object bound as Forge evidence custody uploads.
REPORT_ARTIFACT_MAX_BYTES = 625 * 1024**3


class ReportArtifactDownloadOptions(Model):
    timeout_seconds: float = Field(default=900, gt=0, le=900)
    maximum_bytes: StrictInt = Field(
        default=REPORT_ARTIFACT_MAX_BYTES, ge=1, le=REPORT_ARTIFACT_MAX_BYTES
    )


class ReportArtifactDownloadReceipt(Model):
    scope: Scope
    version_sha256: Digest
    artifact_id: Identifier
    role: str
    access: str
    destination: Path
    byte_size: StrictInt = Field(ge=1)
    digest_sha256: Digest
    media_type: str


def invalid(message, code="report_artifact_invalid"):
    return ContractError(
        message, code=code, stage="report_artifact_download", operation=REPORT_READER_DOWNLOAD.id
    )


def manifest_artifact(document, artifact_id):
    """The verified version's manifest entry, or a local ``not_in_manifest``."""
    for item in document.version.manifest.artifacts:
        if item.artifact_id == artifact_id:
            return item
    raise invalid(
        f"report version {document.version_sha256[:12]} pins no artifact {artifact_id}",
        "not_in_manifest",
    )


class ReportArtifactPlan(ArchiveTransfer):
    operation = REPORT_READER_DOWNLOAD

    def bind(self, document, item):
        self.document, self.item = document, item
        self.accept = "*/*"
        return self

    def query(self):
        return {}

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
            "X-Synth-Content-SHA256": self.item.content_sha256,
            "X-Forge-Report-Version-SHA256": self.document.version_sha256,
            "X-Forge-Report-Artifact-Id": self.item.artifact_id,
        }
        if any(response.headers.get(key) != value for key, value in expected.items()):
            raise invalid("Artifact response changed the selected identity")
        media = response.headers.get("Content-Type", "").split(";", 1)[0].strip().lower()
        if self.item.media_type is not None and media != self.item.media_type.lower():
            raise invalid("Artifact media type differs from the manifest")
        length = response.headers.get("Content-Length", "")
        if re.fullmatch(r"[0-9]{1,20}", length) is None or int(length) < 1:
            raise invalid("Artifact response lacks its exact length")
        size = int(length)
        if self.item.byte_count is not None and size != self.item.byte_count:
            raise invalid("Artifact length differs from the manifest")
        if size > self.options.maximum_bytes:
            raise invalid("Artifact exceeds the selected byte bound")
        self.size, self.digest, self.media_type = size, self.item.content_sha256, media
        self.access = response.headers.get("X-Forge-Report-Access", "")

    def consume(self, chunk):
        self.remaining()
        self.received += len(chunk)
        if self.received > self.size:
            raise invalid("Artifact exceeded its exact length")
        self.hasher.update(chunk)

    def finish_bytes(self):
        self.remaining()
        if self.received != self.size:
            raise invalid("Artifact ended before its exact length", "report_artifact_truncated")
        if self.hasher.hexdigest() != self.digest:
            raise invalid("Artifact bytes differ from the pinned digest", "digest_mismatch")

    def prepare(self):
        self.remaining()
        self.scratch = Path(
            tempfile.mkdtemp(prefix=".synth-report-artifact-", dir=self.destination.parent)
        )
        return self.scratch / "artifact"

    def extract(self, archive):
        self.remaining()
        return archive

    async def extract_owned(self, archive):
        return self.extract(archive)

    def publish(self, tree):
        with self.publication_lock:
            self.remaining()
            receipt = ReportArtifactDownloadReceipt(
                scope=self.document.scope,
                version_sha256=self.document.version_sha256,
                artifact_id=self.item.artifact_id,
                role=self.item.role,
                access=self.access,
                destination=self.destination,
                byte_size=self.received,
                digest_sha256=self.digest,
                media_type=self.media_type,
            )
            os.link(tree, self.destination)
            return receipt


class SyncReportArtifact(ReportArtifactPlan, SyncArchive):
    pass


class AsyncReportArtifact(ReportArtifactPlan, AsyncArchive):
    pass


def _target(project_id, digest, artifact_id):
    return REPORT_READER_DOWNLOAD.path_for(
        project_id=project_id, digest=digest, artifact_id=artifact_id
    )


def _identifier(artifact_id):
    if (
        not isinstance(artifact_id, str)
        or re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_.-]{0,127}", artifact_id) is None
    ):
        raise ValueError("artifact_id must be a manifest artifact identifier")
    return artifact_id


def _sync_transfer(reader, opened, project_id, artifact_id, destination, options):
    deadline = time.monotonic() + options.timeout_seconds
    item = manifest_artifact(opened.document, artifact_id)
    state = SyncReportArtifact(reader._transport, None, destination, options, deadline)
    state.bind(opened.document, item)
    target = _target(project_id, opened.document.version_sha256, artifact_id)
    return cast(ReportArtifactDownloadReceipt, _download_owned(reader._transport, target, state))


async def _async_transfer(reader, opened, project_id, artifact_id, destination, options):
    deadline = time.monotonic() + options.timeout_seconds
    item = manifest_artifact(opened.document, artifact_id)
    state = AsyncReportArtifact(None, destination, options, deadline)
    state.bind(opened.document, item)
    target = _target(project_id, opened.document.version_sha256, artifact_id)
    return cast(
        ReportArtifactDownloadReceipt,
        await _download_owned_async(reader._transport, target, state),
    )


def download_artifact(reader, project_id, digest, artifact_id, destination, *, options=None):
    options = options or ReportArtifactDownloadOptions()
    project_id, artifact_id = project_identity(project_id), _identifier(artifact_id)
    destination = archive_destination(destination)
    opened = reader.open(project_id, digest)
    return _sync_transfer(reader, opened, project_id, artifact_id, destination, options)


async def download_artifact_async(
    reader, project_id, digest, artifact_id, destination, *, options=None
):
    options = options or ReportArtifactDownloadOptions()
    project_id, artifact_id = project_identity(project_id), _identifier(artifact_id)
    destination = archive_destination(destination)
    opened = await reader.open(project_id, digest)
    return await _async_transfer(reader, opened, project_id, artifact_id, destination, options)


def _inline_options(maximum_bytes):
    if not 1 <= maximum_bytes <= REPORT_ARTIFACT_INLINE_MAX_BYTES:
        raise ValueError("inline artifact bound exceeds REPORT_ARTIFACT_INLINE_MAX_BYTES")
    return ReportArtifactDownloadOptions(maximum_bytes=maximum_bytes)


def _content(receipt, opened):
    content = receipt.destination.read_bytes()
    return ReportArtifactContentResponse(
        artifact=ReportArtifactContent(
            scope=receipt.scope,
            version_sha256=receipt.version_sha256,
            artifact_id=receipt.artifact_id,
            role=receipt.role,
            access=receipt.access,
            content_sha256=receipt.digest_sha256,
            size_bytes=receipt.byte_size,
            media_type=receipt.media_type,
            content_base64=base64.b64encode(content).decode("ascii"),
        ),
        metadata=opened.metadata,
    )


def read_artifact(reader, project_id, digest, artifact_id, *, maximum_bytes):
    """Bounded verified bytes as data (MCP); larger artifacts use ``download``."""
    options = _inline_options(maximum_bytes)
    project_id, artifact_id = project_identity(project_id), _identifier(artifact_id)
    opened = reader.open(project_id, digest)
    scratch = Path(tempfile.mkdtemp(prefix=".synth-report-read-"))
    try:
        receipt = _sync_transfer(
            reader, opened, project_id, artifact_id, scratch / "artifact", options
        )
        return _content(receipt, opened)
    finally:
        remove_owned_tree(scratch)


async def read_artifact_async(reader, project_id, digest, artifact_id, *, maximum_bytes):
    options = _inline_options(maximum_bytes)
    project_id, artifact_id = project_identity(project_id), _identifier(artifact_id)
    opened = await reader.open(project_id, digest)
    scratch = Path(tempfile.mkdtemp(prefix=".synth-report-read-"))
    try:
        receipt = await _async_transfer(
            reader, opened, project_id, artifact_id, scratch / "artifact", options
        )
        return _content(receipt, opened)
    finally:
        remove_owned_tree(scratch)
