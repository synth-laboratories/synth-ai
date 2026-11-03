"""Resumable, digest-verified download of a retained project trace (SYN-5347).

``mnemosyne_read_project_trace`` with ``range`` returns one window of at most
1 MiB of an immutable retained archive of at most 8 MiB. Every window is bound
to the archive's full ``archive_sha256``/``archive_size_bytes`` (from the
project trace list) and carries its own ``chunk_sha256`` and ``next_offset``.

The downloader streams windows to ``<dir>/.<name>.<archive_sha256>.part`` so
memory stays bounded by one window. An interrupted download resumes from the
partial file's length on the next call with the same source and destination.
Each window is verified before it is appended; the reassembled file is then
re-hashed in 64 KiB blocks and must match the source size and digest before it
is atomically renamed onto the destination. A full-digest mismatch deletes the
partial (its bytes are unusable) and refuses; a window mismatch refuses and
keeps the verified prefix for a later resume.
"""

from __future__ import annotations

import base64
import binascii
import hashlib
import os
from pathlib import Path
from typing import Annotated, Any, Literal

from pydantic import Field, StrictBool, StrictInt, ValidationError, model_validator

from synth_ai.types.base import Digest, Model, ResponseBase

from .actor_tools import (
    PROJECT_TRACE_ARCHIVE_BYTES_MAX,
    PROJECT_TRACE_CHUNK_BYTES_MAX,
    READ_PROJECT_TRACE_TOOL,
    ActorToolCaller,
    ActorTraceRefusal,
    AsyncActorToolCaller,
    refuse,
    require_tool,
    structured_result,
)

AssociationId = Annotated[str, Field(strict=True, pattern=r"^mna_[0-9a-f]{64}$")]
ArchiveSize = Annotated[StrictInt, Field(ge=1, le=PROJECT_TRACE_ARCHIVE_BYTES_MAX)]
ChunkLength = Annotated[StrictInt, Field(ge=1, le=PROJECT_TRACE_CHUNK_BYTES_MAX)]
# base64 of one maximal window; larger text is refused before decoding.
_ENCODED_CHUNK_MAX = 4 * ((PROJECT_TRACE_CHUNK_BYTES_MAX + 2) // 3)
_HASH_BLOCK = 64 * 1024
CHUNK_SCHEMA = "synth.mnemosyne.actor-project-trace-chunk.v1"


class RetainedTraceSource(Model):
    """The exact retained archive to fetch, as listed by the project."""

    association_id: AssociationId
    archive_sha256: Digest
    archive_size_bytes: ArchiveSize


class ProjectTraceRange(Model):
    archive_sha256: Digest
    archive_size_bytes: ArchiveSize
    offset: Annotated[StrictInt, Field(ge=0, lt=PROJECT_TRACE_ARCHIVE_BYTES_MAX)]
    length: ChunkLength = PROJECT_TRACE_CHUNK_BYTES_MAX

    @model_validator(mode="after")
    def offset_inside_archive(self):
        if self.offset >= self.archive_size_bytes:
            raise ValueError("trace range offset outside archive")
        return self


class ProjectTraceRead(Model):
    """Arguments of ``mnemosyne_read_project_trace`` with a range window."""

    association_id: AssociationId
    range: ProjectTraceRange


class ProjectTraceChunk(ResponseBase):
    schema_version: Literal["synth.mnemosyne.actor-project-trace-chunk.v1"]
    association_id: AssociationId
    rollout_id: Annotated[str, Field(min_length=1, max_length=128)]
    execution_epoch: Annotated[str, Field(min_length=1, max_length=128)]
    archive_sha256: Digest
    archive_size_bytes: ArchiveSize
    offset: Annotated[StrictInt, Field(ge=0, lt=PROJECT_TRACE_ARCHIVE_BYTES_MAX)]
    byte_count: Annotated[StrictInt, Field(ge=1)]
    chunk_sha256: Digest
    next_offset: Annotated[StrictInt, Field(ge=1)] | None
    complete: StrictBool
    encoding: Literal["base64"]
    content: Annotated[str, Field(max_length=_ENCODED_CHUNK_MAX)]


class TraceDownloadReceipt(Model):
    association_id: AssociationId
    rollout_id: str | None
    execution_epoch: str | None
    archive_sha256: Digest
    archive_size_bytes: ArchiveSize
    path: str
    resumed_from: Annotated[StrictInt, Field(ge=0)]
    chunks_fetched: Annotated[StrictInt, Field(ge=0)]


def partial_path(source: RetainedTraceSource, destination: Path) -> Path:
    """The partial file is named by the archive digest: it never mixes sources."""
    return destination.parent / f".{destination.name}.{source.archive_sha256}.part"


class _Download:
    """Pure window state machine shared by the sync and async loops."""

    def __init__(self, source: RetainedTraceSource, destination: Path, chunk_bytes: int):
        if not 1 <= chunk_bytes <= PROJECT_TRACE_CHUNK_BYTES_MAX:
            raise refuse(
                ActorTraceRefusal.CHUNK_TOO_LARGE,
                "request",
                f"chunk_bytes {chunk_bytes} outside 1..{PROJECT_TRACE_CHUNK_BYTES_MAX}",
            )
        self.source = source
        self.destination = destination
        self.partial = partial_path(source, destination)
        self.chunk_bytes = chunk_bytes
        self.offset = self.partial.stat().st_size if self.partial.exists() else 0
        self.resumed_from = self.offset
        self.fetched = 0
        self.origin: tuple[str, str] | None = None
        if self.offset > source.archive_size_bytes:
            raise refuse(
                ActorTraceRefusal.PARTIAL_INVALID,
                "resume",
                f"partial holds {self.offset} bytes; archive is {source.archive_size_bytes}",
            )

    @property
    def done(self) -> bool:
        return self.offset == self.source.archive_size_bytes

    def arguments(self) -> dict[str, Any]:
        window = ProjectTraceRange(
            archive_sha256=self.source.archive_sha256,
            archive_size_bytes=self.source.archive_size_bytes,
            offset=self.offset,
            length=min(self.chunk_bytes, self.source.archive_size_bytes - self.offset),
        )
        read = ProjectTraceRead(association_id=self.source.association_id, range=window)
        return read.model_dump(mode="json")

    def accept(self, result: Any, requested: int) -> None:
        stage = f"chunk@{self.offset}"
        try:
            chunk = ProjectTraceChunk.model_validate_json(structured_result(result, stage))
        except ValidationError as error:
            raise refuse(ActorTraceRefusal.RESULT_INVALID, stage, _first_error(error)) from None
        source = self.source
        if (
            chunk.association_id != source.association_id
            or chunk.archive_sha256 != source.archive_sha256
            or chunk.archive_size_bytes != source.archive_size_bytes
            or chunk.offset != self.offset
        ):
            raise refuse(ActorTraceRefusal.CHUNK_MISMATCH, stage, "window bound to another source")
        end = chunk.offset + chunk.byte_count
        if chunk.byte_count > requested or end > source.archive_size_bytes:
            raise refuse(
                ActorTraceRefusal.CHUNK_TOO_LARGE,
                stage,
                f"{chunk.byte_count} bytes exceed the requested {requested}",
            )
        expected_next = None if end == source.archive_size_bytes else end
        if chunk.next_offset != expected_next or chunk.complete != (expected_next is None):
            raise refuse(ActorTraceRefusal.CHUNK_MISMATCH, stage, "next_offset disagrees")
        if self.origin not in (None, (chunk.rollout_id, chunk.execution_epoch)):
            raise refuse(ActorTraceRefusal.CHUNK_MISMATCH, stage, "rollout changed mid-download")
        try:
            data = base64.b64decode(chunk.content, validate=True)
        except (binascii.Error, ValueError):
            raise refuse(ActorTraceRefusal.RESULT_INVALID, stage, "content is not base64") from None
        if len(data) != chunk.byte_count:
            raise refuse(ActorTraceRefusal.CHUNK_MISMATCH, stage, "byte_count disagrees")
        if hashlib.sha256(data).hexdigest() != chunk.chunk_sha256:
            raise refuse(ActorTraceRefusal.CHUNK_DIGEST_MISMATCH, stage, "chunk sha256 differs")
        self.origin = (chunk.rollout_id, chunk.execution_epoch)
        self.partial.parent.mkdir(parents=True, exist_ok=True)
        with self.partial.open("ab") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        self.offset = end
        self.fetched += 1

    def finish(self) -> TraceDownloadReceipt:
        digest, size = hashlib.sha256(), 0
        with self.partial.open("rb") as handle:
            while block := handle.read(_HASH_BLOCK):
                digest.update(block)
                size += len(block)
        if (
            size != self.source.archive_size_bytes
            or digest.hexdigest() != self.source.archive_sha256
        ):
            self.partial.unlink()
            raise refuse(
                ActorTraceRefusal.ARCHIVE_DIGEST_MISMATCH,
                "reassemble",
                "reassembled archive differs from the listed sha256/size; partial removed",
            )
        os.replace(self.partial, self.destination)
        rollout, epoch = self.origin if self.origin is not None else (None, None)
        return TraceDownloadReceipt(
            association_id=self.source.association_id,
            rollout_id=rollout,
            execution_epoch=epoch,
            archive_sha256=self.source.archive_sha256,
            archive_size_bytes=self.source.archive_size_bytes,
            path=str(self.destination),
            resumed_from=self.resumed_from,
            chunks_fetched=self.fetched,
        )


def _first_error(error: ValidationError) -> str:
    first = error.errors(include_input=False)[0]
    return f"{'.'.join(str(p) for p in first['loc'])}: {first['msg']}"


def download_project_trace(
    caller: ActorToolCaller,
    source: RetainedTraceSource,
    destination: str | os.PathLike[str],
    *,
    chunk_bytes: int = PROJECT_TRACE_CHUNK_BYTES_MAX,
) -> TraceDownloadReceipt:
    """Download (or resume) ``source`` to ``destination`` through the actor tool."""
    require_tool(caller, READ_PROJECT_TRACE_TOOL)
    state = _Download(source, Path(destination), chunk_bytes)
    while not state.done:
        arguments = state.arguments()
        state.accept(
            caller.call_tool(READ_PROJECT_TRACE_TOOL, arguments),
            arguments["range"]["length"],
        )
    return state.finish()


async def adownload_project_trace(
    caller: AsyncActorToolCaller,
    source: RetainedTraceSource,
    destination: str | os.PathLike[str],
    *,
    chunk_bytes: int = PROJECT_TRACE_CHUNK_BYTES_MAX,
) -> TraceDownloadReceipt:
    require_tool(caller, READ_PROJECT_TRACE_TOOL)
    state = _Download(source, Path(destination), chunk_bytes)
    while not state.done:
        arguments = state.arguments()
        state.accept(
            await caller.call_tool(READ_PROJECT_TRACE_TOOL, arguments),
            arguments["range"]["length"],
        )
    return state.finish()
