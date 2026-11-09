"""Stable replayable upload sources with bounded, cancellation-safe disk work."""

from __future__ import annotations

import asyncio
import hashlib
import os
import stat
from collections.abc import AsyncGenerator, Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, TypeVar

from ._limits import UPLOAD_MAX_BYTES
from .errors import DataIntegrityError

T = TypeVar("T")
CHUNK_BYTES = 64 << 10


class FileIO:
    def __init__(self) -> None:
        self._slots = asyncio.Semaphore(4)

    async def run(self, action: Callable[[], T]) -> T:
        async with self._slots:
            task = asyncio.create_task(asyncio.to_thread(action))
            try:
                return await asyncio.shield(task)
            except asyncio.CancelledError:
                # A Python thread cannot be cancelled. Join it before allowing
                # the caller to close/reuse a descriptor it may still access.
                try:
                    await task
                finally:
                    raise


@dataclass(frozen=True)
class FileIdentity:
    device: int
    inode: int
    size: int
    modified_ns: int
    changed_ns: int

    @classmethod
    def read(cls, value: os.stat_result) -> FileIdentity:
        return cls(value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns, value.st_ctime_ns)


@dataclass(frozen=True)
class FileSource:
    path: Path
    stream: BinaryIO
    identity: FileIdentity
    digest_sha256: str

    @property
    def size(self) -> int:
        return self.identity.size

    def verify(self) -> None:
        try:
            current = FileIdentity.read(self.path.stat())
        except OSError as error:
            raise DataIntegrityError("upload source is no longer available") from error
        if (
            current != self.identity
            or FileIdentity.read(os.fstat(self.stream.fileno())) != self.identity
        ):
            raise DataIntegrityError("upload source changed during transfer")

    @classmethod
    def prepare(cls, path: Path) -> FileSource:
        descriptor = os.open(
            path, os.O_RDONLY | getattr(os, "O_NONBLOCK", 0) | getattr(os, "O_BINARY", 0)
        )
        try:
            info = os.fstat(descriptor)
            if not stat.S_ISREG(info.st_mode):
                raise ValueError("upload source must be a regular file")
            stream = os.fdopen(descriptor, "rb")
        except BaseException:
            os.close(descriptor)
            raise
        try:
            identity = FileIdentity.read(info)
            if identity.size > UPLOAD_MAX_BYTES:
                raise ValueError("File exceeds the 32 MiB limit")
            digest = hashlib.sha256()
            size = 0
            while chunk := stream.read(CHUNK_BYTES):
                size += len(chunk)
                if size > UPLOAD_MAX_BYTES:
                    raise ValueError("File exceeds the 32 MiB limit")
                digest.update(chunk)
            source = cls(path, stream, identity, digest.hexdigest())
            source.verify()
            if size != identity.size:
                raise DataIntegrityError("upload size changed during hashing")
            stream.seek(0)
            return source
        except BaseException:
            stream.close()
            raise

    async def chunks(self, io: FileIO) -> AsyncGenerator[bytes, None]:
        await io.run(self.verify)
        digest, size = hashlib.sha256(), 0
        while chunk := await io.run(lambda: self.stream.read(CHUNK_BYTES)):
            size += len(chunk)
            if size > self.size:
                raise DataIntegrityError("upload source grew during transfer")
            digest.update(chunk)
            yield chunk
        await io.run(self.verify)
        if size != self.size or digest.hexdigest() != self.digest_sha256:
            raise DataIntegrityError("upload source differs from the prepared digest")


@asynccontextmanager
async def open_source(path: Path, io: FileIO) -> AsyncGenerator[FileSource, None]:
    # Ownership is acquired inside the joined worker, including on cancellation.
    owned: list[FileSource] = []

    def prepare() -> FileSource:
        source = FileSource.prepare(path)
        owned.append(source)
        return source

    try:
        yield await io.run(prepare)
    finally:
        for source in owned:
            await io.run(source.stream.close)
