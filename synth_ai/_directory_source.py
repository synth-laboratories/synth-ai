"""Freeze a descriptor-relative directory into one bounded private disk spool."""

from __future__ import annotations

import hashlib
import os
import re
import stat
import tempfile
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, BinaryIO

from ._file_source import CHUNK_BYTES, FileIdentity, FileIO
from ._generated.file_limits import (
    FILE_MAX_BYTES,
    LAUNCH_MAX_BYTES,
    LAUNCH_MAX_FILES,
    NAME_MAX_BYTES,
    NAME_PATTERN,
    NAME_SEGMENT_MAX_BYTES,
)
from .errors import DataIntegrityError

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator
    from pathlib import Path

_NAME = re.compile(NAME_PATTERN)


@dataclass(frozen=True)
class DirectoryFile:
    name: str
    offset: int
    size: int
    digest_sha256: str


@dataclass(frozen=True)
class DirectorySource:
    stream: BinaryIO
    file: DirectoryFile
    snapshot_size: int

    @property
    def size(self) -> int:
        return self.file.size

    @property
    def digest_sha256(self) -> str:
        return self.file.digest_sha256

    def verify(self) -> None:
        if os.fstat(self.stream.fileno()).st_size != self.snapshot_size:
            raise DataIntegrityError("directory snapshot size changed")

    async def chunks(self, io: FileIO) -> AsyncGenerator[bytes, None]:
        await io.run(self.verify)
        await io.run(lambda: self.stream.seek(self.file.offset))
        digest, remaining = hashlib.sha256(), self.size
        while remaining:
            count = min(CHUNK_BYTES, remaining)
            chunk = await io.run(lambda count=count: self.stream.read(count))
            if not chunk:
                raise DataIntegrityError("directory snapshot is truncated")
            remaining -= len(chunk)
            digest.update(chunk)
            yield chunk
        await io.run(self.verify)
        if digest.hexdigest() != self.digest_sha256:
            raise DataIntegrityError("directory snapshot differs from its prepared digest")


@dataclass
class _DirectoryWriter:
    stream: BinaryIO
    files: list[DirectoryFile] = field(default_factory=list)
    size: int = 0
    entries: int = 0

    def walk(self, descriptor: int, prefix: str) -> None:
        before = FileIdentity.read(os.fstat(descriptor))
        names: list[str] = []
        with os.scandir(descriptor) as iterator:
            for entry in iterator:
                self.entries += 1
                # Directories count too: empty trees must not make traversal unbounded.
                if self.entries > LAUNCH_MAX_FILES:
                    raise ValueError("directory exceeds the maximum scanned entries")
                names.append(entry.name)
        for segment in sorted(names):
            name = prefix + segment
            if (
                len(name) > NAME_MAX_BYTES
                or len(segment) > NAME_SEGMENT_MAX_BYTES
                or _NAME.fullmatch(name) is None
            ):
                raise ValueError("directory file name is outside the canonical limits")
            before_child = os.stat(segment, dir_fd=descriptor, follow_symlinks=False)
            is_directory = stat.S_ISDIR(before_child.st_mode)
            if not is_directory and not stat.S_ISREG(before_child.st_mode):
                raise ValueError("directory entries must be regular files or directories")
            flags = os.O_RDONLY | os.O_NONBLOCK | os.O_NOFOLLOW
            if is_directory:
                flags |= os.O_DIRECTORY
            child = os.open(segment, flags, dir_fd=descriptor)
            try:
                identity = FileIdentity.read(before_child)
                if FileIdentity.read(os.fstat(child)) != identity:
                    raise DataIntegrityError("directory entry changed before snapshot")
                if is_directory:
                    self.walk(child, name + "/")
                else:
                    self.copy(child, name, identity)
                after_child = os.stat(segment, dir_fd=descriptor, follow_symlinks=False)
                if FileIdentity.read(after_child) != identity:
                    raise DataIntegrityError("directory entry changed during snapshot")
            finally:
                os.close(child)
        if FileIdentity.read(os.fstat(descriptor)) != before:
            raise DataIntegrityError("directory changed during snapshot")

    def copy(self, descriptor: int, name: str, identity: FileIdentity) -> None:
        if identity.size > FILE_MAX_BYTES:
            raise ValueError("directory file exceeds the maximum file bytes")
        offset, size, digest = self.stream.tell(), 0, hashlib.sha256()
        while chunk := os.read(descriptor, CHUNK_BYTES):
            size += len(chunk)
            if size > FILE_MAX_BYTES:
                raise ValueError("directory file exceeds the maximum file bytes")
            self.size += len(chunk)
            if self.size > LAUNCH_MAX_BYTES:
                raise ValueError("directory exceeds the maximum aggregate bytes")
            digest.update(chunk)
            self.stream.write(chunk)
        if size != identity.size or FileIdentity.read(os.fstat(descriptor)) != identity:
            raise DataIntegrityError("directory file changed during snapshot")
        self.files.append(DirectoryFile(name, offset, size, digest.hexdigest()))


@dataclass(frozen=True)
class DirectorySnapshot:
    stream: BinaryIO
    files: tuple[DirectoryFile, ...]
    size: int

    @classmethod
    def prepare(cls, path: Path | int) -> DirectorySnapshot:
        if os.name != "posix":
            raise NotImplementedError("directory uploads require POSIX descriptor-relative I/O")
        if isinstance(path, bool):
            raise ValueError("directory descriptor must be an integer, not a boolean")
        stream = tempfile.TemporaryFile(mode="w+b")  # noqa: SIM115
        try:
            descriptor = (
                os.open(".", os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=path)
                if isinstance(path, int)
                else os.open(path, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
            )
            try:
                writer = _DirectoryWriter(stream)
                writer.walk(descriptor, "")
            finally:
                os.close(descriptor)
            stream.flush()
            return cls(stream, tuple(sorted(writer.files, key=lambda file: file.name)), writer.size)
        except BaseException:
            stream.close()
            raise

    def source(self, file: DirectoryFile) -> DirectorySource:
        return DirectorySource(self.stream, file, self.size)


@asynccontextmanager
async def open_directory(path: Path | int, io: FileIO) -> AsyncGenerator[DirectorySnapshot, None]:
    owned: list[DirectorySnapshot] = []

    def prepare() -> DirectorySnapshot:
        snapshot = DirectorySnapshot.prepare(path)
        owned.append(snapshot)
        return snapshot

    try:
        yield await io.run(prepare)
    finally:
        for snapshot in owned:
            await io.run(snapshot.stream.close)
