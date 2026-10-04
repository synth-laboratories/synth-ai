"""Confine automatic directory uploads to a selected, bounded byte snapshot.

See specifications/tanha/index_selected_directory_custody.md in testing.
No-follow descriptor traversal prevents unselected symlink reads. Git metadata
is excluded; credential-like inputs fail before any HTTP call. The cheap text
scan mirrors backend code_bundle_secrets; it is not complete secret detection.
"""

from __future__ import annotations

import mimetypes
import os
import re
import stat
from pathlib import Path
from typing import Any

# Match the maintained source-bundle member/count/expanded-byte ceilings.
MAX_FILE_BYTES = 2 * 1024 * 1024
MAX_TOTAL_BYTES = 64 * 1024 * 1024
MAX_FILES = 512
MAX_ENTRIES = 1_024
MAX_DEPTH = 32
TEXT_SCAN_BYTES = 256 * 1024
SECRET_MARKERS = (
    "BEGIN OPENSSH PRIVATE KEY",
    "BEGIN RSA PRIVATE KEY",
    "BEGIN EC PRIVATE KEY",
    "OPENAI_API_KEY=",
    "ANTHROPIC_API_KEY=",
    "SYNTH_API_KEY=",
    "AWS_SECRET_ACCESS_KEY=",
    "GITHUB_TOKEN=",
)
SECRET_PATTERNS = (
    ("ghp_", re.compile(r"(?<![A-Za-z0-9_])ghp_[A-Za-z0-9]{36}")),
    ("sk-", re.compile(r"(?<![A-Za-z0-9_])sk-[A-Za-z0-9_-]{20,}")),
)
PRIVATE_KEY_NAMES = frozenset({"id_rsa", "id_dsa", "id_ecdsa", "id_ed25519"})


class DirectoryInputError(ValueError):
    """A selection/snapshot preflight refused; its message contains no file content."""


class DirectoryPorts:
    """Actual local filesystem port; explicit subclass injection is for source tests."""

    def root_stat(self, path: Path) -> os.stat_result:
        return path.lstat()

    def open_root(self, path: Path) -> int:
        return os.open(path, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)

    def names(self, descriptor: int) -> list[str]:
        names = []
        with os.scandir(descriptor) as entries:
            for entry in entries:
                if len(names) >= MAX_ENTRIES:
                    raise DirectoryInputError("directory entry limit exceeded")
                names.append(entry.name)
        return sorted(names)

    def child_stat(self, descriptor: int, name: str) -> os.stat_result:
        return os.stat(name, dir_fd=descriptor, follow_symlinks=False)

    def open_child(self, descriptor: int, name: str, *, directory: bool) -> int:
        flags = os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK
        if directory:
            flags |= os.O_DIRECTORY
        return os.open(name, flags, dir_fd=descriptor)

    def status(self, descriptor: int) -> os.stat_result:
        return os.fstat(descriptor)

    def read(self, descriptor: int, count: int) -> bytes:
        return os.read(descriptor, count)

    def close(self, descriptor: int) -> None:
        os.close(descriptor)


def _identity(status: os.stat_result) -> tuple[int, ...]:
    return (
        status.st_dev,
        status.st_ino,
        status.st_mode,
        status.st_nlink,
        status.st_size,
        status.st_mtime_ns,
        status.st_ctime_ns,
    )


def _scan_text(path: str, content: bytes) -> None:
    try:
        text = content[:TEXT_SCAN_BYTES].decode("utf-8")
    except UnicodeDecodeError:
        return
    for marker in SECRET_MARKERS:
        if marker in text:
            raise DirectoryInputError(f"credential-like text refused: {path}")
    for _, pattern in SECRET_PATTERNS:
        if pattern.search(text):
            raise DirectoryInputError(f"credential-like text refused: {path}")


def collect_directory_entries(
    directory: str | os.PathLike[str],
    *,
    ports: DirectoryPorts | None = None,
) -> list[dict[str, Any]]:
    """Read a selected local root before upload; see the directory-custody spec."""
    if any(not hasattr(os, option) for option in ("O_DIRECTORY", "O_NOFOLLOW", "O_NONBLOCK")):
        raise DirectoryInputError("no-follow directory input port unavailable on this platform")
    filesystem = ports if ports is not None else DirectoryPorts()
    root = Path(os.path.abspath(directory))
    if any(part.casefold() == ".git" for part in root.parts):
        raise DirectoryInputError("selected input root is Git metadata")
    if any(
        part.casefold() == ".env"
        or part.casefold().startswith(".env.")
        or part.casefold() in PRIVATE_KEY_NAMES
        for part in root.parts
    ):
        raise DirectoryInputError("selected input root is credential-like")
    files: list[dict[str, Any]] = []
    total_bytes = 0
    entries_seen = 0

    def visit(descriptor: int, prefix: str, depth: int, expected: os.stat_result) -> None:
        nonlocal total_bytes, entries_seen
        if depth > MAX_DEPTH:
            raise DirectoryInputError("directory depth limit exceeded")
        if _identity(filesystem.status(descriptor)) != _identity(expected):
            raise DirectoryInputError("selected directory identity changed")
        for name in filesystem.names(descriptor):
            entries_seen += 1
            if entries_seen > MAX_ENTRIES:
                raise DirectoryInputError("directory entry limit exceeded")
            if (
                name in (".", "..")
                or name != name.strip()
                or "/" in name
                or "\\" in name
                or any(
                    ord(character) < 32 or 0xD800 <= ord(character) <= 0xDFFF for character in name
                )
            ):
                raise DirectoryInputError("ambiguous directory input name refused")
            relative = f"{prefix}/{name}" if prefix else name
            expected_child = filesystem.child_stat(descriptor, name)
            if stat.S_ISLNK(expected_child.st_mode):
                raise DirectoryInputError(f"symlink input refused: {relative}")
            # Local Git metadata is not selected source, including worktree pointers.
            if name.casefold() == ".git":
                continue
            folded = name.casefold()
            if folded == ".env" or folded.startswith(".env.") or folded in PRIVATE_KEY_NAMES:
                raise DirectoryInputError(f"credential-like input path refused: {relative}")
            is_directory = stat.S_ISDIR(expected_child.st_mode)
            if not is_directory and not stat.S_ISREG(expected_child.st_mode):
                raise DirectoryInputError(f"non-regular input refused: {relative}")
            if not is_directory:
                if expected_child.st_nlink != 1:
                    raise DirectoryInputError(f"hard-linked input refused: {relative}")
                if expected_child.st_size > MAX_FILE_BYTES:
                    raise DirectoryInputError(
                        f"directory input file byte limit exceeded: {relative}"
                    )
                if len(files) >= MAX_FILES:
                    raise DirectoryInputError("directory input file count limit exceeded")
            child = filesystem.open_child(descriptor, name, directory=is_directory)
            try:
                if _identity(filesystem.status(child)) != _identity(expected_child):
                    raise DirectoryInputError(f"directory input identity changed: {relative}")
                if is_directory:
                    visit(child, relative, depth + 1, expected_child)
                    continue
                content = bytearray()
                while True:
                    chunk = filesystem.read(
                        child, min(64 * 1024, MAX_FILE_BYTES - len(content) + 1)
                    )
                    if not chunk:
                        break
                    content.extend(chunk)
                    if len(content) > MAX_FILE_BYTES:
                        raise DirectoryInputError(
                            f"directory input file byte limit exceeded: {relative}"
                        )
                    if total_bytes + len(content) > MAX_TOTAL_BYTES:
                        raise DirectoryInputError("directory input total byte limit exceeded")
                if len(content) != expected_child.st_size or _identity(
                    filesystem.status(child)
                ) != _identity(expected_child):
                    raise DirectoryInputError(f"directory input changed during read: {relative}")
                snapshot = bytes(content)
                _scan_text(relative, snapshot)
                total_bytes += len(snapshot)
                content_type = mimetypes.guess_type(name)[0] or "application/octet-stream"
                upload_content: str | bytes = snapshot
                if not name.lower().endswith(".zip") and content_type not in {
                    "application/zip",
                    "application/x-zip",
                    "application/x-zip-compressed",
                    "multipart/x-zip",
                }:
                    try:
                        upload_content = snapshot.decode("utf-8")
                    except UnicodeDecodeError:
                        # The unchanged normalizer sends captured binary bytes as base64.
                        upload_content = snapshot
                files.append(
                    {
                        "path": relative,
                        "content": upload_content,
                        "content_type": content_type,
                    }
                )
            finally:
                filesystem.close(child)
        if _identity(filesystem.status(descriptor)) != _identity(expected):
            raise DirectoryInputError("selected directory changed during snapshot")

    root_descriptor = None
    try:
        expected_root = filesystem.root_stat(root)
        if not stat.S_ISDIR(expected_root.st_mode):
            raise DirectoryInputError("selected input root must be a non-symlink directory")
        root_descriptor = filesystem.open_root(root)
        visit(root_descriptor, "", 0, expected_root)
    except OSError as exc:
        # Preserve the OS cause without embedding paths/contents in the public message.
        raise DirectoryInputError(
            f"directory input filesystem refusal (errno {exc.errno})"
        ) from exc
    finally:
        if root_descriptor is not None:
            filesystem.close(root_descriptor)
    if not files:
        raise DirectoryInputError("selected directory contains no uploadable source files")
    return files
