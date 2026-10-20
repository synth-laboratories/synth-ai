"""Bounded raw-tar decoding; no extractall, sparse expansion or implicit links.

Only the fixed 512-byte header decoder is borrowed from tarfile. Metadata is
bounded before reading its payload, including GNU names/links and local PAX.
The actual materialized tree is verified by the shared commit-digest authority.
"""

from __future__ import annotations

import errno
import hashlib
import ntpath
import os
import re
import stat
import tarfile
from collections.abc import Callable
from pathlib import Path
from typing import BinaryIO

from synth_ai.errors import ContractError

from .commit_digest import MAX_PATH_BYTES, verify_tree
from .types import RepositoryArchiveOptions

CHUNK_BYTES = 64 * 1024
MAX_PAX_BYTES = 64 * 1024
MAX_EXTENSIONS = 2


class ArchiveFormatError(ContractError):
    service = "sdk"


def invalid(message: str, code: str = "archive_invalid") -> ArchiveFormatError:
    return ArchiveFormatError(
        message, code=code, stage="archive_extract", operation="repos.archive"
    )


def exact(stream: BinaryIO, size: int, check: Callable[[], None]) -> bytes:
    check()
    value = stream.read(size)
    if len(value) != size:
        raise invalid("Archive ended before a declared member was complete")
    return value


def padding(stream: BinaryIO, size: int, check: Callable[[], None]) -> None:
    amount = (-size) % 512
    if amount and any(exact(stream, amount, check)):
        raise invalid("Archive member padding is not zero")


def pax_fields(data: bytes) -> dict[str, str]:
    result = {}
    position = 0
    while position < len(data):
        space = data.find(b" ", position)
        digits = data[position:space] if space >= 0 else b""
        if not 1 <= len(digits) <= 6 or not digits.isdigit():
            raise invalid("Archive PAX record has invalid length framing")
        size = int(digits)
        record = data[space + 1 : position + size]
        key, separator, value = record[:-1].partition(b"=")
        if (
            size < 5
            or position + size > len(data)
            or not record.endswith(b"\n")
            or not key
            or separator != b"="
        ):
            raise invalid("Archive PAX record is incomplete")
        try:
            name, text = key.decode("utf-8"), value.decode("utf-8")
        except UnicodeError as cause:
            raise invalid("Archive PAX metadata is not UTF-8") from cause
        if name in result or name == "size" or name.startswith("GNU.sparse."):
            raise invalid("Archive PAX metadata changes framing or repeats a key")
        result[name] = text
        position += size
    return result


def safe_path(name: str) -> tuple[str, ...]:
    try:
        raw = name.encode("utf-8")
    except UnicodeError as cause:
        raise invalid("Archive path is not UTF-8", "archive_path_invalid") from cause
    parts = tuple(name.split("/"))
    if (
        not 1 <= len(raw) <= MAX_PATH_BYTES
        or name.startswith("/")
        or "\x00" in name
        or any(part in {"", ".", ".."} or part.casefold() == ".git" for part in parts)
        or (
            os.name == "nt"
            and (
                "\\" in name
                or ntpath.splitdrive(name)[0]
                or any(windows_reserved(part) for part in parts)
            )
        )
    ):
        raise invalid("Archive path is not a safe canonical relative name", "archive_path_invalid")
    return parts


def windows_reserved(part: str) -> bool:
    """NTFS streams, devices and Win32 normalization are not ordinary files."""
    return (
        part.endswith((".", " "))
        or any(ord(character) < 32 or character in ':<>"|?*' for character in part)
        or re.fullmatch(
            r"(?:CON|PRN|AUX|NUL|COM[1-9¹²³]|LPT[1-9¹²³])", part.split(".", 1)[0], re.IGNORECASE
        )
        is not None
    )


class Directories:
    def __init__(self, root: Path, maximum: int, members: set[str], check: Callable[[], None]):
        self.root, self.maximum, self.members, self.check = root, maximum, members, check
        self.count = 0
        self.identities: dict[tuple[int, int], bytes] = {}

    def parent(self, parts: tuple[str, ...]) -> Path:
        current = self.root
        for index, part in enumerate(parts[:-1], 1):
            self.check()
            prefix = "/".join(parts[:index])
            if prefix in self.members:
                raise invalid("Archive member is also used as a parent directory")
            current = current / part
            try:
                current.mkdir(mode=0o700)
                self.count += 1
            except FileExistsError:
                pass
            if self.count > self.maximum:
                raise invalid(
                    "Archive exceeds the selected directory count", "archive_directory_limit"
                )
            info = current.lstat()
            if not stat.S_ISDIR(info.st_mode):
                raise invalid("Archive parent is not a real directory")
            identity = (info.st_dev, info.st_ino)
            canonical = hashlib.sha256(prefix.encode()).digest()
            prior = self.identities.setdefault(identity, canonical)
            if prior != canonical:
                raise invalid("Archive directory names collide on the destination filesystem")
        return current

    def gitlink(self, path: Path) -> None:
        path.mkdir(mode=0o700)
        self.count += 1
        if self.count > self.maximum:
            raise invalid("Archive exceeds the selected directory count", "archive_directory_limit")


def extract_verified(
    stream: BinaryIO,
    root: Path,
    expected_digest: str,
    expected_entries: int,
    options: RepositoryArchiveOptions,
    check: Callable[[], None],
) -> dict[str, str]:
    """Extract only into a fresh private directory; verify before publication."""
    if root.is_symlink() or not root.is_dir() or any(root.iterdir()):
        raise invalid("Archive extraction requires a fresh real directory")
    members: set[str] = set()
    directories = Directories(root, options.maximum_directories, members, check)
    gitlinks = {}
    symlinks: list[tuple[Path, bytes]] = []
    long_name = long_link = None
    local_pax: dict[str, str] = {}
    extensions = 0
    expanded = 0
    while True:
        block = exact(stream, 512, check)
        if not any(block):
            if extensions or any(exact(stream, 512, check)):
                raise invalid("Archive lacks its complete zero terminator")
            while chunk := stream.read(CHUNK_BYTES):
                check()
                if any(chunk):
                    raise invalid("Archive contains data after its zero terminator")
            break
        try:
            member = tarfile.TarInfo.frombuf(block, "utf-8", "surrogateescape")
        except tarfile.HeaderError as cause:
            raise invalid("Archive header failed validation") from cause
        if member.size < 0 or member.size > options.maximum_bytes:
            raise invalid("Archive member exceeds the selected byte bound")
        if member.type in {tarfile.XHDTYPE, tarfile.GNUTYPE_LONGNAME, tarfile.GNUTYPE_LONGLINK}:
            extensions += 1
            maximum = MAX_PAX_BYTES if member.type == tarfile.XHDTYPE else MAX_PATH_BYTES + 1
            if extensions > MAX_EXTENSIONS or member.size > maximum:
                raise invalid(
                    "Archive extension metadata exceeds its bound", "archive_metadata_limit"
                )
            value = exact(stream, member.size, check)
            padding(stream, member.size, check)
            if member.type == tarfile.XHDTYPE:
                fields = pax_fields(value)
                if local_pax.keys() & fields.keys():
                    raise invalid("Archive repeats local PAX metadata")
                local_pax.update(fields)
            else:
                if not value.endswith(b"\x00") or not value[:-1] or b"\x00" in value[:-1]:
                    raise invalid("Archive long name/link is not correctly terminated")
                if member.type == tarfile.GNUTYPE_LONGNAME:
                    if long_name is not None:
                        raise invalid("Archive repeats a long name")
                    try:
                        long_name = value[:-1].decode("utf-8", errors="strict")
                    except UnicodeError as cause:
                        raise invalid("Archive long path is not UTF-8") from cause
                else:
                    if long_link is not None:
                        raise invalid("Archive repeats a long link")
                    long_link = value[:-1]
            continue
        if member.type not in {tarfile.REGTYPE, tarfile.AREGTYPE, tarfile.SYMTYPE, tarfile.DIRTYPE}:
            raise invalid("Archive contains an unsupported entry type")
        if not member.issym() and (long_link is not None or "linkpath" in local_pax):
            raise invalid("Archive non-symlink entry has link metadata")
        name = local_pax.get("path", long_name if long_name is not None else member.name)
        if member.isdir():
            name = name.rstrip("/")
        parts = safe_path(name)
        if name in members or len(members) >= options.maximum_files:
            raise invalid("Archive repeats a path or exceeds the selected entry count")
        parent = directories.parent(parts)
        target = parent / parts[-1]
        if target.exists() or target.is_symlink():
            raise invalid("Archive members collide on the destination filesystem")
        members.add(name)
        gitlink = local_pax.get("SYNTH.gitlink")
        if gitlink is not None:
            if (
                not member.isdir()
                or member.size != 0
                or re.fullmatch(r"[0-9a-f]{40}", gitlink) is None
            ):
                raise invalid("Archive gitlink identity is malformed")
            directories.gitlink(target)
            gitlinks[name] = gitlink
        elif member.issym():
            link = local_pax.get("linkpath")
            value = (
                link.encode("utf-8")
                if link is not None
                else long_link
                if long_link is not None
                else os.fsencode(member.linkname)
            )
            if member.size != 0 or not value or b"\x00" in value or len(value) > MAX_PATH_BYTES:
                raise invalid("Archive symlink target is malformed")
            symlinks.append((target, value))
        elif member.isreg():
            if long_link is not None or "linkpath" in local_pax:
                raise invalid("Archive regular file has link metadata")
            expanded += member.size
            if expanded > options.maximum_bytes:
                raise invalid("Archive exceeds the selected content-byte bound")
            descriptor = os.open(
                target, os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0), 0o600
            )
            with os.fdopen(descriptor, "wb") as output:
                remaining = member.size
                while remaining:
                    chunk = exact(stream, min(CHUNK_BYTES, remaining), check)
                    output.write(chunk)
                    remaining -= len(chunk)
                output.flush()
                os.chmod(target, 0o755 if member.mode & stat.S_IXUSR else 0o644)
                os.fsync(output.fileno())
            padding(stream, member.size, check)
        else:
            raise invalid("Raw commit archives do not contain ordinary directory entries")
        long_name = long_link = None
        local_pax = {}
        extensions = 0
    if len(members) != expected_entries:
        raise invalid("Archive entry count differs from its delivery receipt")
    # No symlink exists while files/directories are written. Validate the final
    # link graph before publication, including symlink-prefix/.. combinations.
    for path, value in symlinks:
        check()
        try:
            os.symlink(os.fsdecode(value), path)
        except FileExistsError as cause:
            raise invalid("Archive symlink names collide on the destination filesystem") from cause
    resolved_root = root.resolve()
    for path, _ in symlinks:
        check()
        # Python 3.13 resolve(strict=False) returns a looping link unchanged instead of
        # raising; stat() follows the chain and reports ELOOP on every version. A
        # dangling link (ENOENT) is legitimate and is still checked for escape below.
        try:
            os.stat(path)
        except OSError as cause:
            if cause.errno == errno.ELOOP:
                raise invalid("Archive symlink graph cannot be safely resolved") from cause
        try:
            resolved = path.resolve(strict=False)
        except (RuntimeError, OSError) as cause:
            raise invalid("Archive symlink graph cannot be safely resolved") from cause
        if not resolved.is_relative_to(resolved_root):
            raise invalid("Archive symlink escapes its destination", "archive_path_invalid")
    verify_tree(root, expected_digest, gitlinks=gitlinks, _check=check)
    return gitlinks
