"""Canonical synth.git-commit-manifest.v1 integrity verification.

Entries sort by raw UTF-8 path bytes, not locale. Symlinks hash their target
bytes; gitlinks hash ASCII ``gitlink:<commit>``. Identical trees have identical
digests regardless of commit identity. Limits match the git authority.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import stat
from collections.abc import Callable, Iterable, Mapping
from pathlib import Path

from synth_ai.errors import ContractError

MAX_ENTRIES = 100_000
MAX_BLOB_BYTES = 2 * 1024**3
MAX_PATH_BYTES = 4096
MODES = {"100644", "100755", "120000", "160000"}


class CommitDigestMismatch(ContractError):  # noqa: N818
    def __init__(self, expected: str, actual: str, path: str | None = None) -> None:
        self.expected = expected
        self.actual = actual
        self.path = path
        self.service = "sdk"
        super().__init__(
            "Commit tree failed digest verification",
            code="commit_digest_mismatch",
            stage="commit_verification",
            cause=f"expected={expected} actual={actual}",
        )


def commit_manifest(entries: Iterable[Mapping[str, str]]) -> bytes:
    """Validate a complete, bounded manifest and return its canonical UTF-8 bytes."""
    ordered = []
    seen: set[str] = set()
    for entry in entries:
        if len(ordered) >= MAX_ENTRIES:
            raise ContractError(
                f"Commit manifest exceeds entry limit {MAX_ENTRIES}",
                code="commit_manifest_too_large",
                stage="commit_manifest",
            )
        if set(entry) != {"path", "mode", "sha256"}:
            raise ContractError(
                "Commit manifest exceeds entry bounds or has unknown fields",
                code="commit_manifest_invalid",
                stage="commit_manifest",
            )
        path, mode, digest = entry["path"], entry["mode"], entry["sha256"]
        if not all(isinstance(value, str) for value in (path, mode, digest)):
            raise ContractError(
                "Commit entry fields must be strings",
                code="commit_manifest_invalid",
                stage="commit_manifest",
            )
        try:
            raw_path = path.encode("utf-8")
        except UnicodeEncodeError as error:
            raise ContractError(
                "Commit path is not UTF-8", code="commit_path_invalid", stage="commit_manifest"
            ) from error
        if len(raw_path) > MAX_PATH_BYTES:
            raise ContractError(
                f"Commit path exceeds byte limit {MAX_PATH_BYTES}",
                code="commit_manifest_too_large",
                stage="commit_manifest",
            )
        if (
            not raw_path
            or path.startswith("/")
            or any(part in {"", ".", "..", ".git"} for part in path.split("/"))
            or "\x00" in path
            or path in seen
            or mode not in MODES
            or re.fullmatch(r"[0-9a-f]{64}", digest) is None
        ):
            raise ContractError(
                "Commit entry path, mode or digest is invalid",
                code="commit_manifest_invalid",
                stage="commit_manifest",
            )
        seen.add(path)
        ordered.append({"path": path, "mode": mode, "sha256": digest})
    ordered.sort(key=lambda entry: entry["path"].encode("utf-8"))
    return json.dumps(
        {"entries": ordered, "schema": "synth.git-commit-manifest.v1"},
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")


def commit_manifest_digest(entries: Iterable[Mapping[str, str]]) -> str:
    return hashlib.sha256(commit_manifest(entries)).hexdigest()


def verify_tree(
    path: str | Path,
    expected_digest: str,
    *,
    gitlinks: Mapping[str, str] | None = None,
    _check: Callable[[], None] | None = None,
) -> str:
    """Verify an extracted tree without following symlinks or reading Git metadata.

    Gitlink identities require authority-bound metadata and an empty directory.
    Omitted, missing or changed entries fail the final comparison. No partial
    digest is returned on a limit overrun. Delivery owners may check their own
    deadline before filesystem reads.
    """
    root = Path(path)
    if not root.is_dir() or root.is_symlink():
        raise ContractError(
            "Commit tree root must be a directory",
            code="commit_tree_invalid",
            stage="commit_verification",
        )
    entries: list[dict[str, str]] = []
    total = 0
    gitlinks = dict(gitlinks or {})
    remaining_gitlinks = set(gitlinks)
    directories = [root]
    for name, sha in gitlinks.items():
        if not isinstance(sha, str) or re.fullmatch(r"[0-9a-f]{40}", sha) is None:
            raise ContractError(
                "Gitlink lacks its exact commit identity",
                code="gitlink_invalid",
                stage="commit_verification",
            )
        commit_manifest(
            [
                {
                    "path": name,
                    "mode": "160000",
                    "sha256": hashlib.sha256(f"gitlink:{sha}".encode()).hexdigest(),
                }
            ]
        )

    def visit(directory: Path) -> None:
        nonlocal total
        for entry in directory.iterdir():
            if _check is not None:
                _check()
            if entry.name == ".git":
                continue
            info = entry.lstat()
            relative = entry.relative_to(root).as_posix()
            if stat.S_ISDIR(info.st_mode) and relative not in gitlinks:
                directories.append(entry)
                continue
            if len(entries) >= MAX_ENTRIES:
                raise ContractError(
                    "Commit tree has too many entries",
                    code="commit_manifest_too_large",
                    stage="commit_verification",
                )
            digest = hashlib.sha256()
            if relative in gitlinks:
                if not stat.S_ISDIR(info.st_mode) or any(entry.iterdir()):
                    raise ContractError(
                        "Gitlink materialization changed type or contains files",
                        code="gitlink_invalid",
                        stage="commit_verification",
                    )
                content = f"gitlink:{gitlinks[relative]}".encode("ascii")
                total += len(content)
                digest.update(content)
                mode = "160000"
                remaining_gitlinks.remove(relative)
            elif stat.S_ISLNK(info.st_mode):
                content = os.fsencode(os.readlink(entry))
                total += len(content)
                digest.update(content)
                mode = "120000"
            elif stat.S_ISREG(info.st_mode):
                # O_NOFOLLOW closes a symlink substitution between lstat and open.
                descriptor = os.open(entry, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
                with os.fdopen(descriptor, "rb") as stream:
                    if not stat.S_ISREG(os.fstat(stream.fileno()).st_mode):
                        raise ContractError(
                            "Commit entry changed type",
                            code="commit_tree_invalid",
                            stage="commit_verification",
                        )
                    while True:
                        if _check is not None:
                            _check()
                        chunk = stream.read(1024 * 1024)
                        if not chunk:
                            break
                        total += len(chunk)
                        if total > MAX_BLOB_BYTES:
                            raise ContractError(
                                "Commit tree exceeds blob limit",
                                code="commit_manifest_too_large",
                                stage="commit_verification",
                            )
                        digest.update(chunk)
                mode = "100755" if info.st_mode & stat.S_IXUSR else "100644"
            else:
                raise ContractError(
                    "Commit contains a non-file entry",
                    code="commit_tree_invalid",
                    stage="commit_verification",
                )
            if total > MAX_BLOB_BYTES:
                raise ContractError(
                    "Commit tree exceeds blob limit",
                    code="commit_manifest_too_large",
                    stage="commit_verification",
                )
            entries.append(
                {
                    "path": relative,
                    "mode": mode,
                    "sha256": digest.hexdigest(),
                }
            )

    while directories:
        if _check is not None:
            _check()
        visit(directories.pop())
    if remaining_gitlinks:
        raise ContractError(
            "Declared Gitlink is missing from the materialized tree",
            code="gitlink_invalid",
            stage="commit_verification",
        )
    actual = commit_manifest_digest(entries)
    if actual != expected_digest:
        raise CommitDigestMismatch(expected_digest, actual, str(root))
    return actual
