"""Admit an explicitly selected native Codex task read without inventing history.

See notes/specifications/synth-index/research-archive-release.md. read_thread v1
is a partial retrieval response, not a full rollout export. It omits tool outputs
and reasoning details and may contain in-progress turns. Preserve its exact bytes
and declare those gaps. No session discovery, credential loading or live I/O here.
"""

from __future__ import annotations

import hashlib
import json
import os
import stat
import tempfile
from datetime import datetime
from pathlib import Path

from ..artifacts import ArtifactObjectDeclaration
from .build import FrozenBuildError
from .contracts import FrozenObject, SessionExport, canonical_bytes

EXPORT_BYTES_MAX = 4 * 1024 * 1024


def admit_codex_task_read(
    raw: bytes,
    *,
    expected_thread_id: str,
    captured_at: datetime,
    cutoff_at: datetime,
    object_id: str,
    logical_path: str,
) -> SessionExport:
    """Bind exact native bytes to the selected thread; completeness stays partial."""
    if len(raw) > EXPORT_BYTES_MAX:
        raise FrozenBuildError("session_export_too_large", "native task read exceeds 4 MiB")
    try:
        payload = json.loads(raw)
    except (ValueError, UnicodeDecodeError) as error:
        raise FrozenBuildError(
            "session_export_invalid", "native task read is invalid JSON"
        ) from error
    if not isinstance(payload, dict) or payload.get("schemaVersion") != 1:
        raise FrozenBuildError(
            "session_export_version_unknown", "expected Codex task read schemaVersion 1"
        )
    thread = payload.get("thread")
    turns = payload.get("turns")
    if (
        not isinstance(thread, dict)
        or thread.get("kind") != "codex"
        or thread.get("id") != expected_thread_id
    ):
        raise FrozenBuildError(
            "session_identity_mismatch",
            "native response differs from selected Codex thread",
        )
    if not isinstance(turns, list) or len(turns) > 1000:
        raise FrozenBuildError("session_export_invalid", "native task read has invalid turn set")
    count = 0
    for turn in turns:
        if not isinstance(turn, dict) or not isinstance(turn.get("items"), list):
            raise FrozenBuildError("session_export_invalid", "native task read has invalid items")
        count += len(turn["items"])
        if count > 100_000:
            raise FrozenBuildError(
                "session_export_too_large", "native task read has too many items"
            )
    frozen = FrozenObject(
        object_id=object_id,
        purpose="session",
        object=ArtifactObjectDeclaration(
            logical_path=logical_path,
            digest_sha256=hashlib.sha256(raw).hexdigest(),
            size_bytes=len(raw),
            media_type="application/json",
        ),
    )
    return SessionExport(
        source="codex",
        native_session_id=expected_thread_id,
        event_start=0,
        event_end_exclusive=count,
        captured_at=captured_at,
        cutoff_at=cutoff_at,
        completeness="partial",
        gaps=(
            "Task retrieval response; not a complete native rollout export.",
            "Tool outputs and reasoning details may be omitted or truncated.",
            "Normalized item indexes describe this response only, not global event sequence.",
            "Parent/fork identity is not supplied by this read contract.",
        ),
        native_objects=(frozen,),
    )


def freeze_codex_task_read(
    source: Path,
    destination: Path,
    *,
    expected_thread_id: str,
    captured_at: datetime,
    cutoff_at: datetime,
) -> SessionExport:
    """Freeze an explicitly selected native response privately, without discovery.

    See notes/specifications/synth-index/research-archive-release.md. A retry must
    preserve the same admitted bytes and capture metadata. This partial response
    supplies neither a complete rollout nor parent/fork identity.
    """
    if captured_at.utcoffset() is None or cutoff_at.utcoffset() is None:
        raise FrozenBuildError("session_time_invalid", "capture timestamps require offsets")
    if source.is_symlink() or not source.is_file():
        raise FrozenBuildError("unsafe_session_path", "native input must be a regular file")
    with source.open("rb") as native:
        raw = native.read(EXPORT_BYTES_MAX + 1)
    export = admit_codex_task_read(
        raw,
        expected_thread_id=expected_thread_id,
        captured_at=captured_at,
        cutoff_at=cutoff_at,
        object_id="native-task-read",
        logical_path="objects/native-task-read.json",
    )
    expected = {
        "session-export.json": canonical_bytes(export),
        "objects/native-task-read.json": raw,
    }
    if destination.is_symlink():
        raise FrozenBuildError("unsafe_session_path", "capture root may not be a link")
    if destination.exists():
        actual = {
            p.relative_to(destination).as_posix()
            for p in destination.rglob("*")
            if p.is_file() or p.is_symlink()
        }
        if actual != set(expected):
            raise FrozenBuildError("capture_retry_mismatch", "capture object set changed")
        for name, content in expected.items():
            path = destination / name
            if path.is_symlink() or path.stat().st_mode & 0o077 or path.read_bytes() != content:
                raise FrozenBuildError(
                    "capture_retry_mismatch", "capture bytes or private permissions changed"
                )
        if destination.stat().st_mode & 0o077 or (destination / "objects").is_symlink():
            raise FrozenBuildError("unsafe_session_path", "capture directory must remain private")
        return export
    destination.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    with tempfile.TemporaryDirectory(
        prefix=".native-capture-", dir=destination.parent
    ) as temporary:
        staging = Path(temporary) / "capture"
        staging.mkdir(mode=0o700)
        (staging / "objects").mkdir(mode=0o700)
        for name, content in expected.items():
            path = staging / name
            descriptor = os.open(
                path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, stat.S_IRUSR | stat.S_IWUSR
            )
            with os.fdopen(descriptor, "wb") as output:
                output.write(content)
                output.flush()
                os.fsync(output.fileno())
        staging.rename(destination)
    return export
