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
import re
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


def _native_payload(raw: bytes) -> dict:
    """Bounded exact JSON; ambiguous duplicate keys cannot enter a frozen root."""
    if len(raw) > EXPORT_BYTES_MAX:
        raise FrozenBuildError("session_export_too_large", "native record exceeds 4 MiB")

    def unique_pairs(pairs):
        value = {}
        for key, item in pairs:
            if key in value:
                raise ValueError("duplicate native JSON key")
            value[key] = item
        return value

    try:
        value = json.loads(
            raw,
            object_pairs_hook=unique_pairs,
            parse_constant=lambda _value: (_ for _ in ()).throw(
                ValueError("nonfinite native JSON")
            ),
        )
    except (ValueError, UnicodeDecodeError) as error:
        raise FrozenBuildError("session_export_invalid", "native record is invalid JSON") from error
    if not isinstance(value, dict):
        raise FrozenBuildError("session_export_invalid", "native record must be an object")
    return value


def _native_object(raw: bytes, *, object_id: str, logical_path: str) -> FrozenObject:
    return FrozenObject(
        object_id=object_id,
        purpose="session",
        object=ArtifactObjectDeclaration(
            logical_path=logical_path,
            digest_sha256=hashlib.sha256(raw).hexdigest(),
            size_bytes=len(raw),
            media_type="application/json",
        ),
    )


def admit_swarms_evidence(
    raw: bytes,
    *,
    expected_run_id: str,
    expected_project_id: str,
    captured_at: datetime,
    cutoff_at: datetime,
    object_id: str,
    logical_path: str,
) -> SessionExport:
    """Admit the actual bounded SMR evidence response as a partial run projection.

    See notes/specifications/synth-index/research-archive-release.md. The run
    evidence API limits selected records and does not export a native rollout.
    """
    value = _native_payload(raw)
    if value.get("schema_version") != 1:
        raise FrozenBuildError("session_export_version_unknown", "expected Swarms evidence v1")
    if value.get("run_id") != expected_run_id or value.get("project_id") != expected_project_id:
        raise FrozenBuildError("session_identity_mismatch", "Swarms run/project identity differs")
    freshness = value.get("freshness")
    if not isinstance(freshness, dict):
        raise FrozenBuildError("session_export_invalid", "Swarms freshness is required")
    generated = freshness.get("generated_at")
    try:
        generated_at = datetime.fromisoformat(generated) if isinstance(generated, str) else None
    except ValueError as error:
        raise FrozenBuildError("session_export_invalid", "Swarms time is invalid") from error
    if generated_at is None or generated_at.utcoffset() is None:
        raise FrozenBuildError("session_export_invalid", "Swarms time requires an offset")
    if cutoff_at != generated_at or captured_at < generated_at:
        raise FrozenBuildError(
            "session_time_invalid", "Swarms cutoff must equal source observation time"
        )
    groups = (
        ("artifacts", "artifact_count", 1000),
        ("work_products", "work_product_count", 1000),
        ("tool_calls", "tool_call_count", 250),
    )
    for name, count, maximum in groups:
        items = value.get(name)
        if (
            not isinstance(items, list)
            or len(items) > maximum
            or any(not isinstance(item, dict) for item in items)
        ):
            raise FrozenBuildError("session_export_invalid", "Swarms evidence set is invalid")
        if type(freshness.get(count)) is not int or freshness[count] != len(items):
            raise FrozenBuildError("session_export_invalid", "Swarms freshness count differs")
    for name in ("selected_artifact_contents", "trace_publications"):
        if not isinstance(value.get(name), list):
            raise FrozenBuildError("session_export_invalid", "Swarms evidence set is missing")
    required = {
        "artifacts": {"artifact_id", "artifact_type", "created_at", "content_url", "download_url"},
        "work_products": {
            "work_product_id",
            "kind",
            "title",
            "status",
            "readiness",
            "artifact_links",
            "content_url",
            "created_at",
            "updated_at",
        },
        "tool_calls": {
            "tool_call_id",
            "actor_role",
            "tool_name",
            "arguments_digest",
            "status",
            "retryable",
            "duration_ms",
            "occurred_at",
        },
        "selected_artifact_contents": {
            "artifact_id",
            "artifact_type",
            "content_type",
            "size_bytes",
            "content",
        },
        "trace_publications": {
            "publication_id",
            "factory_id",
            "project_id",
            "run_id",
            "bundle_id",
            "bundle_schema_version",
            "manifest_digest",
            "status",
            "trace_count",
            "evidence_count",
        },
    }
    for name, fields in required.items():
        if any(not isinstance(item, dict) or not fields.issubset(item) for item in value[name]):
            raise FrozenBuildError("session_export_invalid", "Swarms evidence item is incomplete")
    if type(freshness.get("run_is_terminal")) is not bool:
        raise FrozenBuildError("session_export_invalid", "Swarms terminal status is invalid")
    return SessionExport(
        source="swarms",
        native_session_id=expected_run_id,
        event_start=0,
        event_end_exclusive=len(value["tool_calls"]),
        captured_at=captured_at,
        cutoff_at=cutoff_at,
        completeness="partial",
        gaps=(
            "Bounded SMR evidence projection, not a native actor rollout or full run journal.",
            "Tool-call indexes count returned records only; earlier calls may be omitted.",
            "Actor ancestry, failed attempts and full trace payloads are not supplied.",
        ),
        native_objects=(_native_object(raw, object_id=object_id, logical_path=logical_path),),
    )


def admit_mlok_policy_capture(
    raw: bytes,
    *,
    expected_thread_id: str,
    captured_at: datetime,
    cutoff_at: datetime,
    object_id: str,
    logical_path: str,
) -> SessionExport:
    """Verify native mlok context hash and retain exact bytes with partial gaps.

    See notes/specifications/synth-index/research-archive-release.md. The
    model-context capture is not a committed causal participant cut or journal.
    """
    value = _native_payload(raw)
    if set(value) != {"snapshot", "digest"} or not isinstance(value["snapshot"], dict):
        raise FrozenBuildError("session_export_invalid", "mlok capture wrapper is invalid")
    snapshot = value["snapshot"]
    if (
        snapshot.get("schema") != "mlok.policy-snapshot.v1"
        or snapshot.get("restoreScope") != "model_context"
    ):
        raise FrozenBuildError("session_export_version_unknown", "expected mlok policy snapshot v1")
    if snapshot.get("sourceThreadId") != expected_thread_id:
        raise FrozenBuildError("session_identity_mismatch", "mlok thread identity differs")
    history = snapshot.get("history")
    sequence = snapshot.get("contextSeq")
    if (
        not isinstance(history, list)
        or type(sequence) is not int
        or sequence != len(history)
        or sequence > 100000
    ):
        raise FrozenBuildError("session_export_invalid", "mlok context sequence differs")
    for name in ("sourceSessionDigest", "configDigest"):
        if (
            not isinstance(snapshot.get(name), str)
            or re.fullmatch(r"sha256:[0-9a-f]{64}", snapshot[name]) is None
        ):
            raise FrozenBuildError("session_export_invalid", "mlok source digest is invalid")
    try:
        serialized = json.dumps(
            snapshot, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
        ).encode("utf-8")
    except (ValueError, UnicodeError) as error:
        raise FrozenBuildError("session_export_invalid", "mlok context JSON is invalid") from error
    actual = "sha256:" + hashlib.sha256(serialized).hexdigest()
    if value["digest"] != actual:
        raise FrozenBuildError("session_digest_mismatch", "mlok context digest differs")
    if cutoff_at != captured_at:
        raise FrozenBuildError(
            "session_time_invalid",
            "mlok capture has no native clock; cutoff must equal capture time",
        )
    return SessionExport(
        source="mlok",
        native_session_id=expected_thread_id,
        event_start=0,
        event_end_exclusive=sequence,
        captured_at=captured_at,
        cutoff_at=cutoff_at,
        completeness="partial",
        gaps=(
            "Policy model-context snapshot, not a complete native turn or session journal.",
            "Context sequence counts retained history only; pruned history and parent identity are absent.",
            "No causal participant cut, durable commit or exact replay capability is proven.",
        ),
        native_objects=(_native_object(raw, object_id=object_id, logical_path=logical_path),),
    )


def _freeze_selected_native(
    source: Path,
    destination: Path,
    *,
    admission,
    object_id: str,
    captured_at: datetime,
    cutoff_at: datetime,
    expected_identity: dict[str, str],
    object_suffix: str = ".json",
) -> SessionExport:
    """Freeze one selected native response using the same create-only custody as Codex."""
    if captured_at.utcoffset() is None or cutoff_at.utcoffset() is None:
        raise FrozenBuildError("session_time_invalid", "capture timestamps require offsets")
    if source.is_symlink() or not source.is_file():
        raise FrozenBuildError("unsafe_session_path", "native input must be a regular file")
    with source.open("rb") as native:
        raw = native.read(EXPORT_BYTES_MAX + 1)
    export = admission(
        raw,
        **expected_identity,
        captured_at=captured_at,
        cutoff_at=cutoff_at,
        object_id=object_id,
        logical_path=f"objects/{object_id}{object_suffix}",
    )
    expected = {
        "session-export.json": canonical_bytes(export),
        f"objects/{object_id}{object_suffix}": raw,
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


def freeze_swarms_evidence(
    source: Path,
    destination: Path,
    *,
    expected_run_id: str,
    expected_project_id: str,
    captured_at: datetime,
    cutoff_at: datetime,
) -> SessionExport:
    """Freeze one explicitly selected SMR run evidence response privately."""
    return _freeze_selected_native(
        source,
        destination,
        admission=admit_swarms_evidence,
        object_id="native-swarms-evidence",
        captured_at=captured_at,
        cutoff_at=cutoff_at,
        expected_identity={
            "expected_run_id": expected_run_id,
            "expected_project_id": expected_project_id,
        },
    )


def freeze_mlok_policy_capture(
    source: Path,
    destination: Path,
    *,
    expected_thread_id: str,
    captured_at: datetime,
    cutoff_at: datetime,
) -> SessionExport:
    """Freeze one selected policy context snapshot privately; not a causal cut."""
    return _freeze_selected_native(
        source,
        destination,
        admission=admit_mlok_policy_capture,
        object_id="native-mlok-policy-context",
        captured_at=captured_at,
        cutoff_at=cutoff_at,
        expected_identity={"expected_thread_id": expected_thread_id},
    )


def admit_codex_rollout_prefix(
    raw: bytes,
    *,
    expected_thread_id: str,
    captured_at: datetime,
    cutoff_at: datetime,
    object_id: str,
    logical_path: str,
) -> SessionExport:
    """Admit exact native JSONL records through a declared capture cutoff.

    Unlike task-read summaries, this retains native response/tool events. A
    journal prefix still cannot establish inherited history or omitted provider
    state, so completeness remains explicit rather than inferred from byte count.
    """
    if not raw or len(raw) > EXPORT_BYTES_MAX or not raw.endswith(b"\n"):
        raise FrozenBuildError("session_export_invalid", "bounded complete JSONL records required")
    records = [_native_payload(line) for line in raw.splitlines()]
    if len(records) > 100_000 or records[0].get("type") != "session_meta":
        raise FrozenBuildError("session_export_invalid", "native session metadata must be first")
    meta = records[0].get("payload")
    if not isinstance(meta, dict) or meta.get("id") != expected_thread_id:
        raise FrozenBuildError(
            "session_identity_mismatch", "native rollout differs from selected thread"
        )
    if meta.get("session_id", expected_thread_id) != expected_thread_id:
        raise FrozenBuildError("session_identity_mismatch", "native session aliases disagree")
    if cutoff_at.tzinfo is None or captured_at.tzinfo is None or captured_at < cutoff_at:
        raise FrozenBuildError(
            "session_export_invalid", "timezone-aware capture after cutoff required"
        )
    kinds = {
        "session_meta",
        "event_msg",
        "response_item",
        "world_state",
        "turn_context",
        "token_usage_record",
        "compacted",
    }
    previous = None
    for offset, record in enumerate(records):
        if record.get("type") not in kinds or (offset and record.get("type") == "session_meta"):
            raise FrozenBuildError(
                "session_export_version_unknown", "unsupported native record shape"
            )
        try:
            at = datetime.fromisoformat(record["timestamp"].replace("Z", "+00:00"))
        except (KeyError, ValueError, TypeError, AttributeError) as error:
            raise FrozenBuildError("session_export_invalid", "native timestamp required") from error
        if at.tzinfo is None or at > cutoff_at or (previous is not None and at < previous):
            raise FrozenBuildError("session_export_invalid", "native record outside ordered cutoff")
        if not isinstance(record.get("payload"), dict):
            raise FrozenBuildError("session_export_invalid", "native object payload required")
        previous = at
    parents = tuple(
        dict.fromkeys(
            value
            for field in ("parent_thread_id", "forked_from_id")
            if (value := meta.get(field)) is not None
        )
    )
    if any(
        not isinstance(value, str) or not value.strip() or value == expected_thread_id
        for value in parents
    ):
        raise FrozenBuildError("session_identity_mismatch", "invalid native parent identity")
    return SessionExport(
        source="codex",
        native_session_id=expected_thread_id,
        parent_native_ids=parents,
        event_start=0,
        event_end_exclusive=len(records),
        captured_at=captured_at,
        cutoff_at=cutoff_at,
        completeness="partial",
        gaps=(
            "Exact local native journal prefix; records after cutoff are excluded.",
            "Native line indexes are not a provider-wide causal event sequence.",
            "Inherited history and external provider state are not established by this export.",
        ),
        native_objects=(
            FrozenObject(
                object_id=object_id,
                purpose="session",
                object=ArtifactObjectDeclaration(
                    logical_path=logical_path,
                    digest_sha256=hashlib.sha256(raw).hexdigest(),
                    size_bytes=len(raw),
                    media_type="application/x-ndjson",
                ),
            ),
        ),
    )


def freeze_codex_rollout_prefix(
    source: Path,
    destination: Path,
    *,
    expected_thread_id: str,
    captured_at: datetime,
    cutoff_at: datetime,
) -> SessionExport:
    return _freeze_selected_native(
        source,
        destination,
        admission=admit_codex_rollout_prefix,
        object_id="native-codex-rollout",
        object_suffix=".jsonl",
        captured_at=captured_at,
        cutoff_at=cutoff_at,
        expected_identity={"expected_thread_id": expected_thread_id},
    )
