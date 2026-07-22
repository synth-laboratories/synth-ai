"""Typed SDK namespace for immutable SMR runtime image releases."""

from __future__ import annotations

import hashlib
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Literal, TypedDict, cast

import httpx

from synth_ai.managed_research.errors import SmrApiError
from synth_ai.managed_research.sdk._base import _ClientNamespace

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_DIGEST = re.compile(r"^sha256:[0-9a-f]{64}$")
_GIT_SHA = re.compile(r"^[0-9a-f]{40}$")
_RELEASE_ID = re.compile(r"^imgrel_[0-9a-f]{64}$")
_UPLOAD_ID = re.compile(r"^imgup_[0-9a-f]{32}$")
_IMAGE_REF = re.compile(r"^[a-z0-9]+(?:[._/-][a-z0-9]+)*(?::[A-Za-z0-9_][A-Za-z0-9_.-]{0,127})$")
_DECLARATION_FIELDS = frozenset(
    {
        "kind",
        "archive_sha256",
        "archive_size_bytes",
        "image_manifest_digest",
        "image_ref",
        "platform_os",
        "platform_architecture",
        "source_repository",
        "source_commit_sha",
        "fixture_manifest_sha256",
        "fixture_binary_sha256",
    }
)


class ImageReleaseDeclaration(TypedDict):
    kind: Literal["craftax_scorer"]
    archive_sha256: str
    archive_size_bytes: int
    image_manifest_digest: str
    image_ref: str
    platform_os: Literal["linux"]
    platform_architecture: Literal["amd64", "arm64"]
    source_repository: str
    source_commit_sha: str
    fixture_manifest_sha256: str
    fixture_binary_sha256: str


class ImageReleaseArtifact(TypedDict):
    artifact_id: str
    archive_sha256: str
    archive_size_bytes: int


class ImageReleaseInspection(TypedDict):
    archive_format: Literal["oci-image-layout-tar-v1"]
    image_manifest_digest: str
    image_config_digest: str
    image_ref: str
    platform_os: Literal["linux"]
    platform_architecture: Literal["amd64", "arm64"]


class ImageRelease(TypedDict):
    schema_version: Literal["smr-image-release-v1"]
    release_id: str
    org_id: str
    artifact: ImageReleaseArtifact
    declaration: ImageReleaseDeclaration
    inspection: ImageReleaseInspection
    package_release_timestamps: dict[str, str]


class ImageReleaseUpload(TypedDict):
    schema_version: Literal["smr-image-release-upload-v2"]
    upload_id: str
    release_id: str
    upload_url: str
    upload_required: bool
    upload_mode: Literal["content_addressed_quarantine"]
    storage_admission: dict[str, Any] | None
    expires_in: int
    declaration: ImageReleaseDeclaration
    package_release_timestamps: dict[str, str]


class ImageReleaseUploadReconciliation(TypedDict):
    upload_id: str
    status: Literal["verified_and_published", "already_published"]
    object_key: str


class ImageReleaseFinalize(TypedDict):
    schema_version: Literal["smr-image-release-finalize-v1"]
    release: ImageRelease
    upload_reconciliation: ImageReleaseUploadReconciliation


def _exact_fields(
    payload: Mapping[str, object],
    expected: frozenset[str],
    *,
    label: str,
) -> None:
    actual = frozenset(payload)
    if actual != expected:
        raise ValueError(
            f"{label} fields do not match the contract "
            f"(missing={sorted(expected - actual)}, "
            f"unexpected={sorted(actual - expected)})"
        )


def _text(payload: Mapping[str, object], key: str, *, label: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError(f"{label}.{key} must be a nonempty trimmed string")
    return value


def image_release_declaration(
    payload: ImageReleaseDeclaration | Mapping[str, object],
) -> ImageReleaseDeclaration:
    if not isinstance(payload, Mapping):
        raise ValueError("image release declaration must be an object")
    _exact_fields(payload, _DECLARATION_FIELDS, label="declaration")
    kind = _text(payload, "kind", label="declaration")
    archive_sha256 = _text(payload, "archive_sha256", label="declaration")
    archive_size_bytes = payload.get("archive_size_bytes")
    image_manifest_digest = _text(
        payload,
        "image_manifest_digest",
        label="declaration",
    )
    image_ref = _text(payload, "image_ref", label="declaration")
    platform_os = _text(payload, "platform_os", label="declaration")
    platform_architecture = _text(
        payload,
        "platform_architecture",
        label="declaration",
    )
    source_repository = _text(payload, "source_repository", label="declaration")
    source_commit_sha = _text(payload, "source_commit_sha", label="declaration")
    fixture_manifest_sha256 = _text(
        payload,
        "fixture_manifest_sha256",
        label="declaration",
    )
    fixture_binary_sha256 = _text(
        payload,
        "fixture_binary_sha256",
        label="declaration",
    )
    if kind != "craftax_scorer":
        raise ValueError("declaration.kind must be craftax_scorer")
    if not _SHA256.fullmatch(archive_sha256):
        raise ValueError("declaration.archive_sha256 must be lowercase SHA-256")
    if (
        isinstance(archive_size_bytes, bool)
        or not isinstance(archive_size_bytes, int)
        or not 1 <= archive_size_bytes <= 16 * 1024**3
    ):
        raise ValueError("declaration.archive_size_bytes must be between 1 byte and 16 GiB")
    if not _DIGEST.fullmatch(image_manifest_digest):
        raise ValueError("declaration.image_manifest_digest must be a sha256 digest")
    if not _IMAGE_REF.fullmatch(image_ref):
        raise ValueError("declaration.image_ref is invalid")
    if platform_os != "linux" or platform_architecture not in {"amd64", "arm64"}:
        raise ValueError("declaration platform must be linux/amd64 or linux/arm64")
    if not source_repository.startswith("https://github.com/"):
        raise ValueError("declaration.source_repository must be an HTTPS GitHub URL")
    if not _GIT_SHA.fullmatch(source_commit_sha):
        raise ValueError("declaration.source_commit_sha must be a lowercase full Git SHA")
    if not _SHA256.fullmatch(fixture_manifest_sha256) or not _SHA256.fullmatch(
        fixture_binary_sha256
    ):
        raise ValueError("declaration fixture identities must be lowercase SHA-256")
    return cast(
        ImageReleaseDeclaration,
        {
            "kind": kind,
            "archive_sha256": archive_sha256,
            "archive_size_bytes": archive_size_bytes,
            "image_manifest_digest": image_manifest_digest,
            "image_ref": image_ref,
            "platform_os": platform_os,
            "platform_architecture": platform_architecture,
            "source_repository": source_repository,
            "source_commit_sha": source_commit_sha,
            "fixture_manifest_sha256": fixture_manifest_sha256,
            "fixture_binary_sha256": fixture_binary_sha256,
        },
    )


def _image_release_from_wire(payload: object) -> ImageRelease:
    if not isinstance(payload, Mapping):
        raise ValueError("image release response must be an object")
    _exact_fields(
        payload,
        frozenset(
            {
                "schema_version",
                "release_id",
                "org_id",
                "artifact",
                "declaration",
                "inspection",
                "package_release_timestamps",
            }
        ),
        label="image_release",
    )
    if _text(payload, "schema_version", label="image_release") != "smr-image-release-v1":
        raise ValueError("image release schema_version is unsupported")
    release_id = _text(payload, "release_id", label="image_release")
    if not _RELEASE_ID.fullmatch(release_id):
        raise ValueError("image release release_id is invalid")
    declaration_payload = payload.get("declaration")
    artifact_payload = payload.get("artifact")
    inspection_payload = payload.get("inspection")
    if not isinstance(artifact_payload, Mapping) or not isinstance(inspection_payload, Mapping):
        raise ValueError("image release artifact and inspection must be objects")
    package_timestamps = payload.get("package_release_timestamps")
    if not isinstance(package_timestamps, Mapping) or any(
        not isinstance(name, str) or not isinstance(timestamp, str)
        for name, timestamp in package_timestamps.items()
    ):
        raise ValueError("image release package_release_timestamps must be a string map")
    declaration = image_release_declaration(cast(Mapping[str, object], declaration_payload))
    _exact_fields(
        artifact_payload,
        frozenset({"artifact_id", "archive_sha256", "archive_size_bytes"}),
        label="image_release.artifact",
    )
    artifact_id = _text(artifact_payload, "artifact_id", label="image_release.artifact")
    if artifact_id != f"imgobj_{declaration['archive_sha256']}":
        raise ValueError("image release artifact_id does not bind archive_sha256")
    if artifact_payload.get("archive_sha256") != declaration["archive_sha256"] or (
        artifact_payload.get("archive_size_bytes") != declaration["archive_size_bytes"]
    ):
        raise ValueError("image release artifact does not bind the declaration")
    _exact_fields(
        inspection_payload,
        frozenset(
            {
                "archive_format",
                "image_manifest_digest",
                "image_config_digest",
                "image_ref",
                "platform_os",
                "platform_architecture",
            }
        ),
        label="image_release.inspection",
    )
    if inspection_payload.get("archive_format") != "oci-image-layout-tar-v1":
        raise ValueError("image release inspection archive_format is unsupported")
    for key in (
        "image_manifest_digest",
        "image_ref",
        "platform_os",
        "platform_architecture",
    ):
        if inspection_payload.get(key) != declaration[key]:
            raise ValueError(f"image release inspection.{key} does not bind declaration")
    image_config_digest = _text(
        inspection_payload,
        "image_config_digest",
        label="image_release.inspection",
    )
    if not _DIGEST.fullmatch(image_config_digest):
        raise ValueError("image release inspection.image_config_digest is invalid")
    return cast(ImageRelease, dict(payload))


def _image_release_upload_from_wire(payload: object) -> ImageReleaseUpload:
    if not isinstance(payload, Mapping):
        raise ValueError("image release upload response must be an object")
    _exact_fields(
        payload,
        frozenset(
            {
                "schema_version",
                "upload_id",
                "release_id",
                "upload_url",
                "upload_required",
                "upload_mode",
                "storage_admission",
                "expires_in",
                "declaration",
                "package_release_timestamps",
            }
        ),
        label="image_release_upload",
    )
    if payload.get("schema_version") != "smr-image-release-upload-v2":
        raise ValueError("image release upload schema_version is unsupported")
    upload_id = _text(payload, "upload_id", label="image_release_upload")
    release_id = _text(payload, "release_id", label="image_release_upload")
    upload_url = _text(payload, "upload_url", label="image_release_upload")
    if not _UPLOAD_ID.fullmatch(upload_id) or not _RELEASE_ID.fullmatch(release_id):
        raise ValueError("image release upload identifiers are invalid")
    if not upload_url.startswith(("https://", "http://localhost", "http://127.")):
        raise ValueError("image release upload_url must use HTTPS or loopback HTTP")
    expires_in = payload.get("expires_in")
    if isinstance(expires_in, bool) or not isinstance(expires_in, int):
        raise ValueError("image release upload expires_in must be an integer")
    if not isinstance(payload.get("upload_required"), bool):
        raise ValueError("image release upload_required must be a boolean")
    if payload.get("upload_mode") != "content_addressed_quarantine":
        raise ValueError("image release upload_mode is unsupported")
    admission = payload.get("storage_admission")
    if admission is not None and not isinstance(admission, Mapping):
        raise ValueError("image release storage_admission must be an object or null")
    package_timestamps = payload.get("package_release_timestamps")
    if not isinstance(package_timestamps, Mapping) or any(
        not isinstance(name, str) or not isinstance(timestamp, str)
        for name, timestamp in package_timestamps.items()
    ):
        raise ValueError("image release package_release_timestamps must be a string map")
    image_release_declaration(cast(Mapping[str, object], payload.get("declaration")))
    return cast(ImageReleaseUpload, dict(payload))


def _image_release_finalize_from_wire(payload: object) -> ImageReleaseFinalize:
    if not isinstance(payload, Mapping):
        raise ValueError("image release finalize response must be an object")
    _exact_fields(
        payload,
        frozenset({"schema_version", "release", "upload_reconciliation"}),
        label="image_release_finalize",
    )
    if payload.get("schema_version") != "smr-image-release-finalize-v1":
        raise ValueError("image release finalize schema_version is unsupported")
    release = _image_release_from_wire(payload.get("release"))
    reconciliation = payload.get("upload_reconciliation")
    if not isinstance(reconciliation, Mapping):
        raise ValueError("image release upload_reconciliation must be an object")
    _exact_fields(
        reconciliation,
        frozenset({"upload_id", "status", "object_key"}),
        label="image_release_finalize.upload_reconciliation",
    )
    upload_id = _text(
        reconciliation,
        "upload_id",
        label="image_release_finalize.upload_reconciliation",
    )
    object_key = _text(
        reconciliation,
        "object_key",
        label="image_release_finalize.upload_reconciliation",
    )
    if not _UPLOAD_ID.fullmatch(upload_id) or reconciliation.get("status") not in {
        "verified_and_published",
        "already_published",
    }:
        raise ValueError("image release upload reconciliation evidence is invalid")
    if not object_key.startswith("smr/env-images/objects/"):
        raise ValueError("image release upload reconciliation object_key is invalid")
    return cast(
        ImageReleaseFinalize,
        {
            "schema_version": "smr-image-release-finalize-v1",
            "release": release,
            "upload_reconciliation": dict(reconciliation),
        },
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class ImageReleasesAPI(_ClientNamespace):
    def create_upload(
        self,
        declaration: ImageReleaseDeclaration | Mapping[str, object],
        *,
        expires_in: int = 3600,
    ) -> ImageReleaseUpload:
        return self._client.create_image_release_upload(
            declaration=image_release_declaration(declaration),
            expires_in=expires_in,
        )

    def finalize(
        self,
        *,
        upload_id: str,
        declaration: ImageReleaseDeclaration | Mapping[str, object],
    ) -> ImageReleaseFinalize:
        return self._client.finalize_image_release(
            upload_id=upload_id,
            declaration=image_release_declaration(declaration),
        )

    def get(self, *, release_id: str) -> ImageRelease:
        return self._client.get_image_release(release_id=release_id)

    def status(self, *, release_id: str) -> ImageRelease:
        """Read and validate the immutable release receipt."""
        return self.get(release_id=release_id)

    def reconcile(
        self,
        *,
        upload_id: str,
        declaration: ImageReleaseDeclaration | Mapping[str, object],
    ) -> ImageReleaseFinalize:
        """Idempotently finalize and prove content-addressed publication."""
        return self.finalize(upload_id=upload_id, declaration=declaration)

    def upload_archive(
        self,
        archive_path: str | Path,
        *,
        declaration: ImageReleaseDeclaration | Mapping[str, object],
        expires_in: int = 3600,
        upload_timeout_seconds: float = 1800.0,
    ) -> ImageReleaseFinalize:
        """Upload and finalize one exact OCI archive without API-key forwarding."""
        normalized = image_release_declaration(declaration)
        if (
            isinstance(upload_timeout_seconds, bool)
            or not isinstance(upload_timeout_seconds, (int, float))
            or not 1 <= float(upload_timeout_seconds) <= 7200
        ):
            raise ValueError("upload_timeout_seconds must be between 1 and 7200")
        path = Path(archive_path).expanduser().resolve()
        if not path.is_file():
            raise ValueError(f"image release archive is not a file: {path}")
        if path.stat().st_size != normalized["archive_size_bytes"] or (
            _sha256(path) != normalized["archive_sha256"]
        ):
            raise ValueError("image release archive does not match its declaration")
        upload = self.create_upload(normalized, expires_in=expires_in)
        if upload["declaration"] != normalized:
            raise ValueError("image release upload response changed the declaration")
        if upload["upload_required"]:
            upload_error: httpx.TransportError | None = None
            with httpx.Client(
                timeout=upload_timeout_seconds,
                follow_redirects=False,
            ) as upload_client:
                for _attempt in range(2):
                    try:
                        with path.open("rb") as handle:
                            response = upload_client.put(
                                upload["upload_url"],
                                content=handle,
                                headers={
                                    "Content-Length": str(normalized["archive_size_bytes"]),
                                },
                            )
                        upload_error = None
                        break
                    except httpx.TransportError as exc:
                        # Exact-key PUT is idempotent. One replay reconciles the only
                        # ambiguous outcome without crossing the storage boundary.
                        upload_error = exc
                else:
                    raise RuntimeError(
                        "image release archive upload outcome is uncertain "
                        f"(upload_id={upload['upload_id']}, "
                        f"release_id={upload['release_id']})"
                    ) from upload_error
            if response.is_error:
                raise RuntimeError(
                    f"image release archive upload failed with HTTP {response.status_code}"
                )
        try:
            finalized = self.finalize(
                upload_id=upload["upload_id"],
                declaration=normalized,
            )
        except SmrApiError as exc:
            if exc.status_code is not None:
                raise
            # The backend finalize operation is idempotent and always proves
            # content publication, so replay exactly once after response loss.
            try:
                finalized = self.reconcile(
                    upload_id=upload["upload_id"],
                    declaration=normalized,
                )
            except Exception as reconcile_exc:
                raise RuntimeError(
                    "image release finalize outcome is uncertain "
                    f"(upload_id={upload['upload_id']}, "
                    f"release_id={upload['release_id']})"
                ) from reconcile_exc
        if finalized["upload_reconciliation"]["upload_id"] != upload["upload_id"]:
            raise ValueError("image release finalize reconciled a different upload_id")
        if finalized["release"]["release_id"] != upload["release_id"]:
            raise ValueError("image release finalize returned a different release_id")
        return finalized


__all__ = [
    "ImageRelease",
    "ImageReleaseArtifact",
    "ImageReleaseDeclaration",
    "ImageReleaseInspection",
    "ImageReleaseFinalize",
    "ImageReleaseUploadReconciliation",
    "ImageReleasesAPI",
    "ImageReleaseUpload",
    "image_release_declaration",
    "_image_release_from_wire",
    "_image_release_finalize_from_wire",
    "_image_release_upload_from_wire",
]
