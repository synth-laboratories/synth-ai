"""Typed SDK namespace for immutable SMR runtime image releases."""

from __future__ import annotations

import hashlib
import ipaddress
import json
import re
import time
import urllib.parse
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal, NotRequired, TypedDict, cast

import httpx

from synth_ai.managed_research.errors import SmrApiError
from synth_ai.managed_research.sdk._base import _ClientNamespace

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_DIGEST = re.compile(r"^sha256:[0-9a-f]{64}$")
_GIT_SHA = re.compile(r"^[0-9a-f]{40}$")
_RELEASE_ID = re.compile(r"^imgrel_[0-9a-f]{64}$")
_UPLOAD_ID = re.compile(r"^imgup_[0-9a-f]{32}$")
_IMAGE_REF = re.compile(r"^[a-z0-9]+(?:[._/-][a-z0-9]+)*(?::[A-Za-z0-9_][A-Za-z0-9_.-]{0,127})$")
IMAGE_RELEASE_SINGLE_PUT_MAX_BYTES = 5_000_000_000
_UPLOAD_ERROR_DETAIL_MAX_BYTES = 4 * 1024
_UPLOAD_ERROR_DETAIL_MAX_CHARS = 600
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
    python_packages: NotRequired[list[str]]
    platform_os: Literal["linux"]
    platform_architecture: Literal["amd64", "arm64"]


class ImageReleaseStorageIdentity(TypedDict):
    object_key: str
    etag: str
    size_bytes: int


def _bounded_upload_error_detail(response: httpx.Response) -> str:
    """Read a small diagnostic prefix without buffering a provider response."""

    body = bytearray()
    for chunk in response.iter_bytes(chunk_size=1024):
        remaining = _UPLOAD_ERROR_DETAIL_MAX_BYTES - len(body)
        if remaining <= 0:
            break
        body.extend(chunk[:remaining])
        if len(body) >= _UPLOAD_ERROR_DETAIL_MAX_BYTES:
            break
    return " ".join(body.decode("utf-8", errors="replace").split())[
        :_UPLOAD_ERROR_DETAIL_MAX_CHARS
    ]


class ImageRelease(TypedDict):
    schema_version: Literal["smr-image-release-v1"]
    release_id: str
    org_id: str
    artifact: ImageReleaseArtifact
    declaration: ImageReleaseDeclaration
    inspection: ImageReleaseInspection
    package_release_timestamps: dict[str, str]
    storage: NotRequired[ImageReleaseStorageIdentity]


class ImageReleaseUpload(TypedDict):
    schema_version: Literal["smr-image-release-upload-v3"]
    upload_id: str
    release_id: str
    upload_url: str
    upload_headers: dict[str, str]
    upload_required: bool
    upload_mode: Literal["content_addressed_quarantine"]
    storage_admission: dict[str, Any] | None
    expires_in: int
    finalize_deadline_at: str
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


def _valid_upload_url(value: str) -> bool:
    parsed = urllib.parse.urlparse(value)
    if (
        not parsed.hostname
        or parsed.username is not None
        or parsed.password is not None
        or parsed.fragment
    ):
        return False
    if parsed.scheme == "https":
        return True
    if parsed.scheme != "http":
        return False
    host = parsed.hostname.lower()
    if host == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def _declaration_sha256(declaration: Mapping[str, object]) -> str:
    encoded = json.dumps(
        dict(declaration),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _utc_deadline(value: object, *, label: str) -> datetime:
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError(f"{label} must be a nonempty datetime string")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(f"{label} must be an ISO-8601 datetime") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError(f"{label} must include a timezone")
    normalized = parsed.astimezone(timezone.utc)
    if normalized <= datetime.now(timezone.utc):
        raise ValueError(f"{label} must be in the future")
    return normalized


def _remaining_deadline_seconds(
    deadline_at: str,
    *,
    requested_seconds: float,
    label: str,
) -> float:
    requested = float(requested_seconds)
    if requested <= 0:
        raise ValueError(f"{label} timeout must be positive")
    remaining = (
        _utc_deadline(deadline_at, label=label) - datetime.now(timezone.utc)
    ).total_seconds() - 5.0
    if remaining <= 0:
        raise RuntimeError(f"{label} deadline expired before the request started")
    return min(requested, remaining)


def _remaining_upload_seconds(
    *,
    acquired_monotonic: float,
    expires_in: int,
    requested_seconds: float,
) -> float:
    remaining = (
        acquired_monotonic + float(expires_in) - time.monotonic() - 5.0
    )
    if remaining <= 0:
        raise RuntimeError("image release upload URL expired before transfer")
    return min(float(requested_seconds), remaining)


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
        or not 1 <= archive_size_bytes <= IMAGE_RELEASE_SINGLE_PUT_MAX_BYTES
    ):
        raise ValueError(
            "declaration.archive_size_bytes must be between 1 byte and "
            "5,000,000,000 bytes (single-PUT limit)"
        )
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
    base_fields = frozenset(
        {
            "schema_version",
            "release_id",
            "org_id",
            "artifact",
            "declaration",
            "inspection",
            "package_release_timestamps",
        }
    )
    if frozenset(payload) not in {base_fields, base_fields | {"storage"}}:
        _exact_fields(payload, base_fields | {"storage"}, label="image_release")
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
    inspection_fields = frozenset(
        {
            "archive_format",
            "image_manifest_digest",
            "image_config_digest",
            "image_ref",
            "platform_os",
            "platform_architecture",
        }
    )
    if frozenset(inspection_payload) not in {
        inspection_fields,
        inspection_fields | {"python_packages"},
    }:
        _exact_fields(
            inspection_payload,
            inspection_fields | {"python_packages"},
            label="image_release.inspection",
        )
    python_packages = inspection_payload.get("python_packages")
    if python_packages is not None and python_packages != []:
        raise ValueError(
            "scorer image release inspection.python_packages must be empty"
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
    storage_payload = payload.get("storage")
    if "storage" in payload:
        if not isinstance(storage_payload, Mapping):
            raise ValueError("image release storage must be an object")
        _exact_fields(
            storage_payload,
            frozenset({"object_key", "etag", "size_bytes"}),
            label="image_release.storage",
        )
        expected_object_key = (
            f"smr/env-images/objects/{declaration['archive_sha256']}.tar"
        )
        if (
            _text(storage_payload, "object_key", label="image_release.storage")
            != expected_object_key
            or not _text(storage_payload, "etag", label="image_release.storage")
            or storage_payload.get("size_bytes") != declaration["archive_size_bytes"]
        ):
            raise ValueError("image release storage does not bind the declaration")
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
                "upload_headers",
                "upload_required",
                "upload_mode",
                "storage_admission",
                "expires_in",
                "finalize_deadline_at",
                "declaration",
                "package_release_timestamps",
            }
        ),
        label="image_release_upload",
    )
    if payload.get("schema_version") != "smr-image-release-upload-v3":
        raise ValueError("image release upload schema_version is unsupported")
    upload_id = _text(payload, "upload_id", label="image_release_upload")
    release_id = _text(payload, "release_id", label="image_release_upload")
    upload_url = _text(payload, "upload_url", label="image_release_upload")
    if not _UPLOAD_ID.fullmatch(upload_id) or not _RELEASE_ID.fullmatch(release_id):
        raise ValueError("image release upload identifiers are invalid")
    if not _valid_upload_url(upload_url):
        raise ValueError("image release upload_url must use HTTPS or loopback HTTP")
    declaration = image_release_declaration(
        cast(Mapping[str, object], payload.get("declaration"))
    )
    upload_headers = payload.get("upload_headers")
    expected_upload_headers = {
        "Content-Length": str(declaration["archive_size_bytes"]),
        "If-None-Match": "*",
        "x-amz-meta-synth-upload-id": upload_id,
        "x-amz-meta-synth-declaration-sha256": _declaration_sha256(declaration),
    }
    if upload_headers != expected_upload_headers:
        raise ValueError(
            "image release upload_headers must bind create-only content length"
        )
    expires_in = payload.get("expires_in")
    if isinstance(expires_in, bool) or not isinstance(expires_in, int):
        raise ValueError("image release upload expires_in must be an integer")
    _utc_deadline(
        payload.get("finalize_deadline_at"),
        label="image release upload finalize_deadline_at",
    )
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
    if not isinstance(release.get("storage"), Mapping):
        raise ValueError(
            "image release finalize must include the immutable storage identity"
        )
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
    expected_object_key = (
        "smr/env-images/objects/"
        f"{release['declaration']['archive_sha256']}.tar"
    )
    if object_key != expected_object_key:
        raise ValueError(
            "image release upload reconciliation does not bind the release object"
        )
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
        timeout_seconds: float | None = None,
    ) -> ImageReleaseFinalize:
        return self._client.finalize_image_release(
            upload_id=upload_id,
            declaration=image_release_declaration(declaration),
            timeout_seconds=timeout_seconds,
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
        timeout_seconds: float | None = None,
    ) -> ImageReleaseFinalize:
        """Idempotently finalize and prove content-addressed publication."""
        return self.finalize(
            upload_id=upload_id,
            declaration=declaration,
            timeout_seconds=timeout_seconds,
        )

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
        upload_acquired_monotonic = time.monotonic()
        if upload["declaration"] != normalized:
            raise ValueError("image release upload response changed the declaration")
        if upload["upload_required"]:
            status_code: int | None = None
            response_detail = ""
            with httpx.Client(
                timeout=upload_timeout_seconds,
                follow_redirects=False,
            ) as upload_client:
                for attempt_index in range(2):
                    try:
                        with path.open("rb") as handle:
                            with upload_client.stream(
                                "PUT",
                                upload["upload_url"],
                                content=handle,
                                headers=upload["upload_headers"],
                                timeout=_remaining_upload_seconds(
                                    acquired_monotonic=upload_acquired_monotonic,
                                    expires_in=upload["expires_in"],
                                    requested_seconds=upload_timeout_seconds,
                                ),
                            ) as response:
                                status_code = response.status_code
                                response_detail = ""
                                if (
                                    not response.is_success
                                    and status_code not in {409, 412}
                                ):
                                    response_detail = _bounded_upload_error_detail(
                                        response
                                    )
                        if status_code == 409 and attempt_index == 0:
                            continue
                        break
                    except httpx.TransportError:
                        # A create-only retry can return 412 when the first request
                        # committed but its response was lost. Finalize verifies the
                        # exact digest and reconciles that ambiguous success.
                        continue
                else:
                    # A transport failure is an in-doubt write, not proof that
                    # the create-only PUT failed. The backend finalizer is the
                    # authority that re-HEADs and verifies the exact object.
                    status_code = None
            if (
                status_code is not None
                and not 200 <= status_code < 300
                and status_code not in {409, 412}
            ):
                suffix = f": {response_detail}" if response_detail else ""
                raise RuntimeError(
                    "image release archive upload failed with HTTP "
                    f"{status_code}{suffix}"
                )
        finalize_timeout_seconds = _remaining_deadline_seconds(
            upload["finalize_deadline_at"],
            requested_seconds=upload_timeout_seconds,
            label="image release finalize",
        )
        try:
            finalized = self.finalize(
                upload_id=upload["upload_id"],
                declaration=normalized,
                timeout_seconds=finalize_timeout_seconds,
            )
        except SmrApiError as exc:
            if exc.status_code is not None:
                raise
            if isinstance(exc.__cause__, httpx.ConnectError):
                # A failed TCP connect proves the request was not admitted.
                finalized = self.reconcile(
                    upload_id=upload["upload_id"],
                    declaration=normalized,
                    timeout_seconds=_remaining_deadline_seconds(
                        upload["finalize_deadline_at"],
                        requested_seconds=upload_timeout_seconds,
                        label="image release finalize reconciliation",
                    ),
                )
            else:
                # A read timeout or disconnect can happen after publication
                # starts. Automatic replay could duplicate registry work; let
                # the caller explicitly reconcile the durable upload id.
                raise RuntimeError(
                    "image release finalize outcome is uncertain; call reconcile "
                    f"with upload_id={upload['upload_id']}"
                ) from exc
        if finalized["upload_reconciliation"]["upload_id"] != upload["upload_id"]:
            raise ValueError("image release finalize reconciled a different upload_id")
        if finalized["release"]["release_id"] != upload["release_id"]:
            raise ValueError("image release finalize returned a different release_id")
        return finalized


__all__ = [
    "IMAGE_RELEASE_SINGLE_PUT_MAX_BYTES",
    "ImageRelease",
    "ImageReleaseArtifact",
    "ImageReleaseDeclaration",
    "ImageReleaseInspection",
    "ImageReleaseStorageIdentity",
    "ImageReleaseFinalize",
    "ImageReleaseUploadReconciliation",
    "ImageReleasesAPI",
    "ImageReleaseUpload",
    "image_release_declaration",
    "_image_release_from_wire",
    "_image_release_finalize_from_wire",
    "_image_release_upload_from_wire",
]
