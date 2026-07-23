"""Customer actor image namespace: upload OCI archives, receive executable releases.

``client.research.images`` turns a customer-built OCI layout
archive into an immutable, org-scoped, digest-pinned runtime image release.
The returned ``release_id`` binds a run's worker role through
``actor_image_overrides``; the imgrel artifact identity stays available for
audits alongside it.
"""

from __future__ import annotations

import hashlib
import ipaddress
import json
import os
import re
import selectors
import signal
import shutil
import subprocess
import tarfile
import tempfile
import time
import urllib.parse
from collections.abc import Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from pathlib import PurePosixPath
from typing import Any, List

import httpx

from synth_ai.managed_research.errors import SmrApiError
from synth_ai.managed_research.sdk._base import _ClientNamespace
from synth_ai.managed_research.sdk.image_releases import (
    IMAGE_RELEASE_SINGLE_PUT_MAX_BYTES,
    _bounded_upload_error_detail,
)

try:
    import fcntl
except ImportError:  # pragma: no cover - Craftax managed images require Unix Docker hosts.
    fcntl = None

ACTOR_RUNTIME_IMAGE_KIND = "actor_runtime"
ACTOR_RUNTIME_INTERFACE_MODE = "synth_actor_runtime"
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_DIGEST = re.compile(r"^sha256:[0-9a-f]{64}$")
_GIT_SHA = re.compile(r"^[0-9a-f]{40}$")
_RELEASE_ID = re.compile(r"^imgrel_[0-9a-f]{64}$")
_UPLOAD_ID = re.compile(r"^imgup_[0-9a-f]{32}$")
_IMAGE_NAME = re.compile(r"^[a-z0-9]+(?:[._/-][a-z0-9]+)*$")
_IMAGE_TAG = re.compile(r"^[A-Za-z0-9_][A-Za-z0-9_.-]{0,127}$")
_CAPABILITY = re.compile(r"^[a-z0-9][a-z0-9_]{0,63}$")
_PACKAGE_REQUIREMENT = re.compile(
    r"^(?P<name>[a-z0-9][a-z0-9._-]*)==(?P<version>[A-Za-z0-9][A-Za-z0-9.!+_-]{0,127})$"
)
_PLATFORMS = ("linux/amd64", "linux/arm64")
_ACTOR_IMAGE_CACHE_MAX_ENTRIES = 2
_ACTOR_IMAGE_CACHE_MAX_BYTES = 4 * 1024**3
_ACTOR_IMAGE_BUILD_TIMEOUT_SECONDS = 1800.0
_CRAFTAX_SCORER_MAX_BYTES = 64 * 1024**2
_OCI_ARCHIVE_MAX_MEMBERS = 4_096
_OCI_ARCHIVE_MAX_MEMBER_NAME_BYTES = 192
_OCI_ARCHIVE_MAX_MEMBER_NAMES_BYTES = 512 * 1024
_OCI_ARCHIVE_METADATA_MAX_BYTES = 4 * 1024 * 1024
_OCI_ARCHIVE_MAX_TRAILING_PADDING_BYTES = 1024 * 1024
_ACTOR_DECLARATION_FIELDS = frozenset(
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
        "actor_role",
        "interface_mode",
        "capabilities",
        "python_packages",
        "recipe_digest",
    }
)
CUSTOMER_ACTOR_IMAGE_PACKAGE_LABEL = "io.synth.actor-runtime.python-packages"
CRAFTAX_WORKER_BASE_IMAGE = "synth-local-open-research-craftax:latest"
CRAFTAX_WORKER_SCORER_PATH = "/opt/synth/task-assets/craftax_repl"
CRAFTAX_WORKER_CAPABILITIES = (
    "craftax_eval",
    "jax_cpu",
    "managed_research_sdk",
    "mcp_client",
    "synth_sdk",
)
CUSTOMER_ACTOR_IMAGE_PYPI_ALLOWLIST = frozenset(
    {
        "chex", "craftax", "crafter", "distrax", "flax", "gymnasium", "gymnax",
        "httpx", "imageio", "imageio-ffmpeg", "jax", "modal", "nle", "numpy",
        "optax", "orbax-checkpoint", "pillow", "pydantic", "pyyaml", "synth-ai",
    }
)


@dataclass(frozen=True, slots=True)
class ActorImage:
    """An uploaded customer actor image and its executable release identity."""

    release_id: str
    """Org-owned runtime image release ID; bind this in actor_image_overrides."""

    image_release_id: str
    """Content-addressed immutable artifact identity (imgrel_...)."""

    status: str
    image_ref: str
    digest: str
    platform: str
    actor_role: str
    capabilities: tuple[str, ...]
    archive_sha256: str
    python_packages: tuple[str, ...] = ()
    package_release_timestamps: Mapping[str, str] | None = None
    recipe_digest: str | None = None

    image_substrates: tuple[str, ...] = ()
    """Where the executable image lives: 'org_registry' means registry-pulling
    hosts (Daytona) can run it; 'wasabi_artifact' means docker-load hosts can."""

    daytona_pullable: bool = False


class ActorImageFinalizeUncertainError(RuntimeError):
    """A finalize request may have committed and must be reconciled by ID."""

    def __init__(self, *, upload_id: str, declaration: Mapping[str, Any]) -> None:
        self.upload_id = upload_id
        self.declaration = dict(declaration)
        super().__init__(
            "actor image finalize outcome is uncertain; call "
            "images.reconcile_upload with this error's upload_id and declaration "
            f"(upload_id={upload_id})"
        )


@dataclass(frozen=True, slots=True)
class ActorImageRecipe:
    """A constrained, cacheable worker-image recipe built by the public SDK.

    This deliberately exposes no Dockerfile shell commands, arbitrary base
    references, or unchecked package resolver behavior. The resulting OCI
    archive is uploaded through the same immutable release protocol as a
    customer-built image.
    """

    name: str
    base_image: str
    source_repository: str
    source_commit_sha: str
    base_image_id: str
    capabilities: tuple[str, ...]
    python_packages: tuple[str, ...]
    assets: tuple[tuple[Path, str], ...] = ()
    platform: str = "linux/arm64"

    @classmethod
    def craftax_worker(
        cls,
        *,
        craftax_repl_path: str | Path,
        source: Mapping[str, Any],
        python_packages: Sequence[str] = (),
    ) -> "ActorImageRecipe":
        repository, commit_sha = _normalized_source(source)
        scorer = Path(craftax_repl_path).expanduser().resolve()
        if not scorer.is_file():
            raise ValueError(f"Craftax scorer asset is not a file: {scorer}")
        if not 1 <= scorer.stat().st_size <= _CRAFTAX_SCORER_MAX_BYTES:
            raise ValueError("Craftax scorer asset must be between 1 byte and 64 MiB")
        with scorer.open("rb") as asset:
            header = asset.read(20)
        if header[:4] != b"\x7fELF" or header[18:20] != b"\xb7\x00":
            raise ValueError("Craftax scorer asset must be an ELF Linux aarch64 binary")
        return cls(
            name="craftax-worker",
            base_image=CRAFTAX_WORKER_BASE_IMAGE,
            source_repository=repository,
            source_commit_sha=commit_sha,
            base_image_id=_local_image_id(CRAFTAX_WORKER_BASE_IMAGE),
            capabilities=CRAFTAX_WORKER_CAPABILITIES,
            python_packages=tuple(_normalized_python_packages(python_packages)),
            assets=((scorer, CRAFTAX_WORKER_SCORER_PATH),),
        )

    def recipe_digest(self) -> str:
        payload = {
            "name": self.name,
            "base_image": self.base_image,
            "base_image_id": self.base_image_id,
            "source_repository": self.source_repository,
            "source_commit_sha": self.source_commit_sha,
            "capabilities": sorted(self.capabilities),
            "python_packages": sorted(self.python_packages),
            "assets": [
                {
                    "sha256": _sha256_file(path),
                    "destination": destination,
                }
                for path, destination in self.assets
            ],
            "platform": self.platform,
        }
        return "sha256:" + hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()

    def validate(self) -> None:
        if self.name != "craftax-worker":
            raise ValueError("only the craftax-worker managed recipe is currently supported")
        if self.base_image != CRAFTAX_WORKER_BASE_IMAGE:
            raise ValueError("craftax-worker must use the screened Craftax actor base image")
        if not _DIGEST.fullmatch(self.base_image_id):
            raise ValueError("craftax-worker must record a local sha256 base-image identity")
        if self.platform != "linux/arm64":
            raise ValueError("craftax-worker is currently available only for linux/arm64")
        _normalized_source(
            {"repository": self.source_repository, "commit_sha": self.source_commit_sha}
        )
        _normalized_python_packages(self.python_packages)
        if tuple(sorted(self.capabilities)) != CRAFTAX_WORKER_CAPABILITIES:
            raise ValueError("craftax-worker capabilities must match the screened recipe")
        if not self.assets:
            raise ValueError("craftax-worker requires the Craftax scorer asset")
        for path, destination in self.assets:
            if not path.is_file():
                raise ValueError(f"managed image asset is not a file: {path}")
            if not 1 <= path.stat().st_size <= _CRAFTAX_SCORER_MAX_BYTES:
                raise ValueError(
                    "managed Craftax scorer asset must be between 1 byte and 64 MiB"
                )
            with path.open("rb") as asset:
                header = asset.read(20)
            if header[:4] != b"\x7fELF" or header[18:20] != b"\xb7\x00":
                raise ValueError("Craftax scorer asset must be an ELF Linux aarch64 binary")
            normalized_destination = PurePosixPath(destination)
            if str(normalized_destination) != CRAFTAX_WORKER_SCORER_PATH:
                raise ValueError(
                    "managed Craftax assets may only target " + CRAFTAX_WORKER_SCORER_PATH
                )


@dataclass(frozen=True, slots=True)
class _ActorImageCachePolicy:
    """Bounded local cache policy for rebuildable managed-image archives."""

    max_entries: int = _ACTOR_IMAGE_CACHE_MAX_ENTRIES
    max_bytes: int = _ACTOR_IMAGE_CACHE_MAX_BYTES

    def __post_init__(self) -> None:
        if self.max_entries < 1:
            raise ValueError("actor image cache max_entries must be >= 1")
        if self.max_bytes < 1:
            raise ValueError("actor image cache max_bytes must be >= 1")


@dataclass(frozen=True, slots=True)
class _ImageUploadLease:
    """Client-side deadline for one backend-admitted presigned upload."""

    upload_id: str
    release_id: str
    upload_url: str
    upload_required: bool
    upload_headers: dict[str, str]
    expires_in_seconds: float
    finalize_deadline_at: datetime
    acquired_monotonic: float

    @classmethod
    def from_response(
        cls,
        upload: Mapping[str, Any],
        *,
        archive_size_bytes: int,
        declaration: Mapping[str, Any],
        expires_in: int,
    ) -> "_ImageUploadLease":
        expected_fields = frozenset(
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
        )
        actual_fields = frozenset(upload)
        if actual_fields != expected_fields:
            raise ValueError(
                "image_upload fields do not match the v3 contract "
                f"(missing={sorted(expected_fields - actual_fields)}, "
                f"unexpected={sorted(actual_fields - expected_fields)})"
            )
        if upload.get("schema_version") != "smr-image-release-upload-v3":
            raise ValueError("image_upload.schema_version is unsupported")
        upload_id = upload.get("upload_id")
        release_id = upload.get("release_id")
        if (
            not isinstance(upload_id, str)
            or upload_id != upload_id.strip()
            or not _UPLOAD_ID.fullmatch(upload_id)
            or not isinstance(release_id, str)
            or release_id != release_id.strip()
            or not _RELEASE_ID.fullmatch(release_id)
        ):
            raise ValueError("image_upload identifiers are invalid")
        upload_url = upload.get("upload_url")
        if (
            not isinstance(upload_url, str)
            or upload_url != upload_url.strip()
            or not _valid_upload_url(upload_url)
        ):
            raise ValueError("image_upload.upload_url must use HTTPS or loopback HTTP")
        response_declaration = upload.get("declaration")
        if (
            type(response_declaration) is not dict
            or response_declaration != dict(declaration)
        ):
            raise ValueError("image_upload.declaration did not bind the request")
        upload_headers = upload.get("upload_headers")
        expected_headers = {
            "Content-Length": str(archive_size_bytes),
            "If-None-Match": "*",
            "x-amz-meta-synth-upload-id": upload_id,
            "x-amz-meta-synth-declaration-sha256": _declaration_sha256(
                declaration
            ),
        }
        if (
            type(upload_headers) is not dict
            or any(
                type(key) is not str or type(value) is not str
                for key, value in upload_headers.items()
            )
            or upload_headers != expected_headers
        ):
            raise ValueError(
                "image_upload.upload_headers must bind create-only content length"
            )
        upload_required = upload.get("upload_required")
        if type(upload_required) is not bool:
            raise ValueError("image_upload.upload_required must be a boolean")
        if upload.get("upload_mode") != "content_addressed_quarantine":
            raise ValueError("image_upload.upload_mode is unsupported")
        storage_admission = upload.get("storage_admission")
        if storage_admission is not None and type(storage_admission) is not dict:
            raise ValueError("image_upload.storage_admission must be an object or null")
        raw_expires_in = upload.get("expires_in")
        if type(raw_expires_in) is not int or raw_expires_in != expires_in:
            raise ValueError("image_upload.expires_in did not bind the request")
        expires_in_seconds = float(raw_expires_in)
        if expires_in_seconds <= 0:
            raise ValueError("image_upload.expires_in must be a positive number")
        finalize_deadline_at = _parse_utc_deadline(
            upload.get("finalize_deadline_at"),
            label="image_upload.finalize_deadline_at",
        )
        package_timestamps = upload.get("package_release_timestamps")
        if type(package_timestamps) is not dict or any(
            type(name) is not str or type(timestamp) is not str
            for name, timestamp in package_timestamps.items()
        ):
            raise ValueError(
                "image_upload.package_release_timestamps must be a string map"
            )
        return cls(
            upload_id=upload_id,
            release_id=release_id,
            upload_url=upload_url,
            upload_required=upload_required,
            upload_headers=dict(upload_headers),
            expires_in_seconds=expires_in_seconds,
            finalize_deadline_at=finalize_deadline_at,
            acquired_monotonic=time.monotonic(),
        )

    def timeout_seconds(self, *, requested_seconds: float) -> float:
        """Return one attempt's budget without crossing the lease deadline."""

        requested = float(requested_seconds)
        if requested <= 0:
            raise ValueError("upload_timeout_seconds must be positive")
        remaining = (
            self.acquired_monotonic
            + self.expires_in_seconds
            - time.monotonic()
            - 5.0
        )
        if remaining <= 0:
            raise RuntimeError(
                "actor image upload lease expired before transfer completed "
                f"(upload_id={self.upload_id})"
            )
        return min(requested, remaining)

    def finalize_timeout_seconds(self, *, requested_seconds: float) -> float:
        """Bound one finalize request by the server-owned eligibility window."""

        requested = float(requested_seconds)
        if requested <= 0:
            raise ValueError("upload_timeout_seconds must be positive")
        remaining = (
            self.finalize_deadline_at - datetime.now(timezone.utc)
        ).total_seconds() - 5.0
        if remaining <= 0:
            raise RuntimeError(
                "actor image finalize deadline expired before verification "
                f"(upload_id={self.upload_id})"
            )
        return min(requested, remaining)


def _text(payload: Mapping[str, Any], key: str, *, label: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label}.{key} must be a nonempty string")
    return value.strip()


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


def _declaration_sha256(declaration: Mapping[str, Any]) -> str:
    payload = json.dumps(
        dict(declaration),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _parse_utc_deadline(value: object, *, label: str) -> datetime:
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


def _actor_release_from_finalize(
    payload: Mapping[str, Any],
    *,
    expected_upload_id: str,
    expected_release_id: str | None,
    declaration: Mapping[str, Any],
) -> Mapping[str, Any]:
    expected_finalize_fields = frozenset(
        {"schema_version", "release", "upload_reconciliation"}
    )
    if type(payload) is not dict or frozenset(payload) != expected_finalize_fields:
        raise ValueError("image finalize response fields do not match the contract")
    if payload.get("schema_version") != "smr-image-release-finalize-v1":
        raise ValueError("image finalize schema_version is unsupported")

    reconciliation = payload.get("upload_reconciliation")
    expected_reconciliation_fields = frozenset(
        {"upload_id", "status", "object_key"}
    )
    if (
        type(reconciliation) is not dict
        or frozenset(reconciliation) != expected_reconciliation_fields
        or reconciliation.get("upload_id") != expected_upload_id
        or reconciliation.get("status")
        not in {"verified_and_published", "already_published"}
    ):
        raise ValueError("image finalize reconciliation evidence is invalid")
    object_key = reconciliation.get("object_key")
    expected_object_key = (
        "smr/env-images/objects/"
        f"{declaration.get('archive_sha256')}.tar"
    )
    if not isinstance(object_key, str) or object_key != expected_object_key:
        raise ValueError("image finalize reconciliation object_key is invalid")

    release = payload.get("release")
    base_release_fields = frozenset(
        {
            "schema_version",
            "release_id",
            "org_id",
            "artifact",
            "declaration",
            "inspection",
            "package_release_timestamps",
            "runtime_image_release",
        }
    )
    if (
        type(release) is not dict
        or frozenset(release) != base_release_fields | {"storage"}
    ):
        raise ValueError("image finalize release fields do not match the contract")
    if (
        release.get("schema_version") != "smr-image-release-v1"
        or release.get("declaration") != dict(declaration)
    ):
        raise ValueError("image finalize release did not bind the upload")
    release_id = release.get("release_id")
    if not isinstance(release_id, str) or not _RELEASE_ID.fullmatch(release_id):
        raise ValueError("image finalize release_id is invalid")
    if (
        expected_release_id is not None
        and release_id != expected_release_id
    ):
        raise ValueError("image finalize release_id did not bind the upload")
    if not isinstance(release.get("org_id"), str) or not release["org_id"].strip():
        raise ValueError("image finalize release org_id is invalid")
    for field_name in ("artifact", "inspection", "runtime_image_release"):
        if type(release.get(field_name)) is not dict:
            raise ValueError(f"image finalize release {field_name} must be an object")
    storage = release.get("storage")
    expected_storage_fields = {"object_key", "etag", "size_bytes"}
    if (
        type(storage) is not dict
        or set(storage) != expected_storage_fields
        or storage.get("object_key") != object_key
        or not isinstance(storage.get("etag"), str)
        or not storage["etag"]
        or storage.get("size_bytes") != declaration.get("archive_size_bytes")
    ):
        raise ValueError("image finalize storage identity is invalid")
    package_timestamps = release.get("package_release_timestamps")
    if type(package_timestamps) is not dict or any(
        type(name) is not str or type(timestamp) is not str
        for name, timestamp in package_timestamps.items()
    ):
        raise ValueError(
            "image finalize release package_release_timestamps must be a string map"
        )
    return release


def _actor_image_from_materialization(
    materialization: Mapping[str, Any],
    *,
    platform: str,
    archive_sha256: str,
) -> ActorImage:
    release_id = _text(materialization, "runtime_image_release_id", label="runtime_image_release")
    digest = _text(materialization, "resolved_digest", label="runtime_image_release")
    if not _DIGEST.fullmatch(digest):
        raise ValueError("runtime_image_release.resolved_digest is not a sha256 digest")
    image_release_id = _text(materialization, "image_release_id", label="runtime_image_release")
    if not _RELEASE_ID.fullmatch(image_release_id):
        raise ValueError("runtime_image_release.image_release_id is invalid")
    capabilities = materialization.get("capabilities")
    if not isinstance(capabilities, Sequence) or isinstance(capabilities, str):
        raise ValueError("runtime_image_release.capabilities must be a list")
    return ActorImage(
        release_id=release_id,
        image_release_id=image_release_id,
        status=_text(materialization, "status", label="runtime_image_release"),
        image_ref=_text(materialization, "image_ref", label="runtime_image_release"),
        digest=digest,
        platform=platform,
        actor_role=_text(materialization, "actor_role", label="runtime_image_release"),
        capabilities=tuple(str(item) for item in capabilities),
        archive_sha256=archive_sha256,
        python_packages=tuple(
            str(item) for item in (materialization.get("python_packages") or ())
        ),
        package_release_timestamps=(
            {
                str(key): str(value)
                for key, value in materialization.get("package_release_timestamps", {}).items()
            }
            if isinstance(materialization.get("package_release_timestamps"), Mapping)
            else None
        ),
        recipe_digest=(
            str(materialization.get("recipe_digest") or "").strip() or None
        ),
        image_substrates=tuple(
            str(item) for item in (materialization.get("image_substrates") or ())
        ),
        daytona_pullable=bool(materialization.get("daytona_pullable")),
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _plain_tar_octal(field: bytes, *, label: str) -> int:
    normalized = field.rstrip(b"\0 ").lstrip(b" ")
    if not normalized:
        return 0
    if any(byte not in b"01234567" for byte in normalized):
        raise ValueError(f"OCI archive {label} is not a plain-tar integer")
    return int(normalized, 8)


def _validate_plain_oci_tar_envelope(archive_path: Path) -> None:
    """Bound outer tar parsing before Python indexes any TarInfo objects."""

    archive_size = archive_path.stat().st_size
    names: set[str] = set()
    total_name_bytes = 0
    member_count = 0
    offset = 0
    zero_blocks = 0
    with archive_path.open("rb") as archive:
        while offset + 512 <= archive_size:
            archive.seek(offset)
            header = archive.read(512)
            if len(header) != 512:
                raise ValueError("OCI archive has a truncated tar header")
            if header == b"\0" * 512:
                zero_blocks += 1
                offset += 512
                if zero_blocks >= 2:
                    break
                continue
            if zero_blocks:
                raise ValueError("OCI archive contains data after its tar terminator")
            stored_checksum = _plain_tar_octal(header[148:156], label="checksum")
            calculated_checksum = sum(header[:148]) + (8 * ord(" ")) + sum(header[156:])
            if stored_checksum != calculated_checksum:
                raise ValueError("OCI archive tar header checksum is invalid")
            raw_name = header[:100].split(b"\0", 1)[0]
            raw_prefix = header[345:500].split(b"\0", 1)[0]
            name_bytes = raw_prefix + (b"/" if raw_prefix else b"") + raw_name
            try:
                name = name_bytes.decode("ascii")
            except UnicodeDecodeError as exc:
                raise ValueError("OCI archive member names must be ASCII") from exc
            member_size = _plain_tar_octal(header[124:136], label="member size")
            type_flag = header[156:157]
            if type_flag not in {b"\0", b"0", b"5"}:
                raise ValueError(
                    "OCI archive outer tar must contain only plain files and directories"
                )
            member_count += 1
            total_name_bytes += len(name_bytes)
            member_path = PurePosixPath(name)
            if (
                member_count > _OCI_ARCHIVE_MAX_MEMBERS
                or not name_bytes
                or len(name_bytes) > _OCI_ARCHIVE_MAX_MEMBER_NAME_BYTES
                or total_name_bytes > _OCI_ARCHIVE_MAX_MEMBER_NAMES_BYTES
                or member_path.is_absolute()
                or ".." in member_path.parts
                or name in names
                or (type_flag == b"5" and member_size != 0)
            ):
                raise ValueError("OCI archive has an unsafe or oversized member envelope")
            names.add(name)
            padded_size = ((member_size + 511) // 512) * 512
            next_offset = offset + 512 + padded_size
            if member_size > archive_size or next_offset > archive_size:
                raise ValueError("OCI archive member exceeds the tar envelope")
            offset = next_offset
    if zero_blocks < 2:
        raise ValueError("OCI archive is missing its plain-tar terminator")
    if offset < archive_size:
        trailing_bytes = archive_size - offset
        if trailing_bytes > _OCI_ARCHIVE_MAX_TRAILING_PADDING_BYTES:
            raise ValueError("OCI archive has excessive trailing tar padding")
        with archive_path.open("rb") as archive:
            archive.seek(offset)
            for block in iter(lambda: archive.read(1024 * 1024), b""):
                if any(block):
                    raise ValueError(
                        "OCI archive contains data after its tar terminator"
                    )


def _read_archive_member(archive: tarfile.TarFile, name: str) -> bytes:
    try:
        member = archive.getmember(name)
    except KeyError as exc:
        raise ValueError(f"OCI archive is missing {name}") from exc
    if not member.isfile() or not 0 <= member.size <= _OCI_ARCHIVE_METADATA_MAX_BYTES:
        raise ValueError(f"OCI archive member {name} is invalid or too large")
    handle = archive.extractfile(member)
    if handle is None:
        raise ValueError(f"OCI archive member {name} is unreadable")
    payload = handle.read(_OCI_ARCHIVE_METADATA_MAX_BYTES + 1)
    if len(payload) > _OCI_ARCHIVE_METADATA_MAX_BYTES:
        raise ValueError(f"OCI archive member {name} is too large")
    return payload


def inspect_oci_archive(archive_path: Path) -> dict[str, str]:
    """Read the manifest digest and platform out of a single-image OCI archive."""
    archive_size = archive_path.stat().st_size
    if not 1 <= archive_size <= IMAGE_RELEASE_SINGLE_PUT_MAX_BYTES:
        raise ValueError(
            "OCI archive must be between 1 byte and 5,000,000,000 bytes"
        )
    _validate_plain_oci_tar_envelope(archive_path)
    with tarfile.open(archive_path, mode="r:") as archive:
        index = json.loads(_read_archive_member(archive, "index.json"))
        manifests = index.get("manifests")
        if not isinstance(manifests, list) or len(manifests) != 1:
            raise ValueError("OCI archive must contain exactly one image manifest")
        descriptor = manifests[0]
        if (
            isinstance(descriptor, Mapping)
            and descriptor.get("mediaType") == "application/vnd.oci.image.index.v1+json"
        ):
            # containerd-backed `docker save` nests a single-image index one
            # level below the top index; chase exactly one hop.
            nested_digest = str(descriptor.get("digest") or "")
            if not _DIGEST.fullmatch(nested_digest):
                raise ValueError("OCI archive nested index digest is invalid")
            nested = json.loads(
                _read_archive_member(
                    archive,
                    f"blobs/sha256/{nested_digest.removeprefix('sha256:')}",
                )
            )
            nested_manifests = [
                item
                for item in (nested.get("manifests") or [])
                if isinstance(item, Mapping)
                and not (
                    isinstance(item.get("annotations"), Mapping)
                    and item["annotations"].get("vnd.docker.reference.type")
                    == "attestation-manifest"
                )
            ]
            if len(nested_manifests) != 1:
                raise ValueError("OCI archive nested index must contain exactly one image manifest")
            descriptor = nested_manifests[0]
        manifest_digest = str(descriptor.get("digest") or "")
        if not _DIGEST.fullmatch(manifest_digest):
            raise ValueError("OCI archive manifest digest is invalid")
        platform = descriptor.get("platform")
        if isinstance(platform, Mapping) and platform.get("os"):
            platform_os = str(platform.get("os") or "").lower()
            platform_architecture = str(platform.get("architecture") or "").lower()
        else:
            manifest = json.loads(
                _read_archive_member(
                    archive,
                    f"blobs/sha256/{manifest_digest.removeprefix('sha256:')}",
                )
            )
            config_digest = str((manifest.get("config") or {}).get("digest") or "")
            if not _DIGEST.fullmatch(config_digest):
                raise ValueError("OCI archive config digest is invalid")
            config = json.loads(
                _read_archive_member(
                    archive,
                    f"blobs/sha256/{config_digest.removeprefix('sha256:')}",
                )
            )
            platform_os = str(config.get("os") or "").lower()
            platform_architecture = str(config.get("architecture") or "").lower()
    return {
        "image_manifest_digest": manifest_digest,
        "platform_os": platform_os,
        "platform_architecture": platform_architecture,
    }


def _normalized_source(source: Mapping[str, Any]) -> tuple[str, str]:
    if not isinstance(source, Mapping):
        raise ValueError("source must be a mapping with repository and commit_sha")
    unexpected = set(source) - {"repository", "commit_sha"}
    if unexpected:
        raise ValueError(f"source has unexpected fields: {sorted(unexpected)}")
    repository = str(source.get("repository") or "").strip()
    commit_sha = str(source.get("commit_sha") or "").strip().lower()
    if not repository.startswith("https://github.com/"):
        raise ValueError("source.repository must be an HTTPS GitHub repository URL")
    if not _GIT_SHA.fullmatch(commit_sha):
        raise ValueError("source.commit_sha must be a full lowercase Git commit SHA")
    return repository, commit_sha


def _normalized_actor_declaration(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Validate the exact canonical actor declaration used for reconciliation."""

    if type(payload) is not dict or frozenset(payload) != _ACTOR_DECLARATION_FIELDS:
        raise ValueError("actor image declaration fields do not match the contract")
    archive_sha256 = payload.get("archive_sha256")
    archive_size_bytes = payload.get("archive_size_bytes")
    image_manifest_digest = payload.get("image_manifest_digest")
    image_ref = payload.get("image_ref")
    if not isinstance(archive_sha256, str) or not _SHA256.fullmatch(archive_sha256):
        raise ValueError("actor image archive_sha256 is invalid")
    if (
        type(archive_size_bytes) is not int
        or not 1 <= archive_size_bytes <= IMAGE_RELEASE_SINGLE_PUT_MAX_BYTES
    ):
        raise ValueError("actor image archive_size_bytes is outside the contract")
    if (
        not isinstance(image_manifest_digest, str)
        or not _DIGEST.fullmatch(image_manifest_digest)
    ):
        raise ValueError("actor image image_manifest_digest is invalid")
    if not isinstance(image_ref, str) or ":" not in image_ref:
        raise ValueError("actor image image_ref is invalid")
    image_name, image_tag = image_ref.rsplit(":", 1)
    if not _IMAGE_NAME.fullmatch(image_name) or not _IMAGE_TAG.fullmatch(image_tag):
        raise ValueError("actor image image_ref is invalid")
    platform_os = payload.get("platform_os")
    platform_architecture = payload.get("platform_architecture")
    if platform_os != "linux" or platform_architecture not in {"amd64", "arm64"}:
        raise ValueError("actor image platform is invalid")
    repository, commit_sha = _normalized_source(
        {
            "repository": payload.get("source_repository"),
            "commit_sha": payload.get("source_commit_sha"),
        }
    )
    capabilities = payload.get("capabilities")
    if not isinstance(capabilities, list):
        raise ValueError("actor image capabilities must be a canonical list")
    normalized_capabilities = sorted(
        str(item or "").strip().lower() for item in capabilities
    )
    if not normalized_capabilities or len(normalized_capabilities) > 32 or any(
        not _CAPABILITY.fullmatch(item) for item in normalized_capabilities
    ):
        raise ValueError("actor image capabilities are invalid")
    if len(set(normalized_capabilities)) != len(normalized_capabilities):
        raise ValueError("actor image capabilities must not repeat entries")
    python_packages = payload.get("python_packages")
    if not isinstance(python_packages, list):
        raise ValueError("actor image python_packages must be a canonical list")
    normalized_python_packages = _normalized_python_packages(python_packages)
    recipe_digest = payload.get("recipe_digest")
    if recipe_digest is not None and (
        not isinstance(recipe_digest, str) or not _DIGEST.fullmatch(recipe_digest)
    ):
        raise ValueError("actor image recipe_digest is invalid")
    normalized = {
        "kind": ACTOR_RUNTIME_IMAGE_KIND,
        "archive_sha256": archive_sha256,
        "archive_size_bytes": archive_size_bytes,
        "image_manifest_digest": image_manifest_digest,
        "image_ref": image_ref,
        "platform_os": platform_os,
        "platform_architecture": platform_architecture,
        "source_repository": repository,
        "source_commit_sha": commit_sha,
        "actor_role": "worker",
        "interface_mode": ACTOR_RUNTIME_INTERFACE_MODE,
        "capabilities": normalized_capabilities,
        "python_packages": normalized_python_packages,
        "recipe_digest": recipe_digest,
    }
    if normalized != dict(payload):
        raise ValueError("actor image declaration is not canonical")
    return normalized


def _local_image_id(image_ref: str) -> str:
    """Resolve the screened local base and make tag movement cache-visible."""
    try:
        result = subprocess.run(
            ["docker", "image", "inspect", "--format", "{{.Id}}", image_ref],
            capture_output=True,
            text=True,
            check=False,
            timeout=30,
        )
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(
            "screened Craftax base-image identity lookup exceeded 30 seconds"
        ) from exc
    image_id = result.stdout.strip().lower()
    if result.returncode or not _DIGEST.fullmatch(image_id):
        detail = (result.stderr or result.stdout or "image was not found").strip()
        raise RuntimeError(
            f"screened Craftax base image is unavailable: {image_ref} ({detail[-500:]})"
        )
    return image_id


def _pinned_base_image_ref(recipe: ActorImageRecipe) -> str:
    """Return the exact local base reference used by the constrained builder."""
    if recipe.base_image != CRAFTAX_WORKER_BASE_IMAGE:
        raise ValueError("only the screened Craftax base image may be pinned")
    return recipe.base_image.removesuffix(":latest") + "@" + recipe.base_image_id


def _terminate_build_process(process: subprocess.Popen[bytes]) -> None:
    """Stop the exact build process group and wait for its cleanup receipt."""

    process_group_id = process.pid
    try:
        os.killpg(process_group_id, signal.SIGTERM)
    except ProcessLookupError:
        pass
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process_group_id, signal.SIGKILL)
        except ProcessLookupError:
            pass
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError(
                "managed actor image build process did not exit after SIGKILL"
            ) from exc
    group_deadline = time.monotonic() + 5.0
    while _process_group_exists(process_group_id) and time.monotonic() < group_deadline:
        time.sleep(0.05)
    if _process_group_exists(process_group_id):
        try:
            os.killpg(process_group_id, signal.SIGKILL)
        except ProcessLookupError:
            pass
        group_deadline = time.monotonic() + 5.0
        while (
            _process_group_exists(process_group_id)
            and time.monotonic() < group_deadline
        ):
            time.sleep(0.05)
    if _process_group_exists(process_group_id):
        raise RuntimeError(
            "managed actor image build process group survived SIGKILL"
        )


def _process_group_exists(process_group_id: int) -> bool:
    try:
        os.killpg(process_group_id, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


@contextmanager
def _recipe_cache_lock(archive_path: Path):
    """Serialize one local recipe build/upload across concurrent benchmark lanes."""
    lock_path = archive_path.with_suffix(archive_path.suffix + ".lock")
    with lock_path.open("a+", encoding="utf-8") as lock:
        if fcntl is not None:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            if fcntl is not None:
                fcntl.flock(lock.fileno(), fcntl.LOCK_UN)


@contextmanager
def _recipe_build_capacity_lock(cache_root: Path):
    """Admit one managed-image build/upload peak across recipe digests."""

    if fcntl is None:
        raise RuntimeError("managed actor recipe builds require Unix file locking")
    lock_path = cache_root / ".build-capacity.lock"
    with lock_path.open("a+", encoding="utf-8") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)


def _cached_recipe_archive_is_valid(
    archive_path: Path,
    *,
    expected_platform: str,
) -> bool:
    """Validate a rebuildable cache entry before reuse, evicting corruption."""

    if not archive_path.is_file():
        return False
    try:
        inspection = inspect_oci_archive(archive_path)
    except (OSError, ValueError, json.JSONDecodeError, tarfile.TarError):
        archive_path.unlink(missing_ok=True)
        return False
    actual_platform = (
        f"{inspection['platform_os']}/{inspection['platform_architecture']}"
    )
    if actual_platform != expected_platform:
        archive_path.unlink(missing_ok=True)
        return False
    return True


def _prune_recipe_cache(
    cache_root: Path,
    *,
    protected_recipe_digests: set[str],
    policy: _ActorImageCachePolicy = _ActorImageCachePolicy(),
) -> None:
    """Bound rebuildable archives while never deleting an in-use recipe."""
    if not cache_root.is_dir() or fcntl is None:
        return
    prune_lock_path = cache_root / ".prune.lock"
    with prune_lock_path.open("a+", encoding="utf-8") as prune_lock:
        fcntl.flock(prune_lock.fileno(), fcntl.LOCK_EX)
        try:
            protected_names = {
                digest.removeprefix("sha256:") + ".oci.tar"
                for digest in protected_recipe_digests
            }

            def _stat(path: Path):
                try:
                    return path.stat()
                except FileNotFoundError:
                    return None

            archives_with_stat = [
                (archive, archive_stat)
                for archive in cache_root.glob("*.oci.tar")
                if (archive_stat := _stat(archive)) is not None
            ]
            archives_with_stat.sort(
                key=lambda item: item[1].st_mtime_ns,
                reverse=True,
            )
            retained_entries = 0
            retained_bytes = 0
            for archive, observed_stat in archives_with_stat:
                size_bytes = observed_stat.st_size
                protected = archive.name in protected_names
                within_policy = (
                    retained_entries < policy.max_entries
                    and retained_bytes + size_bytes <= policy.max_bytes
                )
                if protected or within_policy:
                    retained_entries += 1
                    retained_bytes += size_bytes
                    continue
                lock_path = archive.with_suffix(archive.suffix + ".lock")
                with lock_path.open("a+", encoding="utf-8") as lock:
                    try:
                        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                    except BlockingIOError:
                        continue
                    current_stat = _stat(archive)
                    if current_stat is not None and (
                        current_stat.st_mtime_ns == observed_stat.st_mtime_ns
                        and current_stat.st_size == observed_stat.st_size
                    ):
                        archive.unlink(missing_ok=True)
                    fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
        finally:
            fcntl.flock(prune_lock.fileno(), fcntl.LOCK_UN)


def _normalized_python_packages(values: Sequence[str]) -> list[str]:
    if len(values) > 32:
        raise ValueError("python_packages must list at most 32 exact-pinned packages")
    normalized: list[str] = []
    for raw_value in values:
        value = str(raw_value or "").strip()
        match = _PACKAGE_REQUIREMENT.fullmatch(value)
        if match is None:
            raise ValueError("python_packages entries must use the exact package==version form")
        package_name = re.sub(r"[-_.]+", "-", match.group("name").lower())
        if package_name not in CUSTOMER_ACTOR_IMAGE_PYPI_ALLOWLIST:
            raise ValueError(f"python package is not allowlisted: {package_name}")
        normalized.append(f"{package_name}=={match.group('version')}")
    if len(set(normalized)) != len(normalized):
        raise ValueError("python_packages must not repeat packages")
    return sorted(normalized)


class ImagesAPI(_ClientNamespace):
    """Customer actor images: upload, read, list, and archive org-owned releases."""

    def upload_archive(
        self,
        *,
        name: str,
        archive_path: str | Path,
        source: Mapping[str, Any],
        capabilities: Sequence[str],
        python_packages: Sequence[str],
        recipe_digest: str | None = None,
        kind: str = ACTOR_RUNTIME_IMAGE_KIND,
        role: str = "worker",
        platform: str | None = None,
        tag: str | None = None,
        expires_in: int = 3600,
        upload_timeout_seconds: float = 1800.0,
    ) -> ActorImage:
        """Upload a declared, allowlisted OCI runtime image and return its release.

        The OCI config must set ``io.synth.actor-runtime.python-packages`` to the
        canonical JSON list of the same exact-pinned requirements. The control
        plane independently rejects versions published less than seven days ago.
        """
        if kind != ACTOR_RUNTIME_IMAGE_KIND:
            raise ValueError(
                "images.upload_archive only uploads actor_runtime images; "
                "scorer releases use client.research.image_releases"
            )
        if (
            isinstance(upload_timeout_seconds, bool)
            or not isinstance(upload_timeout_seconds, (int, float))
            or not 1 <= float(upload_timeout_seconds) <= 7200
        ):
            raise ValueError("upload_timeout_seconds must be between 1 and 7200")
        upload_timeout_seconds = float(upload_timeout_seconds)
        if (
            isinstance(expires_in, bool)
            or not isinstance(expires_in, int)
            or not 60 <= expires_in <= 86400
        ):
            raise ValueError("expires_in must be between 60 and 86400 seconds")
        normalized_name = str(name or "").strip()
        if not _IMAGE_NAME.fullmatch(normalized_name):
            raise ValueError(
                "name must be a lowercase image repository name (for example 'craftax-worker')"
            )
        normalized_capabilities = sorted(
            str(item or "").strip().lower() for item in capabilities
        )
        if not normalized_capabilities or len(normalized_capabilities) > 32 or any(
            not _CAPABILITY.fullmatch(item) for item in normalized_capabilities
        ):
            raise ValueError(
                "capabilities must list between 1 and 32 lowercase snake_case slugs"
            )
        if len(set(normalized_capabilities)) != len(normalized_capabilities):
            raise ValueError("capabilities must not repeat entries")
        normalized_python_packages = _normalized_python_packages(python_packages)
        normalized_recipe_digest = str(recipe_digest or "").strip() or None
        if normalized_recipe_digest is not None and not _DIGEST.fullmatch(normalized_recipe_digest):
            raise ValueError("recipe_digest must be sha256:<64 lowercase hex>")
        if platform is not None and platform not in _PLATFORMS:
            raise ValueError(f"platform must be one of: {', '.join(_PLATFORMS)}")
        repository, commit_sha = _normalized_source(source)
        path = Path(archive_path).expanduser().resolve()
        if not path.is_file():
            raise ValueError(f"actor image archive is not a file: {path}")
        archive_size_bytes = path.stat().st_size
        if not 1 <= archive_size_bytes <= IMAGE_RELEASE_SINGLE_PUT_MAX_BYTES:
            raise ValueError(
                "actor image archive must be between 1 byte and "
                "5,000,000,000 bytes (single-PUT limit)"
            )
        archive_sha256 = _sha256_file(path)
        inspection = inspect_oci_archive(path)
        actual_platform = f"{inspection['platform_os']}/{inspection['platform_architecture']}"
        if platform is not None and platform != actual_platform:
            raise ValueError(
                f"platform mismatch: archive is {actual_platform}, declaration requested {platform}"
            )
        normalized_tag = str(tag or archive_sha256[:12]).strip()
        if not _IMAGE_TAG.fullmatch(normalized_tag):
            raise ValueError("tag must be a valid image tag")
        declaration = _normalized_actor_declaration(
            {
                "kind": ACTOR_RUNTIME_IMAGE_KIND,
                "archive_sha256": archive_sha256,
                "archive_size_bytes": archive_size_bytes,
                "image_manifest_digest": inspection["image_manifest_digest"],
                "image_ref": f"{normalized_name}:{normalized_tag}",
                "platform_os": inspection["platform_os"],
                "platform_architecture": inspection["platform_architecture"],
                "source_repository": repository,
                "source_commit_sha": commit_sha,
                "actor_role": str(role or "").strip().lower(),
                "interface_mode": ACTOR_RUNTIME_INTERFACE_MODE,
                "capabilities": normalized_capabilities,
                "python_packages": normalized_python_packages,
                "recipe_digest": normalized_recipe_digest,
            }
        )
        upload = self._client._request_json(
            "POST",
            "/smr/v1/image-releases/upload-url-v3",
            json_body={"declaration": declaration, "expires_in": expires_in},
        )
        if not isinstance(upload, Mapping):
            raise ValueError("image upload response must be an object")
        upload_lease = _ImageUploadLease.from_response(
            upload,
            archive_size_bytes=archive_size_bytes,
            declaration=declaration,
            expires_in=expires_in,
        )
        upload_id = upload_lease.upload_id
        if upload_lease.upload_required:
            self._put_archive(
                upload_lease.upload_url,
                path,
                upload_timeout_seconds=upload_timeout_seconds,
                upload_lease=upload_lease,
            )
        finalized = self._finalize(
            upload_id=upload_id,
            declaration=declaration,
            upload_lease=upload_lease,
            requested_timeout_seconds=upload_timeout_seconds,
        )
        release = _actor_release_from_finalize(
            finalized,
            expected_upload_id=upload_lease.upload_id,
            expected_release_id=upload_lease.release_id,
            declaration=declaration,
        )
        materialization = release.get("runtime_image_release")
        image = _actor_image_from_materialization(
            materialization,
            platform=actual_platform,
            archive_sha256=archive_sha256,
        )
        if image.image_release_id != upload_lease.release_id:
            raise ValueError("image materialization did not bind the uploaded release")
        return image

    def ensure_recipe(self, recipe: ActorImageRecipe) -> ActorImage:
        """Build once, reuse the local OCI archive, and reuse an admitted release.

        A release UUID is an implementation detail.  Recipe identity is the
        stable API: the SDK first finds a matching active release, otherwise it
        builds a deterministic OCI archive in its user cache and uploads it.
        """
        recipe.validate()
        recipe_digest = recipe.recipe_digest()
        cache_root = Path.home() / ".cache" / "synth-ai" / "actor-images"
        active_release = self._active_recipe_release(recipe_digest)
        if active_release is not None:
            _prune_recipe_cache(cache_root, protected_recipe_digests=set())
            return active_release
        cache_root.mkdir(parents=True, exist_ok=True)
        archive_path = cache_root / f"{recipe_digest.removeprefix('sha256:')}.oci.tar"
        with _recipe_build_capacity_lock(cache_root):
            # Prune before admission so stale cache bytes do not compound the
            # one globally serialized build/export peak.
            _prune_recipe_cache(cache_root, protected_recipe_digests=set())
            try:
                with _recipe_cache_lock(archive_path):
                    image = self._active_recipe_release(recipe_digest)
                    if image is None:
                        if not _cached_recipe_archive_is_valid(
                            archive_path,
                            expected_platform=recipe.platform,
                        ):
                            self._build_recipe_oci_archive(
                                recipe,
                                archive_path=archive_path,
                            )
                        image = self.upload_archive(
                            name=recipe.name,
                            archive_path=archive_path,
                            source={
                                "repository": recipe.source_repository,
                                "commit_sha": recipe.source_commit_sha,
                            },
                            capabilities=recipe.capabilities,
                            python_packages=recipe.python_packages,
                            platform=recipe.platform,
                            tag=(
                                "recipe-"
                                f"{recipe_digest.removeprefix('sha256:')[:12]}"
                            ),
                            recipe_digest=recipe_digest,
                        )
            finally:
                # Success and failure both converge rebuildable cache state;
                # a failed sequence of distinct recipes cannot bypass policy.
                _prune_recipe_cache(cache_root, protected_recipe_digests=set())
        return image

    def _active_recipe_release(self, recipe_digest: str) -> ActorImage | None:
        for item in self.list():
            if (
                str(item.get("recipe_digest") or "").strip() == recipe_digest
                and str(item.get("status") or "").strip().lower() != "archived"
            ):
                image_release_id = str(item.get("image_release_id") or "").strip()
                if image_release_id:
                    return self.get(image_release_id=image_release_id)
        return None

    @staticmethod
    def _build_recipe_oci_archive(recipe: ActorImageRecipe, *, archive_path: Path) -> None:
        """Build a constrained managed recipe as a single-image OCI archive."""
        if _local_image_id(recipe.base_image) != recipe.base_image_id:
            raise RuntimeError(
                "screened Craftax base image changed while creating the managed recipe; "
                "retry to derive a new cache key"
            )
        package_label = json.dumps(list(sorted(recipe.python_packages)))
        with (
            tempfile.TemporaryDirectory(prefix="synth-actor-image-") as temporary,
            tempfile.TemporaryDirectory(
                prefix=f".{archive_path.name}.",
                dir=archive_path.parent,
            ) as staging_directory,
        ):
            context = Path(temporary)
            assets = context / "assets"
            assets.mkdir()
            dockerfile_lines = [
                f"FROM {_pinned_base_image_ref(recipe)}",
                f"LABEL {CUSTOMER_ACTOR_IMAGE_PACKAGE_LABEL}={json.dumps(package_label)}",
            ]
            for index, (source, destination) in enumerate(recipe.assets):
                asset_name = f"asset-{index}"
                shutil.copy2(source, assets / asset_name)
                dockerfile_lines.append(
                    f"COPY --chmod=0755 assets/{asset_name} {destination}"
                )
            if recipe.python_packages:
                dockerfile_lines.append(
                    "RUN python -m pip install --no-cache-dir --no-deps "
                    + " ".join(recipe.python_packages)
                )
            (context / "Dockerfile").write_text(
                "\n".join(dockerfile_lines) + "\n", encoding="utf-8"
            )
            staging_archive = Path(staging_directory) / "image.oci.tar"
            process = subprocess.Popen(
                [
                    "docker",
                    "buildx",
                    "build",
                    "--platform",
                    recipe.platform,
                    "--output",
                    (
                        f"type=oci,dest={staging_archive},compression=gzip,"
                        "compression-level=6,force-compression=true"
                    ),
                    str(context),
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            build_log_tail = bytearray()
            if process.stdout is None:
                _terminate_build_process(process)
                raise RuntimeError("managed actor image build log pipe was unavailable")
            deadline = time.monotonic() + _ACTOR_IMAGE_BUILD_TIMEOUT_SECONDS
            exited_at: float | None = None
            selector: selectors.BaseSelector | None = None
            try:
                selector = selectors.DefaultSelector()
                selector.register(process.stdout, selectors.EVENT_READ)
                while True:
                    now = time.monotonic()
                    if now >= deadline:
                        raise subprocess.TimeoutExpired(
                            process.args,
                            _ACTOR_IMAGE_BUILD_TIMEOUT_SECONDS,
                        )
                    if process.poll() is not None and exited_at is None:
                        exited_at = now
                    if exited_at is not None and now - exited_at >= 5.0:
                        break
                    wait_seconds = min(1.0, deadline - now)
                    if exited_at is not None:
                        wait_seconds = min(wait_seconds, 5.0 - (now - exited_at))
                    events = selector.select(timeout=max(0.0, wait_seconds))
                    if not events:
                        continue
                    block = os.read(process.stdout.fileno(), 64 * 1024)
                    if not block:
                        break
                    build_log_tail.extend(block)
                    if len(build_log_tail) > 8_000:
                        del build_log_tail[:-8_000]
                return_code = process.wait(
                    timeout=max(0.1, deadline - time.monotonic())
                )
            except subprocess.TimeoutExpired as exc:
                _terminate_build_process(process)
                detail = bytes(build_log_tail).decode(
                    "utf-8", errors="replace"
                ).strip()
                suffix = f": {detail[-2000:]}" if detail else ""
                raise RuntimeError(
                    "managed actor image build exceeded its 1800-second deadline"
                    f"{suffix}"
                ) from exc
            except BaseException:
                # Cancellation and caller failures must not orphan the process
                # group or leave the recipe cache lock held by active work.
                _terminate_build_process(process)
                raise
            finally:
                if selector is not None:
                    selector.close()
                process.stdout.close()
            if _process_group_exists(process.pid):
                _terminate_build_process(process)
            if return_code != 0 or not staging_archive.is_file():
                detail = bytes(build_log_tail).decode(
                    "utf-8", errors="replace"
                ).strip()
                detail = detail or "Docker buildx produced no archive"
                raise RuntimeError(
                    f"managed actor image build failed: {detail[-2000:]}"
                )
            if not 1 <= staging_archive.stat().st_size <= IMAGE_RELEASE_SINGLE_PUT_MAX_BYTES:
                raise RuntimeError(
                    "managed actor image build exceeded the 5,000,000,000-byte "
                    "single-PUT contract"
                )
            inspection = inspect_oci_archive(staging_archive)
            actual_platform = (
                f"{inspection['platform_os']}/{inspection['platform_architecture']}"
            )
            if actual_platform != recipe.platform:
                raise RuntimeError(
                    "managed actor image build returned the wrong platform "
                    f"({actual_platform} != {recipe.platform})"
                )
            os.replace(staging_archive, archive_path)

    def _put_archive(
        self,
        upload_url: str,
        path: Path,
        *,
        upload_timeout_seconds: float,
        upload_lease: _ImageUploadLease,
    ) -> None:
        status_code: int | None = None
        response_detail = ""
        with httpx.Client(follow_redirects=False) as upload_client:
            for attempt_index in range(2):
                try:
                    attempt_timeout = upload_lease.timeout_seconds(
                        requested_seconds=upload_timeout_seconds,
                    )
                    with path.open("rb") as handle:
                        with upload_client.stream(
                            "PUT",
                            upload_url,
                            content=handle,
                            headers=upload_lease.upload_headers,
                            timeout=attempt_timeout,
                        ) as response:
                            status_code = response.status_code
                            response_detail = ""
                            if (
                                not response.is_success
                                and status_code not in {409, 412}
                            ):
                                response_detail = _bounded_upload_error_detail(response)
                    if status_code == 409 and attempt_index == 0:
                        continue
                    break
                except httpx.TransportError:
                    # A create-only retry can return 412 when the first request
                    # committed but its response was lost. Finalize verifies the
                    # exact digest and reconciles that ambiguous success.
                    continue
            else:
                # Lost responses leave a create-only PUT in doubt. Finalize is
                # the storage authority: it re-HEADs and verifies the object.
                status_code = None
        if status_code is None:
            return
        if not 200 <= status_code < 300 and status_code not in {409, 412}:
            # Object stores put the actionable SigV4 failure code in the
            # response body (for example, a signed-header mismatch). Keep a
            # bounded, whitespace-normalized diagnostic so launch receipts
            # identify a deterministic storage-contract failure without
            # dumping a potentially large provider error document.
            suffix = f": {response_detail}" if response_detail else ""
            raise RuntimeError(
                f"actor image archive upload failed with HTTP {status_code}{suffix}"
            )

    def _finalize(
        self,
        *,
        upload_id: str,
        declaration: Mapping[str, Any],
        upload_lease: _ImageUploadLease,
        requested_timeout_seconds: float,
    ) -> Mapping[str, Any]:
        for attempt_index in range(2):
            try:
                finalized = self._client._request_json(
                    "POST",
                    "/smr/v1/image-releases/finalize",
                    json_body={
                        "upload_id": upload_id,
                        "declaration": dict(declaration),
                    },
                    timeout_seconds=upload_lease.finalize_timeout_seconds(
                        requested_seconds=requested_timeout_seconds,
                    ),
                )
                break
            except SmrApiError as exc:
                if exc.status_code is not None:
                    raise
                if isinstance(exc.__cause__, httpx.ConnectError):
                    if attempt_index == 0:
                        # A failed TCP connect proves the request was not
                        # admitted. Recompute the remaining absolute deadline
                        # before the one safe retry.
                        continue
                    raise
                # A read timeout or disconnect can happen after the backend has
                # begun a long registry publication. Replaying here would run
                # that materialization twice and multiply storage/network use.
                raise ActorImageFinalizeUncertainError(
                    upload_id=upload_id,
                    declaration=declaration,
                ) from exc
        else:  # pragma: no cover - both loop exits are explicit.
            raise RuntimeError("actor image finalize retry state is invalid")
        if not isinstance(finalized, Mapping):
            raise ValueError("image finalize response must be an object")
        return finalized

    def reconcile_upload(
        self,
        *,
        upload_id: str,
        declaration: Mapping[str, Any],
        timeout_seconds: float = 1800.0,
    ) -> ActorImage:
        """Reconcile one in-doubt finalize without creating a second upload."""

        normalized_upload_id = str(upload_id or "").strip()
        if not _UPLOAD_ID.fullmatch(normalized_upload_id):
            raise ValueError("upload_id must look like imgup_<32 lowercase hex>")
        if (
            isinstance(timeout_seconds, bool)
            or not isinstance(timeout_seconds, (int, float))
            or not 1 <= float(timeout_seconds) <= 7200
        ):
            raise ValueError("timeout_seconds must be between 1 and 7200")
        normalized_declaration = _normalized_actor_declaration(declaration)
        finalized = self._client._request_json(
            "POST",
            "/smr/v1/image-releases/finalize",
            json_body={
                "upload_id": normalized_upload_id,
                "declaration": normalized_declaration,
            },
            timeout_seconds=float(timeout_seconds),
        )
        if not isinstance(finalized, Mapping):
            raise ValueError("image finalize response must be an object")
        release = _actor_release_from_finalize(
            finalized,
            expected_upload_id=normalized_upload_id,
            expected_release_id=None,
            declaration=normalized_declaration,
        )
        materialization = release["runtime_image_release"]
        image = _actor_image_from_materialization(
            materialization,
            platform=(
                f"{normalized_declaration['platform_os']}/"
                f"{normalized_declaration['platform_architecture']}"
            ),
            archive_sha256=normalized_declaration["archive_sha256"],
        )
        if image.image_release_id != release["release_id"]:
            raise ValueError("image materialization did not bind the reconciled release")
        return image

    def get(self, *, image_release_id: str) -> ActorImage:
        """Read one uploaded actor image by its immutable artifact identity."""
        normalized = str(image_release_id or "").strip()
        if not _RELEASE_ID.fullmatch(normalized):
            raise ValueError("image_release_id must look like imgrel_<sha256>")
        receipt = self._client._request_json(
            "GET",
            f"/smr/v1/image-releases/{normalized}",
        )
        if not isinstance(receipt, Mapping):
            raise ValueError("image release response must be an object")
        declaration = receipt.get("declaration")
        if not isinstance(declaration, Mapping):
            raise ValueError("image release response must include its declaration")
        if declaration.get("kind") != ACTOR_RUNTIME_IMAGE_KIND:
            raise ValueError(
                "release is not an actor_runtime image; read scorer releases "
                "through client.research.image_releases"
            )
        materialization = receipt.get("runtime_image_release")
        if not isinstance(materialization, Mapping):
            raise ValueError("image release exists but has no executable runtime image release")
        return _actor_image_from_materialization(
            materialization,
            platform=(
                f"{declaration.get('platform_os')}/{declaration.get('platform_architecture')}"
            ),
            archive_sha256=str(declaration.get("archive_sha256") or ""),
        )

    def status(self, *, image_release_id: str) -> str:
        """Return the executable release status for an uploaded actor image."""
        return self.get(image_release_id=image_release_id).status

    def list(self) -> List[dict[str, Any]]:
        """List this org's customer actor runtime image releases."""
        payload = self._client._request_json("GET", "/smr/v1/image-releases")
        if not isinstance(payload, Mapping) or not isinstance(payload.get("releases"), list):
            raise ValueError("image list response must include releases")
        return [dict(item) for item in payload["releases"] if isinstance(item, Mapping)]

    def archive(self, *, release_id: str) -> dict[str, Any]:
        """Archive an org-owned actor runtime image release (blocks new runs)."""
        normalized = str(release_id or "").strip()
        if not normalized:
            raise ValueError("release_id must be a nonempty string")
        payload = self._client._request_json(
            "POST",
            f"/smr/v1/image-releases/{normalized}/archive",
        )
        if not isinstance(payload, Mapping) or not isinstance(
            payload.get("runtime_image_release"), Mapping
        ):
            raise ValueError("image archive response must include the release")
        return dict(payload["runtime_image_release"])


__all__ = [
    "ACTOR_RUNTIME_IMAGE_KIND",
    "ACTOR_RUNTIME_INTERFACE_MODE",
    "ActorImage",
    "ActorImageFinalizeUncertainError",
    "ActorImageRecipe",
    "CRAFTAX_WORKER_BASE_IMAGE",
    "CRAFTAX_WORKER_CAPABILITIES",
    "CRAFTAX_WORKER_SCORER_PATH",
    "CUSTOMER_ACTOR_IMAGE_PACKAGE_LABEL",
    "CUSTOMER_ACTOR_IMAGE_PYPI_ALLOWLIST",
    "ImagesAPI",
    "inspect_oci_archive",
]
