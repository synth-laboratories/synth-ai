"""Lean customer image-release contracts.
# See: specifications/sdk/core_research_migration.md
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from types import MappingProxyType
from typing import TypeAlias, TypeVar, cast
from uuid import UUID

from synth_ai.core.contracts.json_value import JsonObject, JsonValue
from synth_ai.core.research.contracts._environment_wire import (
    digest,
    integer,
    optional_digest,
    text,
)
from synth_ai.core.research.contracts._wire import (
    array_value,
    object_value,
    required_bool,
    required_datetime,
)
from synth_ai.core.research.contracts.common import OrganizationId, require_text

_ARCHIVE = re.compile(r"^[0-9a-f]{64}$")
_GIT = re.compile(r"^[0-9a-f]{40}$")
_PACKAGE = re.compile(r"^[a-z0-9][a-z0-9._-]*==[A-Za-z0-9][A-Za-z0-9.!+_-]{0,127}$")
_RELEASE = re.compile(r"^imgrel_[0-9a-f]{64}$")
_UPLOAD = re.compile(r"^imgup_[0-9a-f]{32}$")
_ARTIFACT = re.compile(r"^imgobj_[0-9a-f]{64}$")
_MAX_ARCHIVE_BYTES = 16 * 1024**3
TimestampMap: TypeAlias = Mapping[str, datetime]
_IdT = TypeVar("_IdT", bound=str)


def _id(
    cls: type[_IdT],
    value: str,
    pattern: re.Pattern[str],
    field: str,
    label: str,
) -> _IdT:
    normalized = require_text(value, field_name=field)
    if pattern.fullmatch(normalized) is None:
        raise ValueError(f"{field} must match {label}")
    return cast(_IdT, str.__new__(cls, normalized))


class ImageReleaseId(str):
    """Backend-owned immutable image-release receipt identifier."""

    def __new__(cls, value: str) -> ImageReleaseId:
        return _id(cls, value, _RELEASE, "release_id", "imgrel_<64 lowercase hex>")


class ImageUploadId(str):
    """Backend-owned staging upload identifier."""

    def __new__(cls, value: str) -> ImageUploadId:
        return _id(cls, value, _UPLOAD, "upload_id", "imgup_<32 lowercase hex>")


class RuntimeImageReleaseId(str):
    """Backend-owned executable actor runtime image-release UUID."""

    def __new__(cls, value: str) -> RuntimeImageReleaseId:
        return str.__new__(
            cls, str(UUID(require_text(value, field_name="runtime_image_release_id")))
        )


class ImageReleaseKind(StrEnum):
    ACTOR_RUNTIME = "actor_runtime"
    CRAFTAX_SCORER = "craftax_scorer"


class RuntimeImageReleaseStatus(StrEnum):
    ACTIVE = "active"
    ARCHIVED = "archived"


def _obj(
    value: JsonValue, label: str, required: frozenset[str], optional: frozenset[str] = frozenset()
) -> JsonObject:
    payload = object_value(value, operation_id=label)
    actual = frozenset(payload)
    missing = required - actual
    extra = actual - required - optional
    if missing or extra:
        raise ValueError(
            f"{label} fields drifted: missing={sorted(missing)!r} extra={sorted(extra)!r}"
        )
    return payload


def _t(value: object, field: str, *, minimum: int = 1, maximum: int = 4000) -> str:
    normalized = text(value, field=field, maximum=maximum)
    if len(normalized) < minimum:
        raise ValueError(f"{field} must be at least {minimum} characters")
    return normalized


def _rx(value: object, field: str, pattern: re.Pattern[str], label: str) -> str:
    normalized = text(value, field=field, maximum=4000)
    if pattern.fullmatch(normalized) is None:
        raise ValueError(f"{field} must be {label}")
    return normalized


def _const(value: object, field: str, expected: str) -> str:
    normalized = text(value, field=field, maximum=4000)
    if normalized != expected:
        raise ValueError(f"{field} must be {expected!r}")
    return normalized


def _one_of(value: object, field: str, allowed: frozenset[str]) -> str:
    normalized = text(value, field=field, maximum=4000)
    if normalized not in allowed:
        raise ValueError(f"{field} must be one of {sorted(allowed)!r}")
    return normalized


def _strings(
    value: JsonValue,
    field: str,
    *,
    minimum: int = 0,
    maximum: int = 32,
    pattern: re.Pattern[str] | None = None,
) -> tuple[str, ...]:
    items = array_value(value, operation_id=field)
    if not minimum <= len(items) <= maximum:
        raise ValueError(f"{field} must contain {minimum} through {maximum} items")
    out: list[str] = []
    for index, item in enumerate(items):
        normalized = _t(item, f"{field}[{index}]")
        if pattern is not None and pattern.fullmatch(normalized) is None:
            raise ValueError(f"{field}[{index}] has an invalid format")
        out.append(normalized)
    return tuple(out)


def _datetime(value: object, field: str) -> datetime:
    parsed = (
        value
        if isinstance(value, datetime)
        else required_datetime({"value": _t(value, field, maximum=128)}, "value")
    )
    if parsed.tzinfo is None:
        raise ValueError(f"{field} must include a timezone")
    return parsed


def _timestamp_map(value: object | None) -> TimestampMap:
    if value is None:
        return MappingProxyType({})
    if isinstance(value, Mapping):
        payload = dict(value)
    else:
        payload = object_value(cast(JsonValue, value), operation_id="package_release_timestamps")
    return MappingProxyType(
        {
            _t(key, "package name"): _datetime(timestamp, f"package_release_timestamps[{key!r}]")
            for key, timestamp in payload.items()
        }
    )


def _timestamp_wire(timestamps: TimestampMap) -> JsonObject:
    return {key: timestamp.isoformat() for key, timestamp in timestamps.items()}


@dataclass(frozen=True, slots=True)
class _DeclarationBase:
    kind: ImageReleaseKind
    archive_sha256: str
    archive_size_bytes: int
    image_manifest_digest: str
    image_ref: str
    platform_os: str
    platform_architecture: str
    source_repository: str
    source_commit_sha: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "kind", ImageReleaseKind(self.kind))
        object.__setattr__(
            self,
            "archive_sha256",
            _rx(self.archive_sha256, "archive_sha256", _ARCHIVE, "64 lowercase hex characters"),
        )
        object.__setattr__(
            self,
            "archive_size_bytes",
            integer(
                self.archive_size_bytes,
                field="archive_size_bytes",
                minimum=1,
                maximum=_MAX_ARCHIVE_BYTES,
            ),
        )
        object.__setattr__(
            self,
            "image_manifest_digest",
            digest(self.image_manifest_digest, field="image_manifest_digest"),
        )
        object.__setattr__(
            self, "image_ref", _t(self.image_ref, "image_ref", minimum=3, maximum=255)
        )
        object.__setattr__(self, "platform_os", _const(self.platform_os, "platform_os", "linux"))
        object.__setattr__(
            self,
            "platform_architecture",
            _one_of(
                self.platform_architecture, "platform_architecture", frozenset({"amd64", "arm64"})
            ),
        )
        object.__setattr__(
            self,
            "source_repository",
            _t(self.source_repository, "source_repository", minimum=20, maximum=512),
        )
        object.__setattr__(
            self,
            "source_commit_sha",
            _rx(
                self.source_commit_sha,
                "source_commit_sha",
                _GIT,
                "a lowercase 40-character Git SHA",
            ),
        )

    def _base_wire(self) -> JsonObject:
        return {
            "kind": self.kind.value,
            "archive_sha256": self.archive_sha256,
            "archive_size_bytes": self.archive_size_bytes,
            "image_manifest_digest": self.image_manifest_digest,
            "image_ref": self.image_ref,
            "platform_os": self.platform_os,
            "platform_architecture": self.platform_architecture,
            "source_repository": self.source_repository,
            "source_commit_sha": self.source_commit_sha,
        }


@dataclass(frozen=True, slots=True)
class CraftaxScorerImageReleaseDeclaration(_DeclarationBase):
    fixture_manifest_sha256: str
    fixture_binary_sha256: str

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.kind is not ImageReleaseKind.CRAFTAX_SCORER:
            raise ValueError("craftax scorer declaration kind must be craftax_scorer")
        object.__setattr__(
            self,
            "fixture_manifest_sha256",
            _rx(
                self.fixture_manifest_sha256,
                "fixture_manifest_sha256",
                _ARCHIVE,
                "64 lowercase hex characters",
            ),
        )
        object.__setattr__(
            self,
            "fixture_binary_sha256",
            _rx(
                self.fixture_binary_sha256,
                "fixture_binary_sha256",
                _ARCHIVE,
                "64 lowercase hex characters",
            ),
        )

    def to_wire(self) -> JsonObject:
        payload = self._base_wire()
        payload.update(
            {
                "fixture_manifest_sha256": self.fixture_manifest_sha256,
                "fixture_binary_sha256": self.fixture_binary_sha256,
            }
        )
        return payload


@dataclass(frozen=True, slots=True)
class ActorRuntimeImageReleaseDeclaration(_DeclarationBase):
    actor_role: str
    interface_mode: str
    capabilities: tuple[str, ...]
    python_packages: tuple[str, ...]
    recipe_digest: str | None = None

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.kind is not ImageReleaseKind.ACTOR_RUNTIME:
            raise ValueError("actor runtime declaration kind must be actor_runtime")
        object.__setattr__(self, "actor_role", _const(self.actor_role, "actor_role", "worker"))
        object.__setattr__(
            self,
            "interface_mode",
            _const(self.interface_mode, "interface_mode", "synth_actor_runtime"),
        )
        object.__setattr__(
            self, "capabilities", _strings(list(self.capabilities), "capabilities", minimum=1)
        )
        object.__setattr__(
            self,
            "python_packages",
            _strings(list(self.python_packages), "python_packages", pattern=_PACKAGE),
        )
        object.__setattr__(
            self, "recipe_digest", optional_digest(self.recipe_digest, field="recipe_digest")
        )

    def to_wire(self) -> JsonObject:
        payload = self._base_wire()
        payload.update(
            {
                "actor_role": self.actor_role,
                "interface_mode": self.interface_mode,
                "capabilities": list(self.capabilities),
                "python_packages": list(self.python_packages),
                "recipe_digest": self.recipe_digest,
            }
        )
        return payload


ImageReleaseDeclaration: TypeAlias = (
    ActorRuntimeImageReleaseDeclaration | CraftaxScorerImageReleaseDeclaration
)
_DECL_BASE = frozenset(
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
    }
)


def declaration_from_wire(value: JsonValue) -> ImageReleaseDeclaration:
    raw_payload = object_value(value, operation_id="image release declaration")
    kind = ImageReleaseKind(_t(raw_payload.get("kind"), "kind", maximum=64))
    payload = (
        _obj(
            value,
            "craftax scorer declaration",
            _DECL_BASE | {"fixture_manifest_sha256", "fixture_binary_sha256"},
        )
        if kind is ImageReleaseKind.CRAFTAX_SCORER
        else _obj(
            value,
            "actor runtime declaration",
            _DECL_BASE | {"actor_role", "interface_mode", "capabilities", "python_packages"},
            frozenset({"recipe_digest"}),
        )
    )
    archive_sha256 = _rx(
        payload["archive_sha256"],
        "archive_sha256",
        _ARCHIVE,
        "64 lowercase hex characters",
    )
    archive_size_bytes = integer(
        payload["archive_size_bytes"],
        field="archive_size_bytes",
        minimum=1,
        maximum=_MAX_ARCHIVE_BYTES,
    )
    image_manifest_digest = digest(payload["image_manifest_digest"], field="image_manifest_digest")
    image_ref = _t(payload["image_ref"], "image_ref", minimum=3, maximum=255)
    platform_os = _const(payload["platform_os"], "platform_os", "linux")
    platform_architecture = _one_of(
        payload["platform_architecture"],
        "platform_architecture",
        frozenset({"amd64", "arm64"}),
    )
    source_repository = _t(
        payload["source_repository"], "source_repository", minimum=20, maximum=512
    )
    source_commit_sha = _rx(
        payload["source_commit_sha"],
        "source_commit_sha",
        _GIT,
        "a lowercase 40-character Git SHA",
    )
    if kind is ImageReleaseKind.CRAFTAX_SCORER:
        return CraftaxScorerImageReleaseDeclaration(
            kind=kind,
            archive_sha256=archive_sha256,
            archive_size_bytes=archive_size_bytes,
            image_manifest_digest=image_manifest_digest,
            image_ref=image_ref,
            platform_os=platform_os,
            platform_architecture=platform_architecture,
            source_repository=source_repository,
            source_commit_sha=source_commit_sha,
            fixture_manifest_sha256=_rx(
                payload["fixture_manifest_sha256"],
                "fixture_manifest_sha256",
                _ARCHIVE,
                "64 lowercase hex characters",
            ),
            fixture_binary_sha256=_rx(
                payload["fixture_binary_sha256"],
                "fixture_binary_sha256",
                _ARCHIVE,
                "64 lowercase hex characters",
            ),
        )
    return ActorRuntimeImageReleaseDeclaration(
        kind=kind,
        archive_sha256=archive_sha256,
        archive_size_bytes=archive_size_bytes,
        image_manifest_digest=image_manifest_digest,
        image_ref=image_ref,
        platform_os=platform_os,
        platform_architecture=platform_architecture,
        source_repository=source_repository,
        source_commit_sha=source_commit_sha,
        actor_role=_const(payload["actor_role"], "actor_role", "worker"),
        interface_mode=_const(payload["interface_mode"], "interface_mode", "synth_actor_runtime"),
        capabilities=_strings(payload["capabilities"], "capabilities", minimum=1),
        python_packages=_strings(payload["python_packages"], "python_packages", pattern=_PACKAGE),
        recipe_digest=optional_digest(payload.get("recipe_digest"), field="recipe_digest"),
    )


@dataclass(frozen=True, slots=True)
class ImageReleaseUploadRequest:
    declaration: ImageReleaseDeclaration
    expires_in: int = 3600

    def __post_init__(self) -> None:
        if not isinstance(
            self.declaration,
            (ActorRuntimeImageReleaseDeclaration, CraftaxScorerImageReleaseDeclaration),
        ):
            raise ValueError("declaration must be an image-release declaration")
        object.__setattr__(
            self,
            "expires_in",
            integer(self.expires_in, field="expires_in", minimum=60, maximum=86400),
        )

    @classmethod
    def from_wire(cls, value: JsonValue) -> ImageReleaseUploadRequest:
        payload = _obj(
            value,
            "image release upload request",
            frozenset({"declaration"}),
            frozenset({"expires_in"}),
        )
        return cls(
            declaration=declaration_from_wire(payload["declaration"]),
            expires_in=integer(
                payload.get("expires_in", 3600),
                field="expires_in",
                minimum=60,
                maximum=86400,
            ),
        )

    def to_wire(self) -> JsonObject:
        return {"declaration": self.declaration.to_wire(), "expires_in": self.expires_in}


@dataclass(frozen=True, slots=True)
class ImageReleaseFinalizeRequest:
    upload_id: ImageUploadId
    declaration: ImageReleaseDeclaration

    def __post_init__(self) -> None:
        object.__setattr__(self, "upload_id", ImageUploadId(self.upload_id))
        if not isinstance(
            self.declaration,
            (ActorRuntimeImageReleaseDeclaration, CraftaxScorerImageReleaseDeclaration),
        ):
            raise ValueError("declaration must be an image-release declaration")

    @classmethod
    def from_wire(cls, value: JsonValue) -> ImageReleaseFinalizeRequest:
        payload = _obj(
            value, "image release finalize request", frozenset({"upload_id", "declaration"})
        )
        return cls(
            upload_id=ImageUploadId(str(payload["upload_id"])),
            declaration=declaration_from_wire(payload["declaration"]),
        )

    def to_wire(self) -> JsonObject:
        return {"upload_id": self.upload_id, "declaration": self.declaration.to_wire()}


@dataclass(frozen=True, slots=True)
class ImageReleaseArtifact:
    artifact_id: str
    archive_sha256: str
    archive_size_bytes: int

    @classmethod
    def from_wire(cls, value: JsonValue) -> ImageReleaseArtifact:
        payload = _obj(
            value,
            "image release artifact",
            frozenset({"artifact_id", "archive_sha256", "archive_size_bytes"}),
        )
        artifact_id = _rx(
            payload["artifact_id"], "artifact_id", _ARTIFACT, "imgobj_<64 lowercase hex>"
        )
        return cls(
            artifact_id,
            _rx(
                payload["archive_sha256"], "archive_sha256", _ARCHIVE, "64 lowercase hex characters"
            ),
            integer(
                payload["archive_size_bytes"],
                field="archive_size_bytes",
                minimum=1,
                maximum=_MAX_ARCHIVE_BYTES,
            ),
        )

    def to_wire(self) -> JsonObject:
        return {
            "artifact_id": self.artifact_id,
            "archive_sha256": self.archive_sha256,
            "archive_size_bytes": self.archive_size_bytes,
        }


@dataclass(frozen=True, slots=True)
class ImageReleaseInspection:
    archive_format: str
    image_manifest_digest: str
    image_config_digest: str
    image_ref: str
    python_packages: tuple[str, ...]
    platform_os: str
    platform_architecture: str

    @classmethod
    def from_wire(cls, value: JsonValue) -> ImageReleaseInspection:
        payload = _obj(
            value,
            "image release inspection",
            frozenset(
                {
                    "archive_format",
                    "image_manifest_digest",
                    "image_config_digest",
                    "image_ref",
                    "python_packages",
                    "platform_os",
                    "platform_architecture",
                }
            ),
        )
        return cls(
            _const(payload["archive_format"], "archive_format", "oci-image-layout-tar-v1"),
            digest(payload["image_manifest_digest"], field="image_manifest_digest"),
            digest(payload["image_config_digest"], field="image_config_digest"),
            _t(payload["image_ref"], "image_ref", minimum=3, maximum=255),
            _strings(payload["python_packages"], "python_packages", pattern=_PACKAGE),
            _const(payload["platform_os"], "platform_os", "linux"),
            _one_of(
                payload["platform_architecture"],
                "platform_architecture",
                frozenset({"amd64", "arm64"}),
            ),
        )

    def to_wire(self) -> JsonObject:
        return {
            "archive_format": self.archive_format,
            "image_manifest_digest": self.image_manifest_digest,
            "image_config_digest": self.image_config_digest,
            "image_ref": self.image_ref,
            "python_packages": list(self.python_packages),
            "platform_os": self.platform_os,
            "platform_architecture": self.platform_architecture,
        }


@dataclass(frozen=True, slots=True)
class ActorRuntimeImageMaterialization:
    schema_version: str
    runtime_image_release_id: RuntimeImageReleaseId
    status: RuntimeImageReleaseStatus
    image_ref: str
    resolved_digest: str
    interface_mode: str
    actor_role: str
    selection_kind: str
    capabilities: tuple[str, ...]
    python_packages: tuple[str, ...]
    image_release_id: ImageReleaseId
    image_substrates: tuple[str, ...]
    daytona_pullable: bool
    package_release_timestamps: TimestampMap = field(default_factory=dict)
    recipe_digest: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "schema_version",
            _const(self.schema_version, "schema_version", "smr-actor-image-materialization-v1"),
        )
        object.__setattr__(
            self, "runtime_image_release_id", RuntimeImageReleaseId(self.runtime_image_release_id)
        )
        object.__setattr__(self, "status", RuntimeImageReleaseStatus(self.status))
        object.__setattr__(self, "image_ref", _t(self.image_ref, "image_ref", minimum=3))
        object.__setattr__(
            self, "resolved_digest", digest(self.resolved_digest, field="resolved_digest")
        )
        object.__setattr__(
            self,
            "interface_mode",
            _const(self.interface_mode, "interface_mode", "synth_actor_runtime"),
        )
        object.__setattr__(self, "actor_role", _const(self.actor_role, "actor_role", "worker"))
        object.__setattr__(
            self,
            "selection_kind",
            _const(self.selection_kind, "selection_kind", "customer_actor_runtime"),
        )
        object.__setattr__(
            self, "capabilities", _strings(list(self.capabilities), "capabilities", minimum=1)
        )
        object.__setattr__(
            self,
            "python_packages",
            _strings(list(self.python_packages), "python_packages", pattern=_PACKAGE),
        )
        object.__setattr__(
            self, "package_release_timestamps", _timestamp_map(self.package_release_timestamps)
        )
        object.__setattr__(
            self, "recipe_digest", optional_digest(self.recipe_digest, field="recipe_digest")
        )
        object.__setattr__(self, "image_release_id", ImageReleaseId(self.image_release_id))
        object.__setattr__(
            self,
            "image_substrates",
            _strings(list(self.image_substrates), "image_substrates", maximum=2),
        )
        object.__setattr__(
            self, "daytona_pullable", required_bool({"value": self.daytona_pullable}, "value")
        )
        expected = (
            ("org_registry", "wasabi_artifact") if self.daytona_pullable else ("wasabi_artifact",)
        )
        if self.image_substrates != expected:
            raise ValueError("image_substrates must exactly describe the admitted execution path")

    @classmethod
    def from_wire(cls, value: JsonValue) -> ActorRuntimeImageMaterialization:
        payload = _obj(
            value,
            "actor runtime image materialization",
            frozenset(
                {
                    "schema_version",
                    "runtime_image_release_id",
                    "status",
                    "image_ref",
                    "resolved_digest",
                    "interface_mode",
                    "actor_role",
                    "selection_kind",
                    "capabilities",
                    "python_packages",
                    "package_release_timestamps",
                    "image_release_id",
                    "image_substrates",
                    "daytona_pullable",
                }
            ),
            frozenset({"recipe_digest"}),
        )
        return cls(
            schema_version=_const(
                payload["schema_version"],
                "schema_version",
                "smr-actor-image-materialization-v1",
            ),
            runtime_image_release_id=RuntimeImageReleaseId(
                _t(payload["runtime_image_release_id"], "runtime_image_release_id", maximum=64)
            ),
            status=RuntimeImageReleaseStatus(_t(payload["status"], "status", maximum=64)),
            image_ref=_t(payload["image_ref"], "image_ref", minimum=3),
            resolved_digest=digest(payload["resolved_digest"], field="resolved_digest"),
            interface_mode=_const(
                payload["interface_mode"], "interface_mode", "synth_actor_runtime"
            ),
            actor_role=_const(payload["actor_role"], "actor_role", "worker"),
            selection_kind=_const(
                payload["selection_kind"], "selection_kind", "customer_actor_runtime"
            ),
            capabilities=_strings(payload["capabilities"], "capabilities", minimum=1),
            python_packages=_strings(
                payload["python_packages"], "python_packages", pattern=_PACKAGE
            ),
            package_release_timestamps=_timestamp_map(payload["package_release_timestamps"]),
            recipe_digest=optional_digest(payload.get("recipe_digest"), field="recipe_digest"),
            image_release_id=ImageReleaseId(
                _t(payload["image_release_id"], "image_release_id", maximum=80)
            ),
            image_substrates=_strings(payload["image_substrates"], "image_substrates", maximum=2),
            daytona_pullable=required_bool(payload, "daytona_pullable"),
        )

    def to_wire(self) -> JsonObject:
        return {
            "schema_version": self.schema_version,
            "runtime_image_release_id": self.runtime_image_release_id,
            "status": self.status.value,
            "image_ref": self.image_ref,
            "resolved_digest": self.resolved_digest,
            "interface_mode": self.interface_mode,
            "actor_role": self.actor_role,
            "selection_kind": self.selection_kind,
            "capabilities": list(self.capabilities),
            "python_packages": list(self.python_packages),
            "package_release_timestamps": _timestamp_wire(self.package_release_timestamps),
            "recipe_digest": self.recipe_digest,
            "image_release_id": self.image_release_id,
            "image_substrates": list(self.image_substrates),
            "daytona_pullable": self.daytona_pullable,
        }


def _check_receipt(
    declaration: ImageReleaseDeclaration,
    artifact: ImageReleaseArtifact,
    inspection: ImageReleaseInspection,
    timestamps: TimestampMap,
) -> None:
    if (
        artifact.archive_sha256 != declaration.archive_sha256
        or artifact.archive_size_bytes != declaration.archive_size_bytes
    ):
        raise ValueError("artifact must bind the declaration archive")
    if (
        inspection.image_manifest_digest != declaration.image_manifest_digest
        or inspection.image_ref != declaration.image_ref
    ):
        raise ValueError("inspection must bind the declaration image identity")
    if (
        inspection.platform_os != declaration.platform_os
        or inspection.platform_architecture != declaration.platform_architecture
    ):
        raise ValueError("inspection platform must bind the declaration")
    expected_packages = (
        declaration.python_packages
        if isinstance(declaration, ActorRuntimeImageReleaseDeclaration)
        else ()
    )
    if inspection.python_packages != expected_packages or set(timestamps) != set(expected_packages):
        raise ValueError("package evidence must bind the declared packages")


@dataclass(frozen=True, slots=True)
class _ImageReleaseBase:
    schema_version: str
    release_id: ImageReleaseId
    organization_id: OrganizationId
    artifact: ImageReleaseArtifact
    declaration: ImageReleaseDeclaration
    inspection: ImageReleaseInspection
    package_release_timestamps: TimestampMap

    def _base_wire(self) -> JsonObject:
        return {
            "schema_version": self.schema_version,
            "release_id": self.release_id,
            "org_id": self.organization_id,
            "artifact": self.artifact.to_wire(),
            "declaration": self.declaration.to_wire(),
            "inspection": self.inspection.to_wire(),
            "package_release_timestamps": _timestamp_wire(self.package_release_timestamps),
        }


def _release_base(
    payload: JsonObject,
) -> tuple[
    str,
    ImageReleaseId,
    OrganizationId,
    ImageReleaseArtifact,
    ImageReleaseDeclaration,
    ImageReleaseInspection,
    TimestampMap,
]:
    return (
        _const(payload["schema_version"], "schema_version", "smr-image-release-v1"),
        ImageReleaseId(str(payload["release_id"])),
        OrganizationId(str(UUID(_t(payload["org_id"], "org_id")))),
        ImageReleaseArtifact.from_wire(payload["artifact"]),
        declaration_from_wire(payload["declaration"]),
        ImageReleaseInspection.from_wire(payload["inspection"]),
        _timestamp_map(payload.get("package_release_timestamps")),
    )


@dataclass(frozen=True, slots=True)
class CraftaxScorerImageRelease(_ImageReleaseBase):
    @classmethod
    def from_wire(cls, value: JsonValue) -> CraftaxScorerImageRelease:
        payload = _obj(
            value,
            "craftax scorer image release",
            frozenset(
                {"schema_version", "release_id", "org_id", "artifact", "declaration", "inspection"}
            ),
            frozenset({"package_release_timestamps"}),
        )
        release = cls(*_release_base(payload))
        if not isinstance(release.declaration, CraftaxScorerImageReleaseDeclaration):
            raise ValueError("craftax scorer release declaration kind drifted")
        _check_receipt(
            release.declaration,
            release.artifact,
            release.inspection,
            release.package_release_timestamps,
        )
        return release

    def to_wire(self) -> JsonObject:
        return self._base_wire()


@dataclass(frozen=True, slots=True)
class ActorRuntimeImageRelease(_ImageReleaseBase):
    runtime_image_release: ActorRuntimeImageMaterialization

    @classmethod
    def from_wire(cls, value: JsonValue) -> ActorRuntimeImageRelease:
        payload = _obj(
            value,
            "actor runtime image release",
            frozenset(
                {
                    "schema_version",
                    "release_id",
                    "org_id",
                    "artifact",
                    "declaration",
                    "inspection",
                    "runtime_image_release",
                }
            ),
            frozenset({"package_release_timestamps"}),
        )
        release = cls(
            *_release_base(payload),
            ActorRuntimeImageMaterialization.from_wire(payload["runtime_image_release"]),
        )
        if not isinstance(release.declaration, ActorRuntimeImageReleaseDeclaration):
            raise ValueError("actor runtime release declaration kind drifted")
        _check_receipt(
            release.declaration,
            release.artifact,
            release.inspection,
            release.package_release_timestamps,
        )
        materialization = release.runtime_image_release
        if (
            materialization.image_release_id != release.release_id
            or materialization.resolved_digest != release.declaration.image_manifest_digest
        ):
            raise ValueError("runtime materialization must bind the image release")
        if (
            materialization.interface_mode != release.declaration.interface_mode
            or materialization.actor_role != release.declaration.actor_role
        ):
            raise ValueError("runtime materialization interface must bind the declaration")
        if (
            materialization.capabilities != release.declaration.capabilities
            or materialization.python_packages != release.declaration.python_packages
        ):
            raise ValueError("runtime materialization packages must bind the declaration")
        if materialization.recipe_digest != release.declaration.recipe_digest or dict(
            materialization.package_release_timestamps
        ) != dict(release.package_release_timestamps):
            raise ValueError("runtime materialization evidence must bind the release")
        return release

    def to_wire(self) -> JsonObject:
        payload = self._base_wire()
        payload["runtime_image_release"] = self.runtime_image_release.to_wire()
        return payload


ImageRelease: TypeAlias = CraftaxScorerImageRelease | ActorRuntimeImageRelease


def image_release_from_wire(value: JsonValue) -> ImageRelease:
    payload = object_value(value, operation_id="image release")
    declaration = object_value(payload.get("declaration"), operation_id="image release declaration")
    kind = ImageReleaseKind(_t(declaration.get("kind"), "kind", maximum=64))
    return (
        ActorRuntimeImageRelease.from_wire(value)
        if kind is ImageReleaseKind.ACTOR_RUNTIME
        else CraftaxScorerImageRelease.from_wire(value)
    )


@dataclass(frozen=True, slots=True)
class ImageReleaseUpload:
    schema_version: str
    upload_id: ImageUploadId
    release_id: ImageReleaseId
    upload_url: str
    upload_required: bool
    expires_in: int
    declaration: ImageReleaseDeclaration
    package_release_timestamps: TimestampMap = field(default_factory=dict)

    @classmethod
    def from_wire(cls, value: JsonValue) -> ImageReleaseUpload:
        payload = _obj(
            value,
            "image release upload",
            frozenset(
                {
                    "schema_version",
                    "upload_id",
                    "release_id",
                    "upload_url",
                    "upload_required",
                    "expires_in",
                    "declaration",
                    "package_release_timestamps",
                }
            ),
        )
        result = cls(
            _const(payload["schema_version"], "schema_version", "smr-image-release-upload-v1"),
            ImageUploadId(str(payload["upload_id"])),
            ImageReleaseId(str(payload["release_id"])),
            _t(payload["upload_url"], "upload_url"),
            required_bool(payload, "upload_required"),
            integer(payload["expires_in"], field="expires_in", minimum=60, maximum=86400),
            declaration_from_wire(payload["declaration"]),
            _timestamp_map(payload["package_release_timestamps"]),
        )
        _check_packages(result.declaration, result.package_release_timestamps)
        return result

    def to_wire(self) -> JsonObject:
        return {
            "schema_version": self.schema_version,
            "upload_id": self.upload_id,
            "release_id": self.release_id,
            "upload_url": self.upload_url,
            "upload_required": self.upload_required,
            "expires_in": self.expires_in,
            "declaration": self.declaration.to_wire(),
            "package_release_timestamps": _timestamp_wire(self.package_release_timestamps),
        }


def _check_packages(declaration: ImageReleaseDeclaration, timestamps: TimestampMap) -> None:
    expected = (
        declaration.python_packages
        if isinstance(declaration, ActorRuntimeImageReleaseDeclaration)
        else ()
    )
    if set(timestamps) != set(expected):
        raise ValueError("package release evidence must cover the declared packages")


@dataclass(frozen=True, slots=True)
class ImageReleaseStagingCleanup:
    upload_id: ImageUploadId
    status: str

    @classmethod
    def from_wire(cls, value: JsonValue) -> ImageReleaseStagingCleanup:
        payload = _obj(value, "image release staging cleanup", frozenset({"upload_id", "status"}))
        return cls(
            ImageUploadId(str(payload["upload_id"])),
            _one_of(payload["status"], "status", frozenset({"deleted", "absent"})),
        )

    def to_wire(self) -> JsonObject:
        return {"upload_id": self.upload_id, "status": self.status}


@dataclass(frozen=True, slots=True)
class ImageReleaseFinalize:
    schema_version: str
    release: ImageRelease
    staging_cleanup: ImageReleaseStagingCleanup

    @classmethod
    def from_wire(cls, value: JsonValue) -> ImageReleaseFinalize:
        payload = _obj(
            value,
            "image release finalize",
            frozenset({"schema_version", "release", "staging_cleanup"}),
        )
        return cls(
            _const(payload["schema_version"], "schema_version", "smr-image-release-finalize-v1"),
            image_release_from_wire(payload["release"]),
            ImageReleaseStagingCleanup.from_wire(payload["staging_cleanup"]),
        )

    def to_wire(self) -> JsonObject:
        return {
            "schema_version": self.schema_version,
            "release": self.release.to_wire(),
            "staging_cleanup": self.staging_cleanup.to_wire(),
        }


@dataclass(frozen=True, slots=True)
class ActorRuntimeImageReleaseList:
    schema_version: str
    releases: tuple[ActorRuntimeImageMaterialization, ...]

    @classmethod
    def from_wire(cls, value: JsonValue) -> ActorRuntimeImageReleaseList:
        payload = _obj(
            value, "actor runtime image release list", frozenset({"schema_version", "releases"})
        )
        return cls(
            _const(payload["schema_version"], "schema_version", "smr-actor-image-list-v1"),
            tuple(
                ActorRuntimeImageMaterialization.from_wire(item)
                for item in array_value(payload["releases"], operation_id="releases")
            ),
        )

    def to_wire(self) -> JsonObject:
        return {
            "schema_version": self.schema_version,
            "releases": [release.to_wire() for release in self.releases],
        }


@dataclass(frozen=True, slots=True)
class ActorRuntimeImageReleaseArchive:
    schema_version: str
    runtime_image_release: ActorRuntimeImageMaterialization

    @classmethod
    def from_wire(cls, value: JsonValue) -> ActorRuntimeImageReleaseArchive:
        payload = _obj(
            value,
            "actor runtime image release archive",
            frozenset({"schema_version", "runtime_image_release"}),
        )
        return cls(
            _const(payload["schema_version"], "schema_version", "smr-actor-image-archive-v1"),
            ActorRuntimeImageMaterialization.from_wire(payload["runtime_image_release"]),
        )

    def to_wire(self) -> JsonObject:
        return {
            "schema_version": self.schema_version,
            "runtime_image_release": self.runtime_image_release.to_wire(),
        }


ImageReleaseResponse = ImageRelease
ImageReleaseUploadResponse = ImageReleaseUpload
ImageReleaseFinalizeResponse = ImageReleaseFinalize
RuntimeImageReleaseListResponse = ActorRuntimeImageReleaseList
RuntimeImageReleaseArchiveResponse = ActorRuntimeImageReleaseArchive
__all__ = [
    "ActorRuntimeImageMaterialization",
    "ActorRuntimeImageRelease",
    "ActorRuntimeImageReleaseArchive",
    "ActorRuntimeImageReleaseDeclaration",
    "ActorRuntimeImageReleaseList",
    "CraftaxScorerImageRelease",
    "CraftaxScorerImageReleaseDeclaration",
    "ImageRelease",
    "ImageReleaseDeclaration",
    "ImageReleaseFinalize",
    "ImageReleaseFinalizeRequest",
    "ImageReleaseFinalizeResponse",
    "ImageReleaseId",
    "ImageReleaseKind",
    "ImageReleaseStagingCleanup",
    "ImageReleaseUpload",
    "ImageReleaseUploadRequest",
    "ImageReleaseUploadResponse",
    "ImageUploadId",
    "RuntimeImageReleaseArchiveResponse",
    "RuntimeImageReleaseId",
    "RuntimeImageReleaseListResponse",
    "RuntimeImageReleaseStatus",
    "declaration_from_wire",
    "image_release_from_wire",
]
