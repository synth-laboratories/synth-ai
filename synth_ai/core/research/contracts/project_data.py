"""Strict project repository and dataset contracts.

# See: specifications/sdk/core_research_migration.md
"""

from __future__ import annotations

import base64
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from math import isfinite
from types import MappingProxyType
from typing import TypeAlias, cast

from synth_ai.core.contracts.json_value import JsonObject, JsonValue
from synth_ai.core.research.contracts._wire import (
    object_value,
    optional_text,
    required_bool,
    required_datetime,
    required_text,
)
from synth_ai.core.research.contracts.common import (
    OrganizationId,
    ProjectDatasetId,
    ProjectId,
    ProjectRepositoryId,
    require_text,
)

ResourceJsonScalar: TypeAlias = str | int | float | bool | None
ResourceJsonValue: TypeAlias = (
    ResourceJsonScalar | Sequence["ResourceJsonValue"] | Mapping[str, "ResourceJsonValue"]
)
FrozenResourceJsonValue: TypeAlias = (
    ResourceJsonScalar
    | tuple["FrozenResourceJsonValue", ...]
    | Mapping[str, "FrozenResourceJsonValue"]
)


def _freeze_json(value: object, *, field_name: str) -> FrozenResourceJsonValue:
    if value is None or isinstance(value, (str, bool)):
        return value
    if type(value) is int:
        return value
    if type(value) is float:
        if not isfinite(value):
            raise ValueError(f"{field_name} must contain finite JSON numbers")
        return value
    if isinstance(value, Mapping):
        frozen: dict[str, FrozenResourceJsonValue] = {}
        for key, child in value.items():
            if not isinstance(key, str) or not key:
                raise ValueError(f"{field_name} object keys must be non-empty strings")
            frozen[key] = _freeze_json(child, field_name=f"{field_name}.{key}")
        return MappingProxyType(frozen)
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return tuple(
            _freeze_json(child, field_name=f"{field_name}[{index}]")
            for index, child in enumerate(value)
        )
    raise ValueError(f"{field_name} must contain only JSON values")


def _thaw_json(value: FrozenResourceJsonValue) -> JsonValue:
    if isinstance(value, Mapping):
        return cast(
            JsonValue,
            {
                str(key): _thaw_json(cast(FrozenResourceJsonValue, child))
                for key, child in value.items()
            },
        )
    if isinstance(value, tuple):
        return cast(JsonValue, [_thaw_json(child) for child in value])
    return value


def _exact_object(
    value: JsonValue,
    *,
    label: str,
    fields: frozenset[str],
) -> JsonObject:
    payload = object_value(value, operation_id=label)
    actual_fields = frozenset(payload)
    if actual_fields != fields:
        raise ValueError(
            f"{label} fields drifted: missing={sorted(fields - actual_fields)!r} "
            f"extra={sorted(actual_fields - fields)!r}"
        )
    return payload


def _optional_non_negative_int(payload: JsonObject, name: str) -> int | None:
    value = payload[name]
    if value is None:
        return None
    if type(value) is not int or value < 0:
        raise ValueError(f"{name} must be a non-negative integer when provided")
    return value


def _optional_normalized_text(value: str | None, *, field_name: str) -> str | None:
    if value is None:
        return None
    return require_text(value, field_name=field_name)


class ProjectRepositoryRole(StrEnum):
    PRIMARY = "primary"
    DEPENDENCY = "dependency"


class ProjectRepositoryScope(StrEnum):
    PROJECT = "project"
    RUN = "run"


class ProjectDatasetEncoding(StrEnum):
    BASE64 = "base64"
    TEXT = "text"
    UTF_8 = "utf-8"


class ProjectDatasetSourceKind(StrEnum):
    UPLOAD = "upload"


@dataclass(frozen=True, slots=True)
class ResourceMetadata:
    """Recursively immutable metadata limited to finite JSON values."""

    values: Mapping[str, ResourceJsonValue] = field(default_factory=dict)

    def __post_init__(self) -> None:
        frozen = _freeze_json(self.values, field_name="resource metadata")
        if not isinstance(frozen, Mapping):
            raise ValueError("resource metadata must be an object")
        object.__setattr__(self, "values", frozen)

    @classmethod
    def from_wire(cls, value: JsonValue) -> ResourceMetadata:
        return cls(values=object_value(value, operation_id="resource metadata"))

    def to_wire(self) -> JsonObject:
        frozen = _freeze_json(self.values, field_name="resource metadata")
        thawed = _thaw_json(frozen)
        if not isinstance(thawed, dict):
            raise ValueError("resource metadata must be an object")
        return thawed


@dataclass(frozen=True, slots=True)
class ProjectRepositorySpec:
    name: str
    url: str
    default_branch: str | None = None
    role: ProjectRepositoryRole = ProjectRepositoryRole.DEPENDENCY
    metadata: ResourceMetadata = field(default_factory=ResourceMetadata)

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", require_text(self.name, field_name="name"))
        object.__setattr__(self, "url", require_text(self.url, field_name="url"))
        object.__setattr__(
            self,
            "default_branch",
            _optional_normalized_text(
                self.default_branch,
                field_name="default_branch",
            ),
        )
        if not isinstance(self.role, ProjectRepositoryRole):
            raise ValueError("role must be a ProjectRepositoryRole")
        if not isinstance(self.metadata, ResourceMetadata):
            raise ValueError("metadata must be ResourceMetadata")

    def to_wire(self) -> JsonObject:
        payload: JsonObject = {
            "name": self.name,
            "url": self.url,
            "role": self.role.value,
            "metadata": self.metadata.to_wire(),
        }
        if self.default_branch is not None:
            payload["default_branch"] = self.default_branch
        return payload


@dataclass(frozen=True, slots=True)
class ProjectRepositoryPatch:
    url: str | None = None
    default_branch: str | None = None
    role: ProjectRepositoryRole | None = None
    metadata: ResourceMetadata | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "url",
            _optional_normalized_text(self.url, field_name="url"),
        )
        object.__setattr__(
            self,
            "default_branch",
            _optional_normalized_text(
                self.default_branch,
                field_name="default_branch",
            ),
        )
        if self.role is not None and not isinstance(self.role, ProjectRepositoryRole):
            raise ValueError("role must be a ProjectRepositoryRole when provided")
        if self.metadata is not None and not isinstance(self.metadata, ResourceMetadata):
            raise ValueError("metadata must be ResourceMetadata when provided")
        if all(
            value is None for value in (self.url, self.default_branch, self.role, self.metadata)
        ):
            raise ValueError("project repository patch requires at least one field")

    def to_wire(self) -> JsonObject:
        payload: JsonObject = {}
        if self.url is not None:
            payload["url"] = self.url
        if self.default_branch is not None:
            payload["default_branch"] = self.default_branch
        if self.role is not None:
            payload["role"] = self.role.value
        if self.metadata is not None:
            payload["metadata"] = self.metadata.to_wire()
        return payload


@dataclass(frozen=True, slots=True)
class ProjectRepository:
    repository_id: ProjectRepositoryId
    organization_id: OrganizationId
    project_id: ProjectId
    scope: ProjectRepositoryScope
    name: str
    url: str
    default_branch: str | None
    role: ProjectRepositoryRole
    metadata: ResourceMetadata
    created_at: datetime
    updated_at: datetime

    @classmethod
    def from_wire(cls, value: JsonValue) -> ProjectRepository:
        payload = _exact_object(
            value,
            label="project repository",
            fields=frozenset(
                {
                    "repository_id",
                    "org_id",
                    "project_id",
                    "run_id",
                    "scope_kind",
                    "name",
                    "url",
                    "default_branch",
                    "role",
                    "metadata",
                    "created_at",
                    "updated_at",
                }
            ),
        )
        scope = ProjectRepositoryScope(required_text(payload, "scope_kind"))
        if scope is not ProjectRepositoryScope.PROJECT:
            raise ValueError("project repository response must be project-scoped")
        if payload["run_id"] is not None:
            raise ValueError("project repository response must not carry a run_id")
        return cls(
            repository_id=ProjectRepositoryId(required_text(payload, "repository_id")),
            organization_id=OrganizationId(required_text(payload, "org_id")),
            project_id=ProjectId(required_text(payload, "project_id")),
            scope=scope,
            name=required_text(payload, "name"),
            url=required_text(payload, "url"),
            default_branch=optional_text(payload, "default_branch"),
            role=ProjectRepositoryRole(required_text(payload, "role")),
            metadata=ResourceMetadata.from_wire(payload["metadata"]),
            created_at=required_datetime(payload, "created_at"),
            updated_at=required_datetime(payload, "updated_at"),
        )

    def to_wire(self) -> JsonObject:
        return {
            "repository_id": self.repository_id,
            "org_id": self.organization_id,
            "project_id": self.project_id,
            "run_id": None,
            "scope_kind": self.scope.value,
            "name": self.name,
            "url": self.url,
            "default_branch": self.default_branch,
            "role": self.role.value,
            "metadata": self.metadata.to_wire(),
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }


@dataclass(frozen=True, slots=True)
class ProjectRepositoryDeletion:
    repository_id: ProjectRepositoryId
    deleted: bool

    def __post_init__(self) -> None:
        if self.deleted is not True:
            raise ValueError("project repository deletion must confirm deleted=true")

    @classmethod
    def from_wire(cls, value: JsonValue) -> ProjectRepositoryDeletion:
        payload = _exact_object(
            value,
            label="project repository deletion",
            fields=frozenset({"deleted", "repository_id"}),
        )
        deleted = required_bool(payload, "deleted")
        if not deleted:
            raise ValueError("project repository deletion must confirm deleted=true")
        return cls(
            repository_id=ProjectRepositoryId(required_text(payload, "repository_id")),
            deleted=True,
        )

    def to_wire(self) -> JsonObject:
        return {"deleted": self.deleted, "repository_id": self.repository_id}


@dataclass(frozen=True, slots=True)
class ProjectDatasetUpload:
    name: str
    content: str
    encoding: ProjectDatasetEncoding = ProjectDatasetEncoding.TEXT
    content_type: str | None = None
    format: str | None = None
    row_count: int | None = None
    metadata: ResourceMetadata = field(default_factory=ResourceMetadata)

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", require_text(self.name, field_name="name"))
        if not isinstance(self.content, str):
            raise ValueError("content must be a string")
        if not isinstance(self.encoding, ProjectDatasetEncoding):
            raise ValueError("encoding must be a ProjectDatasetEncoding")
        object.__setattr__(
            self,
            "content_type",
            _optional_normalized_text(self.content_type, field_name="content_type"),
        )
        object.__setattr__(
            self,
            "format",
            _optional_normalized_text(self.format, field_name="format"),
        )
        if self.row_count is not None and (type(self.row_count) is not int or self.row_count < 0):
            raise ValueError("row_count must be a non-negative integer when provided")
        if not isinstance(self.metadata, ResourceMetadata):
            raise ValueError("metadata must be ResourceMetadata")
        if self.encoding is ProjectDatasetEncoding.BASE64:
            try:
                base64.b64decode(self.content.encode("ascii"), validate=True)
            except (UnicodeEncodeError, ValueError) as error:
                raise ValueError("base64 dataset content is invalid") from error

    @classmethod
    def from_bytes(
        cls,
        *,
        name: str,
        content: bytes,
        content_type: str | None = None,
        format: str | None = None,
        row_count: int | None = None,
        metadata: ResourceMetadata | None = None,
    ) -> ProjectDatasetUpload:
        return cls(
            name=name,
            content=base64.b64encode(content).decode("ascii"),
            encoding=ProjectDatasetEncoding.BASE64,
            content_type=content_type,
            format=format,
            row_count=row_count,
            metadata=metadata if metadata is not None else ResourceMetadata(),
        )

    def to_wire(self) -> JsonObject:
        payload: JsonObject = {
            "name": self.name,
            "content": self.content,
            "encoding": self.encoding.value,
            "metadata": self.metadata.to_wire(),
        }
        if self.content_type is not None:
            payload["content_type"] = self.content_type
        if self.format is not None:
            payload["format"] = self.format
        if self.row_count is not None:
            payload["row_count"] = self.row_count
        return payload


@dataclass(frozen=True, slots=True)
class ProjectDataset:
    dataset_id: ProjectDatasetId
    project_id: ProjectId
    name: str
    source_kind: ProjectDatasetSourceKind
    format: str | None
    row_count: int | None
    size_bytes: int | None
    created_at: datetime
    download_url: str

    @classmethod
    def from_wire(cls, value: JsonValue) -> ProjectDataset:
        payload = _exact_object(
            value,
            label="project dataset",
            fields=frozenset(
                {
                    "id",
                    "project_id",
                    "run_id",
                    "name",
                    "source_kind",
                    "format",
                    "row_count",
                    "size_bytes",
                    "mount_path",
                    "created_at",
                    "download_url",
                }
            ),
        )
        if payload["run_id"] is not None:
            raise ValueError("project dataset response must not carry a run_id")
        if payload["mount_path"] is not None:
            raise ValueError("project dataset response must not carry a run mount path")
        dataset_id = ProjectDatasetId(required_text(payload, "id"))
        project_id = ProjectId(required_text(payload, "project_id"))
        download_url = required_text(payload, "download_url")
        expected_download_url = f"/smr/projects/{project_id}/datasets/{dataset_id}/download"
        if download_url != expected_download_url:
            raise ValueError("project dataset download_url drifted from its resource identity")
        return cls(
            dataset_id=dataset_id,
            project_id=project_id,
            name=required_text(payload, "name"),
            source_kind=ProjectDatasetSourceKind(required_text(payload, "source_kind")),
            format=optional_text(payload, "format"),
            row_count=_optional_non_negative_int(payload, "row_count"),
            size_bytes=_optional_non_negative_int(payload, "size_bytes"),
            created_at=required_datetime(payload, "created_at"),
            download_url=download_url,
        )

    def to_wire(self) -> JsonObject:
        return {
            "id": self.dataset_id,
            "project_id": self.project_id,
            "run_id": None,
            "name": self.name,
            "source_kind": self.source_kind.value,
            "format": self.format,
            "row_count": self.row_count,
            "size_bytes": self.size_bytes,
            "mount_path": None,
            "created_at": self.created_at.isoformat(),
            "download_url": self.download_url,
        }


__all__ = [
    "ProjectDataset",
    "ProjectDatasetEncoding",
    "ProjectDatasetSourceKind",
    "ProjectDatasetUpload",
    "ProjectRepository",
    "ProjectRepositoryDeletion",
    "ProjectRepositoryPatch",
    "ProjectRepositoryRole",
    "ProjectRepositoryScope",
    "ProjectRepositorySpec",
    "ResourceMetadata",
]
