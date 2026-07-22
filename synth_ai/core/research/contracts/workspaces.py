"""Strict project workspace-input contracts.

# See: specifications/sdk/core_research_migration.md
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from math import isfinite
from pathlib import PurePosixPath
from types import MappingProxyType
from typing import Optional, TypeAlias, cast
from urllib.parse import urlsplit

from synth_ai.core.contracts.json_value import JsonObject, JsonValue
from synth_ai.core.research.contracts._wire import (
    array_value,
    object_value,
    optional_text,
    required_bool,
    required_datetime,
    required_text,
)
from synth_ai.core.research.contracts.common import (
    ConfigurationVersionId,
    OrganizationId,
    ProjectEventId,
    ProjectId,
    SwarmId,
    WorkspaceFileId,
    require_text,
)

WorkspaceJsonScalar: TypeAlias = str | int | float | bool | None
WorkspaceJsonValue: TypeAlias = (
    WorkspaceJsonScalar | Sequence["WorkspaceJsonValue"] | Mapping[str, "WorkspaceJsonValue"]
)
FrozenWorkspaceJsonValue: TypeAlias = (
    WorkspaceJsonScalar
    | tuple["FrozenWorkspaceJsonValue", ...]
    | Mapping[str, "FrozenWorkspaceJsonValue"]
)

WORKSPACE_UPLOAD_FILE_LIMIT = 100
WORKSPACE_BATCH_UPLOAD_FILE_LIMIT = 10_000


def _freeze_json(value: object, *, field_name: str) -> FrozenWorkspaceJsonValue:
    if value is None or isinstance(value, (str, bool)):
        return value
    if type(value) is int:
        return value
    if type(value) is float:
        if not isfinite(value):
            raise ValueError(f"{field_name} must contain finite JSON numbers")
        return value
    if isinstance(value, Mapping):
        frozen: dict[str, FrozenWorkspaceJsonValue] = {}
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


def _thaw_json(value: FrozenWorkspaceJsonValue) -> JsonValue:
    if isinstance(value, Mapping):
        return cast(
            JsonValue,
            {
                str(key): _thaw_json(cast(FrozenWorkspaceJsonValue, child))
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
    required_fields: frozenset[str],
    optional_fields: frozenset[str] = frozenset(),
) -> JsonObject:
    payload = object_value(value, operation_id=label)
    fields = frozenset(payload)
    missing = required_fields - fields
    extra = fields - required_fields - optional_fields
    if missing or extra:
        raise ValueError(
            f"{label} fields drifted: missing={sorted(missing)!r} extra={sorted(extra)!r}"
        )
    return payload


def _non_negative_int(payload: JsonObject, name: str) -> int:
    value = payload[name]
    if type(value) is not int or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value


def _optional_non_negative_int(payload: JsonObject, name: str) -> Optional[int]:
    value = payload[name]
    if value is None:
        return None
    if type(value) is not int or value < 0:
        raise ValueError(f"{name} must be a non-negative integer when provided")
    return value


def _optional_datetime(payload: JsonObject, name: str) -> Optional[datetime]:
    if payload[name] is None:
        return None
    return required_datetime(payload, name)


def _string_tuple(value: JsonValue, *, label: str) -> tuple[str, ...]:
    items = array_value(value, operation_id=label)
    result: list[str] = []
    for index, item in enumerate(items):
        if not isinstance(item, str) or not item.strip():
            raise ValueError(f"{label}[{index}] must be a non-empty string")
        result.append(item.strip())
    return tuple(result)


def _git_sha(value: str, *, field_name: str) -> str:
    normalized = require_text(value, field_name=field_name)
    if len(normalized) != 40 or any(
        character not in "0123456789abcdef" for character in normalized
    ):
        raise ValueError(f"{field_name} must be a lowercase 40-character git SHA")
    return normalized


def _sha256(value: str, *, field_name: str) -> str:
    normalized = require_text(value, field_name=field_name)
    if len(normalized) != 64 or any(
        character not in "0123456789abcdef" for character in normalized
    ):
        raise ValueError(f"{field_name} must be a lowercase 64-character SHA-256 digest")
    return normalized


def _source_repository_url(value: str) -> str:
    normalized = require_text(value, field_name="source repository url")
    parsed = urlsplit(normalized)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc or not parsed.path:
        raise ValueError("source repository url must be a public http(s) repository URL")
    if parsed.username is not None or parsed.password is not None:
        raise ValueError("source repository url must not embed credentials")
    return normalized


def _workspace_path(value: str) -> str:
    normalized = require_text(value, field_name="workspace file path")
    path = PurePosixPath(normalized)
    if path.is_absolute() or ".." in path.parts or path.as_posix() in {".", ".."}:
        raise ValueError("workspace file path must stay beneath the workspace root")
    if path.as_posix() != normalized:
        raise ValueError("workspace file path must be a normalized POSIX relative path")
    return normalized


class WorkspaceInputState(StrEnum):
    EMPTY = "empty"
    CONFIGURED = "configured"


class WorkspaceSourceRepositoryKind(StrEnum):
    EXTERNAL_GIT = "external_git"


class WorkspaceRepositoryAuthMode(StrEnum):
    NONE = "none"


class WorkspaceRepositoryBootstrapMode(StrEnum):
    WORKSPACE_CLONE = "workspace_clone"


class WorkspaceFileScope(StrEnum):
    PROJECT = "project"
    RUN = "run"


class WorkspaceFileVisibility(StrEnum):
    MODEL = "model"
    VERIFIER = "verifier"


class WorkspaceFileEncoding(StrEnum):
    BASE64 = "base64"
    UTF_8 = "utf-8"


class WorkspaceFileKind(StrEnum):
    FILE = "file"
    SOURCE_BUNDLE = "source_bundle"


@dataclass(frozen=True, slots=True)
class WorkspaceMetadata:
    """Immutable JSON metadata owned by the backend resource contract."""

    values: Mapping[str, WorkspaceJsonValue] = field(default_factory=dict)

    def __post_init__(self) -> None:
        frozen = _freeze_json(self.values, field_name="workspace metadata")
        if not isinstance(frozen, Mapping):
            raise ValueError("workspace metadata must be an object")
        object.__setattr__(self, "values", frozen)

    @classmethod
    def from_wire(cls, value: JsonValue) -> WorkspaceMetadata:
        payload = object_value(value, operation_id="workspace metadata")
        return cls(values=payload)

    def to_wire(self) -> JsonObject:
        frozen = _freeze_json(self.values, field_name="workspace metadata")
        thawed = _thaw_json(frozen)
        if not isinstance(thawed, dict):
            raise ValueError("workspace metadata must be an object")
        return thawed


@dataclass(frozen=True, slots=True)
class WorkspaceSourceRepositorySpec:
    url: str
    default_branch: Optional[str] = None
    commit_sha: Optional[str] = None

    def __post_init__(self) -> None:
        _source_repository_url(self.url)
        if self.default_branch is not None:
            require_text(self.default_branch, field_name="default_branch")
        if self.commit_sha is not None:
            _git_sha(self.commit_sha, field_name="commit_sha")

    def to_wire(self) -> JsonObject:
        payload: JsonObject = {"url": self.url}
        if self.default_branch is not None:
            payload["default_branch"] = self.default_branch
        if self.commit_sha is not None:
            payload["commit_sha"] = self.commit_sha
        return payload


@dataclass(frozen=True, slots=True)
class WorkspaceSourceRepository:
    kind: WorkspaceSourceRepositoryKind
    url: str
    display_url: str
    is_public: bool
    auth_mode: WorkspaceRepositoryAuthMode
    bootstrap_mode: WorkspaceRepositoryBootstrapMode
    remote_name: str
    default_branch: Optional[str]
    commit_sha: Optional[str]

    @classmethod
    def from_wire(cls, value: JsonValue) -> WorkspaceSourceRepository:
        required_fields = frozenset(
            {
                "kind",
                "url",
                "display_url",
                "public",
                "auth_mode",
                "bootstrap_mode",
                "remote_name",
                "default_branch",
            }
        )
        payload = _exact_object(
            value,
            label="workspace source repository",
            required_fields=required_fields,
            optional_fields=frozenset({"commit_sha"}),
        )
        commit_sha = optional_text(payload, "commit_sha") if "commit_sha" in payload else None
        if commit_sha is not None:
            _git_sha(commit_sha, field_name="source_repository.commit_sha")
        return cls(
            kind=WorkspaceSourceRepositoryKind(required_text(payload, "kind")),
            url=_source_repository_url(required_text(payload, "url")),
            display_url=_source_repository_url(required_text(payload, "display_url")),
            is_public=required_bool(payload, "public"),
            auth_mode=WorkspaceRepositoryAuthMode(required_text(payload, "auth_mode")),
            bootstrap_mode=WorkspaceRepositoryBootstrapMode(
                required_text(payload, "bootstrap_mode")
            ),
            remote_name=required_text(payload, "remote_name"),
            default_branch=optional_text(payload, "default_branch"),
            commit_sha=commit_sha,
        )

    def to_wire(self) -> JsonObject:
        payload: JsonObject = {
            "kind": self.kind.value,
            "url": self.url,
            "display_url": self.display_url,
            "public": self.is_public,
            "auth_mode": self.auth_mode.value,
            "bootstrap_mode": self.bootstrap_mode.value,
            "remote_name": self.remote_name,
            "default_branch": self.default_branch,
        }
        if self.commit_sha is not None:
            payload["commit_sha"] = self.commit_sha
        return payload


@dataclass(frozen=True, slots=True)
class WorkspaceProjectRepository:
    repository_id: str
    organization_slug: str
    project_slug: str
    storage_backend: str
    storage_bucket: Optional[str]
    storage_prefix: str
    vcs_provider: Optional[str]
    remote_repository: Optional[str]
    default_branch: Optional[str]
    current_archive_key: Optional[str]
    current_manifest_key: Optional[str]
    current_commit_sha: Optional[str]
    metadata: WorkspaceMetadata
    updated_at: datetime

    @classmethod
    def from_wire(cls, value: JsonValue) -> Optional[WorkspaceProjectRepository]:
        payload = object_value(value, operation_id="workspace project repository")
        if not payload:
            return None
        payload = _exact_object(
            payload,
            label="workspace project repository",
            required_fields=frozenset(
                {
                    "repo_id",
                    "repo_org_slug",
                    "repo_project_slug",
                    "storage_backend",
                    "s3_bucket",
                    "s3_prefix",
                    "vcs_provider",
                    "remote_repo",
                    "default_branch",
                    "current_archive_key",
                    "current_manifest_key",
                    "current_commit_sha",
                    "metadata",
                    "updated_at",
                }
            ),
        )
        commit_sha = optional_text(payload, "current_commit_sha")
        if commit_sha is not None:
            _git_sha(commit_sha, field_name="project_repository.current_commit_sha")
        return cls(
            repository_id=required_text(payload, "repo_id"),
            organization_slug=required_text(payload, "repo_org_slug"),
            project_slug=required_text(payload, "repo_project_slug"),
            storage_backend=required_text(payload, "storage_backend"),
            storage_bucket=optional_text(payload, "s3_bucket"),
            storage_prefix=required_text(payload, "s3_prefix"),
            vcs_provider=optional_text(payload, "vcs_provider"),
            remote_repository=optional_text(payload, "remote_repo"),
            default_branch=optional_text(payload, "default_branch"),
            current_archive_key=optional_text(payload, "current_archive_key"),
            current_manifest_key=optional_text(payload, "current_manifest_key"),
            current_commit_sha=commit_sha,
            metadata=WorkspaceMetadata.from_wire(payload["metadata"]),
            updated_at=required_datetime(payload, "updated_at"),
        )

    def to_wire(self) -> JsonObject:
        return {
            "repo_id": self.repository_id,
            "repo_org_slug": self.organization_slug,
            "repo_project_slug": self.project_slug,
            "storage_backend": self.storage_backend,
            "s3_bucket": self.storage_bucket,
            "s3_prefix": self.storage_prefix,
            "vcs_provider": self.vcs_provider,
            "remote_repo": self.remote_repository,
            "default_branch": self.default_branch,
            "current_archive_key": self.current_archive_key,
            "current_manifest_key": self.current_manifest_key,
            "current_commit_sha": self.current_commit_sha,
            "metadata": self.metadata.to_wire(),
            "updated_at": self.updated_at.isoformat(),
        }


@dataclass(frozen=True, slots=True)
class WorkspaceFileUpload:
    path: str
    content: str
    content_type: Optional[str] = None
    encoding: Optional[WorkspaceFileEncoding] = None
    kind: Optional[WorkspaceFileKind] = None
    metadata: WorkspaceMetadata = field(default_factory=WorkspaceMetadata)

    def __post_init__(self) -> None:
        _workspace_path(self.path)
        if self.content_type is not None:
            require_text(self.content_type, field_name="content_type")

    def to_wire(self) -> JsonObject:
        payload: JsonObject = {
            "path": self.path,
            "content": self.content,
            "metadata": self.metadata.to_wire(),
        }
        if self.content_type is not None:
            payload["content_type"] = self.content_type
        if self.encoding is not None:
            payload["encoding"] = self.encoding.value
        if self.kind is not None:
            payload["kind"] = self.kind.value
        return payload


@dataclass(frozen=True, slots=True)
class WorkspaceFilesUploadRequest:
    files: tuple[WorkspaceFileUpload, ...]

    def __post_init__(self) -> None:
        if not self.files:
            raise ValueError("workspace file upload requires at least one file")
        if len(self.files) > WORKSPACE_UPLOAD_FILE_LIMIT:
            raise ValueError(
                "workspace file upload exceeds the "
                f"{WORKSPACE_UPLOAD_FILE_LIMIT}-file request limit"
            )
        paths = tuple(item.path for item in self.files)
        if len(set(paths)) != len(paths):
            raise ValueError("workspace file upload paths must be unique")

    def to_wire(self) -> JsonObject:
        return {"files": [item.to_wire() for item in self.files]}


@dataclass(frozen=True, slots=True)
class WorkspaceFilesBatchUploadRequest:
    """Client-side composite request partitioned into bounded server mutations."""

    files: tuple[WorkspaceFileUpload, ...]

    def __post_init__(self) -> None:
        if not self.files:
            raise ValueError("workspace batch upload requires at least one file")
        if len(self.files) > WORKSPACE_BATCH_UPLOAD_FILE_LIMIT:
            raise ValueError(
                "workspace batch upload exceeds the "
                f"{WORKSPACE_BATCH_UPLOAD_FILE_LIMIT}-file composite limit"
            )
        paths = tuple(item.path for item in self.files)
        if len(set(paths)) != len(paths):
            raise ValueError("workspace batch upload paths must be globally unique")

    @property
    def batch_count(self) -> int:
        return (len(self.files) + WORKSPACE_UPLOAD_FILE_LIMIT - 1) // (WORKSPACE_UPLOAD_FILE_LIMIT)

    def partitions(self) -> tuple[tuple[WorkspaceFileUpload, ...], ...]:
        return tuple(
            self.files[index : index + WORKSPACE_UPLOAD_FILE_LIMIT]
            for index in range(0, len(self.files), WORKSPACE_UPLOAD_FILE_LIMIT)
        )


@dataclass(frozen=True, slots=True)
class WorkspaceStoredFile:
    file_id: WorkspaceFileId
    organization_id: OrganizationId
    project_id: ProjectId
    run_id: Optional[SwarmId]
    scope: WorkspaceFileScope
    path: str
    logical_name: Optional[str]
    content_type: Optional[str]
    encoding: Optional[WorkspaceFileEncoding]
    visibility: WorkspaceFileVisibility
    storage_backend: str
    content_sha256: Optional[str]
    content_bytes: Optional[int]
    metadata: WorkspaceMetadata
    created_at: datetime
    updated_at: datetime

    @classmethod
    def from_wire(cls, value: JsonValue) -> WorkspaceStoredFile:
        payload = _exact_object(
            value,
            label="workspace stored file",
            required_fields=frozenset(
                {
                    "file_id",
                    "org_id",
                    "project_id",
                    "run_id",
                    "scope_kind",
                    "path",
                    "logical_name",
                    "content_type",
                    "encoding",
                    "visibility",
                    "storage_backend",
                    "content_sha256",
                    "content_bytes",
                    "metadata",
                    "created_at",
                    "updated_at",
                }
            ),
        )
        encoding = optional_text(payload, "encoding")
        digest = optional_text(payload, "content_sha256")
        run_id = optional_text(payload, "run_id")
        if digest is not None:
            _sha256(digest, field_name="stored_file.content_sha256")
        return cls(
            file_id=WorkspaceFileId(required_text(payload, "file_id")),
            organization_id=OrganizationId(required_text(payload, "org_id")),
            project_id=ProjectId(required_text(payload, "project_id")),
            run_id=SwarmId(run_id) if run_id is not None else None,
            scope=WorkspaceFileScope(required_text(payload, "scope_kind")),
            path=_workspace_path(required_text(payload, "path")),
            logical_name=optional_text(payload, "logical_name"),
            content_type=optional_text(payload, "content_type"),
            encoding=(WorkspaceFileEncoding(encoding) if encoding is not None else None),
            visibility=WorkspaceFileVisibility(required_text(payload, "visibility")),
            storage_backend=required_text(payload, "storage_backend"),
            content_sha256=digest,
            content_bytes=_optional_non_negative_int(payload, "content_bytes"),
            metadata=WorkspaceMetadata.from_wire(payload["metadata"]),
            created_at=required_datetime(payload, "created_at"),
            updated_at=required_datetime(payload, "updated_at"),
        )

    def to_wire(self) -> JsonObject:
        return {
            "file_id": self.file_id,
            "org_id": self.organization_id,
            "project_id": self.project_id,
            "run_id": self.run_id,
            "scope_kind": self.scope.value,
            "path": self.path,
            "logical_name": self.logical_name,
            "content_type": self.content_type,
            "encoding": self.encoding.value if self.encoding is not None else None,
            "visibility": self.visibility.value,
            "storage_backend": self.storage_backend,
            "content_sha256": self.content_sha256,
            "content_bytes": self.content_bytes,
            "metadata": self.metadata.to_wire(),
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }


@dataclass(frozen=True, slots=True)
class ProjectWorkspaceInputs:
    project_id: ProjectId
    state: WorkspaceInputState
    source_repository: Optional[WorkspaceSourceRepository]
    files: tuple[WorkspaceStoredFile, ...]
    project_repository: Optional[WorkspaceProjectRepository]
    updated_at: Optional[datetime]

    @classmethod
    def from_wire(cls, value: JsonValue) -> ProjectWorkspaceInputs:
        payload = _exact_object(
            value,
            label="project workspace inputs",
            required_fields=frozenset(
                {
                    "project_id",
                    "state",
                    "source_repo",
                    "files",
                    "file_count",
                    "project_repo",
                    "updated_at",
                }
            ),
        )
        source_value = payload["source_repo"]
        source_repository = (
            WorkspaceSourceRepository.from_wire(source_value) if source_value is not None else None
        )
        files = tuple(
            WorkspaceStoredFile.from_wire(item)
            for item in array_value(payload["files"], operation_id="project workspace input files")
        )
        file_count = _non_negative_int(payload, "file_count")
        if file_count != len(files):
            raise ValueError("project workspace input file_count does not match files")
        project_repository = WorkspaceProjectRepository.from_wire(payload["project_repo"])
        state = WorkspaceInputState(required_text(payload, "state"))
        project_id = ProjectId(required_text(payload, "project_id"))
        configured = source_repository is not None or project_repository is not None or bool(files)
        if state is WorkspaceInputState.EMPTY and configured:
            raise ValueError("empty workspace inputs must not contain configured resources")
        if state is WorkspaceInputState.CONFIGURED and not configured:
            raise ValueError("configured workspace inputs must contain at least one resource")
        for stored_file in files:
            if stored_file.project_id != project_id:
                raise ValueError("project workspace input file project_id drifted")
            if stored_file.scope is not WorkspaceFileScope.PROJECT:
                raise ValueError("project workspace inputs may contain only project-scoped files")
            if stored_file.run_id is not None:
                raise ValueError("project workspace input files must not carry a run_id")
        return cls(
            project_id=project_id,
            state=state,
            source_repository=source_repository,
            files=files,
            project_repository=project_repository,
            updated_at=_optional_datetime(payload, "updated_at"),
        )

    def to_wire(self) -> JsonObject:
        return {
            "project_id": self.project_id,
            "state": self.state.value,
            "source_repo": (
                self.source_repository.to_wire() if self.source_repository is not None else None
            ),
            "files": [item.to_wire() for item in self.files],
            "file_count": len(self.files),
            "project_repo": (
                self.project_repository.to_wire() if self.project_repository is not None else {}
            ),
            "updated_at": self.updated_at.isoformat() if self.updated_at is not None else None,
        }


@dataclass(frozen=True, slots=True)
class WorkspaceSourceRepositoryReceipt:
    url: str
    default_branch: Optional[str]
    commit_sha: Optional[str]
    configuration_version_id: ConfigurationVersionId

    @classmethod
    def from_wire(cls, value: JsonValue) -> WorkspaceSourceRepositoryReceipt:
        payload = _exact_object(
            value,
            label="workspace source repository receipt",
            required_fields=frozenset(
                {"ok", "url", "default_branch", "commit_sha", "config_version_id"}
            ),
        )
        if not required_bool(payload, "ok"):
            raise ValueError("workspace source repository mutation did not succeed")
        commit_sha = optional_text(payload, "commit_sha")
        if commit_sha is not None:
            _git_sha(commit_sha, field_name="source_repository_receipt.commit_sha")
        return cls(
            url=_source_repository_url(required_text(payload, "url")),
            default_branch=optional_text(payload, "default_branch"),
            commit_sha=commit_sha,
            configuration_version_id=ConfigurationVersionId(
                required_text(payload, "config_version_id")
            ),
        )

    def to_wire(self) -> JsonObject:
        return {
            "ok": True,
            "url": self.url,
            "default_branch": self.default_branch,
            "commit_sha": self.commit_sha,
            "config_version_id": self.configuration_version_id,
        }


@dataclass(frozen=True, slots=True)
class WorkspaceFilesUploadReceipt:
    project_id: ProjectId
    branch: str
    commit_sha: str
    committed: bool
    committed_paths: tuple[str, ...]
    file_count: int
    bytes_uploaded: int
    uploaded_files: tuple[WorkspaceStoredFile, ...]
    project_event_id: ProjectEventId
    event_summary: str
    message_for_agents: str
    fanout_swarm_ids: tuple[SwarmId, ...]

    @classmethod
    def from_wire(cls, value: JsonValue) -> WorkspaceFilesUploadReceipt:
        payload = _exact_object(
            value,
            label="workspace files upload receipt",
            required_fields=frozenset(
                {
                    "project_id",
                    "branch",
                    "commit_sha",
                    "files",
                    "committed",
                    "file_count",
                    "bytes_uploaded",
                    "uploaded_files",
                    "project_event_id",
                    "event_summary",
                    "message_for_agents",
                    "fanout_runs",
                }
            ),
        )
        uploaded_files = tuple(
            WorkspaceStoredFile.from_wire(item)
            for item in array_value(
                payload["uploaded_files"], operation_id="workspace uploaded files"
            )
        )
        file_count = _non_negative_int(payload, "file_count")
        if file_count > WORKSPACE_UPLOAD_FILE_LIMIT:
            raise ValueError("workspace upload receipt exceeds the bounded server file limit")
        if file_count != len(uploaded_files):
            raise ValueError("workspace upload file_count does not match uploaded_files")
        project_id = ProjectId(required_text(payload, "project_id"))
        committed_paths = tuple(
            _workspace_path(item)
            for item in _string_tuple(payload["files"], label="workspace upload committed paths")
        )
        if len(committed_paths) != file_count:
            raise ValueError("workspace upload committed paths do not match file_count")
        for stored_file in uploaded_files:
            if stored_file.project_id != project_id:
                raise ValueError("workspace upload file project_id drifted")
            if stored_file.scope is not WorkspaceFileScope.PROJECT:
                raise ValueError("workspace upload receipt may contain only project-scoped files")
            if stored_file.run_id is not None:
                raise ValueError("workspace upload receipt files must not carry a run_id")
        return cls(
            project_id=project_id,
            branch=required_text(payload, "branch"),
            commit_sha=_git_sha(
                required_text(payload, "commit_sha"), field_name="upload_receipt.commit_sha"
            ),
            committed=required_bool(payload, "committed"),
            committed_paths=committed_paths,
            file_count=file_count,
            bytes_uploaded=_non_negative_int(payload, "bytes_uploaded"),
            uploaded_files=uploaded_files,
            project_event_id=ProjectEventId(required_text(payload, "project_event_id")),
            event_summary=required_text(payload, "event_summary"),
            message_for_agents=required_text(payload, "message_for_agents"),
            fanout_swarm_ids=tuple(
                SwarmId(item)
                for item in _string_tuple(
                    payload["fanout_runs"], label="workspace upload fanout runs"
                )
            ),
        )

    def to_wire(self) -> JsonObject:
        return {
            "project_id": self.project_id,
            "branch": self.branch,
            "commit_sha": self.commit_sha,
            "files": list(self.committed_paths),
            "committed": self.committed,
            "file_count": self.file_count,
            "bytes_uploaded": self.bytes_uploaded,
            "uploaded_files": [item.to_wire() for item in self.uploaded_files],
            "project_event_id": self.project_event_id,
            "event_summary": self.event_summary,
            "message_for_agents": self.message_for_agents,
            "fanout_runs": list(self.fanout_swarm_ids),
        }


@dataclass(frozen=True, slots=True)
class WorkspaceFilesBatchUploadReceipt:
    """Complete ordered receipt for a deterministic composite upload."""

    project_id: ProjectId
    requested_file_count: int
    batches: tuple[WorkspaceFilesUploadReceipt, ...]

    def __post_init__(self) -> None:
        if self.requested_file_count < 1:
            raise ValueError("batch upload receipt requires a positive requested_file_count")
        if self.requested_file_count > WORKSPACE_BATCH_UPLOAD_FILE_LIMIT:
            raise ValueError("batch upload receipt exceeds the composite file limit")
        if not self.batches:
            raise ValueError("batch upload receipt requires at least one batch")
        expected_batch_count = (
            self.requested_file_count + WORKSPACE_UPLOAD_FILE_LIMIT - 1
        ) // WORKSPACE_UPLOAD_FILE_LIMIT
        if len(self.batches) != expected_batch_count:
            raise ValueError("batch upload receipt does not cover every bounded batch")
        if any(receipt.project_id != self.project_id for receipt in self.batches):
            raise ValueError("batch upload receipt crossed its requested project boundary")
        if self.file_count != self.requested_file_count:
            raise ValueError("batch upload receipt file count does not match the composite request")
        if len(set(self.committed_paths)) != len(self.committed_paths):
            raise ValueError("batch upload receipt contains duplicate committed paths")

    @property
    def batch_count(self) -> int:
        return len(self.batches)

    @property
    def file_count(self) -> int:
        return sum(receipt.file_count for receipt in self.batches)

    @property
    def bytes_uploaded(self) -> int:
        return sum(receipt.bytes_uploaded for receipt in self.batches)

    @property
    def committed_paths(self) -> tuple[str, ...]:
        return tuple(path for receipt in self.batches for path in receipt.committed_paths)

    @property
    def uploaded_files(self) -> tuple[WorkspaceStoredFile, ...]:
        return tuple(item for receipt in self.batches for item in receipt.uploaded_files)

    @property
    def final_commit_sha(self) -> str:
        return self.batches[-1].commit_sha

    @property
    def committed_batch_count(self) -> int:
        return sum(1 for receipt in self.batches if receipt.committed)

    def to_wire(self) -> JsonObject:
        return {
            "complete": True,
            "project_id": self.project_id,
            "requested_file_count": self.requested_file_count,
            "batch_count": self.batch_count,
            "committed_batch_count": self.committed_batch_count,
            "file_count": self.file_count,
            "bytes_uploaded": self.bytes_uploaded,
            "committed_paths": list(self.committed_paths),
            "final_commit_sha": self.final_commit_sha,
            "batches": [receipt.to_wire() for receipt in self.batches],
        }


@dataclass(frozen=True, slots=True)
class WorkspaceFilesBatchUploadProgress:
    """Exact partial receipt attached to a failed composite upload."""

    project_id: ProjectId
    requested_file_count: int
    total_batch_count: int
    completed_batches: tuple[WorkspaceFilesUploadReceipt, ...]
    failed_batch_index: int
    failed_paths: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.requested_file_count < 1:
            raise ValueError("batch upload progress requires a positive file count")
        if self.requested_file_count > WORKSPACE_BATCH_UPLOAD_FILE_LIMIT:
            raise ValueError("batch upload progress exceeds the composite file limit")
        if self.total_batch_count < 1:
            raise ValueError("batch upload progress requires a positive batch count")
        if self.failed_batch_index != len(self.completed_batches):
            raise ValueError("failed batch index must follow the completed batch receipts")
        if not 0 <= self.failed_batch_index < self.total_batch_count:
            raise ValueError("failed batch index is outside the composite request")
        if not self.failed_paths:
            raise ValueError("batch upload progress requires failed batch paths")
        if len(self.failed_paths) > WORKSPACE_UPLOAD_FILE_LIMIT:
            raise ValueError("failed batch paths exceed the bounded server file limit")
        if len(set(self.failed_paths)) != len(self.failed_paths):
            raise ValueError("failed batch paths must be unique")
        if any(receipt.project_id != self.project_id for receipt in self.completed_batches):
            raise ValueError("batch upload progress crossed its requested project boundary")
        if self.completed_file_count + len(self.failed_paths) > self.requested_file_count:
            raise ValueError("batch upload progress exceeds the composite request")

    @property
    def completed_file_count(self) -> int:
        return sum(receipt.file_count for receipt in self.completed_batches)

    @property
    def completed_paths(self) -> tuple[str, ...]:
        return tuple(path for receipt in self.completed_batches for path in receipt.committed_paths)

    @property
    def remaining_file_count(self) -> int:
        return self.requested_file_count - self.completed_file_count

    def to_wire(self) -> JsonObject:
        return {
            "complete": False,
            "project_id": self.project_id,
            "requested_file_count": self.requested_file_count,
            "total_batch_count": self.total_batch_count,
            "completed_batch_count": len(self.completed_batches),
            "completed_file_count": self.completed_file_count,
            "completed_paths": list(self.completed_paths),
            "failed_batch_index": self.failed_batch_index,
            "failed_paths": list(self.failed_paths),
            "remaining_file_count": self.remaining_file_count,
            "completed_batches": [receipt.to_wire() for receipt in self.completed_batches],
        }


__all__ = [
    "WORKSPACE_BATCH_UPLOAD_FILE_LIMIT",
    "WORKSPACE_UPLOAD_FILE_LIMIT",
    "ProjectWorkspaceInputs",
    "WorkspaceFileEncoding",
    "WorkspaceFileKind",
    "WorkspaceFileScope",
    "WorkspaceFilesUploadReceipt",
    "WorkspaceFilesUploadRequest",
    "WorkspaceFilesBatchUploadProgress",
    "WorkspaceFilesBatchUploadReceipt",
    "WorkspaceFilesBatchUploadRequest",
    "WorkspaceFileUpload",
    "WorkspaceFileVisibility",
    "WorkspaceInputState",
    "WorkspaceMetadata",
    "WorkspaceProjectRepository",
    "WorkspaceRepositoryAuthMode",
    "WorkspaceRepositoryBootstrapMode",
    "WorkspaceSourceRepository",
    "WorkspaceSourceRepositoryKind",
    "WorkspaceSourceRepositoryReceipt",
    "WorkspaceSourceRepositorySpec",
    "WorkspaceStoredFile",
]
