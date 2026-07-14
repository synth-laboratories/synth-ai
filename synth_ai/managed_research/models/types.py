"""Public models for Managed Research wire shapes (kickoff contracts, workspace, preflight).

Parse and validate JSON-like payloads at the trust boundary into dataclasses; callers
should use typed attributes after ``from_wire`` rather than probing raw mappings.

# See: Synth Style — ``specifications/tanha/references/synthstyle.md`` in the
# backend repo; backend SMR owns authoritative kickoff and work-product contracts.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, cast

from synth_ai.managed_research.models.smr_actor_models import (
    SmrActorModelAssignment,
    SmrActorType,
    normalize_actor_model_assignments,
)
from synth_ai.managed_research.models.smr_agent_kinds import (
    SmrAgentKind,
    coerce_smr_agent_kind,
)
from synth_ai.managed_research.models.smr_environment_kinds import (
    SmrEnvironmentKind,
    coerce_smr_environment_kind,
)
from synth_ai.managed_research.models.smr_network_topology import (
    SmrNetworkTopology,
    coerce_smr_network_topology,
)
from synth_ai.managed_research.models.smr_providers import (
    ActorResourceCapability,
    ProviderBinding,
    UsageLimit,
    coerce_provider_bindings,
    coerce_usage_limit,
)
from synth_ai.managed_research.models.smr_runtime_kinds import (
    SmrRuntimeKind,
    coerce_smr_runtime_kind,
)


def _require_mapping(payload: object, *, label: str) -> Mapping[str, object]:
    if not isinstance(payload, Mapping):
        raise ValueError(f"{label} must be an object")
    return cast(Mapping[str, object], payload)


def _optional_string(
    payload: Mapping[str, object],
    key: str,
) -> str | None:
    value = payload.get(key)
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError(f"{key} must be a string when provided")
    normalized = value.strip()
    return normalized or None


def _require_string(payload: Mapping[str, object], key: str, *, label: str) -> str:
    value = _optional_string(payload, key)
    if value is None:
        raise ValueError(f"{label} is required")
    return value


def _float_value(
    payload: Mapping[str, object],
    key: str,
    *,
    default: float | None = None,
) -> float:
    value = payload.get(key)
    if value is None and default is not None:
        return float(default)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{key} must be numeric")
    return float(value)


def _int_value(payload: Mapping[str, object], key: str) -> int:
    value = payload.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{key} must be an integer")
    return value


def _object_dict(payload: object) -> dict[str, object]:
    mapping = _require_mapping(payload, label="metadata")
    return dict(mapping)


def _optional_object_dict(payload: object) -> dict[str, object]:
    if payload is None:
        return {}
    return _object_dict(payload)


def _require_array(payload: Mapping[str, object], key: str, *, label: str) -> list[object]:
    value = payload.get(key)
    if not isinstance(value, list):
        raise ValueError(f"{label} must be an array")
    return cast(list[object], value)


def _optional_array(payload: Mapping[str, object], key: str) -> list[object]:
    value = payload.get(key)
    if value is None:
        return []
    if not isinstance(value, list):
        raise ValueError(f"{key} must be an array when provided")
    return cast(list[object], value)


def _optional_int(payload: Mapping[str, object], key: str) -> int | None:
    value = payload.get(key)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{key} must be an integer when provided")
    return value


def _optional_bool(payload: Mapping[str, object], key: str) -> bool | None:
    value = payload.get(key)
    if value is None:
        return None
    if not isinstance(value, bool):
        raise ValueError(f"{key} must be a boolean when provided")
    return value


def _string_list(payload: object, *, label: str) -> list[str]:
    if payload is None:
        return []
    if not isinstance(payload, list):
        raise ValueError(f"{label} must be an array when provided")
    values: list[str] = []
    for item in payload:
        if not isinstance(item, str):
            raise ValueError(f"{label} entries must be strings")
        normalized = item.strip()
        if normalized:
            values.append(normalized)
    return values


@dataclass(frozen=True)
class RecommendedAction:
    tool_name: str | None = None
    arguments: dict[str, object] = field(default_factory=dict)
    description: str | None = None

    @classmethod
    def from_wire(cls, payload: object) -> RecommendedAction:
        mapping = _require_mapping(payload, label="recommended action")
        return cls(
            tool_name=_optional_string(mapping, "tool_name"),
            arguments=_optional_object_dict(mapping.get("arguments")),
            description=_optional_string(mapping, "description"),
        )


@dataclass(frozen=True)
class WorkspaceSourceRepo:
    kind: str | None = None
    url: str | None = None
    display_url: str | None = None
    default_branch: str | None = None
    public: bool | None = None
    auth_mode: str | None = None
    bootstrap_mode: str | None = None
    remote_name: str | None = None

    @classmethod
    def from_wire(cls, payload: object) -> WorkspaceSourceRepo | None:
        if payload is None:
            return None
        mapping = _require_mapping(payload, label="workspace source repo")
        return cls(
            kind=_optional_string(mapping, "kind"),
            url=_optional_string(mapping, "url"),
            display_url=_optional_string(mapping, "display_url"),
            default_branch=_optional_string(mapping, "default_branch"),
            public=_optional_bool(mapping, "public"),
            auth_mode=_optional_string(mapping, "auth_mode"),
            bootstrap_mode=_optional_string(mapping, "bootstrap_mode"),
            remote_name=_optional_string(mapping, "remote_name"),
        )


@dataclass(frozen=True)
class WorkspaceFileInput:
    path: str | None = None
    content: str | None = None
    content_type: str | None = None
    encoding: str | None = None

    @classmethod
    def from_wire(cls, payload: object) -> WorkspaceFileInput:
        mapping = _require_mapping(payload, label="workspace file input")
        return cls(
            path=_optional_string(mapping, "path"),
            content=_optional_string(mapping, "content"),
            content_type=_optional_string(mapping, "content_type"),
            encoding=_optional_string(mapping, "encoding"),
        )


@dataclass(frozen=True)
class WorkspaceInputsState:
    project_id: str | None = None
    state: str | None = None
    source_repo: WorkspaceSourceRepo | None = None
    files: list[WorkspaceFileInput] = field(default_factory=list)
    file_count: int | None = None
    project_repo: dict[str, object] = field(default_factory=dict)
    updated_at: str | None = None

    @classmethod
    def from_wire(cls, payload: object) -> WorkspaceInputsState:
        mapping = _require_mapping(payload, label="workspace inputs state")
        files_payload = _optional_array(mapping, "files")
        return cls(
            project_id=_optional_string(mapping, "project_id"),
            state=_optional_string(mapping, "state"),
            source_repo=WorkspaceSourceRepo.from_wire(mapping.get("source_repo")),
            files=[WorkspaceFileInput.from_wire(item) for item in files_payload],
            file_count=_optional_int(mapping, "file_count"),
            project_repo=_optional_object_dict(mapping.get("project_repo")),
            updated_at=_optional_string(mapping, "updated_at"),
        )


@dataclass(frozen=True)
class WorkspaceUploadResult:
    project_id: str | None = None
    file_count: int | None = None
    bytes_uploaded: int | None = None
    uploaded_files: list[WorkspaceFileInput] = field(default_factory=list)

    @classmethod
    def from_wire(cls, payload: object) -> WorkspaceUploadResult:
        mapping = _require_mapping(payload, label="workspace upload result")
        uploaded_files_payload = _optional_array(mapping, "uploaded_files")
        return cls(
            project_id=_optional_string(mapping, "project_id"),
            file_count=_optional_int(mapping, "file_count"),
            bytes_uploaded=_optional_int(mapping, "bytes_uploaded"),
            uploaded_files=[WorkspaceFileInput.from_wire(item) for item in uploaded_files_payload],
        )


@dataclass(frozen=True)
class KickoffContractFile:
    path: str
    file_id: str | None = None
    content_type: str | None = None

    @classmethod
    def from_wire(cls, payload: object) -> KickoffContractFile:
        mapping = _require_mapping(payload, label="kickoff contract file")
        return cls(
            path=_require_string(
                mapping,
                "path",
                label="kickoff contract file.path",
            ),
            file_id=_optional_string(mapping, "file_id"),
            content_type=_optional_string(mapping, "content_type"),
        )

    def to_wire(self) -> dict[str, object]:
        payload: dict[str, object] = {"path": self.path}
        if self.file_id is not None:
            payload["file_id"] = self.file_id
        if self.content_type is not None:
            payload["content_type"] = self.content_type
        return payload


@dataclass(frozen=True)
class RequiredWorkProductSpec:
    """Structured obligation the run must publish (report, model, or container eval).

    Replaces legacy path-or-filename kickoff fields so completion criteria stay
    aligned with backend ``WorkProduct`` kinds instead of ad-hoc file lists.

    # See: backend ``services/smr/work_products`` and Synth Style "contracts" rules.
    """

    kind: str
    subtype: str | None = None
    title: str | None = None
    description: str | None = None
    required: bool = True

    @classmethod
    def from_wire(cls, payload: object) -> RequiredWorkProductSpec:
        mapping = _require_mapping(payload, label="kickoff contract required_work_product")
        kind = _require_string(
            mapping,
            "kind",
            label="kickoff contract required_work_product.kind",
        )
        if kind not in {"report", "model", "container_eval"}:
            raise ValueError(
                "kickoff contract required_work_product.kind must be one of "
                "report, model, container_eval"
            )
        required_value = mapping.get("required", True)
        if not isinstance(required_value, bool):
            raise ValueError("kickoff contract required_work_product.required must be a boolean")
        return cls(
            kind=kind,
            subtype=_optional_string(mapping, "subtype"),
            title=_optional_string(mapping, "title") or _optional_string(mapping, "label"),
            description=_optional_string(mapping, "description"),
            required=required_value,
        )

    def to_wire(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "kind": self.kind,
            "required": self.required,
        }
        if self.subtype is not None:
            payload["subtype"] = self.subtype
        if self.title is not None:
            payload["title"] = self.title
        if self.description is not None:
            payload["description"] = self.description
        return payload


@dataclass(frozen=True)
class KickoffContract:
    """Run kickoff contract mirrored from the backend SMR API.

    Legacy fields such as ``required_output_files`` are rejected so SDK clients
    cannot accidentally emit shapes the server stopped accepting.

    # See: backend run/kickoff handlers; Synth Style — fail fast on unknown contract keys.
    """

    schema_version: int
    contract_kind: str
    run_objective: str
    scenario: str | None = None
    task_id: str | None = None
    task_title: str | None = None
    task_kind: str | None = None
    repo_url: str | None = None
    worker_pool_id: str | None = None
    project_notes_framing: str | None = None
    dispatch_requirements: dict[str, object] = field(default_factory=dict)
    tasks: list[dict[str, object]] = field(default_factory=list)
    plan_task_payloads: list[dict[str, object]] = field(default_factory=list)
    task_briefs: list[str] = field(default_factory=list)
    required_work_products: list[RequiredWorkProductSpec] = field(default_factory=list)
    model_visible_contract_files: list[KickoffContractFile] = field(default_factory=list)
    kickoff_contract_file: str | None = None
    kickoff_contract_ref: str | None = None

    @classmethod
    def from_wire(cls, payload: object) -> KickoffContract:
        mapping = _require_mapping(payload, label="kickoff contract")
        _reject_legacy_file_contract_fields(mapping)
        task_payload = _optional_array(mapping, "tasks")
        plan_task_payload = _optional_array(mapping, "plan_task_payloads")
        file_payload = _optional_array(mapping, "model_visible_contract_files")
        required_work_products_payload = _optional_array(mapping, "required_work_products")
        return cls(
            schema_version=_int_value(mapping, "schema_version"),
            contract_kind=_require_string(
                mapping,
                "contract_kind",
                label="kickoff contract.contract_kind",
            ),
            run_objective=_require_string(
                mapping,
                "run_objective",
                label="kickoff contract.run_objective",
            ),
            scenario=_optional_string(mapping, "scenario"),
            task_id=_optional_string(mapping, "task_id"),
            task_title=_optional_string(mapping, "task_title"),
            task_kind=_optional_string(mapping, "task_kind"),
            repo_url=_optional_string(mapping, "repo_url"),
            worker_pool_id=_optional_string(mapping, "worker_pool_id"),
            project_notes_framing=_optional_string(mapping, "project_notes_framing"),
            dispatch_requirements=_optional_object_dict(mapping.get("dispatch_requirements")),
            tasks=[_optional_object_dict(item) for item in task_payload],
            plan_task_payloads=[_optional_object_dict(item) for item in plan_task_payload],
            task_briefs=_string_list(
                mapping.get("task_briefs"),
                label="kickoff contract.task_briefs",
            ),
            required_work_products=[
                RequiredWorkProductSpec.from_wire(item) for item in required_work_products_payload
            ],
            model_visible_contract_files=[
                KickoffContractFile.from_wire(item) for item in file_payload
            ],
            kickoff_contract_file=_optional_string(mapping, "kickoff_contract_file"),
            kickoff_contract_ref=_optional_string(mapping, "kickoff_contract_ref"),
        )

    def to_wire(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "schema_version": self.schema_version,
            "contract_kind": self.contract_kind,
            "run_objective": self.run_objective,
            "tasks": [dict(item) for item in self.tasks],
            "plan_task_payloads": [dict(item) for item in self.plan_task_payloads],
            "task_briefs": list(self.task_briefs),
            "required_work_products": [item.to_wire() for item in self.required_work_products],
            "model_visible_contract_files": [
                item.to_wire() for item in self.model_visible_contract_files
            ],
        }
        if self.dispatch_requirements:
            payload["dispatch_requirements"] = dict(self.dispatch_requirements)
        if self.scenario is not None:
            payload["scenario"] = self.scenario
        if self.task_id is not None:
            payload["task_id"] = self.task_id
        if self.task_title is not None:
            payload["task_title"] = self.task_title
        if self.task_kind is not None:
            payload["task_kind"] = self.task_kind
        if self.repo_url is not None:
            payload["repo_url"] = self.repo_url
        if self.worker_pool_id is not None:
            payload["worker_pool_id"] = self.worker_pool_id
        if self.project_notes_framing is not None:
            payload["project_notes_framing"] = self.project_notes_framing
        if self.kickoff_contract_file is not None:
            payload["kickoff_contract_file"] = self.kickoff_contract_file
        if self.kickoff_contract_ref is not None:
            payload["kickoff_contract_ref"] = self.kickoff_contract_ref
        return payload


def _reject_legacy_file_contract_fields(mapping: Mapping[str, object]) -> None:
    """Fail fast when wire payloads still use removed kickoff file-list keys.

    Backend migrated to ``required_work_products``; keeping the old keys silent would
    let callers think outputs were specified when the server ignores those fields.
    """

    for field_name in (
        "required_output_files",
        "required_output_paths",
        "allowed_repo_paths",
        "required_files",
        "required_file_paths",
    ):
        if field_name in mapping:
            raise ValueError(
                f"kickoff contract.{field_name} is no longer supported; "
                "use required_work_products instead"
            )


@dataclass(frozen=True)
class StoredFile:
    file_id: str
    org_id: str
    project_id: str
    run_id: str | None = None
    scope_kind: str | None = None
    path: str | None = None
    logical_name: str | None = None
    content_type: str | None = None
    encoding: str | None = None
    visibility: str | None = None
    storage_backend: str | None = None
    content_sha256: str | None = None
    content_bytes: int | None = None
    metadata: dict[str, object] = field(default_factory=dict)
    created_at: str | None = None
    updated_at: str | None = None

    @classmethod
    def from_wire(cls, payload: object) -> StoredFile:
        mapping = _require_mapping(payload, label="stored file")
        return cls(
            file_id=_require_string(mapping, "file_id", label="stored file.file_id"),
            org_id=_require_string(mapping, "org_id", label="stored file.org_id"),
            project_id=_require_string(mapping, "project_id", label="stored file.project_id"),
            run_id=_optional_string(mapping, "run_id"),
            scope_kind=_optional_string(mapping, "scope_kind"),
            path=_optional_string(mapping, "path"),
            logical_name=_optional_string(mapping, "logical_name"),
            content_type=_optional_string(mapping, "content_type"),
            encoding=_optional_string(mapping, "encoding"),
            visibility=_optional_string(mapping, "visibility"),
            storage_backend=_optional_string(mapping, "storage_backend"),
            content_sha256=_optional_string(mapping, "content_sha256"),
            content_bytes=_optional_int(mapping, "content_bytes"),
            metadata=_optional_object_dict(mapping.get("metadata")),
            created_at=_optional_string(mapping, "created_at"),
            updated_at=_optional_string(mapping, "updated_at"),
        )


@dataclass(frozen=True)
class RunFileMount:
    mount_id: str
    run_id: str
    file_id: str
    mount_path: str | None = None
    visibility: str | None = None
    active: bool | None = None
    metadata: dict[str, object] = field(default_factory=dict)
    created_at: str | None = None
    file: StoredFile | None = None

    @classmethod
    def from_wire(cls, payload: object) -> RunFileMount:
        mapping = _require_mapping(payload, label="run file mount")
        file_payload = mapping.get("file")
        return cls(
            mount_id=_require_string(mapping, "mount_id", label="run file mount.mount_id"),
            run_id=_require_string(mapping, "run_id", label="run file mount.run_id"),
            file_id=_require_string(mapping, "file_id", label="run file mount.file_id"),
            mount_path=_optional_string(mapping, "mount_path"),
            visibility=_optional_string(mapping, "visibility"),
            active=_optional_bool(mapping, "active"),
            metadata=_optional_object_dict(mapping.get("metadata")),
            created_at=_optional_string(mapping, "created_at"),
            file=StoredFile.from_wire(file_payload) if file_payload is not None else None,
        )


@dataclass(frozen=True)
class RunOutputFile:
    output_file_id: str
    org_id: str
    project_id: str
    run_id: str
    artifact_type: str | None = None
    title: str | None = None
    uri: str | None = None
    digest: str | None = None
    path: str | None = None
    content_type: str | None = None
    failure_reason: str | None = None
    failure_source: str | None = None
    created_at: str | None = None
    metadata: dict[str, object] = field(default_factory=dict)

    @classmethod
    def from_wire(cls, payload: object) -> RunOutputFile:
        mapping = _require_mapping(payload, label="run output file")
        return cls(
            output_file_id=_require_string(
                mapping, "output_file_id", label="run output file.output_file_id"
            ),
            org_id=_require_string(mapping, "org_id", label="run output file.org_id"),
            project_id=_require_string(mapping, "project_id", label="run output file.project_id"),
            run_id=_require_string(mapping, "run_id", label="run output file.run_id"),
            artifact_type=_optional_string(mapping, "artifact_type"),
            title=_optional_string(mapping, "title"),
            uri=_optional_string(mapping, "uri"),
            digest=_optional_string(mapping, "digest"),
            path=_optional_string(mapping, "path"),
            content_type=_optional_string(mapping, "content_type"),
            failure_reason=_optional_string(mapping, "failure_reason"),
            failure_source=_optional_string(mapping, "failure_source"),
            created_at=_optional_string(mapping, "created_at"),
            metadata=_optional_object_dict(mapping.get("metadata")),
        )


@dataclass(frozen=True)
class RunArtifact:
    artifact_id: str
    project_id: str | None = None
    run_id: str | None = None
    artifact_type: str | None = None
    title: str | None = None
    uri: str | None = None
    digest: str | None = None
    path: str | None = None
    content_type: str | None = None
    size_bytes: int | None = None
    content_url: str | None = None
    download_url: str | None = None
    created_at: str | None = None
    metadata: dict[str, object] = field(default_factory=dict)

    @classmethod
    def from_wire(cls, payload: object) -> RunArtifact:
        mapping = _require_mapping(payload, label="run artifact")
        size_value = mapping.get("size_bytes")
        size_bytes = (
            size_value if isinstance(size_value, int) and not isinstance(size_value, bool) else None
        )
        return cls(
            artifact_id=_require_string(mapping, "artifact_id", label="run artifact.artifact_id"),
            project_id=_optional_string(mapping, "project_id"),
            run_id=_optional_string(mapping, "run_id"),
            artifact_type=_optional_string(mapping, "artifact_type"),
            title=_optional_string(mapping, "title"),
            uri=_optional_string(mapping, "uri"),
            digest=_optional_string(mapping, "digest"),
            path=_optional_string(mapping, "path"),
            content_type=_optional_string(mapping, "content_type"),
            size_bytes=size_bytes,
            content_url=_optional_string(mapping, "content_url"),
            download_url=_optional_string(mapping, "download_url"),
            created_at=_optional_string(mapping, "created_at"),
            metadata=_optional_object_dict(mapping.get("metadata")),
        )


@dataclass(frozen=True)
class RunArtifactManifest:
    schema_version: str
    project_id: str
    run_id: str
    generated_at: str | None = None
    artifact_count: int = 0
    artifacts: list[RunArtifact] = field(default_factory=list)
    output_files: list[RunArtifact] = field(default_factory=list)
    result_json: RunArtifact | None = None
    result_outputs: list[RunArtifact] = field(default_factory=list)
    reports: list[RunArtifact] = field(default_factory=list)
    pull_requests: list[RunArtifact] = field(default_factory=list)
    workspace_archive: dict[str, object] = field(default_factory=dict)
    models: list[dict[str, object]] = field(default_factory=list)
    datasets: list[dict[str, object]] = field(default_factory=list)
    links: dict[str, object] = field(default_factory=dict)
    raw: dict[str, object] = field(default_factory=dict)

    @classmethod
    def from_wire(cls, payload: object) -> RunArtifactManifest:
        mapping = _require_mapping(payload, label="run artifact manifest")
        result_json_payload = mapping.get("result_json")
        artifact_count = mapping.get("artifact_count")
        return cls(
            schema_version=_require_string(
                mapping,
                "schema_version",
                label="run artifact manifest.schema_version",
            ),
            project_id=_require_string(
                mapping, "project_id", label="run artifact manifest.project_id"
            ),
            run_id=_require_string(mapping, "run_id", label="run artifact manifest.run_id"),
            generated_at=_optional_string(mapping, "generated_at"),
            artifact_count=(
                artifact_count
                if isinstance(artifact_count, int) and not isinstance(artifact_count, bool)
                else 0
            ),
            artifacts=[
                RunArtifact.from_wire(item) for item in _optional_array(mapping, "artifacts")
            ],
            output_files=[
                RunArtifact.from_wire(item) for item in _optional_array(mapping, "output_files")
            ],
            result_json=(
                RunArtifact.from_wire(result_json_payload)
                if isinstance(result_json_payload, Mapping)
                else None
            ),
            result_outputs=[
                RunArtifact.from_wire(item) for item in _optional_array(mapping, "result_outputs")
            ],
            reports=[RunArtifact.from_wire(item) for item in _optional_array(mapping, "reports")],
            pull_requests=[
                RunArtifact.from_wire(item) for item in _optional_array(mapping, "pull_requests")
            ],
            workspace_archive=_optional_object_dict(mapping.get("workspace_archive")),
            models=[
                _object_dict(item)
                for item in _optional_array(mapping, "models")
                if isinstance(item, Mapping)
            ],
            datasets=[
                _object_dict(item)
                for item in _optional_array(mapping, "datasets")
                if isinstance(item, Mapping)
            ],
            links=_optional_object_dict(mapping.get("links")),
            raw=dict(mapping),
        )


@dataclass(frozen=True)
class ResourceUploadResult:
    project_id: str
    run_id: str | None = None
    file_count: int | None = None
    bytes_uploaded: int | None = None
    uploaded_files: list[StoredFile] = field(default_factory=list)

    @classmethod
    def from_wire(cls, payload: object) -> ResourceUploadResult:
        mapping = _require_mapping(payload, label="resource upload result")
        uploaded_files_payload = _optional_array(mapping, "uploaded_files")
        return cls(
            project_id=_require_string(
                mapping, "project_id", label="resource upload result.project_id"
            ),
            run_id=_optional_string(mapping, "run_id"),
            file_count=_optional_int(mapping, "file_count"),
            bytes_uploaded=_optional_int(mapping, "bytes_uploaded"),
            uploaded_files=[StoredFile.from_wire(item) for item in uploaded_files_payload],
        )


@dataclass(frozen=True)
class ProjectCodeSource:
    project_id: str
    kind: str
    status: str
    default_branch: str
    upload_id: str
    internal_repo_ref: dict[str, object] = field(default_factory=dict)
    head_commit_sha: str | None = None
    validation_summary: dict[str, object] = field(default_factory=dict)
    metadata: dict[str, object] = field(default_factory=dict)

    @classmethod
    def from_wire(cls, payload: object) -> ProjectCodeSource:
        mapping = _require_mapping(payload, label="project code source")
        return cls(
            project_id=_require_string(
                mapping, "project_id", label="project code source.project_id"
            ),
            kind=_require_string(mapping, "kind", label="project code source.kind"),
            status=_require_string(mapping, "status", label="project code source.status"),
            default_branch=_require_string(
                mapping,
                "default_branch",
                label="project code source.default_branch",
            ),
            upload_id=_require_string(mapping, "upload_id", label="project code source.upload_id"),
            internal_repo_ref=_optional_object_dict(mapping.get("internal_repo_ref")),
            head_commit_sha=_optional_string(mapping, "head_commit_sha"),
            validation_summary=_optional_object_dict(mapping.get("validation_summary")),
            metadata=_optional_object_dict(mapping.get("metadata")),
        )


@dataclass(frozen=True)
class ProjectDataPoolObject:
    object_id: str
    file_id: str
    path: str
    role: str
    content_hash: str | None = None
    size_bytes: int | None = None
    media_type: str | None = None
    metadata: dict[str, object] = field(default_factory=dict)

    @classmethod
    def from_wire(cls, payload: object) -> ProjectDataPoolObject:
        mapping = _require_mapping(payload, label="project data pool object")
        return cls(
            object_id=_require_string(
                mapping,
                "object_id",
                label="project data pool object.object_id",
            ),
            file_id=_require_string(mapping, "file_id", label="project data pool object.file_id"),
            path=_require_string(mapping, "path", label="project data pool object.path"),
            role=_require_string(mapping, "role", label="project data pool object.role"),
            content_hash=_optional_string(mapping, "content_hash"),
            size_bytes=_optional_int(mapping, "size_bytes"),
            media_type=_optional_string(mapping, "media_type"),
            metadata=_optional_object_dict(mapping.get("metadata")),
        )


@dataclass(frozen=True)
class ProjectDataPoolUploadResult:
    project_id: str
    pool_id: str
    name: str
    status: str
    manifest_id: str
    object_prefix: str
    file_count: int | None = None
    bytes_uploaded: int | None = None
    access_policy: dict[str, object] = field(default_factory=dict)
    metadata: dict[str, object] = field(default_factory=dict)
    objects: list[ProjectDataPoolObject] = field(default_factory=list)

    @classmethod
    def from_wire(cls, payload: object) -> ProjectDataPoolUploadResult:
        mapping = _require_mapping(payload, label="project data pool upload result")
        object_payload = _optional_array(mapping, "objects")
        return cls(
            project_id=_require_string(
                mapping,
                "project_id",
                label="project data pool upload result.project_id",
            ),
            pool_id=_require_string(
                mapping, "pool_id", label="project data pool upload result.pool_id"
            ),
            name=_require_string(mapping, "name", label="project data pool upload result.name"),
            status=_require_string(
                mapping, "status", label="project data pool upload result.status"
            ),
            manifest_id=_require_string(
                mapping,
                "manifest_id",
                label="project data pool upload result.manifest_id",
            ),
            object_prefix=_require_string(
                mapping,
                "object_prefix",
                label="project data pool upload result.object_prefix",
            ),
            file_count=_optional_int(mapping, "file_count"),
            bytes_uploaded=_optional_int(mapping, "bytes_uploaded"),
            access_policy=_optional_object_dict(mapping.get("access_policy")),
            metadata=_optional_object_dict(mapping.get("metadata")),
            objects=[ProjectDataPoolObject.from_wire(item) for item in object_payload],
        )


@dataclass(frozen=True)
class ProjectLaunchProfile:
    project_id: str
    status: str
    launch_profile: dict[str, object] = field(default_factory=dict)
    launch_request: dict[str, object] = field(default_factory=dict)
    workspace_profile: dict[str, object] = field(default_factory=dict)
    metadata: dict[str, object] = field(default_factory=dict)

    @classmethod
    def from_wire(cls, payload: object) -> ProjectLaunchProfile:
        mapping = _require_mapping(payload, label="project launch profile")
        return cls(
            project_id=_require_string(
                mapping,
                "project_id",
                label="project launch profile.project_id",
            ),
            status=_require_string(mapping, "status", label="project launch profile.status"),
            launch_profile=_optional_object_dict(mapping.get("launch_profile")),
            launch_request=_optional_object_dict(mapping.get("launch_request")),
            workspace_profile=_optional_object_dict(mapping.get("workspace_profile")),
            metadata=_optional_object_dict(mapping.get("metadata")),
        )


@dataclass(frozen=True)
class ExternalRepository:
    repository_id: str
    org_id: str
    project_id: str
    run_id: str | None = None
    scope_kind: str | None = None
    name: str | None = None
    url: str | None = None
    default_branch: str | None = None
    role: str | None = None
    metadata: dict[str, object] = field(default_factory=dict)
    created_at: str | None = None
    updated_at: str | None = None

    @classmethod
    def from_wire(cls, payload: object) -> ExternalRepository:
        mapping = _require_mapping(payload, label="external repository")
        return cls(
            repository_id=_require_string(
                mapping, "repository_id", label="external repository.repository_id"
            ),
            org_id=_require_string(mapping, "org_id", label="external repository.org_id"),
            project_id=_require_string(
                mapping, "project_id", label="external repository.project_id"
            ),
            run_id=_optional_string(mapping, "run_id"),
            scope_kind=_optional_string(mapping, "scope_kind"),
            name=_optional_string(mapping, "name"),
            url=_optional_string(mapping, "url"),
            default_branch=_optional_string(mapping, "default_branch"),
            role=_optional_string(mapping, "role"),
            metadata=_optional_object_dict(mapping.get("metadata")),
            created_at=_optional_string(mapping, "created_at"),
            updated_at=_optional_string(mapping, "updated_at"),
        )


@dataclass(frozen=True)
class RunRepositoryMount:
    mount_id: str
    run_id: str
    repository_id: str
    mount_name: str | None = None
    role: str | None = None
    metadata: dict[str, object] = field(default_factory=dict)
    created_at: str | None = None
    repository: ExternalRepository | None = None

    @classmethod
    def from_wire(cls, payload: object) -> RunRepositoryMount:
        mapping = _require_mapping(payload, label="run repository mount")
        repository_payload = mapping.get("repository")
        return cls(
            mount_id=_require_string(mapping, "mount_id", label="run repository mount.mount_id"),
            run_id=_require_string(mapping, "run_id", label="run repository mount.run_id"),
            repository_id=_require_string(
                mapping,
                "repository_id",
                label="run repository mount.repository_id",
            ),
            mount_name=_optional_string(mapping, "mount_name"),
            role=_optional_string(mapping, "role"),
            metadata=_optional_object_dict(mapping.get("metadata")),
            created_at=_optional_string(mapping, "created_at"),
            repository=(
                ExternalRepository.from_wire(repository_payload)
                if repository_payload is not None
                else None
            ),
        )


@dataclass(frozen=True)
class CredentialRef:
    credential_ref_id: str
    org_id: str
    project_id: str
    run_id: str | None = None
    scope_kind: str | None = None
    kind: str | None = None
    label: str | None = None
    provider: str | None = None
    funding_source: str | None = None
    credential_name: str | None = None
    metadata: dict[str, object] = field(default_factory=dict)
    created_at: str | None = None
    updated_at: str | None = None

    @classmethod
    def from_wire(cls, payload: object) -> CredentialRef:
        mapping = _require_mapping(payload, label="credential ref")
        return cls(
            credential_ref_id=_require_string(
                mapping, "credential_ref_id", label="credential ref.credential_ref_id"
            ),
            org_id=_require_string(mapping, "org_id", label="credential ref.org_id"),
            project_id=_require_string(mapping, "project_id", label="credential ref.project_id"),
            run_id=_optional_string(mapping, "run_id"),
            scope_kind=_optional_string(mapping, "scope_kind"),
            kind=_optional_string(mapping, "kind"),
            label=_optional_string(mapping, "label"),
            provider=_optional_string(mapping, "provider"),
            funding_source=_optional_string(mapping, "funding_source"),
            credential_name=_optional_string(mapping, "credential_name"),
            metadata=_optional_object_dict(mapping.get("metadata")),
            created_at=_optional_string(mapping, "created_at"),
            updated_at=_optional_string(mapping, "updated_at"),
        )


@dataclass(frozen=True)
class RunCredentialBinding:
    binding_id: str
    run_id: str
    credential_ref_id: str
    metadata: dict[str, object] = field(default_factory=dict)
    created_at: str | None = None
    credential_ref: CredentialRef | None = None

    @classmethod
    def from_wire(cls, payload: object) -> RunCredentialBinding:
        mapping = _require_mapping(payload, label="run credential binding")
        credential_ref_payload = mapping.get("credential_ref")
        return cls(
            binding_id=_require_string(
                mapping, "binding_id", label="run credential binding.binding_id"
            ),
            run_id=_require_string(mapping, "run_id", label="run credential binding.run_id"),
            credential_ref_id=_require_string(
                mapping,
                "credential_ref_id",
                label="run credential binding.credential_ref_id",
            ),
            metadata=_optional_object_dict(mapping.get("metadata")),
            created_at=_optional_string(mapping, "created_at"),
            credential_ref=(
                CredentialRef.from_wire(credential_ref_payload)
                if credential_ref_payload is not None
                else None
            ),
        )


@dataclass(frozen=True)
class Environment:
    environment_id: str | None
    name: str
    digest: str
    manifest_digest: str | None = None
    org_id: str | None = None
    created_by_user_id: str | None = None
    spec: dict[str, object] = field(default_factory=dict)
    manifest: dict[str, object] = field(default_factory=dict)
    created_at: str | None = None

    @classmethod
    def from_wire(cls, payload: object) -> Environment:
        mapping = _require_mapping(payload, label="environment")
        return cls(
            environment_id=_optional_string(mapping, "environment_id"),
            name=_require_string(mapping, "name", label="environment.name"),
            digest=_require_string(mapping, "digest", label="environment.digest"),
            manifest_digest=_optional_string(mapping, "manifest_digest"),
            org_id=_optional_string(mapping, "org_id"),
            created_by_user_id=_optional_string(mapping, "created_by_user_id"),
            spec=_optional_object_dict(mapping.get("spec")),
            manifest=_optional_object_dict(mapping.get("manifest")),
            created_at=_optional_string(mapping, "created_at"),
        )


@dataclass(frozen=True)
class EnvironmentPreflight:
    name: str
    digest: str
    ok: bool
    result: str
    details: dict[str, object] = field(default_factory=dict)
    error: dict[str, object] | None = None

    @classmethod
    def from_wire(cls, payload: object) -> EnvironmentPreflight:
        mapping = _require_mapping(payload, label="environment preflight")
        error_payload = mapping.get("error")
        return cls(
            name=_require_string(mapping, "name", label="environment preflight.name"),
            digest=_require_string(mapping, "digest", label="environment preflight.digest"),
            ok=bool(mapping.get("ok")),
            result=_require_string(mapping, "result", label="environment preflight.result"),
            details=_optional_object_dict(mapping.get("details")),
            error=(
                _optional_object_dict(error_payload) if isinstance(error_payload, Mapping) else None
            ),
        )


@dataclass(frozen=True)
class DevEnvironment:
    dev_environment_id: str
    environment_id: str
    org_id: str
    project_id: str
    name: str
    backend_target: str
    lifecycle_state: str
    topology_id: str
    host_kind: str
    environment_name: str
    topology_version: str | None = None
    environment_digest: str | None = None
    quota_class: str | None = None
    cost_summary: dict[str, object] = field(default_factory=dict)
    service_summary: dict[str, object] = field(default_factory=dict)
    metadata: dict[str, object] = field(default_factory=dict)
    created_by_user_id: str | None = None
    created_at: str | None = None
    updated_at: str | None = None
    deleted_at: str | None = None

    @classmethod
    def from_wire(cls, payload: object) -> DevEnvironment:
        mapping = _require_mapping(payload, label="dev environment")
        dev_environment_id = _require_string(
            mapping,
            "dev_environment_id",
            label="dev_environment.dev_environment_id",
        )
        return cls(
            dev_environment_id=dev_environment_id,
            environment_id=_optional_string(mapping, "environment_id") or dev_environment_id,
            org_id=_require_string(mapping, "org_id", label="dev_environment.org_id"),
            project_id=_require_string(
                mapping,
                "project_id",
                label="dev_environment.project_id",
            ),
            name=_require_string(mapping, "name", label="dev_environment.name"),
            backend_target=_require_string(
                mapping,
                "backend_target",
                label="dev_environment.backend_target",
            ),
            lifecycle_state=_require_string(
                mapping,
                "lifecycle_state",
                label="dev_environment.lifecycle_state",
            ),
            topology_id=_require_string(
                mapping,
                "topology_id",
                label="dev_environment.topology_id",
            ),
            topology_version=_optional_string(mapping, "topology_version"),
            environment_name=_require_string(
                mapping,
                "environment_name",
                label="dev_environment.environment_name",
            ),
            environment_digest=_optional_string(mapping, "environment_digest"),
            host_kind=_require_string(mapping, "host_kind", label="dev_environment.host_kind"),
            quota_class=_optional_string(mapping, "quota_class"),
            cost_summary=_optional_object_dict(mapping.get("cost_summary")),
            service_summary=_optional_object_dict(mapping.get("service_summary")),
            metadata=_optional_object_dict(mapping.get("metadata")),
            created_by_user_id=_optional_string(mapping, "created_by_user_id"),
            created_at=_optional_string(mapping, "created_at"),
            updated_at=_optional_string(mapping, "updated_at"),
            deleted_at=_optional_string(mapping, "deleted_at"),
        )


@dataclass(frozen=True)
class DevEnvironmentTopology:
    topology_id: str
    version: str
    display_name: str
    service_graph: dict[str, object] = field(default_factory=dict)
    backing_services: list[dict[str, object]] = field(default_factory=list)
    manifest_refs: list[dict[str, object]] = field(default_factory=list)
    required_secrets: list[dict[str, object]] = field(default_factory=list)
    network_surfaces: list[dict[str, object]] = field(default_factory=list)
    health_checks: list[dict[str, object]] = field(default_factory=list)
    allowed_substrates: list[str] = field(default_factory=list)
    metadata: dict[str, object] = field(default_factory=dict)

    @classmethod
    def from_wire(cls, payload: object) -> DevEnvironmentTopology:
        mapping = _require_mapping(payload, label="dev environment topology")

        def object_list(key: str) -> list[dict[str, object]]:
            return [_optional_object_dict(item) for item in _optional_array(mapping, key)]

        raw_allowed = _optional_array(mapping, "allowed_substrates")
        allowed_substrates = [
            value.strip() for value in (str(item or "") for item in raw_allowed) if value.strip()
        ]
        return cls(
            topology_id=_require_string(
                mapping,
                "topology_id",
                label="dev_environment_topology.topology_id",
            ),
            version=_require_string(
                mapping,
                "version",
                label="dev_environment_topology.version",
            ),
            display_name=_require_string(
                mapping,
                "display_name",
                label="dev_environment_topology.display_name",
            ),
            service_graph=_optional_object_dict(mapping.get("service_graph")),
            backing_services=object_list("backing_services"),
            manifest_refs=object_list("manifest_refs"),
            required_secrets=object_list("required_secrets"),
            network_surfaces=object_list("network_surfaces"),
            health_checks=object_list("health_checks"),
            allowed_substrates=allowed_substrates,
            metadata=_optional_object_dict(mapping.get("metadata")),
        )


@dataclass(frozen=True)
class DevEnvironmentPreflight:
    dev_environment_id: str
    preflight_ok: bool
    lifecycle_state: str
    checks: list[dict[str, object]] = field(default_factory=list)
    error: dict[str, object] | None = None
    manifest: dict[str, object] | None = None
    topology: dict[str, object] | None = None

    @classmethod
    def from_wire(cls, payload: object) -> DevEnvironmentPreflight:
        mapping = _require_mapping(payload, label="dev environment preflight")
        checks = _optional_array(mapping, "checks")
        error_payload = mapping.get("error")
        manifest_payload = mapping.get("manifest")
        topology_payload = mapping.get("topology")
        return cls(
            dev_environment_id=_require_string(
                mapping,
                "dev_environment_id",
                label="dev_environment_preflight.dev_environment_id",
            ),
            preflight_ok=bool(mapping.get("preflight_ok")),
            lifecycle_state=_require_string(
                mapping,
                "lifecycle_state",
                label="dev_environment_preflight.lifecycle_state",
            ),
            checks=[_optional_object_dict(item) for item in checks],
            error=(
                _optional_object_dict(error_payload) if isinstance(error_payload, Mapping) else None
            ),
            manifest=(
                _optional_object_dict(manifest_payload)
                if isinstance(manifest_payload, Mapping)
                else None
            ),
            topology=(
                _optional_object_dict(topology_payload)
                if isinstance(topology_payload, Mapping)
                else None
            ),
        )


@dataclass(frozen=True)
class DevEnvironmentCollection:
    dev_environment_id: str
    items: list[dict[str, object]] = field(default_factory=list)
    lifecycle_state: str | None = None
    environment: dict[str, object] = field(default_factory=dict)
    usage: dict[str, object] = field(default_factory=dict)
    summary: dict[str, object] = field(default_factory=dict)
    next_cursor: str | None = None
    projection_sources: dict[str, object] = field(default_factory=dict)

    @classmethod
    def from_wire(cls, payload: object, *, key: str) -> DevEnvironmentCollection:
        mapping = _require_mapping(payload, label=f"dev environment {key}")
        raw_items = mapping.get(key)
        if isinstance(raw_items, Mapping):
            items = [dict(raw_items)]
        else:
            items = _optional_array(mapping, key)
        return cls(
            dev_environment_id=_require_string(
                mapping,
                "dev_environment_id",
                label=f"dev_environment_{key}.dev_environment_id",
            ),
            lifecycle_state=_optional_string(mapping, "lifecycle_state"),
            environment=_optional_object_dict(mapping.get("environment")),
            usage=_optional_object_dict(mapping.get("usage")),
            summary=_optional_object_dict(mapping.get("summary")),
            items=[_optional_object_dict(item) for item in items],
            next_cursor=_optional_string(mapping, "next_cursor"),
            projection_sources=_optional_object_dict(mapping.get("projection_sources")),
        )


@dataclass(frozen=True)
class DevEnvironmentAttach:
    dev_environment_id: str
    lifecycle_state: str
    attachable: bool = False
    attach_surfaces: list[dict[str, object]] = field(default_factory=list)
    default_surface: dict[str, object] | None = None
    operator_next_action: str | None = None
    topology: dict[str, object] = field(default_factory=dict)
    service_summary: dict[str, object] = field(default_factory=dict)
    projection_sources: dict[str, object] = field(default_factory=dict)

    @classmethod
    def from_wire(cls, payload: object) -> DevEnvironmentAttach:
        mapping = _require_mapping(payload, label="dev environment attach")
        default_surface = mapping.get("default_surface")
        return cls(
            dev_environment_id=_require_string(
                mapping,
                "dev_environment_id",
                label="dev_environment_attach.dev_environment_id",
            ),
            lifecycle_state=_require_string(
                mapping,
                "lifecycle_state",
                label="dev_environment_attach.lifecycle_state",
            ),
            attachable=bool(mapping.get("attachable")),
            attach_surfaces=[
                _optional_object_dict(item) for item in _optional_array(mapping, "attach_surfaces")
            ],
            default_surface=(
                _optional_object_dict(default_surface)
                if isinstance(default_surface, Mapping)
                else None
            ),
            operator_next_action=_optional_string(mapping, "operator_next_action"),
            topology=_optional_object_dict(mapping.get("topology")),
            service_summary=_optional_object_dict(mapping.get("service_summary")),
            projection_sources=_optional_object_dict(mapping.get("projection_sources")),
        )


@dataclass(frozen=True)
class DevEnvironmentUsage:
    dev_environment_id: str
    summary: dict[str, object] = field(default_factory=dict)
    by_meter: list[dict[str, object]] = field(default_factory=list)
    facts: list[dict[str, object]] = field(default_factory=list)
    limit: int | None = None
    next_cursor: str | None = None

    @classmethod
    def from_wire(cls, payload: object) -> DevEnvironmentUsage:
        mapping = _require_mapping(payload, label="dev environment usage")
        return cls(
            dev_environment_id=_require_string(
                mapping,
                "dev_environment_id",
                label="dev_environment_usage.dev_environment_id",
            ),
            summary=_optional_object_dict(mapping.get("summary")),
            by_meter=[_optional_object_dict(item) for item in _optional_array(mapping, "by_meter")],
            facts=[_optional_object_dict(item) for item in _optional_array(mapping, "facts")],
            limit=_optional_int(mapping, "limit"),
            next_cursor=_optional_string(mapping, "next_cursor"),
        )


@dataclass(frozen=True)
class DevEnvironmentMaterializationWorkItem:
    dev_environment_id: str
    environment_id: str
    org_id: str
    project_id: str
    name: str
    backend_target: str
    lifecycle_state: str
    materialization_action: str
    topology_id: str
    host_kind: str
    topology_version: str | None = None
    environment: dict[str, object] = field(default_factory=dict)
    quota_class: str | None = None
    materialization_request: dict[str, object] = field(default_factory=dict)
    materialization_lease: dict[str, object] = field(default_factory=dict)
    service_summary: dict[str, object] = field(default_factory=dict)
    metadata: dict[str, object] = field(default_factory=dict)
    created_at: str | None = None
    updated_at: str | None = None

    @classmethod
    def from_wire(cls, payload: object) -> DevEnvironmentMaterializationWorkItem:
        mapping = _require_mapping(payload, label="dev environment materialization item")
        dev_environment_id = _require_string(
            mapping,
            "dev_environment_id",
            label="dev_environment_materialization.dev_environment_id",
        )
        return cls(
            dev_environment_id=dev_environment_id,
            environment_id=_optional_string(mapping, "environment_id") or dev_environment_id,
            org_id=_require_string(
                mapping,
                "org_id",
                label="dev_environment_materialization.org_id",
            ),
            project_id=_require_string(
                mapping,
                "project_id",
                label="dev_environment_materialization.project_id",
            ),
            name=_require_string(
                mapping,
                "name",
                label="dev_environment_materialization.name",
            ),
            backend_target=_require_string(
                mapping,
                "backend_target",
                label="dev_environment_materialization.backend_target",
            ),
            lifecycle_state=_require_string(
                mapping,
                "lifecycle_state",
                label="dev_environment_materialization.lifecycle_state",
            ),
            materialization_action=_require_string(
                mapping,
                "materialization_action",
                label="dev_environment_materialization.materialization_action",
            ),
            topology_id=_require_string(
                mapping,
                "topology_id",
                label="dev_environment_materialization.topology_id",
            ),
            topology_version=_optional_string(mapping, "topology_version"),
            host_kind=_require_string(
                mapping,
                "host_kind",
                label="dev_environment_materialization.host_kind",
            ),
            environment=_optional_object_dict(mapping.get("environment")),
            quota_class=_optional_string(mapping, "quota_class"),
            materialization_request=_optional_object_dict(mapping.get("materialization_request")),
            materialization_lease=_optional_object_dict(mapping.get("materialization_lease")),
            service_summary=_optional_object_dict(mapping.get("service_summary")),
            metadata=_optional_object_dict(mapping.get("metadata")),
            created_at=_optional_string(mapping, "created_at"),
            updated_at=_optional_string(mapping, "updated_at"),
        )


@dataclass(frozen=True)
class DevEnvironmentMaterializationQueue:
    items: list[DevEnvironmentMaterializationWorkItem] = field(default_factory=list)
    next_cursor: str | None = None

    @classmethod
    def from_wire(cls, payload: object) -> DevEnvironmentMaterializationQueue:
        mapping = _require_mapping(payload, label="dev environment materialization queue")
        return cls(
            items=[
                DevEnvironmentMaterializationWorkItem.from_wire(item)
                for item in _optional_array(mapping, "items")
            ],
            next_cursor=_optional_string(mapping, "next_cursor"),
        )


Secret = CredentialRef
Repository = ExternalRepository
GitHubInstallation = dict[str, object]


@dataclass(frozen=True)
class InlineExternalRepositoryBinding:
    name: str
    url: str
    default_branch: str | None = None
    role: str | None = None
    metadata: dict[str, object] = field(default_factory=dict)

    @classmethod
    def from_wire(cls, payload: object) -> InlineExternalRepositoryBinding:
        mapping = _require_mapping(payload, label="inline external repository binding")
        return cls(
            name=_require_string(
                mapping,
                "name",
                label="inline external repository binding.name",
            ),
            url=_require_string(
                mapping,
                "url",
                label="inline external repository binding.url",
            ),
            default_branch=_optional_string(mapping, "default_branch"),
            role=_optional_string(mapping, "role"),
            metadata=_optional_object_dict(mapping.get("metadata")),
        )

    def to_wire(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "name": self.name,
            "url": self.url,
            "metadata": dict(self.metadata),
        }
        if self.default_branch is not None:
            payload["default_branch"] = self.default_branch
        if self.role is not None:
            payload["role"] = self.role
        return payload


@dataclass(frozen=True)
class RunResourceBindings:
    external_repository_ids: list[str] = field(default_factory=list)
    external_repositories: list[InlineExternalRepositoryBinding] = field(default_factory=list)
    credential_ref_ids: list[str] = field(default_factory=list)

    @classmethod
    def from_wire(cls, payload: object) -> RunResourceBindings:
        mapping = _require_mapping(payload, label="run resource bindings")
        external_repository_payload = _optional_array(mapping, "external_repositories")
        return cls(
            external_repository_ids=_string_list(
                mapping.get("external_repository_ids"),
                label="run resource bindings.external_repository_ids",
            ),
            external_repositories=[
                InlineExternalRepositoryBinding.from_wire(item)
                for item in external_repository_payload
            ],
            credential_ref_ids=_string_list(
                mapping.get("credential_ref_ids"),
                label="run resource bindings.credential_ref_ids",
            ),
        )

    def to_wire(self) -> dict[str, object]:
        payload: dict[str, object] = {}
        if self.external_repository_ids:
            payload["external_repository_ids"] = list(self.external_repository_ids)
        if self.external_repositories:
            payload["external_repositories"] = [
                item.to_wire() for item in self.external_repositories
            ]
        if self.credential_ref_ids:
            payload["credential_ref_ids"] = list(self.credential_ref_ids)
        return payload


@dataclass(frozen=True)
class ProviderKeyStatus:
    ok: bool | None = None
    project_id: str | None = None
    provider: str | None = None
    funding_source: str | None = None
    configured: bool | None = None
    required: bool | None = None
    auth_mechanism: str | None = None

    @classmethod
    def from_wire(cls, payload: object) -> ProviderKeyStatus:
        mapping = _require_mapping(payload, label="provider key status")
        return cls(
            ok=_optional_bool(mapping, "ok"),
            project_id=_optional_string(mapping, "project_id"),
            provider=_optional_string(mapping, "provider"),
            funding_source=_optional_string(mapping, "funding_source"),
            configured=_optional_bool(mapping, "configured"),
            required=_optional_bool(mapping, "required"),
            auth_mechanism=_optional_string(mapping, "auth_mechanism"),
        )


@dataclass(frozen=True)
class ProjectReadiness:
    project_id: str | None = None
    state: str | None = None
    blockers: list[str] = field(default_factory=list)
    recommended_actions: list[RecommendedAction] = field(default_factory=list)
    entitlement: dict[str, object] = field(default_factory=dict)
    capabilities: dict[str, object] = field(default_factory=dict)
    workspace_inputs: WorkspaceInputsState | None = None
    provider_key_status: ProviderKeyStatus | None = None
    repo_status: dict[str, object] = field(default_factory=dict)
    run_target: dict[str, object] = field(default_factory=dict)

    @classmethod
    def from_wire(cls, payload: object) -> ProjectReadiness:
        mapping = _require_mapping(payload, label="project readiness")
        blockers_payload = _optional_array(mapping, "blockers")
        recommended_actions_payload = _optional_array(mapping, "recommended_actions")
        return cls(
            project_id=_optional_string(mapping, "project_id"),
            state=_optional_string(mapping, "state"),
            blockers=[str(item) for item in blockers_payload if isinstance(item, str)],
            recommended_actions=[
                RecommendedAction.from_wire(item) for item in recommended_actions_payload
            ],
            entitlement=_optional_object_dict(mapping.get("entitlement")),
            capabilities=_optional_object_dict(mapping.get("capabilities")),
            workspace_inputs=(
                WorkspaceInputsState.from_wire(mapping.get("workspace_inputs"))
                if mapping.get("workspace_inputs") is not None
                else None
            ),
            provider_key_status=(
                ProviderKeyStatus.from_wire(mapping.get("provider_key_status"))
                if mapping.get("provider_key_status") is not None
                else None
            ),
            repo_status=_optional_object_dict(mapping.get("repo_status")),
            run_target=_optional_object_dict(mapping.get("run_target")),
        )


class SmrProjectSetupStatus(StrEnum):
    NOT_STARTED = "not_started"
    PREPARING = "preparing"
    BLOCKED = "blocked"
    READY = "ready"


@dataclass(frozen=True)
class SmrProjectSetupReason:
    code: str
    message: str

    @classmethod
    def from_wire(cls, payload: object) -> SmrProjectSetupReason:
        mapping = _require_mapping(payload, label="project setup reason")
        return cls(
            code=_require_string(mapping, "code", label="project setup reason.code"),
            message=_require_string(
                mapping,
                "message",
                label="project setup reason.message",
            ),
        )


@dataclass(frozen=True)
class SmrProjectSetup:
    project_id: str | None = None
    state: SmrProjectSetupStatus | None = None
    blockers: list[str] = field(default_factory=list)
    reasons: list[SmrProjectSetupReason] = field(default_factory=list)
    recommended_actions: list[RecommendedAction] = field(default_factory=list)
    onboarding_state: dict[str, object] = field(default_factory=dict)
    workspace_inputs: WorkspaceInputsState | None = None
    repo_status: dict[str, object] = field(default_factory=dict)
    run_target: dict[str, object] = field(default_factory=dict)

    @classmethod
    def from_wire(cls, payload: object) -> SmrProjectSetup:
        mapping = _require_mapping(payload, label="project setup")
        state_value = _optional_string(mapping, "state")
        state = None if state_value is None else SmrProjectSetupStatus(state_value)
        blockers_payload = _optional_array(mapping, "blockers")
        reasons_payload = _optional_array(mapping, "reasons")
        recommended_actions_payload = _optional_array(mapping, "recommended_actions")
        workspace_inputs_payload = mapping.get("workspace_inputs")
        return cls(
            project_id=_optional_string(mapping, "project_id"),
            state=state,
            blockers=[str(item) for item in blockers_payload if isinstance(item, str)],
            reasons=[SmrProjectSetupReason.from_wire(item) for item in reasons_payload],
            recommended_actions=[
                RecommendedAction.from_wire(item) for item in recommended_actions_payload
            ],
            onboarding_state=_optional_object_dict(mapping.get("onboarding_state")),
            workspace_inputs=(
                WorkspaceInputsState.from_wire(workspace_inputs_payload)
                if workspace_inputs_payload is not None
                else None
            ),
            repo_status=_optional_object_dict(mapping.get("repo_status")),
            run_target=_optional_object_dict(mapping.get("run_target")),
        )


@dataclass(frozen=True)
class SmrLaunchPreflightBlocker:
    stage: str
    http_status: int
    error_code: str | None = None
    message: str | None = None
    detail: object | None = None

    @classmethod
    def from_wire(cls, payload: object) -> SmrLaunchPreflightBlocker:
        mapping = _require_mapping(payload, label="launch preflight blocker")
        return cls(
            stage=_require_string(mapping, "stage", label="launch preflight blocker.stage"),
            http_status=_int_value(mapping, "http_status"),
            error_code=_optional_string(mapping, "error_code"),
            message=_optional_string(mapping, "message"),
            detail=mapping.get("detail"),
        )


@dataclass(frozen=True)
class SmrResolvedProfile:
    """A single platform-resolved actor profile (read-only, hosted launches)."""

    role: SmrActorType
    profile_id: str
    model: str
    agent_harness: SmrAgentKind | None = None
    agent_kind: SmrAgentKind | None = None

    @classmethod
    def from_wire(
        cls, *, role: SmrActorType, profile_id: str, snapshot: object
    ) -> SmrResolvedProfile | None:
        if not isinstance(snapshot, Mapping):
            return None
        return cls(
            role=role,
            profile_id=profile_id,
            model=str(snapshot.get("model") or ""),
            agent_harness=coerce_smr_agent_kind(
                snapshot.get("agent_harness") or snapshot.get("agent_kind"),
                field_name="resolved_profiles.agent_harness",
            ),
            agent_kind=coerce_smr_agent_kind(
                snapshot.get("agent_kind"),
                field_name="resolved_profiles.agent_kind",
            ),
        )


@dataclass(frozen=True)
class SmrResolvedActorProfiles:
    """Platform-resolved actor execution surfaced on the preflight response.

    The platform — not the customer — chooses these on hosted launches.
    """

    orchestrator: SmrResolvedProfile
    workers: tuple[SmrResolvedProfile, ...] = field(default_factory=tuple)
    default_worker_profile_id: str | None = None

    @classmethod
    def from_wire(cls, payload: object) -> SmrResolvedActorProfiles | None:
        if not isinstance(payload, Mapping):
            return None
        resolved = payload.get("resolved_profiles")
        orchestrator_id = payload.get("orchestrator_profile_id")
        if not isinstance(resolved, Mapping) or not orchestrator_id:
            return None
        orchestrator = SmrResolvedProfile.from_wire(
            role=SmrActorType.ORCHESTRATOR,
            profile_id=str(orchestrator_id),
            snapshot=resolved.get(str(orchestrator_id)),
        )
        if orchestrator is None:
            return None
        worker_ids = payload.get("worker_profile_ids")
        workers: list[SmrResolvedProfile] = []
        if isinstance(worker_ids, (list, tuple)):
            for worker_id in worker_ids:
                worker = SmrResolvedProfile.from_wire(
                    role=SmrActorType.WORKER,
                    profile_id=str(worker_id),
                    snapshot=resolved.get(str(worker_id)),
                )
                if worker is not None:
                    workers.append(worker)
        default_worker = payload.get("default_worker_profile_id")
        return cls(
            orchestrator=orchestrator,
            workers=tuple(workers),
            default_worker_profile_id=(str(default_worker) if default_worker else None),
        )


@dataclass(frozen=True)
class SmrLaunchPreflight:
    project_id: str | None = None
    project_alias: str | None = None
    project_kind: str | None = None
    clear_to_trigger: bool | None = None
    checked: list[str] = field(default_factory=list)
    blockers: list[SmrLaunchPreflightBlocker] = field(default_factory=list)
    preferred_lane: str | None = None
    resolved_lane: str | None = None
    resolution_reason: str | None = None
    network_topology: SmrNetworkTopology | None = None
    network_surfaces: dict[str, object] = field(default_factory=dict)
    effective_plan: str | None = None
    using_synth_free_mode: bool | None = None
    compute_pool_payload: dict[str, object] = field(default_factory=dict)
    providers: tuple[ProviderBinding, ...] = field(default_factory=tuple)
    capabilities: frozenset[ActorResourceCapability] = field(default_factory=frozenset)
    required_capabilities: frozenset[ActorResourceCapability] = field(default_factory=frozenset)
    limit: UsageLimit | None = None
    resolved_actor_profiles: SmrResolvedActorProfiles | None = None
    launch_mode: str | None = None
    dev_environment: dict[str, object] = field(default_factory=dict)

    @classmethod
    def from_wire(cls, payload: object) -> SmrLaunchPreflight:
        mapping = _require_mapping(payload, label="launch preflight")
        checked_payload = _optional_array(mapping, "checked")
        blockers_payload = _optional_array(mapping, "blockers")
        return cls(
            project_id=_optional_string(mapping, "project_id"),
            project_alias=_optional_string(mapping, "project_alias"),
            project_kind=_optional_string(mapping, "project_kind"),
            clear_to_trigger=_optional_bool(mapping, "clear_to_trigger"),
            checked=[str(item) for item in checked_payload if isinstance(item, str)],
            blockers=[SmrLaunchPreflightBlocker.from_wire(item) for item in blockers_payload],
            preferred_lane=_optional_string(mapping, "preferred_lane"),
            resolved_lane=_optional_string(mapping, "resolved_lane"),
            resolution_reason=_optional_string(mapping, "resolution_reason"),
            network_topology=coerce_smr_network_topology(
                _optional_string(mapping, "network_topology"),
                field_name="launch_preflight.network_topology",
            ),
            network_surfaces=_optional_object_dict(mapping.get("network_surfaces")),
            effective_plan=_optional_string(mapping, "effective_plan"),
            using_synth_free_mode=_optional_bool(mapping, "using_synth_free_mode"),
            compute_pool_payload=_optional_object_dict(mapping.get("compute_pool_payload")),
            providers=(
                coerce_provider_bindings(
                    cast(Any, mapping.get("providers")),
                    field_name="providers",
                )
                if mapping.get("providers") is not None
                else ()
            ),
            capabilities=frozenset(
                ActorResourceCapability(str(item))
                for item in _optional_array(mapping, "capabilities")
            ),
            required_capabilities=frozenset(
                ActorResourceCapability(str(item))
                for item in _optional_array(mapping, "required_capabilities")
            ),
            limit=coerce_usage_limit(
                cast(Any, mapping.get("limit")),
                field_name="limit",
            ),
            resolved_actor_profiles=SmrResolvedActorProfiles.from_wire(
                mapping.get("resolved_actor_profiles")
            ),
            launch_mode=_optional_string(mapping, "launch_mode"),
            dev_environment=_optional_object_dict(mapping.get("dev_environment")),
        )


@dataclass(frozen=True)
class SmrAgentProfileBindings:
    orchestrator_profile_id: str
    default_worker_profile_id: str
    worker_profile_ids: list[str] = field(default_factory=list)

    def to_wire(self) -> dict[str, object]:
        worker_profile_ids = list(self.worker_profile_ids)
        if not worker_profile_ids:
            worker_profile_ids = [self.default_worker_profile_id]
        return {
            "orchestrator_profile_id": self.orchestrator_profile_id,
            "default_worker_profile_id": self.default_worker_profile_id,
            "worker_profile_ids": worker_profile_ids,
        }


@dataclass(frozen=True)
class SmrRunnableProjectRequest:
    name: str
    timezone: str
    pool_id: str
    runtime_kind: SmrRuntimeKind
    environment_kind: SmrEnvironmentKind
    agent_profiles: SmrAgentProfileBindings
    runtime_artifact_release_id: str | None = None
    worker_profile_ids: list[str] = field(default_factory=list)
    actor_profile_id: str | None = None
    actor_model_assignments: list[SmrActorModelAssignment] = field(default_factory=list)
    budgets: dict[str, object] = field(default_factory=dict)
    key_policy: dict[str, object] = field(default_factory=dict)
    execution_policy: dict[str, object] = field(default_factory=dict)
    research: dict[str, object] = field(default_factory=dict)
    scenario: str | None = None
    notes: str | None = None
    retention_policy: dict[str, object] = field(default_factory=dict)
    metered_infra: dict[str, object] = field(default_factory=dict)
    schedule: dict[str, object] = field(default_factory=dict)
    integrations: dict[str, object] = field(default_factory=dict)
    synth_ai: dict[str, object] = field(default_factory=dict)
    policy: dict[str, object] = field(default_factory=dict)
    trial_matrix: dict[str, object] = field(default_factory=dict)

    @classmethod
    def from_wire(cls, payload: object) -> SmrRunnableProjectRequest:
        mapping = _require_mapping(payload, label="runnable project request")
        agent_profiles_payload = _require_mapping(
            mapping.get("agent_profiles"),
            label="runnable project request.agent_profiles",
        )
        worker_profile_ids = _string_list(
            agent_profiles_payload.get("worker_profile_ids"),
            label="runnable project request.agent_profiles.worker_profile_ids",
        )
        return cls(
            name=_require_string(mapping, "name", label="runnable project request.name"),
            timezone=_require_string(
                mapping,
                "timezone",
                label="runnable project request.timezone",
            ),
            pool_id=_require_string(
                mapping,
                "pool_id",
                label="runnable project request.pool_id",
            ),
            runtime_kind=coerce_smr_runtime_kind(
                _require_string(
                    mapping,
                    "runtime_kind",
                    label="runnable project request.runtime_kind",
                ),
                field_name="runtime_kind",
            )
            or SmrRuntimeKind.SANDBOX_AGENT,
            environment_kind=coerce_smr_environment_kind(
                _require_string(
                    mapping,
                    "environment_kind",
                    label="runnable project request.environment_kind",
                ),
                field_name="environment_kind",
            )
            or SmrEnvironmentKind.HARBOR,
            agent_profiles=SmrAgentProfileBindings(
                orchestrator_profile_id=_require_string(
                    agent_profiles_payload,
                    "orchestrator_profile_id",
                    label="runnable project request.agent_profiles.orchestrator_profile_id",
                ),
                default_worker_profile_id=_require_string(
                    agent_profiles_payload,
                    "default_worker_profile_id",
                    label="runnable project request.agent_profiles.default_worker_profile_id",
                ),
                worker_profile_ids=worker_profile_ids,
            ),
            runtime_artifact_release_id=_optional_string(
                mapping, "runtime_artifact_release_id"
            ),
            worker_profile_ids=worker_profile_ids,
            actor_profile_id=_optional_string(mapping, "actor_profile_id"),
            actor_model_assignments=normalize_actor_model_assignments(
                mapping.get("actor_model_assignments"),
                field_name="actor_model_assignments",
            ),
            budgets=_optional_object_dict(mapping.get("budgets")),
            key_policy=_optional_object_dict(mapping.get("key_policy")),
            execution_policy=_optional_object_dict(mapping.get("execution_policy")),
            research=_optional_object_dict(mapping.get("research")),
            scenario=_optional_string(mapping, "scenario"),
            notes=_optional_string(mapping, "notes"),
            retention_policy=_optional_object_dict(mapping.get("retention_policy")),
            metered_infra=_optional_object_dict(mapping.get("metered_infra")),
            schedule=_optional_object_dict(mapping.get("schedule")),
            integrations=_optional_object_dict(mapping.get("integrations")),
            synth_ai=_optional_object_dict(mapping.get("synth_ai")),
            policy=_optional_object_dict(mapping.get("policy")),
            trial_matrix=_optional_object_dict(mapping.get("trial_matrix")),
        )

    def to_wire(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "name": self.name,
            "timezone": self.timezone,
            "pool_id": self.pool_id,
            "runtime_kind": self.runtime_kind.value,
            "environment_kind": self.environment_kind.value,
            "orchestrator_profile_id": self.agent_profiles.orchestrator_profile_id,
            "default_worker_profile_id": self.agent_profiles.default_worker_profile_id,
            "worker_profile_ids": list(
                self.agent_profiles.worker_profile_ids or self.worker_profile_ids
            ),
            "budgets": dict(self.budgets),
            "key_policy": dict(self.key_policy),
            "execution_policy": dict(self.execution_policy),
            "research": dict(self.research),
            "retention_policy": dict(self.retention_policy),
            "metered_infra": dict(self.metered_infra),
            "schedule": dict(self.schedule),
            "integrations": dict(self.integrations),
            "synth_ai": dict(self.synth_ai),
            "policy": dict(self.policy),
            "trial_matrix": dict(self.trial_matrix),
        }
        if self.actor_model_assignments:
            payload["actor_model_assignments"] = [
                item.as_payload() for item in self.actor_model_assignments
            ]
        if self.actor_profile_id is not None:
            payload["actor_profile_id"] = self.actor_profile_id
        if self.runtime_artifact_release_id is not None:
            payload["runtime_artifact_release_id"] = self.runtime_artifact_release_id
        if self.scenario is not None:
            payload["scenario"] = self.scenario
        if self.notes is not None:
            payload["notes"] = self.notes
        return payload


@dataclass(frozen=True)
class RunProgress:
    state: str | None = None
    phase: str | None = None
    stalled_reason: str | None = None
    last_progress_at: str | None = None
    blocked_task_count: int | None = None
    pending_approval_ids: list[str] = field(default_factory=list)
    pending_question_ids: list[str] = field(default_factory=list)
    recent_artifact_ids: list[str] = field(default_factory=list)
    recent_event_summary: list[dict[str, object]] = field(default_factory=list)
    task_progress: list[dict[str, object]] = field(default_factory=list)
    worker_progress_unavailable: dict[str, object] | None = None
    recommended_actions: list[RecommendedAction] = field(default_factory=list)

    @classmethod
    def from_wire(cls, payload: object) -> RunProgress:
        mapping = _require_mapping(payload, label="run progress")
        recommended_actions_payload = _optional_array(mapping, "recommended_actions")
        pending_approval_ids = _optional_array(mapping, "pending_approval_ids")
        pending_question_ids = _optional_array(mapping, "pending_question_ids")
        recent_artifact_ids = _optional_array(mapping, "recent_artifact_ids")
        recent_event_summary = _optional_array(mapping, "recent_event_summary")
        task_progress = _optional_array(mapping, "task_progress")
        return cls(
            state=_optional_string(mapping, "state"),
            phase=_optional_string(mapping, "phase"),
            stalled_reason=_optional_string(mapping, "stalled_reason"),
            last_progress_at=_optional_string(mapping, "last_progress_at"),
            blocked_task_count=_optional_int(mapping, "blocked_task_count"),
            pending_approval_ids=[
                str(item) for item in pending_approval_ids if isinstance(item, str)
            ],
            pending_question_ids=[
                str(item) for item in pending_question_ids if isinstance(item, str)
            ],
            recent_artifact_ids=[
                str(item) for item in recent_artifact_ids if isinstance(item, str)
            ],
            recent_event_summary=[
                _object_dict(item) for item in recent_event_summary if isinstance(item, Mapping)
            ],
            task_progress=[
                _object_dict(item) for item in task_progress if isinstance(item, Mapping)
            ],
            worker_progress_unavailable=(
                _object_dict(mapping["worker_progress_unavailable"])
                if isinstance(mapping.get("worker_progress_unavailable"), Mapping)
                else None
            ),
            recommended_actions=[
                RecommendedAction.from_wire(item) for item in recommended_actions_payload
            ],
        )


class ParentObjectiveKind(StrEnum):
    OPEN_ENDED_QUESTION = "open_ended_question"
    DIRECTED_EFFORT_OUTCOME = "directed_effort_outcome"


class ParentObjectiveEvaluationState(StrEnum):
    ACTIVE = "active"
    REVIEW_PENDING = "review_pending"
    NEEDS_REVISION = "needs_revision"
    SATISFIED = "satisfied"
    PARTIAL = "partial"
    FAILED = "failed"
    MAX_ITERATIONS_REACHED = "max_iterations_reached"
    INTERRUPTED = "interrupted"
    WITHDRAWN = "withdrawn"


@dataclass(frozen=True)
class PrimaryParentRef:
    kind: ParentObjectiveKind
    id: str

    @classmethod
    def from_wire(cls, payload: object) -> PrimaryParentRef:
        mapping = _require_mapping(payload, label="primary parent ref")
        kind = ParentObjectiveKind(
            _require_string(mapping, "kind", label="primary parent ref.kind")
        )
        return cls(
            kind=kind,
            id=_require_string(mapping, "id", label="primary parent ref.id"),
        )


@dataclass(frozen=True)
class OpenEndedQuestion:
    open_ended_question_id: str
    project_id: str
    run_id: str | None = None
    title: str | None = None
    question_text: str | None = None
    evaluation_state: ParentObjectiveEvaluationState | None = None

    @classmethod
    def from_wire(cls, payload: object) -> OpenEndedQuestion:
        mapping = _require_mapping(payload, label="open ended question")
        raw_state = _optional_string(mapping, "evaluation_state")
        return cls(
            open_ended_question_id=_require_string(
                mapping,
                "open_ended_question_id",
                label="open ended question id",
            ),
            project_id=_require_string(mapping, "project_id", label="project id"),
            run_id=_optional_string(mapping, "run_id"),
            title=_optional_string(mapping, "title"),
            question_text=_optional_string(mapping, "question_text"),
            evaluation_state=(ParentObjectiveEvaluationState(raw_state) if raw_state else None),
        )


@dataclass(frozen=True)
class DirectedEffortOutcome:
    directed_effort_outcome_id: str
    project_id: str
    run_id: str | None = None
    title: str | None = None
    outcome_text: str | None = None
    evaluation_state: ParentObjectiveEvaluationState | None = None

    @classmethod
    def from_wire(cls, payload: object) -> DirectedEffortOutcome:
        mapping = _require_mapping(payload, label="directed effort outcome")
        raw_state = _optional_string(mapping, "evaluation_state")
        return cls(
            directed_effort_outcome_id=_require_string(
                mapping,
                "directed_effort_outcome_id",
                label="directed effort outcome id",
            ),
            project_id=_require_string(mapping, "project_id", label="project id"),
            run_id=_optional_string(mapping, "run_id"),
            title=_optional_string(mapping, "title"),
            outcome_text=_optional_string(mapping, "outcome_text"),
            evaluation_state=(ParentObjectiveEvaluationState(raw_state) if raw_state else None),
        )


@dataclass(frozen=True)
class MilestoneProgress:
    milestone_id: str
    project_id: str
    run_id: str | None = None
    parent_kind: str | None = None
    parent_id: str | None = None
    milestone_kind: str | None = None
    title: str | None = None
    objective: str | None = None
    state: str | None = None
    validation_status: str | None = None
    validation_summary: str | None = None
    acceptance_criteria: list[str] = field(default_factory=list)
    evidence_artifact_ids: list[str] = field(default_factory=list)
    evidence_entry_ids: list[str] = field(default_factory=list)
    position: int | None = None
    revision: int | None = None
    metadata: dict[str, object] = field(default_factory=dict)
    created_at: str | None = None
    updated_at: str | None = None

    @classmethod
    def from_wire(cls, payload: object) -> MilestoneProgress:
        mapping = _require_mapping(payload, label="milestone progress")
        return cls(
            milestone_id=_require_string(
                mapping,
                "milestone_id",
                label="milestone progress.milestone_id",
            ),
            project_id=_require_string(
                mapping,
                "project_id",
                label="milestone progress.project_id",
            ),
            run_id=_optional_string(mapping, "run_id"),
            parent_kind=_optional_string(mapping, "parent_kind"),
            parent_id=_optional_string(mapping, "parent_id"),
            milestone_kind=_optional_string(mapping, "milestone_kind"),
            title=_optional_string(mapping, "title"),
            objective=_optional_string(mapping, "objective"),
            state=_optional_string(mapping, "state"),
            validation_status=_optional_string(mapping, "validation_status"),
            validation_summary=_optional_string(mapping, "validation_summary"),
            acceptance_criteria=_string_list(
                mapping.get("acceptance_criteria"),
                label="milestone progress.acceptance_criteria",
            ),
            evidence_artifact_ids=_string_list(
                mapping.get("evidence_artifact_ids"),
                label="milestone progress.evidence_artifact_ids",
            ),
            evidence_entry_ids=_string_list(
                mapping.get("evidence_entry_ids"),
                label="milestone progress.evidence_entry_ids",
            ),
            position=_optional_int(mapping, "position"),
            revision=_optional_int(mapping, "revision"),
            metadata=_optional_object_dict(mapping.get("metadata")),
            created_at=_optional_string(mapping, "created_at"),
            updated_at=_optional_string(mapping, "updated_at"),
        )


@dataclass(frozen=True)
class ExperimentProgress:
    experiment_id: str
    project_id: str
    run_id: str | None = None
    parent_experiment_id: str | None = None
    milestone_id: str | None = None
    title: str | None = None
    hypothesis: str | None = None
    status: str | None = None
    summary: str | None = None
    recommendation: str | None = None
    disposition: str | None = None
    repos: list[str] = field(default_factory=list)
    branches: list[str] = field(default_factory=list)
    metrics_before: dict[str, object] = field(default_factory=dict)
    metrics_after: dict[str, object] = field(default_factory=dict)
    comparison_metadata: dict[str, object] = field(default_factory=dict)
    linked_artifact_ids: list[str] = field(default_factory=list)
    linked_entry_ids: list[str] = field(default_factory=list)
    revision: int | None = None
    metadata: dict[str, object] = field(default_factory=dict)
    created_at: str | None = None
    updated_at: str | None = None

    @classmethod
    def from_wire(cls, payload: object) -> ExperimentProgress:
        mapping = _require_mapping(payload, label="experiment progress")
        return cls(
            experiment_id=_require_string(
                mapping,
                "experiment_id",
                label="experiment progress.experiment_id",
            ),
            project_id=_require_string(
                mapping,
                "project_id",
                label="experiment progress.project_id",
            ),
            run_id=_optional_string(mapping, "run_id"),
            parent_experiment_id=_optional_string(mapping, "parent_experiment_id"),
            milestone_id=_optional_string(mapping, "milestone_id"),
            title=_optional_string(mapping, "title"),
            hypothesis=_optional_string(mapping, "hypothesis"),
            status=_optional_string(mapping, "status"),
            summary=_optional_string(mapping, "summary"),
            recommendation=_optional_string(mapping, "recommendation"),
            disposition=_optional_string(mapping, "disposition"),
            repos=_string_list(mapping.get("repos"), label="experiment progress.repos"),
            branches=_string_list(
                mapping.get("branches"),
                label="experiment progress.branches",
            ),
            metrics_before=_optional_object_dict(mapping.get("metrics_before")),
            metrics_after=_optional_object_dict(mapping.get("metrics_after")),
            comparison_metadata=_optional_object_dict(mapping.get("comparison_metadata")),
            linked_artifact_ids=_string_list(
                mapping.get("linked_artifact_ids"),
                label="experiment progress.linked_artifact_ids",
            ),
            linked_entry_ids=_string_list(
                mapping.get("linked_entry_ids"),
                label="experiment progress.linked_entry_ids",
            ),
            revision=_optional_int(mapping, "revision"),
            metadata=_optional_object_dict(mapping.get("metadata")),
            created_at=_optional_string(mapping, "created_at"),
            updated_at=_optional_string(mapping, "updated_at"),
        )


@dataclass(frozen=True)
class SemanticProgressSnapshot:
    project_id: str
    run_id: str | None = None
    primary_parent: PrimaryParentRef | None = None
    run_progress: RunProgress | None = None
    open_ended_questions: list[OpenEndedQuestion] = field(default_factory=list)
    directed_effort_outcomes: list[DirectedEffortOutcome] = field(default_factory=list)
    milestones: list[MilestoneProgress] = field(default_factory=list)
    primary_parent_milestones: list[MilestoneProgress] = field(default_factory=list)
    experiments: list[ExperimentProgress] = field(default_factory=list)

    @classmethod
    def from_wire(cls, payload: object) -> SemanticProgressSnapshot:
        mapping = _require_mapping(payload, label="semantic progress snapshot")
        return cls(
            project_id=_require_string(
                mapping,
                "project_id",
                label="semantic progress snapshot.project_id",
            ),
            run_id=_optional_string(mapping, "run_id"),
            primary_parent=(
                PrimaryParentRef.from_wire(mapping.get("primary_parent"))
                if mapping.get("primary_parent") is not None
                else None
            ),
            run_progress=(
                RunProgress.from_wire(mapping.get("run_progress"))
                if mapping.get("run_progress") is not None
                else None
            ),
            open_ended_questions=[
                OpenEndedQuestion.from_wire(item)
                for item in _optional_array(mapping, "open_ended_questions")
            ],
            directed_effort_outcomes=[
                DirectedEffortOutcome.from_wire(item)
                for item in _optional_array(mapping, "directed_effort_outcomes")
            ],
            milestones=[
                MilestoneProgress.from_wire(item) for item in _optional_array(mapping, "milestones")
            ],
            primary_parent_milestones=[
                MilestoneProgress.from_wire(item)
                for item in _optional_array(mapping, "primary_parent_milestones")
            ],
            experiments=[
                ExperimentProgress.from_wire(item)
                for item in _optional_array(mapping, "experiments")
            ],
        )


ProjectSetupAuthorityStatus = SmrProjectSetupStatus
ProjectSetupAuthorityReason = SmrProjectSetupReason
ProjectSetupAuthority = SmrProjectSetup
LaunchPreflightBlocker = SmrLaunchPreflightBlocker
LaunchPreflight = SmrLaunchPreflight


__all__ = [
    "CredentialRef",
    "DevEnvironment",
    "DevEnvironmentAttach",
    "DevEnvironmentCollection",
    "DevEnvironmentMaterializationQueue",
    "DevEnvironmentMaterializationWorkItem",
    "DevEnvironmentPreflight",
    "DevEnvironmentTopology",
    "DevEnvironmentUsage",
    "DirectedEffortOutcome",
    "ExperimentProgress",
    "ExternalRepository",
    "InlineExternalRepositoryBinding",
    "LaunchPreflight",
    "LaunchPreflightBlocker",
    "MilestoneProgress",
    "OpenEndedQuestion",
    "ParentObjectiveEvaluationState",
    "ParentObjectiveKind",
    "PrimaryParentRef",
    "ProjectSetupAuthority",
    "ProjectSetupAuthorityReason",
    "ProjectSetupAuthorityStatus",
    "ProviderKeyStatus",
    "ProjectCodeSource",
    "ProjectDataPoolObject",
    "ProjectDataPoolUploadResult",
    "ProjectLaunchProfile",
    "RecommendedAction",
    "ResourceUploadResult",
    "RunArtifact",
    "RunArtifactManifest",
    "RunProgress",
    "RunCredentialBinding",
    "RunFileMount",
    "RunOutputFile",
    "RunRepositoryMount",
    "RunResourceBindings",
    "SemanticProgressSnapshot",
    "StoredFile",
    "WorkspaceFileInput",
    "WorkspaceInputsState",
    "WorkspaceSourceRepo",
    "WorkspaceUploadResult",
]
