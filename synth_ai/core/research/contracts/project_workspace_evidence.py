"""Typed immutable Git and workspace evidence for Project Computers."""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from synth_ai.core.research.contracts.traces import validate_sha256_digest


class _FrozenContract(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    @classmethod
    def from_wire(cls, value: object):
        return cls.model_validate(value)

    def to_wire(self) -> dict[str, object]:
        return self.model_dump(mode="json", exclude_none=True)


def validate_git_sha(value: str) -> str:
    if len(value) != 40 or value != value.lower():
        raise ValueError("Git SHA must be exactly 40 lowercase hexadecimal characters")
    try:
        int(value, 16)
    except ValueError as error:
        raise ValueError("Git SHA contains non-hexadecimal characters") from error
    return value


class FactoryProjectMembershipReceiptRef(_FrozenContract):
    authority: Literal["factory_project_control_plane"] = "factory_project_control_plane"
    factory_project_membership_id: str = Field(min_length=1, max_length=255)
    org_id: str = Field(min_length=1, max_length=255)
    factory_id: str = Field(min_length=1, max_length=255)
    project_id: str = Field(min_length=1, max_length=255)
    role: Literal["canonical", "auxiliary"]
    status: Literal["active"] = "active"
    generation: int = Field(ge=1)
    authority_receipt_digest: str

    _authority_receipt_digest = field_validator("authority_receipt_digest")(
        validate_sha256_digest
    )


class ProjectInternalGitReceipt(_FrozenContract):
    schema_version: Literal["synth.project-internal-git-receipt.v1"] = (
        "synth.project-internal-git-receipt.v1"
    )
    receipt_id: str = Field(min_length=1, max_length=255)
    org_id: str = Field(min_length=1, max_length=255)
    factory_id: str = Field(min_length=1, max_length=255)
    project_id: str = Field(min_length=1, max_length=255)
    factory_project_membership: FactoryProjectMembershipReceiptRef
    repository_registration_id: str = Field(min_length=1, max_length=255)
    revision_admission_id: str = Field(min_length=1, max_length=255)
    internal_repo_id: str = Field(min_length=1, max_length=255)
    internal_branch: str = Field(min_length=1, max_length=255)
    external_base_sha: str
    external_base_reachable: Literal[True] = True
    internal_head_sha: str
    internal_tree_sha: str
    tree_listing_digest: str
    tree_entry_count: int = Field(ge=0)
    receipt_digest: str

    _external_base_sha = field_validator("external_base_sha")(validate_git_sha)
    _internal_head_sha = field_validator("internal_head_sha")(validate_git_sha)
    _internal_tree_sha = field_validator("internal_tree_sha")(validate_git_sha)
    _tree_listing_digest = field_validator("tree_listing_digest")(validate_sha256_digest)
    _receipt_digest = field_validator("receipt_digest")(validate_sha256_digest)

    @model_validator(mode="after")
    def validate_membership_scope(self) -> ProjectInternalGitReceipt:
        membership = self.factory_project_membership
        if (self.org_id, self.factory_id, self.project_id) != (
            membership.org_id,
            membership.factory_id,
            membership.project_id,
        ):
            raise ValueError("internal Git receipt crossed its FactoryProject boundary")
        return self


class ProjectGitArchiveManifestReceipt(_FrozenContract):
    schema_version: Literal["synth.project-git-archive-manifest-receipt.v1"] = (
        "synth.project-git-archive-manifest-receipt.v1"
    )
    authority: Literal["internal_git_server"] = "internal_git_server"
    receipt_id: str = Field(min_length=1, max_length=255)
    org_id: str = Field(min_length=1, max_length=255)
    factory_id: str = Field(min_length=1, max_length=255)
    project_id: str = Field(min_length=1, max_length=255)
    internal_git_receipt_id: str = Field(min_length=1, max_length=255)
    internal_git_receipt_digest: str
    internal_head_sha: str
    internal_tree_sha: str
    tree_listing_digest: str
    tree_entry_count: int = Field(ge=0)
    archive_format: Literal["git-archive-tar.v1"] = "git-archive-tar.v1"
    archive_digest: str
    archive_size_bytes: int = Field(ge=1)
    receipt_digest: str

    _internal_git_receipt_digest = field_validator("internal_git_receipt_digest")(
        validate_sha256_digest
    )
    _internal_head_sha = field_validator("internal_head_sha")(validate_git_sha)
    _internal_tree_sha = field_validator("internal_tree_sha")(validate_git_sha)
    _tree_listing_digest = field_validator("tree_listing_digest")(validate_sha256_digest)
    _archive_digest = field_validator("archive_digest")(validate_sha256_digest)
    _receipt_digest = field_validator("receipt_digest")(validate_sha256_digest)


class ProjectWorkspaceSnapshotManifest(_FrozenContract):
    schema_version: Literal["synth.project-workspace-snapshot.v1"] = (
        "synth.project-workspace-snapshot.v1"
    )
    snapshot_manifest_id: str = Field(min_length=1, max_length=255)
    snapshot_ref_id: str = Field(min_length=1, max_length=255)
    org_id: str = Field(min_length=1, max_length=255)
    factory_id: str = Field(min_length=1, max_length=255)
    project_id: str = Field(min_length=1, max_length=255)
    internal_git_receipt: ProjectInternalGitReceipt
    git_archive_manifest: ProjectGitArchiveManifestReceipt
    archive_digest: str
    archive_size_bytes: int = Field(ge=1)
    snapshot_source: str = Field(min_length=1, max_length=128)
    manifest_digest: str

    _archive_digest = field_validator("archive_digest")(validate_sha256_digest)
    _manifest_digest = field_validator("manifest_digest")(validate_sha256_digest)

    @model_validator(mode="after")
    def validate_evidence_join(self) -> ProjectWorkspaceSnapshotManifest:
        receipt = self.internal_git_receipt
        archive = self.git_archive_manifest
        scope = (self.org_id, self.factory_id, self.project_id)
        if scope != (receipt.org_id, receipt.factory_id, receipt.project_id):
            raise ValueError("workspace snapshot and internal Git receipt scope drifted")
        if scope != (archive.org_id, archive.factory_id, archive.project_id):
            raise ValueError("workspace snapshot and archive receipt scope drifted")
        if (
            archive.internal_git_receipt_id != receipt.receipt_id
            or archive.internal_git_receipt_digest != receipt.receipt_digest
            or archive.internal_head_sha != receipt.internal_head_sha
            or archive.internal_tree_sha != receipt.internal_tree_sha
            or archive.tree_listing_digest != receipt.tree_listing_digest
            or archive.tree_entry_count != receipt.tree_entry_count
        ):
            raise ValueError("workspace archive does not join its internal Git receipt")
        if (
            self.archive_digest != archive.archive_digest
            or self.archive_size_bytes != archive.archive_size_bytes
        ):
            raise ValueError("workspace snapshot bytes do not match the archive receipt")
        return self


class ProjectComputerWorkspaceSnapshot(_FrozenContract):
    snapshot_ref_id: str = Field(min_length=1, max_length=255)
    org_id: str = Field(min_length=1, max_length=255)
    factory_id: str = Field(min_length=1, max_length=255)
    project_id: str = Field(min_length=1, max_length=255)
    run_id: str = Field(min_length=1, max_length=255)
    bucket: str = Field(min_length=1, max_length=255)
    archive_key: str = Field(min_length=1, max_length=2048)
    uri: str = Field(min_length=1, max_length=4096)
    archive_digest: str
    archive_size_bytes: int = Field(ge=1)
    source: str = Field(min_length=1, max_length=128)
    commit_sha: str
    manifest: ProjectWorkspaceSnapshotManifest

    _archive_digest = field_validator("archive_digest")(validate_sha256_digest)
    _commit_sha = field_validator("commit_sha")(validate_git_sha)

    @model_validator(mode="after")
    def validate_exact_snapshot(self) -> ProjectComputerWorkspaceSnapshot:
        manifest = self.manifest
        if self.uri != f"s3://{self.bucket}/{self.archive_key}":
            raise ValueError("workspace snapshot URI must match exact bucket/key")
        if (
            self.snapshot_ref_id != manifest.snapshot_ref_id
            or (self.org_id, self.factory_id, self.project_id)
            != (manifest.org_id, manifest.factory_id, manifest.project_id)
            or self.archive_digest != manifest.archive_digest
            or self.archive_size_bytes != manifest.archive_size_bytes
            or self.commit_sha != manifest.internal_git_receipt.internal_head_sha
        ):
            raise ValueError("workspace snapshot locator does not match its manifest")
        return self


class WorkspaceGitServerProvenance(_FrozenContract):
    authority: Literal["internal_git_server"] = "internal_git_server"
    org_id: str = Field(min_length=1, max_length=255)
    project_id: str = Field(min_length=1, max_length=255)
    repository_id: str = Field(min_length=1, max_length=255)
    org_slug: str = Field(min_length=1, max_length=255)
    project_slug: str = Field(min_length=1, max_length=255)
    branch: str = Field(min_length=1, max_length=255)
    observed_head_sha: str
    observed_tree_sha: str
    repository_generation_digest: str

    _observed_head_sha = field_validator("observed_head_sha")(validate_git_sha)
    _observed_tree_sha = field_validator("observed_tree_sha")(validate_git_sha)
    _repository_generation_digest = field_validator("repository_generation_digest")(
        validate_sha256_digest
    )


class WorkspacePushConfirmationReceipt(_FrozenContract):
    schema_version: Literal["synth.workspace-push-confirmation-receipt.v1"] = (
        "synth.workspace-push-confirmation-receipt.v1"
    )
    authority: Literal["project_workspace_control_plane"] = (
        "project_workspace_control_plane"
    )
    receipt_id: str = Field(min_length=1, max_length=255)
    org_id: str = Field(min_length=1, max_length=255)
    factory_id: str = Field(min_length=1, max_length=255)
    factory_project_membership_id: str = Field(min_length=1, max_length=255)
    project_id: str = Field(min_length=1, max_length=255)
    run_id: str = Field(min_length=1, max_length=255)
    workspace_archive_id: str = Field(min_length=1, max_length=255)
    commit_sha: str
    verified_internal_head_sha: str
    immutable_archive_uri: str = Field(min_length=1, max_length=2000)
    archive_digest: str
    archive_size_bytes: int = Field(ge=1)
    snapshot_manifest_digest: str
    git_server_provenance: WorkspaceGitServerProvenance
    confirmed_at: datetime
    receipt_digest: str

    _commit_sha = field_validator("commit_sha")(validate_git_sha)
    _verified_internal_head_sha = field_validator("verified_internal_head_sha")(
        validate_git_sha
    )
    _archive_digest = field_validator("archive_digest")(validate_sha256_digest)
    _snapshot_manifest_digest = field_validator("snapshot_manifest_digest")(
        validate_sha256_digest
    )
    _receipt_digest = field_validator("receipt_digest")(validate_sha256_digest)

    @model_validator(mode="after")
    def validate_exact_head(self) -> WorkspacePushConfirmationReceipt:
        provenance = self.git_server_provenance
        if (
            self.commit_sha != self.verified_internal_head_sha
            or provenance.observed_head_sha != self.verified_internal_head_sha
            or provenance.org_id != self.org_id
            or provenance.project_id != self.project_id
        ):
            raise ValueError("workspace confirmation scope or internal HEAD drifted")
        if self.immutable_archive_uri.startswith(("http://", "https://")):
            raise ValueError("workspace confirmation requires an immutable archive URI")
        return self


__all__ = [
    "FactoryProjectMembershipReceiptRef",
    "ProjectComputerWorkspaceSnapshot",
    "ProjectGitArchiveManifestReceipt",
    "ProjectInternalGitReceipt",
    "ProjectWorkspaceSnapshotManifest",
    "WorkspaceGitServerProvenance",
    "WorkspacePushConfirmationReceipt",
    "validate_git_sha",
]
