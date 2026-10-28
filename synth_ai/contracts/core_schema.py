"""Deterministic shared-reference contract export, independent of transport."""

from __future__ import annotations

import json

from synth_ai.sdk.core.artifact_types import (
    Artifact,
    ArtifactAlias,
    ArtifactAliasSet,
    ArtifactLabelsUpdate,
    ArtifactManifest,
    ArtifactObject,
    ArtifactUploadManifest,
    ArtifactUploadReceipt,
    ArtifactUploadRequest,
    ArtifactUploadStatus,
)
from synth_ai.sdk.core.types import (
    CloneToken,
    CloneTokenRequest,
    RepositoryCommit,
    RepositoryDiff,
    RepositoryRef,
    RepositorySnapshot,
)
from synth_ai.sdk.forge.contracts.operations_v2 import PublicWriteV2
from synth_ai.sdk.forge.contracts.projects import ProjectOperation
from synth_ai.sdk.forge.contracts.workspace_science import WorkspaceIntent
from synth_ai.sdk.forge.index_types import (
    ForgeIndexDraftNoticesResponse,
    ForgeIndexDraftRequest,
    ForgeIndexDraftResponse,
    ForgeIndexDraftUploadRequest,
    ForgeIndexDraftUploadResponse,
    ForgeIndexNoticePage,
)
from synth_ai.sdk.forge.types import (
    ProjectListResponse,
    ProjectMutationResponse,
    ProjectResponse,
    ScientificArtifactEvidenceResponse,
    ScientificReceiptResponse,
    ScientificRecordResponse,
)
from synth_ai.types.base import Page
from synth_ai.types.refs import Link, Producer, Ref
from synth_ai.types.response_values import response_schema

from .core import ALL_OPERATIONS
from .forge import ALL_OPERATIONS as FORGE_OPERATIONS
from .operations import PREFIX

MODELS = (
    ForgeIndexDraftNoticesResponse,
    ForgeIndexNoticePage,
    ForgeIndexDraftRequest,
    ForgeIndexDraftResponse,
    ForgeIndexDraftUploadRequest,
    ForgeIndexDraftUploadResponse,
    PublicWriteV2,
    WorkspaceIntent,
    ScientificRecordResponse,
    ScientificReceiptResponse,
    ScientificArtifactEvidenceResponse,
    ProjectOperation,
    ProjectListResponse,
    ProjectResponse,
    ProjectMutationResponse,
    Artifact,
    ArtifactAlias,
    ArtifactAliasSet,
    ArtifactLabelsUpdate,
    ArtifactManifest,
    ArtifactObject,
    ArtifactUploadManifest,
    ArtifactUploadRequest,
    ArtifactUploadStatus,
    ArtifactUploadReceipt,
    Page[Artifact],
    Ref,
    Producer,
    Link,
    RepositorySnapshot,
    RepositoryRef,
    RepositoryCommit,
    RepositoryDiff,
    CloneTokenRequest,
    CloneToken,
    Page[RepositoryRef],
    Page[Link],
)

ARCHIVE_PROTOCOL = {
    "format": "synth.git-raw-tar.v1",
    "media_type": "application/x-tar",
    "content_encoding": "identity",
    "request": {
        "commit": "full lowercase SHA-1",
        "digest_sha256": "exact canonical commit manifest digest",
    },
    "required_headers": {
        "X-Synth-Contract": "core.v1",
        "X-Synth-Archive-Format": "synth.git-raw-tar.v1",
        "X-Synth-Repo-Id": "selected public repo_id",
        "X-Synth-Commit-SHA": "selected commit SHA-1",
        "X-Synth-Commit-Manifest-SHA256": "selected Ref.digest_sha256",
        "X-Synth-Content-SHA256": "SHA-256 of complete uncompressed tar bytes",
        "X-Synth-Archive-Entries": "canonical tree entry count",
        "X-Synth-Archive-Bytes": "complete uncompressed tar byte count",
    },
    "maximum_bytes": 512 * 1024 * 1024,
    "maximum_entries": 100_000,
    "maximum_seconds": 900,
    "gitlinks": "empty directory with local PAX SYNTH.gitlink=<40 lowercase hex>",
    "publication": "exclusive directory rename only after EOF, byte hash and actual tree digest verification",
}


def export_contract() -> str:
    return (
        json.dumps(
            {
                "schema_version": "core.v1",
                "status": "sdk_first_pending_backend",
                "models": {model.__name__: response_schema(model) for model in MODELS},
                "archive_protocol": ARCHIVE_PROTOCOL,
                "repository_identity": "repo_ plus SHA-256 of the UTF-8 internal immutable repo identity; internal registry IDs are not public Ref IDs",
                "operations": [
                    {
                        "id": op.id,
                        "method": op.method,
                        "path": PREFIX + op.path,
                        "keyed": op.keyed,
                        "mutation": op.mutation,
                        "required_scope": op.required_scope,
                        "may_charge": op.may_charge,
                        "idempotent": op.idempotent,
                    }
                    for op in (*ALL_OPERATIONS, *FORGE_OPERATIONS)
                ],
            },
            sort_keys=True,
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    print(export_contract(), end="")
