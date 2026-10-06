"""Consumer references and native data-pool read models.

# See: testing-resources/specifications/sdk/forge_resource_reads.md
Backend SMR owns resource scope, lifecycle, descriptors and file custody.
"""

from __future__ import annotations

from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, JsonValue, field_validator

from synth_ai.sdk.research.contracts.traces import validate_sha256_digest
from synth_ai.sdk.research.contracts.types import StoredFile


class _ResourceReadModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class DatasetRevisionReadReference(_ResourceReadModel):
    """Caller-selected exact identity; a reference grants no authority."""

    org_id: str = Field(min_length=1, max_length=255)
    project_id: str = Field(min_length=1, max_length=255)
    data_binding_id: UUID
    dataset_revision_id: UUID
    revision_digest: str

    _revision_digest = field_validator("revision_digest")(validate_sha256_digest)


class DataPoolReadReference(_ResourceReadModel):
    """Caller-selected organization/project/pool scope for current reads."""

    org_id: str = Field(min_length=1, max_length=255)
    project_id: str = Field(min_length=1, max_length=255)
    pool_id: str = Field(min_length=1, max_length=255)


class ProjectDataPoolDescriptor(_ResourceReadModel):
    """Native descriptor; upload-batch counts and access policy stay opaque."""

    pool_id: str
    name: str
    status: str
    manifest_id: str | None = None
    object_prefix: str | None = None
    access_policy: dict[str, JsonValue] = Field(default_factory=dict)
    metadata: dict[str, JsonValue] = Field(default_factory=dict)
    object_count: int | None = None
    bytes_uploaded: int | None = None


class ProjectDataPoolInventory(_ResourceReadModel):
    """The native complete current declaration set, never an admission receipt."""

    project_id: str
    descriptor: ProjectDataPoolDescriptor
    inventory_digest: str
    file_count: int = Field(ge=0, le=10000, strict=True)
    size_bytes: int = Field(ge=0, strict=True)
    files: tuple[StoredFile, ...]

    _inventory_digest = field_validator("inventory_digest")(validate_sha256_digest)

    @field_validator("files", mode="before")
    @classmethod
    def parse_native_files(cls, value: object) -> tuple[StoredFile, ...]:
        """Preserve the existing strict native file wire boundary.

        # See: testing-resources/specifications/sdk/forge_resource_reads.md
        Pydantic dataclass coercion must not reinterpret byte counts or IDs.
        """
        if not isinstance(value, (list, tuple)) or len(value) > 10000:
            raise ValueError("Native data-pool files must be a complete bounded array")
        return tuple(
            item if isinstance(item, StoredFile) else StoredFile.from_wire(item) for item in value
        )


__all__ = [
    "DataPoolReadReference",
    "DatasetRevisionReadReference",
    "ProjectDataPoolDescriptor",
    "ProjectDataPoolInventory",
]
