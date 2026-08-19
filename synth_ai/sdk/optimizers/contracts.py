"""Typed public contracts for hosted optimizer model discovery and lineage."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class OptimizerContract(BaseModel):
    """Forward-compatible base for optimizer API payloads."""

    model_config = ConfigDict(extra="allow")


class SavedLoraStorage(OptimizerContract):
    backend: str
    bucket: str
    key: str
    version: str | None = None
    etag: str | None = None
    sha256: str | None = None
    size_bytes: int | None = None
    content_type: str


class SavedLoraLineage(OptimizerContract):
    optimizer_algorithm: str | None = None
    run_id: str | None = None
    attempt_id: str | None = None
    source_checkpoint_id: str | None = None
    provider_checkpoint_reference: str | None = None


class SavedLoraCheckpoint(OptimizerContract):
    schema_version: str = "saved_lora_checkpoint.v1"
    checkpoint_id: str
    org_id: str
    owner_user_id: str | None = None
    visibility: str
    name: str
    description: str = ""
    provider: str
    checkpoint_kind: str
    provider_checkpoint_reference: str | None = None
    optimizer_algorithm: str | None = None
    run_id: str | None = None
    attempt_id: str | None = None
    source_checkpoint_id: str | None = None
    base_model: str
    lora_rank: int | None = None
    step: int | None = None
    status: str
    storage: SavedLoraStorage
    lineage: SavedLoraLineage = Field(default_factory=SavedLoraLineage)
    tags: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: str | None = None
    updated_at: str | None = None
    archived_at: str | None = None


class SavedLoraCheckpointPage(OptimizerContract):
    items: list[SavedLoraCheckpoint]
    total: int
    limit: int
    offset: int


class OptimizerRunIdentity(OptimizerContract):
    run_id: str
    attempt_id: str | None = None
    optimizer_algorithm: str
    status: str


class OptimizerRunArtifact(OptimizerContract):
    artifact_id: str
    run_id: str
    artifact_name: str
    content_type: str | None = None
    size_bytes: int
    sha256: str | None = None
    storage_backend: str
    uri: str
    download_path: str
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: str | None = None
    updated_at: str | None = None


class OptimizerRunOutputs(OptimizerContract):
    """All automatically persisted outputs for one optimizer run."""

    schema_version: str = "optimizer.run_outputs.v1"
    run: OptimizerRunIdentity
    result: dict[str, Any] | None = None
    artifacts: list[OptimizerRunArtifact]
    model_checkpoints: list[SavedLoraCheckpoint]
    counts: dict[str, int]


class SavedLoraRunPage(OptimizerContract):
    schema_version: str = "saved_lora_checkpoint.run_page.v1"
    run: OptimizerRunIdentity
    items: list[SavedLoraCheckpoint]
    counts: dict[str, int]
    total: int
    limit: int
    offset: int


class HostedTrainingModel(OptimizerContract):
    model_id: str
    label: str
    provider: str
    provider_revision: str
    architecture: str
    max_context_length: int
    rank: dict[str, int]
    algorithms: dict[str, dict[str, Any]]


class HostedTrainingModelCatalog(OptimizerContract):
    catalog_revision: str
    live_preflight_required: bool
    models: list[HostedTrainingModel]
    total: int


__all__ = [
    "HostedTrainingModel",
    "HostedTrainingModelCatalog",
    "OptimizerRunIdentity",
    "OptimizerRunArtifact",
    "OptimizerRunOutputs",
    "SavedLoraCheckpoint",
    "SavedLoraCheckpointPage",
    "SavedLoraLineage",
    "SavedLoraRunPage",
    "SavedLoraStorage",
]
