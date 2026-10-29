"""``synth.menagerie.v1`` receipts the SDK reads (backend packages/menagerie/contracts.py)."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

ModelKind = Literal["base_model", "artifact", "checkpoint"]
LineageRelation = Literal["base", "parent", "merge", "quantization", "export"]
ServingStrength = Literal["unknown", "provider_reference", "provider_receipt"]
# Path-safe subset of the backend's deployment id grammar.
DEPLOYMENT_ID = r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$"


class MenagerieModel(BaseModel):
    model_config = ConfigDict(extra="ignore", frozen=True, hide_input_in_errors=True)


class ModelBinding(MenagerieModel):
    kind: ModelKind
    id: str


class ModelResolution(MenagerieModel):
    schema_version: Literal["synth.menagerie.v1"]
    canonical: ModelBinding
    resolved_from: str
    base_model: dict[str, Any]
    artifact: dict[str, Any] | None
    checkpoint: dict[str, Any] | None
    aliases: tuple[dict[str, Any], ...]
    locations: tuple[dict[str, Any], ...]
    bytes_verified: bool
    status: Literal["active", "archived"] = "active"


class MenagerieExport(MenagerieModel):
    schema_version: Literal["synth.menagerie.export.v1"]
    org_id: str
    sections: dict[str, tuple[dict[str, Any], ...]]
    manifest_digest: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")


class ModelArchive(MenagerieModel):
    schema_version: Literal["synth.menagerie.v1"]
    model: ModelBinding
    status: Literal["archived"]
    replayed: bool
    retained_aliases: int
    retained_eval_links: int


class ModelRestore(MenagerieModel):
    schema_version: Literal["synth.menagerie.v1"]
    model: ModelBinding
    status: Literal["active"]
    replayed: bool


class RecordArchive(MenagerieModel):
    schema_version: Literal["synth.menagerie.v1"]
    record_id: str
    head_revision: int
    archived: Literal[True]
    replayed: bool
    revoked_grants: int


class RecordRestore(MenagerieModel):
    schema_version: Literal["synth.menagerie.v1"]
    record_id: str
    head_revision: int
    archived: Literal[False]
    replayed: bool


class LineageWrite(MenagerieModel):
    """``source`` derives from ``target`` by ``relation``; ids or aliases the caller owns."""

    source: str = Field(min_length=1, max_length=512)
    relation: LineageRelation
    target: str = Field(min_length=1, max_length=512)


class LineageEdge(MenagerieModel):
    schema_version: Literal["synth.menagerie.v1"]
    source: ModelBinding
    relation: LineageRelation
    target: ModelBinding
    replayed: bool
    created_at: str


class LineageView(MenagerieModel):
    schema_version: Literal["synth.menagerie.v1"]
    model: ModelBinding
    derived_from: tuple[LineageEdge, ...]
    derivatives: tuple[LineageEdge, ...]


class DeploymentWrite(MenagerieModel):
    """A model's actual serving identity. ``digest_verified`` is never claimable;
    ``provider_receipt`` needs an operator key."""

    model: str = Field(min_length=1, max_length=512)
    provider: str = Field(pattern=r"^[a-z0-9][a-z0-9._-]{0,63}$")
    route_id: str | None = Field(default=None, min_length=1, max_length=256)
    serving_revision: str | None = Field(default=None, min_length=1, max_length=256)
    verification_strength: ServingStrength = "provider_reference"


class Deployment(MenagerieModel):
    schema_version: Literal["synth.menagerie.v1"]
    deployment_id: str
    model: ModelBinding
    provider: str
    route_id: str | None
    serving_revision: str | None
    verification_strength: Literal[
        "unknown", "provider_reference", "provider_receipt", "digest_verified"
    ]
    status: Literal["active", "retired"]
    replayed: bool
    created_at: str
