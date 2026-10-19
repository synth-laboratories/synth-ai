"""Closed deployment capability observations. See backend SDK index_contract.md."""

from __future__ import annotations

import hashlib
import json
from typing import Annotated, Literal, Self

from pydantic import Field, JsonValue, TypeAdapter, model_validator

from .base import Digest, Identifier, Model, ResponseModel

ContractName = Literal["index.v1", "swarms.v1", "core.v1"]
Surface = Literal["http", "mcp"]
TrustEnvironment = Literal["local", "development", "staging", "production"]
Version = Annotated[str, Field(strict=True, max_length=64, pattern=r"^[0-9][0-9A-Za-z._+-]*$")]
SourceCommit = Annotated[str, Field(strict=True, pattern=r"^[0-9a-f]{40}$")]
JSON_OBJECT = TypeAdapter(dict[str, JsonValue])


class ComponentIdentity(Model):
    component_id: Identifier
    version: Version
    source_commit: SourceCommit
    artifact_sha256: Digest
    configuration_sha256: Digest


class ReleaseIdentity(Model):
    environment: TrustEnvironment
    release_train: Annotated[str, Field(strict=True, max_length=32, pattern=r"^0\.[0-9]+\.[0-9]+$")]
    candidate_id: Identifier
    manifest_sha256: Digest
    components: tuple[ComponentIdentity, ...] = Field(min_length=1, max_length=32)

    @model_validator(mode="after")
    def validate_components(self) -> Self:
        identifiers = tuple(component.component_id for component in self.components)
        if identifiers != tuple(sorted(set(identifiers))):
            raise ValueError("release component identities must be unique and sorted")
        return self


class ContractIdentity(Model):
    contract: ContractName
    schema_sha256: Digest


class OperationCapability(Model):
    operation_id: Identifier
    contract: ContractName
    state: Literal["enabled", "unavailable"]
    surfaces: tuple[Surface, ...] = Field(max_length=2)
    reason: Literal["not_implemented", "not_qualified", "disabled", "dependency_unavailable"] | None

    @model_validator(mode="after")
    def validate_disposition(self) -> Self:
        if self.surfaces != tuple(sorted(set(self.surfaces))):
            raise ValueError("capability surfaces must be unique and sorted")
        if self.state == "enabled":
            if not self.surfaces or self.reason is not None:
                raise ValueError("enabled capability requires surfaces and no refusal reason")
        elif self.surfaces or self.reason is None:
            raise ValueError("unavailable capability requires a reason and no surfaces")
        return self


class CapabilitySnapshot(Model):
    schema_version: Literal["synth.capabilities.v1"] = "synth.capabilities.v1"
    release: ReleaseIdentity | None
    admission: Literal["candidate", "qualified"] | None
    contracts: tuple[ContractIdentity, ...] = Field(max_length=3)
    operations: tuple[OperationCapability, ...] = Field(max_length=128)

    @model_validator(mode="after")
    def validate_snapshot(self) -> Self:
        contracts = tuple(item.contract for item in self.contracts)
        identifiers = tuple(item.operation_id for item in self.operations)
        if contracts != tuple(sorted(set(contracts))) or identifiers != tuple(
            sorted(set(identifiers))
        ):
            raise ValueError("capability identities must be unique and sorted")
        if any(item.contract not in contracts for item in self.operations):
            raise ValueError("operation requires its contract schema identity")
        if (self.release is None) != (self.admission is None):
            raise ValueError("release identity and admission disposition must be supplied together")
        if (
            self.admission == "candidate"
            and self.release is not None
            and self.release.environment == "production"
        ):
            raise ValueError("candidate admission is forbidden in production")
        if self.release is None and any(item.state == "enabled" for item in self.operations):
            raise ValueError("enabled operations require an explicit release identity")
        return self


def snapshot_sha256(snapshot: CapabilitySnapshot) -> str:
    """Fingerprint canonical nonsecret capability facts, excluding request timing.

    See: the public contract snapshots in contracts/"""
    value = JSON_OBJECT.validate_json(snapshot.model_dump_json())
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


class Capabilities(CapabilitySnapshot, ResponseModel):
    capability_sha256: Digest

    @model_validator(mode="after")
    def validate_digest(self) -> Self:
        snapshot = CapabilitySnapshot(
            schema_version=self.schema_version,
            release=self.release,
            admission=self.admission,
            contracts=self.contracts,
            operations=self.operations,
        )
        if self.capability_sha256 != snapshot_sha256(snapshot):
            raise ValueError("capability receipt digest does not match its snapshot")
        return self


class NegotiatedCapability(Model):
    schema_sha256: Digest
    capability_sha256: Digest
    release_manifest_sha256: Digest
