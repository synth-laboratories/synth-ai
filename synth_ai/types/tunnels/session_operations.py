"""Backend session contracts. See the public contract snapshots in contracts/"""

from __future__ import annotations

from typing import Annotated, Literal
from uuid import UUID

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, field_validator, model_validator


class Contract(BaseModel):
    model_config = ConfigDict(
        extra="forbid", frozen=True, hide_input_in_errors=True, allow_inf_nan=False
    )

    @field_validator("*", mode="after")
    @classmethod
    def non_nil(cls, value):
        if isinstance(value, UUID) and value.int == 0:
            raise ValueError("nil_session_identity")
        return value


class Artifact(Contract):
    namespace: str = Field(min_length=1, max_length=128)
    object_key: str = Field(pattern=r"^sha256/[a-f0-9]{64}$")
    sha256: str = Field(pattern=r"^[a-f0-9]{64}$")

    @field_validator("namespace")
    @classmethod
    def bounded_namespace(cls, value):
        if len(value.encode()) > 128:
            raise ValueError("artifact_namespace_capacity")
        return value

    @model_validator(mode="after")
    def content_address(self):
        if self.object_key != "sha256/" + self.sha256:
            raise ValueError("artifact_content_address_mismatch")
        return self


class SessionOperation(Contract):
    operation: UUID
    session: UUID
    generation: int = Field(ge=1, le=9_007_199_254_740_991)
    deadline_at: AwareDatetime

    def wire_payload(self):
        return self.model_dump(mode="json", exclude={"action", "session", "deadline_at"})


class Pause(SessionOperation):
    action: Literal["pause"]


class SealCheckpoint(SessionOperation):
    action: Literal["checkpoint"]
    checkpoint: UUID
    artifact: Artifact
    semantics: Literal["logical_session", "application_cold_restore", "true_environment_snapshot"]
    source_origin_instance: UUID
    cut_receipt: UUID
    pending_operations: Literal[0]
    unresolved_operations: Literal[0]


class Resume(SessionOperation):
    action: Literal["resume"]


class _CheckpointRestart(SessionOperation):
    checkpoint: UUID
    checkpoint_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")


class Restart(_CheckpointRestart):
    action: Literal["restart"]


class CompleteRestart(_CheckpointRestart):
    action: Literal["restart-complete"]
    service_identity: str = Field(min_length=1, max_length=128)
    restored_origin_instance: UUID
    receipt: UUID
    restart_operation: UUID

    def wire_payload(self):
        payload = super().wire_payload()
        payload["operation"] = payload.pop("restart_operation")
        return payload

    @field_validator("service_identity")
    @classmethod
    def canonical_service(cls, value):
        if value != value.strip() or len(value.encode()) > 128:
            raise ValueError("invalid_restore_service_identity")
        return value


OperationRequest = Annotated[
    Pause | SealCheckpoint | Resume | Restart | CompleteRestart,
    Field(discriminator="action"),
]
