"""Control and recovery receipts. See SDK index_contract.md."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Generic, Literal, Self, TypeVar

from pydantic import SerializerFunctionWrapHandler, model_serializer, model_validator

from .base import ResponseBase, ResponseIdentifier, ResponseMetadata, ResponseModel
from .identity import MutationIdentity


class ResourceReference(ResponseBase):
    kind: Literal[
        "search",
        "contribution",
        "upload",
        "control",
        "search_response",
        "swarm",
        "repo_clone_token",
    ]
    resource_id: ResponseIdentifier


ResolutionState = Literal["accepted", "pending", "not_accepted", "unavailable", "conflict"]


class MutationResolution(ResponseModel):
    identity: MutationIdentity
    state: ResolutionState
    resource: ResourceReference | None

    @model_validator(mode="after")
    def validate_resource(self) -> Self:
        if (self.state == "accepted") != (self.resource is not None):
            raise ValueError("only an accepted resolution carries a resource")
        return self


ControlCommand = Literal["cancel", "pause", "resume"]
ControlState = Literal["accepted", "applied", "quiesced", "refused"]


class ControlReceipt(ResponseModel):
    operation_id: ResponseIdentifier
    resource_id: ResponseIdentifier
    state: ControlState
    command: ControlCommand | None = None
    reason: str | None
    execution_closed: bool
    accounting_final: bool
    cleanup_complete: bool

    @model_serializer(mode="wrap")
    def omit_absent_command(self, handler: SerializerFunctionWrapHandler) -> dict[str, Any]:
        value = handler(self)
        if self.command is None:
            value.pop("command", None)
        return value


T = TypeVar("T")


@dataclass(frozen=True, slots=True)
class CreationResolution(Generic[T]):
    state: ResolutionState
    resource: T | None
    metadata: ResponseMetadata
