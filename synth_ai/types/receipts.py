"""Control and recovery receipts. See SDK index_contract.md."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Generic, Literal, Self, TypeVar

from pydantic import model_validator

from .base import Identifier, Model, ResponseMetadata, ResponseModel
from .identity import MutationIdentity


class ResourceReference(Model):
    kind: Literal["search", "contribution", "upload", "control", "search_response", "swarm"]
    resource_id: Identifier


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


class ControlReceipt(ResponseModel):
    operation_id: Identifier
    resource_id: Identifier
    state: Literal["accepted", "applied", "refused"]
    reason: str | None
    execution_closed: bool
    accounting_final: bool
    cleanup_complete: bool


T = TypeVar("T")


@dataclass(frozen=True, slots=True)
class CreationResolution(Generic[T]):
    state: ResolutionState
    resource: T | None
    metadata: ResponseMetadata
