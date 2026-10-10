"""Origin cleanup custody. See tanha/current/systems/tunnels/v2_catalog.md."""

from typing import Annotated, Literal
from uuid import UUID

from pydantic import AwareDatetime, Field, field_validator

from .lease_custody import CustodyContract


class CallerOperationOwner(CustodyContract):
    kind: Literal["caller"]
    job_id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_-]{0,127}$")
    caller_binding: UUID


class ExecutorOperationOwner(CustodyContract):
    kind: Literal["executor"]
    job_id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_-]{0,127}$")
    offer: UUID
    executor_instance: UUID


OperationOwner = Annotated[
    CallerOperationOwner | ExecutorOperationOwner, Field(discriminator="kind")
]


class OriginOperationRegistration(CustodyContract):
    operation: UUID
    attempt: UUID
    route: UUID
    origin_instance: UUID
    origin_operation_id: str = Field(
        strict=True,
        min_length=1,
        max_length=128,
        pattern=r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}$",
    )
    input_sha256: str = Field(
        strict=True, min_length=64, max_length=64, pattern=r"^[a-f0-9]{64}$"
    )
    service_identity: str = Field(strict=True, min_length=1, max_length=128)
    ownership: OperationOwner
    deadline_at: AwareDatetime

    @field_validator("service_identity")
    @classmethod
    def exact_service_identity(cls, value: str) -> str:
        if value.strip() != value or len(value.encode()) > 128:
            raise ValueError("origin_operation_service_identity_invalid")
        return value


class OriginOperationLookup(CustodyContract):
    ownership: OperationOwner


class OriginManagementRequest(OriginOperationLookup):
    action: Literal["lookup", "cancel"]


class CallerOriginRegistration(OriginOperationRegistration):
    ownership: CallerOperationOwner


class CallerOriginLookup(OriginOperationLookup):
    ownership: CallerOperationOwner


class CallerOriginManagement(OriginManagementRequest):
    ownership: CallerOperationOwner


class OriginManagementRevocation(CallerOriginLookup):
    expected_revision: int = Field(strict=True, ge=1)


class ExecutorOriginRegistration(OriginOperationRegistration):
    org_id: UUID
    ownership: ExecutorOperationOwner


class ExecutorOriginLookup(OriginOperationLookup):
    org_id: UUID
    ownership: ExecutorOperationOwner


class ExecutorOriginManagement(OriginManagementRequest):
    org_id: UUID
    ownership: ExecutorOperationOwner


class ExecutorOriginRevocation(ExecutorOriginLookup):
    expected_revision: int = Field(strict=True, ge=1)
