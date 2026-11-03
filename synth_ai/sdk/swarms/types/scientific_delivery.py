"""Retained scientific publication, independent of run and billing finality.

See notes/specifications/sdk/swarms_scientific_delivery.md. These DTOs carry
owner receipts only; a closed execution never implies scientific delivery.
"""
from __future__ import annotations

from typing import Literal, Self

from pydantic import Field, StrictInt, model_validator

from synth_ai.types.base import Digest, ResponseBase
from synth_ai.types.refs import Ref


class ScientificFamily(ResponseBase):
    id: str
    version: StrictInt = Field(ge=1)


class ScientificRecordReceipt(ResponseBase):
    operation_id: str
    payload_digest_sha256: Digest
    cursor: StrictInt = Field(ge=0)
    recorded_at: str
    reference: Ref
    project_id: str
    observation_receipt_id: str | None = None


class ScientificWritePending(ResponseBase):
    operation_id: str
    project_id: str | None = None


class ScientificWriteRefusal(ResponseBase):
    operation_id: str
    code: str | None = None


class NativeScientificScope(ResponseBase):
    organization_id: str
    project_id: str


class NativeScientificReceipt(ResponseBase):
    schema_version: Literal["forge.receipt.v1"]
    scope: NativeScientificScope
    operation_id: str
    payload_digest_sha256: Digest
    reference: Ref
    cursor: StrictInt = Field(ge=1)
    recorded_at: str


class NativeScientificRecording(ResponseBase):
    schema_version: Literal["forge.native-recording.v1"]
    status: Literal["delivered", "incomplete"]
    receipts: dict[str, NativeScientificReceipt]
    in_doubt: tuple[str, ...]
    incomplete_reasons: tuple[str, ...]
    record_set_digest: Digest


class ScientificDelivery(ResponseBase):
    schema_version: Literal["synth.research-scientific-delivery.v1"]
    family: ScientificFamily
    forge_project_id: str
    record_protocol: str
    status: Literal["delivered", "incomplete"]
    incomplete_reasons: tuple[str, ...]
    receipts: tuple[ScientificRecordReceipt, ...]
    in_doubt: tuple[ScientificWritePending, ...]
    refused: tuple[ScientificWriteRefusal, ...]
    unreadable: tuple[str, ...]
    delivery_digest: Digest
    native_recording: NativeScientificRecording | None = Field(
        default=None, exclude_if=lambda value: value is None
    )

    @model_validator(mode="after")
    def validate_delivery(self) -> Self:
        if self.status == "delivered" and (
            self.incomplete_reasons or self.in_doubt or self.refused or self.unreadable
            or not any(r.project_id == self.forge_project_id for r in self.receipts)
            or (self.native_recording is not None and (
                self.native_recording.status != "delivered"
                or self.native_recording.in_doubt
                or self.native_recording.incomplete_reasons
            ))
        ):
            raise ValueError("Delivered science requires bound receipts without gaps")
        if self.status == "incomplete" and not self.incomplete_reasons:
            raise ValueError("Incomplete science requires explicit reasons")
        return self
