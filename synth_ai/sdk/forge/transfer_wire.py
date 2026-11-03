"""Generated public Forge transfer response envelopes; no new transport."""

from pydantic import JsonValue, field_validator

from synth_ai.sdk.forge.contracts.workspace_transfer_public import (
    ImportPreview,
    NativeImportReceipt,
)
from synth_ai.types.base import ResponseModel


class NativeTransferPreviewResponse(ResponseModel):
    preview: ImportPreview


class NativeTransferReceiptResponse(ResponseModel):
    receipt: dict[str, JsonValue]

    @field_validator("receipt")
    @classmethod
    def validate_original_receipt(cls, value):
        NativeImportReceipt.model_validate(value)
        return value
