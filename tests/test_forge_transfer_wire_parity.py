"""Generated Forge transfer wire declarations do not grant transfer authority."""

import pytest
from pydantic import ValidationError
from synth_ai.contracts import forge
from synth_ai.contracts.registry import OPERATIONS_BY_ID
from synth_ai.sdk.forge.contracts.workspace_transfer_public import PublicImportRequest
from synth_ai.sdk.forge.transfer_wire import NativeTransferReceiptResponse


def test_public_import_selection_is_bounded_and_does_not_accept_principal():
    request = {
        "operation_id": "op-1",
        "source_owner_id": "source-1",
        "declared_cut": 0,
        "bundle_digest_sha256": "a" * 64,
        "source_manifest_header_json": "{}",
    }
    assert PublicImportRequest.model_validate(request).declared_cut == 0
    with pytest.raises(ValidationError):
        PublicImportRequest.model_validate({**request, "principal": "forged"})
    with pytest.raises(ValidationError):
        PublicImportRequest.model_validate({**request, "declared_cut": -1})


def test_transfer_descriptors_and_retained_receipts_are_strict():
    assert OPERATIONS_BY_ID["forge.transfers.import"] is forge.TRANSFER_IMPORT
    assert forge.TRANSFER_IMPORT.mutation and forge.TRANSFER_IMPORT.required_scope == "forge:write"
    assert not forge.TRANSFER_PREVIEW.mutation and not forge.TRANSFER_RECEIPT.mutation
    with pytest.raises(ValidationError):
        NativeTransferReceiptResponse.model_validate(
            {
                "metadata": {
                    "contract": "core.v1",
                    "request_id": "req",
                    "observed_at": "2026-10-31T00:00:00Z",
                },
                "receipt": {"schema": "forge.native-import-custody.v1"},
            }
        )
