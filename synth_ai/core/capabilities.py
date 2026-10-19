"""Fail-closed operation compatibility. See backend SDK index_contract.md."""

from __future__ import annotations

from synth_ai.contracts.fingerprint import schema_sha256
from synth_ai.contracts.operations import Operation
from synth_ai.errors import ContractError, UnsupportedOperationError
from synth_ai.types.capabilities import Capabilities, NegotiatedCapability, Surface


def validate_release(
    receipt: Capabilities,
    expected_manifest_sha256: str | None,
    candidate_manifest_sha256: str | None = None,
) -> Capabilities:
    if expected_manifest_sha256 is not None and (
        receipt.release is None or receipt.release.manifest_sha256 != expected_manifest_sha256
    ):
        raise ContractError(
            "Backend release does not match the selected manifest", code="release_mismatch"
        )
    if candidate_manifest_sha256 is not None and (
        receipt.release is None
        or receipt.release.environment == "production"
        or receipt.release.manifest_sha256 != candidate_manifest_sha256
    ):
        raise ContractError(
            "Backend does not match the selected nonproduction candidate", code="release_mismatch"
        )
    return receipt


def require_operation(
    receipt: Capabilities,
    operation: Operation,
    *,
    surface: Surface,
    expected_manifest_sha256: str | None = None,
    candidate_manifest_sha256: str | None = None,
) -> NegotiatedCapability:
    """Require actual qualified surface and schema identity before dispatch.

    See: backend/notes/specifications/sdk/index_contract.md. This receipt is not
    authorization, health proof or independent verification of service pins."""
    validate_release(receipt, expected_manifest_sha256, candidate_manifest_sha256)
    if receipt.admission == "candidate" and candidate_manifest_sha256 is None:
        raise UnsupportedOperationError(
            "Candidate operations require an explicit nonproduction manifest pin",
            code="candidate_not_qualified",
            status=503,
        )
    if receipt.admission is None:
        raise UnsupportedOperationError(
            "Backend qualification is unavailable", code="capabilities_unavailable", status=503
        )
    selected = next(
        (item for item in receipt.operations if item.operation_id == operation.id), None
    )
    if selected is None or selected.state != "enabled" or surface not in selected.surfaces:
        raise UnsupportedOperationError(
            "Operation is unavailable on the selected backend",
            code="operation_unavailable",
            status=503,
        )
    if selected.contract != operation.contract:
        raise ContractError(
            "Backend operation selected a different protocol", code="schema_mismatch"
        )
    contract = next(
        (item for item in receipt.contracts if item.contract == operation.contract), None
    )
    if contract is None:
        raise ContractError(
            "Backend protocol schema is incompatible",
            code="schema_mismatch",
            stage="capability_negotiation",
            cause=f"contract={operation.contract} operation={operation.id} "
            f"backend_receipt=absent contracts_in_receipt="
            f"{','.join(item.contract for item in receipt.contracts) or 'none'}",
        )
    expected_schema = schema_sha256(operation.contract)
    if contract.schema_sha256 != expected_schema:
        raise ContractError(
            "Backend protocol schema is incompatible",
            code="schema_mismatch",
            stage="capability_negotiation",
            cause=f"contract={operation.contract} operation={operation.id} "
            f"sdk_schema_sha256={expected_schema} backend_schema_sha256={contract.schema_sha256}",
        )
    if receipt.release is None:
        raise ContractError("Backend did not identify a release", code="release_mismatch")
    return NegotiatedCapability(
        schema_sha256=contract.schema_sha256,
        capability_sha256=receipt.capability_sha256,
        release_manifest_sha256=receipt.release.manifest_sha256,
    )


def supports_operation(
    receipt: Capabilities,
    operation: Operation,
    *,
    surface: Surface,
    candidate_manifest_sha256: str | None = None,
) -> bool:
    """Hide unsupported/incompatible tools; transport/auth failures still propagate.

    See: backend/notes/specifications/sdk/index_contract.md."""
    try:
        require_operation(
            receipt, operation, surface=surface, candidate_manifest_sha256=candidate_manifest_sha256
        )
    except (UnsupportedOperationError, ContractError):
        return False
    return True
