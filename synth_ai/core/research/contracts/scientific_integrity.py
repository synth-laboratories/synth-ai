"""Scientific-integrity contracts mirrored from the backend authority.

Mirrors ``services/smr/science/integrity.py`` in ``backend`` (PR #734,
``feat/scientific-verdicts-intervention-gate``). The backend gate at
``smr_attach_experiment_result`` and the Factory candidate-grading intake
(``validate_candidate_grading_record``) both require a typed
``evaluation_mode`` and, for experiment-result attachment, a verified
``intervention_receipt.v1``. These client-side mirrors let SDK callers build
and validate the same payloads before submission and fail with the same
stable failure-class tokens the backend uses, rather than discovering a
rejection only after a round trip.

Every rejection raises ``ValueError`` whose message begins with a stable
failure class token (``evaluation_mode_missing``, ``evaluation_mode_invalid``,
``intervention_receipt_missing``, ``intervention_receipt_invalid``,
``intervention_receipt_schema_unsupported``,
``intervention_receipt_declared_field_not_transmitted``,
``intervention_receipt_declared_field_not_read_back``,
``intervention_receipt_declared_field_readback_mismatch``).
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from typing import Any


class SmrEvaluationMode(StrEnum):
    """How the measured episodes were actually executed."""

    LIVE = "live"
    MOCK = "mock"
    FIXTURE = "fixture"
    OFFLINE_FALLBACK = "offline_fallback"


class SmrScientificStatus(StrEnum):
    """Scientific validity verdict — distinct from lifecycle and verifier."""

    SCIENTIFIC = "scientific"
    NON_SCIENTIFIC_SMOKE = "non_scientific_smoke"


INTERVENTION_RECEIPT_SCHEMA_VERSION = "intervention_receipt.v1"


def parse_evaluation_mode(value: Any) -> SmrEvaluationMode:
    """Parse a declared evaluation mode; absent/unknown is a typed rejection."""
    if value is None or (isinstance(value, str) and not value.strip()):
        raise ValueError(
            "evaluation_mode_missing: experiment-result attachment requires "
            "evaluation_mode (one of "
            f"{sorted(m.value for m in SmrEvaluationMode)})"
        )
    text = str(value).strip().lower()
    try:
        return SmrEvaluationMode(text)
    except ValueError:
        raise ValueError(
            f"evaluation_mode_invalid: {text!r} is not one of "
            f"{sorted(m.value for m in SmrEvaluationMode)}"
        ) from None


def scientific_status_for_mode(
    mode: SmrEvaluationMode | str,
) -> SmrScientificStatus:
    """Only live execution can carry the scientific verdict."""
    if str(mode) == SmrEvaluationMode.LIVE.value:
        return SmrScientificStatus.SCIENTIFIC
    return SmrScientificStatus.NON_SCIENTIFIC_SMOKE


@dataclass(frozen=True)
class InterventionReceiptV1:
    """Versioned receipt binding declaration → transmission → readback.

    ``declared_changed_fields`` maps each intervention field name to its
    declared value. ``transmitted_request_fields`` is the exact field map
    sent to the evaluated service. ``runtime_effective_config`` is the
    runtime's effective-config readback after applying the request.
    """

    declared_changed_fields: dict[str, Any]
    transmitted_request_fields: dict[str, Any]
    runtime_effective_config: dict[str, Any]
    held_constant_fields: tuple[str, ...] = ()
    unexpected_differences: tuple[str, ...] = ()
    schema_version: str = INTERVENTION_RECEIPT_SCHEMA_VERSION

    def payload(self) -> dict[str, Any]:
        """JSON-storable form of this receipt."""
        return {
            "schema_version": self.schema_version,
            "declared_changed_fields": dict(self.declared_changed_fields),
            "transmitted_request_fields": dict(self.transmitted_request_fields),
            "runtime_effective_config": dict(self.runtime_effective_config),
            "held_constant_fields": list(self.held_constant_fields),
            "unexpected_differences": list(self.unexpected_differences),
        }


def parse_intervention_receipt(value: Any) -> InterventionReceiptV1:
    """Parse a receipt payload; absent/malformed is a typed rejection."""
    if value is None:
        raise ValueError(
            "intervention_receipt_missing: experiment-result attachment "
            f"requires an {INTERVENTION_RECEIPT_SCHEMA_VERSION} receipt"
        )
    if not isinstance(value, Mapping):
        raise ValueError(
            f"intervention_receipt_invalid: receipt must be a mapping, got {type(value).__name__}"
        )
    schema = str(value.get("schema_version") or "").strip()
    if schema != INTERVENTION_RECEIPT_SCHEMA_VERSION:
        raise ValueError(
            "intervention_receipt_schema_unsupported: expected "
            f"{INTERVENTION_RECEIPT_SCHEMA_VERSION!r}, got {schema!r}"
        )
    declared = value.get("declared_changed_fields")
    transmitted = value.get("transmitted_request_fields")
    readback = value.get("runtime_effective_config")
    for name, part in (
        ("declared_changed_fields", declared),
        ("transmitted_request_fields", transmitted),
        ("runtime_effective_config", readback),
    ):
        if not isinstance(part, Mapping):
            raise ValueError(
                f"intervention_receipt_invalid: {name} must be a mapping, got {type(part).__name__}"
            )
    return InterventionReceiptV1(
        declared_changed_fields=dict(declared),
        transmitted_request_fields=dict(transmitted),
        runtime_effective_config=dict(readback),
        held_constant_fields=tuple(str(item) for item in (value.get("held_constant_fields") or [])),
        unexpected_differences=tuple(
            str(item) for item in (value.get("unexpected_differences") or [])
        ),
    )


def verify_intervention_receipt(receipt: InterventionReceiptV1) -> None:
    """Reject when a declared field never reached the wire or the runtime.

    A declared intervention field must appear in the transmitted request
    fields AND in the runtime effective-config readback, and the readback
    value must equal the declared value.
    """
    for name, declared_value in receipt.declared_changed_fields.items():
        if name not in receipt.transmitted_request_fields:
            raise ValueError(
                "intervention_receipt_declared_field_not_transmitted: "
                f"declared field {name!r} is absent from "
                "transmitted_request_fields"
            )
        if name not in receipt.runtime_effective_config:
            raise ValueError(
                "intervention_receipt_declared_field_not_read_back: "
                f"declared field {name!r} is absent from "
                "runtime_effective_config"
            )
        readback_value = receipt.runtime_effective_config[name]
        if readback_value != declared_value:
            raise ValueError(
                "intervention_receipt_declared_field_readback_mismatch: "
                f"declared field {name!r} declared={declared_value!r} "
                f"readback={readback_value!r}"
            )


def compile_intervention_receipt(
    *,
    declared_changed_fields: Mapping[str, Any],
    transmitted_request_fields: Mapping[str, Any],
    runtime_effective_config: Mapping[str, Any],
    held_constant_fields: tuple[str, ...] = (),
) -> InterventionReceiptV1:
    """Build and verify a receipt in one call, deriving ``unexpected_differences``.

    ``unexpected_differences`` names any ``held_constant_fields`` entry whose
    transmitted value diverges from its runtime-effective readback — a signal
    that something outside the declared intervention drifted between request
    and execution. Raises the same typed ``ValueError`` as
    :func:`verify_intervention_receipt` when a declared field failed to reach
    the wire or the runtime.
    """
    unexpected_differences = tuple(
        name
        for name in held_constant_fields
        if name in transmitted_request_fields
        and name in runtime_effective_config
        and transmitted_request_fields[name] != runtime_effective_config[name]
    )
    receipt = InterventionReceiptV1(
        declared_changed_fields=dict(declared_changed_fields),
        transmitted_request_fields=dict(transmitted_request_fields),
        runtime_effective_config=dict(runtime_effective_config),
        held_constant_fields=tuple(held_constant_fields),
        unexpected_differences=unexpected_differences,
    )
    verify_intervention_receipt(receipt)
    return receipt


def grading_record_evaluation_mode(
    record: Mapping[str, Any],
) -> SmrEvaluationMode | None:
    """Read the evaluation mode a grading record declares, if any.

    Returns ``None`` for legacy records written before the contract; any
    present-but-invalid value is a typed rejection.
    """
    raw = record.get("evaluation_mode")
    if raw is None:
        return None
    return parse_evaluation_mode(raw)


__all__ = [
    "INTERVENTION_RECEIPT_SCHEMA_VERSION",
    "InterventionReceiptV1",
    "SmrEvaluationMode",
    "SmrScientificStatus",
    "compile_intervention_receipt",
    "grading_record_evaluation_mode",
    "parse_evaluation_mode",
    "parse_intervention_receipt",
    "scientific_status_for_mode",
    "verify_intervention_receipt",
]
