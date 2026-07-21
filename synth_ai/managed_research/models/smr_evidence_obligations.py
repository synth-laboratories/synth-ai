"""Typed terminal evidence obligations for proof-bearing Managed Research runs.

# See: backend/services/smr/api_schemas.py (SmrEvidenceObligationsRequest authority)
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Self

EVIDENCE_OBLIGATIONS_SCHEMA = "smr.evidence_obligations.v1"


class EvidenceObligationKind(StrEnum):
    TRANSCRIPT_ROWS = "transcript_rows"
    TASK_TURN_ROWS = "task_turn_rows"
    RAW_TRACE_ROWS = "raw_trace_rows"
    EVIDENCE_PACKET = "evidence_packet"
    TERMINAL_TASK_ROWS = "terminal_task_rows"


EVIDENCE_OBLIGATION_KIND_VALUES: tuple[str, ...] = tuple(
    kind.value for kind in EvidenceObligationKind
)


@dataclass(frozen=True, slots=True)
class EvidenceObligations:
    """Explicit evidence requirements evaluated when a run reaches terminal state."""

    required: Sequence[EvidenceObligationKind | str]
    schema: str = EVIDENCE_OBLIGATIONS_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != EVIDENCE_OBLIGATIONS_SCHEMA:
            raise ValueError(
                "evidence_obligations.schema must be "
                f"{EVIDENCE_OBLIGATIONS_SCHEMA!r}, got {self.schema!r}"
            )
        if not isinstance(self.required, Sequence) or isinstance(
            self.required, (str, bytes)
        ):
            raise ValueError("evidence_obligations.required must be a non-empty sequence")

        normalized: list[EvidenceObligationKind] = []
        for index, value in enumerate(self.required):
            raw_value = str(getattr(value, "value", value)).strip()
            try:
                kind = EvidenceObligationKind(raw_value)
            except ValueError as exc:
                raise ValueError(
                    f"evidence_obligations.required[{index}] has unknown value {raw_value!r}"
                ) from exc
            if kind not in normalized:
                normalized.append(kind)
        if not normalized:
            raise ValueError("evidence_obligations.required must not be empty")
        object.__setattr__(self, "required", tuple(normalized))

    @classmethod
    def from_wire(cls, payload: Mapping[str, Any]) -> Self:
        if not isinstance(payload, Mapping):
            raise ValueError("evidence_obligations must be a mapping when provided")
        unknown = sorted(set(payload) - {"schema", "required"})
        if unknown:
            raise ValueError(
                "evidence_obligations contains unknown fields: " + ", ".join(unknown)
            )
        schema = payload.get("schema", EVIDENCE_OBLIGATIONS_SCHEMA)
        if not isinstance(schema, str):
            raise ValueError("evidence_obligations.schema must be a string")
        return cls(
            schema=schema,
            required=_required_values(payload),
        )

    def to_wire(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "required": [str(getattr(kind, "value", kind)) for kind in self.required],
        }


def coerce_evidence_obligations(
    value: EvidenceObligations | Mapping[str, Any] | None,
    *,
    field_name: str = "evidence_obligations",
) -> EvidenceObligations | None:
    if value is None:
        return None
    if isinstance(value, EvidenceObligations):
        return value
    if not isinstance(value, Mapping):
        raise ValueError(f"{field_name} must be a mapping when provided")
    try:
        return EvidenceObligations.from_wire(value)
    except ValueError as exc:
        if field_name == "evidence_obligations":
            raise
        raise ValueError(f"invalid {field_name}: {exc}") from exc


def _required_values(payload: Mapping[str, Any]) -> Sequence[EvidenceObligationKind | str]:
    value = payload.get("required")
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError("evidence_obligations.required must be a non-empty sequence")
    return value


__all__ = [
    "EVIDENCE_OBLIGATION_KIND_VALUES",
    "EVIDENCE_OBLIGATIONS_SCHEMA",
    "EvidenceObligationKind",
    "EvidenceObligations",
    "coerce_evidence_obligations",
]
