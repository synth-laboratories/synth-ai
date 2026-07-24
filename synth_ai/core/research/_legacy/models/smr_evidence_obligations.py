"""Compatibility re-export; implementation lives in contracts.smr_evidence_obligations."""

from synth_ai.core.research.contracts.smr_evidence_obligations import (
    EVIDENCE_OBLIGATION_KIND_VALUES,
    EVIDENCE_OBLIGATIONS_SCHEMA,
    EvidenceObligationKind,
    EvidenceObligations,
    coerce_evidence_obligations,
)

__all__ = [
    'EVIDENCE_OBLIGATION_KIND_VALUES',
    'EVIDENCE_OBLIGATIONS_SCHEMA',
    'EvidenceObligationKind',
    'EvidenceObligations',
    'coerce_evidence_obligations',
]
