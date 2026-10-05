# Mirrors backend services/forge/client.py (CitationVerification) and Forge
# forge_service.persistence.CitationVerification (forge.citation-verification.v1).
"""Read-time custodian state of a retained revision's external citations (R06).

The retained record and its exact references are never changed or dropped;
each external citation reports ``retained``, ``dangling`` (deleted/retired or
withdrawn), ``denied`` (revoked grant, foreign scope), ``conflict``,
``unavailable`` or ``refused`` with the custodian's closed code.
"""

from typing import Literal

from synth_ai.sdk.research.contracts.forge.contracts import Contract, ExactReference

CitationStatus = Literal["retained", "dangling", "denied", "conflict", "unavailable", "refused"]


class CitationCheck(Contract):
    reference: ExactReference
    status: CitationStatus
    code: str = ""
    authority_code: str = ""


class CitationVerification(Contract):
    schema_version: Literal["forge.citation-verification.v1"]
    record: ExactReference
    citations: tuple[CitationCheck, ...]
    retained: bool
