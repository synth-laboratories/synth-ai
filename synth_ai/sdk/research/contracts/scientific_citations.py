"""Read-only citation custody, mirrored from backend CitationVerification.

See forge_scientific_delivery.md; citation failure never deletes a record.
"""

from typing import Literal

from synth_ai.sdk.research.contracts.forge.contracts import Contract, ExactReference


class CitationCheck(Contract):
    reference: ExactReference
    status: Literal["retained", "dangling", "denied", "conflict", "unavailable", "refused"]
    code: str = ""
    authority_code: str = ""


class CitationVerification(Contract):
    schema_version: Literal["forge.citation-verification.v1"]
    record: ExactReference
    citations: tuple[CitationCheck, ...]
    retained: bool
