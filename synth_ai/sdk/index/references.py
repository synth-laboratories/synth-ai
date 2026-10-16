"""Exact provenance identities retained by Index; references never grant access.

See: backend/notes/specifications/sdk/surface.md. Moved from the preserved
Research Forge contracts without changing their wire constraints.
"""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, StringConstraints, model_validator

Identifier = Annotated[str, StringConstraints(pattern="^[a-zA-Z0-9][a-zA-Z0-9_.-]{0,127}$")]
Digest = Annotated[str, StringConstraints(pattern="^[0-9a-f]{64}$")]
ReferenceRecordId = Annotated[
    str,
    StringConstraints(pattern="^(?:[a-zA-Z0-9][a-zA-Z0-9_.-]{0,127}|intern-exec:[0-9a-f]{64})$"),
]


class Contract(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=True)


class ExactReference(Contract):
    model_config = ConfigDict(
        json_schema_extra={
            "allOf": [
                {
                    "if": {
                        "properties": {"record_id": {"pattern": "^intern-exec:"}},
                        "required": ["record_id"],
                    },
                    "then": {
                        "properties": {
                            "authority": {"const": "mloky"},
                            "kind": {"const": "mloky_execution"},
                        }
                    },
                }
            ]
        }
    )
    authority: Identifier
    kind: Identifier
    record_id: ReferenceRecordId
    revision: Identifier
    digest_sha256: Digest

    @model_validator(mode="after")
    def native_execution_scope(self) -> ExactReference:
        if ":" in self.record_id and (self.authority, self.kind) != ("mloky", "mloky_execution"):
            raise ValueError("native Intern execution reference authority/kind mismatch")
        return self


class RevisionReference(Contract):
    """Exact identity of one complete retained revision, provenance included.

    See docs/decisions/0001. ``reference.digest_sha256`` keeps its payload-only
    meaning; ``document_digest_sha256`` binds producer, operation and time too.
    It proves the document Forge recorded, not external producer attestation.
    """

    schema_version: Literal["forge.revision-reference.v1"] = "forge.revision-reference.v1"
    reference: ExactReference
    document_digest_sha256: Digest
