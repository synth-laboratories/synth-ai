"""Exact provenance identities retained by Index; references never grant access.

See: the public contract snapshots in contracts/ Moved from the preserved
Research Forge contracts without changing their wire constraints.
"""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import ConfigDict, StringConstraints, WrapValidator, model_validator

from synth_ai.types.base import Identifier, Model, _identifier

Digest = Annotated[str, StringConstraints(pattern="^[0-9a-f]{64}$")]


def _record_identifier(value, handler):
    # This identity embeds a digest, rather than an opaque server identifier.
    if isinstance(value, str) and value.startswith("intern-exec:"):
        return handler(value)
    return _identifier(value, handler)


ReferenceRecordId = Annotated[
    str,
    StringConstraints(pattern="^(?:[a-zA-Z0-9][a-zA-Z0-9_.-]{0,127}|intern-exec:[0-9a-f]{64})$"),
    WrapValidator(_record_identifier),
]


class Contract(Model):
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
        if ":" in self.record_id and not self.record_id.startswith("intern-exec:"):
            raise ValueError("native Intern execution reference must retain its digest identity")
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
