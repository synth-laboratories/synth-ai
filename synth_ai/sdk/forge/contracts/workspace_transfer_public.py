# Generated DTO excerpt from backend packages/forge_contract/workspace_transfer.py.
# The types describe wire integrity, not access authority or an SDK transport.
from typing import Annotated, Literal

from pydantic import AwareDatetime, Field, StringConstraints

from .contracts import Contract, Digest, Identifier, Scope
from .projects import ProjectPrincipal


class PublicImportRequest(Contract):
    """Integrity-only public selection; current principal and proof are server-owned."""

    schema_version: Literal["forge.public-transfer-import.v1"] = "forge.public-transfer-import.v1"
    operation_id: Identifier
    source_owner_id: Identifier
    declared_cut: Annotated[int, Field(strict=True, ge=0, le=9223372036854775807)]
    bundle_digest_sha256: Digest
    source_manifest_header_json: Annotated[
        str, StringConstraints(strip_whitespace=False, min_length=2, max_length=16384)
    ]


class ImportPreview(Contract):
    schema_version: Literal["forge.transfer-preview.v1"] = "forge.transfer-preview.v1"
    operation_id: Identifier
    source_owner_id: Identifier
    scope: Scope
    declared_cut: Annotated[int, Field(strict=True, ge=0, le=9223372036854775807)]
    bundle_digest_sha256: Digest
    revision_count: Annotated[int, Field(strict=True, ge=0, le=100000)]


class NativeImportReceipt(Contract):
    """Exact retained importer custody; its historical credential is not a grant."""

    schema_name: Literal["forge.native-import-custody.v1"] = Field(alias="schema")
    scope: Scope
    operation_id: Identifier
    manifest_sha256: Digest
    authorization_context_sha256: Digest
    source_owner_id: Identifier
    source_cut: Annotated[int, Field(strict=True, ge=0, le=9223372036854775807)]
    custody_owner_id: Identifier
    source_inventory_sha256: Digest
    importer: ProjectPrincipal
    revision_count: Annotated[int, Field(strict=True, ge=0, le=100000)]
    destination_inventory_sha256: Digest
    recorded_at: AwareDatetime
    producer_admission_renewed: Literal[False]
    importer_evidence_sha256: Digest
