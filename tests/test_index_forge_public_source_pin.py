"""ForgePublicResearchSource carries the Forge provenance pin exactly as backend does.

Mirror of backend ``packages/contributions/forge_release.py`` (SYN-3603 finding):
``revision_reference`` must name the exported revision and ``archive_schema_version``
is pinned to ``forge.private-archive.v3``; both stay optional on the wire.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError
from synth_ai.sdk.index import ForgePublicResearchSource, ResearchDraftSpec
from synth_ai.sdk.index.contributions import ARCHIVE_V3
from synth_ai.sdk.research.contracts.forge.contracts import ExactReference
from synth_ai.sdk.research.contracts.forge.operations import RevisionReference

_DIGEST = "a" * 64
_OTHER_DIGEST = "b" * 64
_EXPORT = {
    "authority": "forge",
    "kind": "scientific_export",
    "record_id": "export-1",
    "revision": "rev-1",
    "digest_sha256": _DIGEST,
}
_SOURCE = {
    "schema_version": "synth.index.forge-source.v2",
    "organization_id": "org-1",
    "project_id": "project-1",
    "export": _EXPORT,
    "archive_publication_id": "00000000-0000-4000-8000-000000000001",
    "archive_manifest_digest_sha256": _DIGEST,
    "public_provenance": {
        "provenance_license": "CC-BY-4.0",
        "assets": [{"asset_id": "report", "digest_sha256": _DIGEST, "license": "CC-BY-4.0"}],
        "credits": [{"principal_id": "author", "role": "author", "license": "CC-BY-4.0"}],
    },
}
_PIN = {
    "schema_version": "forge.revision-reference.v1",
    "reference": _EXPORT,
    "document_digest_sha256": _OTHER_DIGEST,
}


def test_pre_pin_public_source_still_decodes_with_both_fields_absent() -> None:
    source = ForgePublicResearchSource.model_validate(_SOURCE)

    assert source.revision_reference is None
    assert source.archive_schema_version is None
    assert "revision_reference" in source.model_dump(mode="json")


def test_pinned_source_round_trips_the_exact_pin() -> None:
    source = ForgePublicResearchSource.model_validate(
        {**_SOURCE, "revision_reference": _PIN, "archive_schema_version": ARCHIVE_V3}
    )

    assert isinstance(source.revision_reference, RevisionReference)
    assert source.revision_reference.reference == ExactReference.model_validate(_EXPORT)
    assert source.revision_reference.document_digest_sha256 == _OTHER_DIGEST
    assert source.archive_schema_version == "forge.private-archive.v3"
    document = source.model_dump(mode="json")
    assert document["revision_reference"] == _PIN
    assert document["archive_schema_version"] == ARCHIVE_V3
    assert ForgePublicResearchSource.model_validate(document) == source


def test_reference_naming_another_revision_never_validates() -> None:
    with pytest.raises(ValidationError, match="must name the exported revision"):
        ForgePublicResearchSource.model_validate(
            {
                **_SOURCE,
                "revision_reference": {**_PIN, "reference": {**_EXPORT, "revision": "rev-2"}},
                "archive_schema_version": ARCHIVE_V3,
            }
        )


@pytest.mark.parametrize("version", ["forge.private-archive.v2", "", "v3"])
def test_archive_below_v3_is_refused(version: str) -> None:
    with pytest.raises(ValidationError):
        ForgePublicResearchSource.model_validate(
            {**_SOURCE, "revision_reference": _PIN, "archive_schema_version": version}
        )


def test_draft_spec_keeps_the_pin_through_the_versioned_source_union() -> None:
    spec = ResearchDraftSpec.model_validate(
        {
            "bundle_digest": f"sha256:{_DIGEST}",
            "source": {**_SOURCE, "revision_reference": _PIN, "archive_schema_version": ARCHIVE_V3},
        }
    )

    assert isinstance(spec.source, ForgePublicResearchSource)
    assert spec.source.revision_reference == RevisionReference.model_validate(_PIN)
    assert spec.model_dump(mode="json")["source"]["archive_schema_version"] == ARCHIVE_V3
