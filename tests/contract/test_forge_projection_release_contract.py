"""SYN-3928 typed installed consumer of the truthful Forge projection recipe."""

from pathlib import Path

import pytest
from pydantic import ValidationError
from synth_ai.sdk.index.research import ForgeProjectionRecipe, decode_derivation
from synth_ai.sdk.index.research.build import FrozenBuildError, build_release
from synth_ai.sdk.index.research.contracts import canonical_bytes, decode_disclosure

FIXTURE = Path(__file__).resolve().parents[1] / "fixtures/forge_public_projection/binding.json"


def test_syn3928_typed_projection_retains_exact_source_and_archive():
    binding = decode_derivation(FIXTURE.read_bytes())
    assert isinstance(binding.recipe, ForgeProjectionRecipe), "SYN-3928"
    assert (
        binding.recipe.source_digest_sha256
        == "35ea1c5921010a42a10a995598ff7ff016c2a392780599cbc42f492deac959aa"
    )
    assert (
        binding.recipe.source_revision_reference.document_digest_sha256
        == "12db87393518f4f306805cc36942f677ca13f840c98a003b14ee2788c9b0a6a6"
    )
    assert canonical_bytes(decode_derivation(canonical_bytes(binding))) == canonical_bytes(binding)
    assert all(
        representation.kind == "structured_text"
        for representation in binding.disclosure.representations
    )


def test_syn3928_projection_never_claims_offline_copy_or_reproduction(tmp_path):
    binding = decode_derivation(FIXTURE.read_bytes())
    with pytest.raises(FrozenBuildError) as failure:
        build_release(tmp_path, tmp_path / "out", binding=binding, descriptor=b"", manifest=None)
    assert failure.value.code == "forge_projection_not_frozen_copy", "SYN-3928"
    assert not (tmp_path / "out").exists()


def test_syn3928_v1_disclosure_cannot_opt_into_new_structured_parser():
    binding = decode_derivation(FIXTURE.read_bytes())
    disclosure = binding.disclosure.model_dump(mode="json")
    disclosure["schema_version"] = "synth.contribution.release-disclosure.v1"
    disclosure.pop("credited_principal_ids")
    with pytest.raises(ValidationError):
        decode_disclosure(disclosure)
