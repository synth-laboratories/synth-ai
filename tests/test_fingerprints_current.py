"""Installed negotiation uses the reviewed bytes, independent of the serializer."""

import json
from importlib.resources import files
from pathlib import Path
from typing import get_args

import pydantic
import pytest
from synth_ai.contracts import fingerprint
from synth_ai.types.capabilities import ContractName

pytestmark = pytest.mark.contract


def test_shipped_fingerprints_match_retained_contract_objects():
    root = Path(__file__).resolve().parents[1]
    shipped = json.loads(files("synth_ai.contracts").joinpath("fingerprints.json").read_text())
    assert set(shipped) == set(get_args(ContractName))
    for contract in get_args(ContractName):
        retained = (root / "contracts" / (contract.replace(".", "-") + ".json")).read_text()
        assert fingerprint.schema_sha256(contract) == shipped[contract]
        assert fingerprint.digest_exported_contract(retained) == shipped[contract]


@pytest.mark.skipif(
    pydantic.__version__ != fingerprint.QUALIFICATION_PYDANTIC_VERSION,
    reason="Live schema generation is qualified with the pinned dev serializer",
)
def test_pinned_live_exports_match_the_shipped_table():
    for contract in get_args(ContractName):
        assert fingerprint.schema_sha256_live(contract) == fingerprint.schema_sha256(contract)


def test_runtime_negotiation_does_not_call_the_live_exporter(monkeypatch):
    def forbidden(*args):
        raise AssertionError("A client call must not generate schemas")

    monkeypatch.setattr(fingerprint, "export_contract_json", forbidden)
    monkeypatch.setattr(fingerprint._FINGERPRINTS, "digest", forbidden)
    fingerprint._shipped_fingerprints.cache_clear()
    for contract in get_args(ContractName):
        assert len(fingerprint.schema_sha256(contract)) == 64
