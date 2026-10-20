"""Canonical protocol schema fingerprints. See backend SDK index_contract.md."""

from __future__ import annotations

import json
import threading
from collections.abc import Callable
from functools import lru_cache
from hashlib import sha256
from importlib.resources import files
from pathlib import Path
from typing import get_args

from pydantic import JsonValue, TypeAdapter

from synth_ai.types.base import Digest
from synth_ai.types.capabilities import ContractName

QUALIFICATION_PYDANTIC_VERSION = "2.12.5"


def export_contract_json(contract: ContractName) -> str:
    if contract == "index.v1":
        from .schema import export_contract
    elif contract == "swarms.v1":
        from .swarms_schema import export_contract
    elif contract == "core.v1":
        from .core_schema import export_contract
    else:
        raise ValueError(f"Unknown contract: {contract}")
    return export_contract()


def digest_exported_contract(exported: str) -> str:
    value = TypeAdapter(dict[str, JsonValue]).validate_json(exported)
    value.pop("status", None)
    return sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


class SchemaFingerprints:
    """Process-wide, thread-safe contract fingerprints.

    Pydantic's ``model_json_schema`` is not safe to run from several threads on a
    cold process: concurrent exports intermittently omit ``$defs`` entries and
    yield a different digest, which ``lru_cache`` would then pin for the process
    (SDK ``schema_mismatch`` on one of two concurrent searches.create calls). One
    lock serializes the export; the first complete digest is the only one cached."""

    def __init__(self, exporter: Callable[[ContractName], str] = export_contract_json) -> None:
        self._exporter = exporter
        self._lock = threading.Lock()
        self._digests: dict[ContractName, str] = {}

    def digest(self, contract: ContractName) -> str:
        with self._lock:
            cached = self._digests.get(contract)
            if cached is None:
                cached = digest_exported_contract(self._exporter(contract))
                self._digests[contract] = cached
            return cached


_FINGERPRINTS = SchemaFingerprints()


def schema_sha256_live(contract: ContractName) -> str:
    """Generate a fingerprint with the qualified serializer, for release tooling.

    See: the public contract snapshots in contracts/"""
    return _FINGERPRINTS.digest(contract)


@lru_cache(maxsize=1)
def _shipped_fingerprints() -> dict[ContractName, str]:
    value = TypeAdapter(dict[ContractName, Digest]).validate_json(
        files("synth_ai.contracts").joinpath("fingerprints.json").read_bytes()
    )
    if set(value) != set(get_args(ContractName)):
        raise ValueError("Shipped fingerprints must identify every supported contract")
    return value


def schema_sha256(contract: ContractName) -> str:
    """Read the shipped identity; never export pydantic schemas during a call."""
    return _shipped_fingerprints()[contract]


def main() -> None:
    import argparse

    import pydantic

    parser = argparse.ArgumentParser(description="Generate qualified SDK contract fingerprints")
    parser.add_argument("--write", action="store_true", help="Update the source fingerprint table")
    arguments = parser.parse_args()
    if pydantic.__version__ != QUALIFICATION_PYDANTIC_VERSION:
        parser.error(f"Generation requires pydantic=={QUALIFICATION_PYDANTIC_VERSION}")
    value = {contract: schema_sha256_live(contract) for contract in get_args(ContractName)}
    encoded = json.dumps(value, indent=2, sort_keys=True) + "\n"
    if arguments.write:
        Path(__file__).with_name("fingerprints.json").write_text(encoded)
    else:
        print(encoded, end="")


if __name__ == "__main__":
    main()
