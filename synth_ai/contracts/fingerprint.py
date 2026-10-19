"""Canonical protocol schema fingerprints. See backend SDK index_contract.md."""

from __future__ import annotations

import json
import threading
from collections.abc import Callable
from hashlib import sha256

from pydantic import JsonValue, TypeAdapter

from synth_ai.types.capabilities import ContractName


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


def schema_sha256(contract: ContractName) -> str:
    """Hash the exported schema and operation inventory, not the release status.

    See: the public contract snapshots in contracts/"""
    return _FINGERPRINTS.digest(contract)
