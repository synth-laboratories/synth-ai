"""Canonical protocol schema fingerprints. See backend SDK index_contract.md."""

from __future__ import annotations

import json
from functools import lru_cache
from hashlib import sha256
from importlib.resources import files

from synth_ai.types.capabilities import ContractName


@lru_cache(maxsize=2)
def schema_sha256(contract: ContractName) -> str:
    """Hash the packaged protocol snapshot, independently of Pydantic's renderer.

    See: backend/notes/specifications/sdk/index_contract.md."""
    name = "index-v1.json" if contract == "index.v1" else "swarms-v1.json"
    value = json.loads(files("synth_ai.contracts").joinpath("snapshots", name).read_text())
    if not isinstance(value, dict):
        raise ValueError("Malformed packaged contract snapshot")
    value.pop("status", None)
    return sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
