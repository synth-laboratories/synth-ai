"""Canonical protocol schema fingerprints. See backend SDK index_contract.md."""

from __future__ import annotations

import json
from functools import lru_cache
from hashlib import sha256

from pydantic import JsonValue, TypeAdapter

from synth_ai.types.capabilities import ContractName


@lru_cache(maxsize=2)
def schema_sha256(contract: ContractName) -> str:
    """Hash the exported schema and operation inventory, not the release status.

    See: backend/notes/specifications/sdk/index_contract.md."""
    if contract == "index.v1":
        from .schema import export_contract
    else:
        from .swarms_schema import export_contract
    value = TypeAdapter(dict[str, JsonValue]).validate_json(export_contract())
    value.pop("status", None)
    return sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
