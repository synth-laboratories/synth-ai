"""The SDK error model must not drift the index.v1 fingerprint the backend attests (SYN-4354)."""

from __future__ import annotations

from synth_ai.contracts.fingerprint import schema_sha256

# SYN-4612 added KeyedSearchOptions.deadline_seconds (DEEP, 1..600); the prior value
# was 96d2a1d7... SYN-4571 changed the index.v1 export (retention modes, retention routes, tolerant
# retention block); the previous value was 4b39082e... The backend must regenerate
# its contract to this value before it vendors a wheel built from this source.
# Equal to the backend's packages.customer_api.contracts.fingerprint value; a
# mismatch hides every hosted MCP tool (schema_mismatch in supports_operation).
INDEX_V1_SCHEMA_SHA256 = "9cc9d30a4fed9f26cc04af5be244f0ad6b831307bb8d6fbebfacb0674e4e3dca"


def test_index_v1_fingerprint_matches_the_backend_attested_value():
    assert schema_sha256("index.v1") == INDEX_V1_SCHEMA_SHA256

