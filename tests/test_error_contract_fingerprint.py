"""The SDK error model must not drift the index.v1 fingerprint the backend attests (SYN-4354)."""

from __future__ import annotations

from synth_ai.contracts.fingerprint import schema_sha256

# SYN-4571 changed the index.v1 export (retention modes, retention routes, tolerant
# retention block); the previous value was 4b39082e... The backend must regenerate
# its contract to this value before it vendors a wheel built from this source.
# Equal to the backend's packages.customer_api.contracts.fingerprint value; a
# mismatch hides every hosted MCP tool (schema_mismatch in supports_operation).
INDEX_V1_SCHEMA_SHA256 = "96d2a1d78f5e00c98af5335dbe81d5d7cc72860a147fbb89b0548816f37bf369"


def test_index_v1_fingerprint_matches_the_backend_attested_value():
    assert schema_sha256("index.v1") == INDEX_V1_SCHEMA_SHA256

