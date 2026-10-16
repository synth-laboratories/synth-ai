"""The SDK error model must not drift the index.v1 fingerprint the backend attests (SYN-4354)."""

from __future__ import annotations

from synth_ai.contracts.fingerprint import schema_sha256

# Equal to the backend's packages.customer_api.contracts.fingerprint value; a
# mismatch hides every hosted MCP tool (schema_mismatch in supports_operation).
INDEX_V1_SCHEMA_SHA256 = "4b39082e1fcb0b443ed8897ddb92a45ae72062decf5e01c21faeb0a0112a23ab"


def test_index_v1_fingerprint_matches_the_backend_attested_value():
    assert schema_sha256("index.v1") == INDEX_V1_SCHEMA_SHA256

