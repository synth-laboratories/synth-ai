"""Pin the SDK-first contract candidate; backend regeneration follows SYN-4884."""

from __future__ import annotations

import pytest
from synth_ai.contracts.fingerprint import schema_sha256

pytestmark = pytest.mark.contract
# SYN-4612 added KeyedSearchOptions.deadline_seconds (DEEP, 1..600); the prior value
# was 96d2a1d7... SYN-4571 changed the index.v1 export (retention modes, retention routes, tolerant
# retention block); the previous value was 4b39082e... The backend must regenerate
# its contract to this value before it vendors a wheel built from this source.
# SYN-4884 adds core.v1 to capability observations. Regenerate backend from
# this candidate before qualification; this source pin is not deployed evidence.
# SYN-4868 adds the optional control command and distinct physical quiescence state.
INDEX_V1_SCHEMA_SHA256 = "c79663ec0f6690d7dce649afb4afd5c6c149595287e000159b503039ee7380d9"


def test_index_v1_fingerprint_matches_the_candidate_export():
    assert schema_sha256("index.v1") == INDEX_V1_SCHEMA_SHA256
