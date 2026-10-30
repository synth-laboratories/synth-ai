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
# SYN-4888 makes response schemas tolerant while request construction stays strict.
# SYN-4885 adds core response metadata and clone-token recovery resources.
# The pin was refreshed to the recomputed export; bd9a6454... was stale.
# Existing dev rollout-usage DTOs have a different reviewed export. Curated
# v0.23.8 promotion retains its deployed Index DTOs and corresponding pin.
# SYN-5019 adds the optional SearchSnapshot creation timestamp.
# SYN-5220 (PR #592) adds ContentRef and Contribution.content_refs; e5caec17 moved
# fingerprints.json to this value but left the test pin at 503e43c8... (stale).
INDEX_V1_SCHEMA_SHA256 = "75563ddda18894c5cf10c18d3ce20a828a779772288c7f08b2007db7050b0f93"


def test_index_v1_fingerprint_matches_the_candidate_export():
    assert schema_sha256("index.v1") == INDEX_V1_SCHEMA_SHA256
