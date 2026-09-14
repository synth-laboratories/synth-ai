"""Swarm usage carries the backend's unattributed_sessions token count.

Since 2026-07-26 the backend's SmrSwarmTokenUsageResponse includes
tokens.unattributed_sessions, and the SDK's exact-field token-usage parser
refused it ("swarm token usage fields drifted: extra=['unattributed_sessions']"),
so research.swarms.usage() raised for every swarm.
"""

from __future__ import annotations

import pytest
from synth_ai.sdk.research.contracts.usage import SwarmUsage

_COUNTS = {
    "input_tokens": 100,
    "cached_input_tokens": 40,
    "non_cached_input_tokens": 60,
    "output_tokens": 30,
    "reasoning_output_tokens": 10,
    "non_reasoning_output_tokens": 20,
    "snapshots": 2,
}


def _usage(**token_overrides):
    tokens = {
        "sessions_seen": 2,
        "session_snapshots_count": 2,
        "unattributed_sessions": 1,
        "totals": dict(_COUNTS),
        "by_model": {"unattributed": dict(_COUNTS)},
    }
    tokens.update(token_overrides)
    return {
        "schema_version": 1,
        "run_id": "run-1",
        "project_id": "proj-1",
        "money": {
            "nominal_cents": 7,
            "billed_cents": 9,
            "internal_cost_cents": 5,
            "nominal_pico_usd": 70_000_000_000,
            "billed_pico_usd": 90_000_000_000,
            "internal_cost_pico_usd": 50_000_000_000,
        },
        "tokens": tokens,
        "actors": [],
        "freshness": {
            "source": "usage_facts",
            "as_of": "2026-09-12T01:00:00+00:00",
            "record_count": 2,
            "run_is_terminal": True,
        },
    }


def test_usage_with_unattributed_sessions_parses() -> None:
    usage = SwarmUsage.from_wire(_usage())

    assert usage.tokens.unattributed_sessions == 1
    assert usage.tokens.sessions_seen == 2


def test_usage_round_trips_through_to_wire() -> None:
    wire = _usage()

    assert SwarmUsage.from_wire(wire).to_wire() == wire
    assert SwarmUsage.from_wire(SwarmUsage.from_wire(wire).to_wire()) == SwarmUsage.from_wire(wire)


def test_usage_without_unattributed_sessions_still_parses() -> None:
    legacy = _usage()
    del legacy["tokens"]["unattributed_sessions"]

    assert SwarmUsage.from_wire(legacy).tokens.unattributed_sessions == 0


def test_token_usage_drift_is_still_refused() -> None:
    with pytest.raises(ValueError, match="swarm token usage fields drifted"):
        SwarmUsage.from_wire(_usage(unexpected=1))


def test_unattributed_sessions_is_typed() -> None:
    with pytest.raises(ValueError, match="non-negative integer"):
        SwarmUsage.from_wire(_usage(unattributed_sessions=None))
    with pytest.raises(ValueError, match="cannot exceed sessions_seen"):
        SwarmUsage.from_wire(_usage(unattributed_sessions=3))
