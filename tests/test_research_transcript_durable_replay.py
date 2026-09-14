"""The SDK decodes durable transcript replay pages served after terminality.

Backend lockstep: services/smr/api_schemas.py TranscriptReplayMode /
TranscriptProjectionAuthority gained ``durable_replay`` so a live run whose
Redis stream was trimmed (or a caller resuming with a durable cursor) replays
durable rows with an explicit mode instead of an empty live tail.
"""

from __future__ import annotations

import pytest

from synth_ai.sdk.research.contracts.transcript import (
    TranscriptFreshness,
    TranscriptProjectionAuthority,
    TranscriptReplayMode,
)


def test_durable_replay_freshness_round_trips() -> None:
    wire = {
        "observed_at": "2026-09-12T21:45:00+00:00",
        "projection_authority": "smr_transcript_events.durable_replay.v1",
        "replay_mode": "durable_replay",
        "live_tail_available": False,
    }
    freshness = TranscriptFreshness.from_wire(wire)
    assert freshness.replay_mode is TranscriptReplayMode.DURABLE_REPLAY
    assert (
        freshness.projection_authority is TranscriptProjectionAuthority.DURABLE_REPLAY
    )
    assert freshness.to_wire() == wire


def test_durable_replay_cannot_advertise_live_tail() -> None:
    with pytest.raises(ValueError, match="cannot advertise a live tail"):
        TranscriptFreshness.from_wire(
            {
                "observed_at": "2026-09-12T21:45:00+00:00",
                "projection_authority": "smr_transcript_events.durable_replay.v1",
                "replay_mode": "durable_replay",
                "live_tail_available": True,
            }
        )
