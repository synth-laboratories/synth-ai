from __future__ import annotations

from datetime import timezone

import pytest

from synth_ai.core.tunnels.synth_tunnel import _parse_lease_response


def _valid_payload() -> dict:
    return {
        "lease_id": "lease-1",
        "route_token": "rt_123",
        "public_base_url": "https://st.usesynth.ai",
        "public_url": "https://st.usesynth.ai/s/rt_123",
        "worker_token": "wk_abc",
        "expires_at": "2026-02-22T10:00:00Z",
        "agent_connect": {"url": "wss://st.usesynth.ai/ws", "agent_token": "agt_abc"},
        "limits": {"max_inflight": 16},
        "heartbeat": {"interval_seconds": 30},
    }


def test_parse_lease_response_rejects_missing_worker_token() -> None:
    payload = _valid_payload()
    payload.pop("worker_token")

    with pytest.raises(RuntimeError, match="missing required field: worker_token"):
        _parse_lease_response(payload)


def test_parse_lease_response_rejects_empty_required_string() -> None:
    payload = _valid_payload()
    payload["route_token"] = "   "

    with pytest.raises(RuntimeError, match="empty required field: route_token"):
        _parse_lease_response(payload)


def test_parse_lease_response_rejects_invalid_expires_at() -> None:
    payload = _valid_payload()
    payload["expires_at"] = "not-a-date"

    with pytest.raises(RuntimeError, match="invalid expires_at"):
        _parse_lease_response(payload)


def test_parse_lease_response_accepts_valid_payload() -> None:
    lease = _parse_lease_response(_valid_payload())

    assert lease.lease_id == "lease-1"
    assert lease.route_token == "rt_123"
    assert lease.agent_token == "agt_abc"
    assert lease.expires_at.tzinfo is not None
    assert lease.expires_at.tzinfo == timezone.utc
    assert lease.limits["max_inflight"] == 16
