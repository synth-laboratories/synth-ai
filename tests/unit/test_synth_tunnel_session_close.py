from __future__ import annotations

import asyncio
from datetime import datetime, timezone

import pytest

from synth_ai.core.tunnels.synth_tunnel import SynthTunnelLease, SynthTunnelSession


def _sample_lease() -> SynthTunnelLease:
    return SynthTunnelLease(
        lease_id="lease-1",
        route_token="route-1",
        public_base_url="https://st.usesynth.ai",
        public_url="https://st.usesynth.ai/s/route-1",
        agent_url="wss://st.usesynth.ai/ws",
        agent_token="agent-token",
        worker_token="worker-token",
        expires_at=datetime.now(timezone.utc),
        limits={},
        heartbeat={},
    )


def test_close_uses_asyncio_run_when_no_active_loop(monkeypatch: pytest.MonkeyPatch) -> None:
    called: dict[str, bool] = {"ran": False}

    def fake_run(coro: object) -> None:
        called["ran"] = True
        # Ensure we don't leak an un-awaited coroutine object in test process.
        close = getattr(coro, "close", None)
        if callable(close):
            close()

    session = SynthTunnelSession(
        lease=_sample_lease(),
        agent=object(),
        task=object(),  # not used by close() in this test path
        client=object(),
        stop_event=asyncio.Event(),
    )
    monkeypatch.setattr(asyncio, "run", fake_run)
    session.close()
    assert called["ran"] is True


@pytest.mark.asyncio
async def test_close_raises_inside_active_event_loop() -> None:
    session = SynthTunnelSession(
        lease=_sample_lease(),
        agent=object(),
        task=object(),  # not used by this test path
        client=object(),
        stop_event=asyncio.Event(),
    )
    with pytest.raises(RuntimeError, match="await close_async"):
        session.close()
