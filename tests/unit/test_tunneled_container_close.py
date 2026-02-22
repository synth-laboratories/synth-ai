from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from synth_ai.core.tunnels.tunneled_api import TunnelBackend, TunneledContainer


class _FakeAgent:
    def __init__(self) -> None:
        self.stopped = False

    def stop(self) -> None:
        self.stopped = True


class _FakeClient:
    def __init__(self) -> None:
        self.closed_lease_id: str | None = None

    async def close_lease(self, lease_id: str) -> None:
        self.closed_lease_id = lease_id


def _sample_tunneled_container() -> tuple[TunneledContainer, _FakeAgent, _FakeClient]:
    agent = _FakeAgent()
    client = _FakeClient()
    lease = SimpleNamespace(lease_id="lease-1")
    tunnel = TunneledContainer(
        url="https://st.usesynth.ai/s/rt_1",
        hostname="st.usesynth.ai",
        local_port=8001,
        backend=TunnelBackend.SynthTunnel,
        _synth_session={"agent": agent, "client": client, "lease": lease},
        _lease_id="lease-1",
        worker_token="worker-token",
    )
    return tunnel, agent, client


def test_close_uses_asyncio_run_when_no_active_loop(monkeypatch: pytest.MonkeyPatch) -> None:
    tunnel, _, _ = _sample_tunneled_container()
    called: dict[str, bool] = {"ran": False}

    def _fake_run(coro: object) -> None:
        called["ran"] = True
        close = getattr(coro, "close", None)
        if callable(close):
            close()

    monkeypatch.setattr(asyncio, "run", _fake_run)
    tunnel.close()
    assert called["ran"] is True


@pytest.mark.asyncio
async def test_close_raises_inside_active_event_loop() -> None:
    tunnel, _, _ = _sample_tunneled_container()
    with pytest.raises(RuntimeError, match="await close_async"):
        tunnel.close()


@pytest.mark.asyncio
async def test_close_async_waits_for_lease_close_and_clears_state() -> None:
    tunnel, agent, client = _sample_tunneled_container()

    await tunnel.close_async()

    assert agent.stopped is True
    assert client.closed_lease_id == "lease-1"
    assert tunnel._synth_session is None
    assert tunnel._lease_id is None
    assert tunnel.worker_token is None

