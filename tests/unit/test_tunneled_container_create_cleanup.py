from __future__ import annotations

import sys
import types

import pytest

from synth_ai.core.tunnels.tunneled_api import TunnelBackend, TunneledContainer


class _FakeLease:
    def __init__(self) -> None:
        self.lease_id = "lease-1"
        self.route_token = "rt_1"
        self.public_url = "https://st.usesynth.ai/s/rt_1"
        self.public_base_url = "https://st.usesynth.ai"
        self.agent_url = "wss://st.usesynth.ai/agent"
        self.agent_token = "agent-token"
        self.worker_token = "worker-token"
        self.limits = {"max_inflight": 16}


class _FakeClient:
    instances: list["_FakeClient"] = []

    def __init__(self, *_args, **_kwargs) -> None:
        self.closed_lease_ids: list[str] = []
        self.created_lease = _FakeLease()
        _FakeClient.instances.append(self)

    async def create_lease(self, **_kwargs) -> _FakeLease:
        return self.created_lease

    async def close_lease(self, lease_id: str) -> None:
        self.closed_lease_ids.append(lease_id)


class _FakeAgent:
    def __init__(self) -> None:
        self.stopped = False

    def stop(self) -> None:
        self.stopped = True


def _patch_synth_tunnel_helpers(monkeypatch: pytest.MonkeyPatch) -> None:
    import synth_ai.core.tunnels.synth_tunnel as synth_tunnel

    _FakeClient.instances.clear()
    monkeypatch.setattr(synth_tunnel, "SynthTunnelClient", _FakeClient)
    monkeypatch.setattr(synth_tunnel, "get_client_instance_id", lambda: "client-12345678")
    monkeypatch.setattr(synth_tunnel, "_collect_container_keys", lambda _env_key: [])
    monkeypatch.setattr(synth_tunnel, "hostname_from_url", lambda _url: "st.usesynth.ai")


@pytest.mark.asyncio
async def test_create_synth_tunnel_cleans_up_lease_when_agent_start_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_synth_tunnel_helpers(monkeypatch)

    async def _inline_to_thread(func, *args, **kwargs):
        return func(*args, **kwargs)

    monkeypatch.setattr("synth_ai.core.tunnels.tunneled_api.asyncio.to_thread", _inline_to_thread)
    monkeypatch.setitem(
        sys.modules,
        "synth_ai_py",
        types.SimpleNamespace(synth_tunnel_start=lambda *_a, **_k: (_ for _ in ()).throw(RuntimeError("start failed"))),
    )

    with pytest.raises(RuntimeError, match="start failed"):
        await TunneledContainer.create(
            local_port=8001,
            backend=TunnelBackend.SynthTunnel,
            api_key="sk_test",
        )

    assert len(_FakeClient.instances) == 1
    assert _FakeClient.instances[0].closed_lease_ids == ["lease-1"]


@pytest.mark.asyncio
async def test_create_synth_tunnel_stops_agent_and_cleans_up_lease_when_online_check_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_synth_tunnel_helpers(monkeypatch)
    fake_agent = _FakeAgent()

    async def _inline_to_thread(func, *args, **kwargs):
        return func(*args, **kwargs)

    monkeypatch.setattr("synth_ai.core.tunnels.tunneled_api.asyncio.to_thread", _inline_to_thread)
    monkeypatch.setitem(
        sys.modules,
        "synth_ai_py",
        types.SimpleNamespace(synth_tunnel_start=lambda *_a, **_k: fake_agent),
    )
    monkeypatch.setenv("SYNTH_TUNNEL_AGENT_ONLINE_TIMEOUT_SEC", "1")

    import httpx

    class _FailingAsyncClient:
        def __init__(self, *args, **kwargs) -> None:
            return None

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb) -> None:
            return None

        async def get(self, *args, **kwargs):
            raise RuntimeError("agent offline")

    monkeypatch.setattr(httpx, "AsyncClient", _FailingAsyncClient)

    with pytest.raises(RuntimeError, match="did not come online"):
        await TunneledContainer.create(
            local_port=8001,
            backend=TunnelBackend.SynthTunnel,
            api_key="sk_test",
        )

    assert fake_agent.stopped is True
    assert len(_FakeClient.instances) == 1
    assert _FakeClient.instances[0].closed_lease_ids == ["lease-1"]

