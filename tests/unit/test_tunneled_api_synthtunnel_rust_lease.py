from __future__ import annotations

import asyncio

import pytest


@pytest.mark.unit
@pytest.mark.asyncio
async def test_synthtunnel_create_and_close_uses_rust_lease(monkeypatch: pytest.MonkeyPatch) -> None:
    from synth_ai.core.tunnels.tunneled_api import TunneledContainer, TunnelBackend

    calls: dict[str, object] = {}

    async def fake_to_thread(func, /, *args, **kwargs):  # type: ignore[no-untyped-def]
        return func(*args, **kwargs)

    monkeypatch.setattr("synth_ai.core.tunnels.tunneled_api.asyncio.to_thread", fake_to_thread)
    monkeypatch.setattr(
        "synth_ai.core.tunnels.synth_tunnel.get_client_instance_id",
        lambda: "client-1",
    )

    import synth_ai_py

    def fake_create_lease(  # type: ignore[no-untyped-def]
        api_key,
        backend_base,
        client_instance_id,
        local_port,
        local_host="127.0.0.1",
        requested_ttl_seconds=3600,
        metadata=None,
        capabilities=None,
        timeout_s=None,
    ):
        calls["create_lease"] = {
            "api_key": api_key,
            "backend_base": backend_base,
            "client_instance_id": client_instance_id,
            "local_port": local_port,
            "local_host": local_host,
        }
        return {
            "lease_id": "lease_1",
            "route_token": "rt_1",
            "public_base_url": "https://st.usesynth.ai/s",
            "public_url": "https://st.usesynth.ai/s/rt_1",
            "agent_url": "wss://agent.example/ws",
            "agent_token": "agent_tok",
            "worker_token": "worker_tok",
            "expires_at": "2026-02-14T00:00:00Z",
            "limits": {"max_inflight": 7},
            "heartbeat": {},
        }

    class FakeAgent:
        def stop(self) -> None:
            calls["agent_stop"] = True

    def fake_start(  # type: ignore[no-untyped-def]
        agent_url,
        agent_token,
        lease_id,
        local_host,
        local_port,
        public_url,
        worker_token,
        container_keys,
        max_inflight,
    ):
        calls["start_agent"] = {
            "agent_url": agent_url,
            "agent_token": agent_token,
            "lease_id": lease_id,
            "local_host": local_host,
            "local_port": local_port,
            "public_url": public_url,
            "worker_token": worker_token,
            "container_keys": container_keys,
            "max_inflight": max_inflight,
        }
        return FakeAgent()

    def fake_close_lease(api_key, backend_base, lease_id, timeout_s=None):  # type: ignore[no-untyped-def]
        calls["close_lease"] = {
            "api_key": api_key,
            "backend_base": backend_base,
            "lease_id": lease_id,
        }

    monkeypatch.setattr(synth_ai_py, "synth_tunnel_create_lease", fake_create_lease)
    monkeypatch.setattr(synth_ai_py, "synth_tunnel_start", fake_start)
    monkeypatch.setattr(synth_ai_py, "synth_tunnel_close_lease", fake_close_lease)

    import httpx

    class FakeResp:
        status_code = 200

    class FakeAsyncClient:
        def __init__(self, *args, **kwargs):  # type: ignore[no-untyped-def]
            pass

        async def __aenter__(self):  # type: ignore[no-untyped-def]
            return self

        async def __aexit__(self, exc_type, exc, tb):  # type: ignore[no-untyped-def]
            return False

        async def get(self, url, headers=None):  # type: ignore[no-untyped-def]
            calls["health_check"] = {"url": url, "headers": headers}
            return FakeResp()

    monkeypatch.setattr(httpx, "AsyncClient", FakeAsyncClient)

    tunnel = await TunneledContainer.create(
        local_port=8001,
        backend=TunnelBackend.SynthTunnel,
        api_key="sk_test",
        env_api_key="env_test_key",
        backend_url="https://api.example.com",
    )

    assert tunnel.backend == TunnelBackend.SynthTunnel
    assert tunnel.url == "https://st.usesynth.ai/s/rt_1"
    assert tunnel.worker_token == "worker_tok"
    assert calls["create_lease"] == {
        "api_key": "sk_test",
        "backend_base": "https://api.example.com",
        "client_instance_id": "client-1",
        "local_port": 8001,
        "local_host": "127.0.0.1",
    }
    assert calls["start_agent"]["max_inflight"] == 7

    tunnel.close()
    await asyncio.sleep(0.01)  # allow scheduled close task to run

    assert calls.get("agent_stop") is True
    assert calls["close_lease"] == {
        "api_key": "sk_test",
        "backend_base": "https://api.example.com",
        "lease_id": "lease_1",
    }

