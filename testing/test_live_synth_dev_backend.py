from __future__ import annotations

import os
import uuid

import httpx
import pytest
from synth_ai import SynthClient
from synth_ai.sdk.containers import ContainerSpec, ContainerType


def _live_client() -> SynthClient:
    base_url = (os.getenv("SYNTH_SDK_LIVE_BACKEND_URL") or "").strip()
    api_key = (os.getenv("SYNTH_API_KEY") or "").strip()
    if not base_url:
        pytest.skip("SYNTH_SDK_LIVE_BACKEND_URL is not set")
    if not api_key:
        pytest.skip("SYNTH_API_KEY is not set")
    return SynthClient(api_key=api_key, base_url=base_url)


def _http_error_detail(exc: Exception) -> str:
    if isinstance(exc, httpx.HTTPStatusError):
        response = exc.response
        text = response.text.strip()
        if text:
            return f"{exc} body={text[:1000]}"
    return str(exc)


@pytest.mark.integration
def test_live_backend_read_surfaces() -> None:
    client = _live_client()

    assert client.tunnels.health()["status"] == "ok"
    assert isinstance(client.tunnels.list_leases(), list)
    assert isinstance(client.containers.list(), list)

    pools = client.pools.list()
    assert "items" in pools
    assert "cursor" in pools
    assert "limit" in pools

    capabilities = client.pools.get_capabilities()
    assert "supported_adapters" in capabilities
    assert "supported_rollout_request_keys" in capabilities

    queue_status = client.pools.get_queue_status()
    assert queue_status["ok"] is True
    assert "queue" in queue_status

    if pools["items"]:
        pool_id = pools["items"][0]["id"]
        pool = client.pools.get(pool_id)
        urls = client.pools.get_urls(pool_id)
        metrics = client.pools.metrics.get(pool_id)
        tasks = client.pools.tasks.list(pool_id)

        assert pool["id"] == pool_id
        assert urls["pool_id"] == pool_id
        assert metrics["pool_id"] == pool_id
        assert "items" in tasks


@pytest.mark.integration
def test_live_backend_container_create_delete() -> None:
    client = _live_client()
    name = f"codex-live-container-{uuid.uuid4().hex[:8]}"

    try:
        created = client.containers.create(
            ContainerSpec(
                name=name,
                task_type=ContainerType.harbor_code,
                definition={},
            )
        )
    except Exception as exc:  # noqa: BLE001
        pytest.fail(
            f"container create failed against live synth-dev backend: {_http_error_detail(exc)}"
        )

    try:
        assert created.id
        fetched = client.containers.get(created.id)
        assert fetched.id == created.id
    finally:
        client.containers.delete(created.id)


@pytest.mark.integration
def test_live_backend_synth_tunnel_lease_flow() -> None:
    client = _live_client()

    try:
        lease = client.tunnels.create_synth_lease(
            client_instance_id=f"codex-synth-{uuid.uuid4().hex[:8]}",
            local_host="127.0.0.1",
            local_port=8000,
            metadata={"source": "codex-live-test"},
        )
    except Exception as exc:  # noqa: BLE001
        pytest.fail(
            "synth tunnel lease create failed against live synth-dev backend: "
            f"{_http_error_detail(exc)}"
        )

    lease_id = lease["lease_id"]
    status = client.tunnels.get_synth_lease(lease_id)
    refreshed = client.tunnels.refresh_synth_worker_token(lease_id)
    closed = client.tunnels.close_synth_lease(lease_id)

    assert status["lease_id"] == lease_id
    assert refreshed["lease_id"] == lease_id
    assert closed["lease_id"] == lease_id


@pytest.mark.integration
def test_live_backend_managed_ngrok_lease_flow() -> None:
    client = _live_client()

    try:
        lease = client.tunnels.create_lease(
            client_instance_id=f"codex-ngrok-{uuid.uuid4().hex[:8]}",
            local_host="127.0.0.1",
            local_port=8000,
            requested_ttl_seconds=900,
        )
    except Exception as exc:  # noqa: BLE001
        pytest.fail(
            "managed tunnel lease create failed against live synth-dev backend: "
            f"{_http_error_detail(exc)}"
        )

    lease_id = lease["lease_id"]
    heartbeat = client.tunnels.heartbeat(
        lease_id,
        connected_to_edge=True,
        gateway_ready=True,
        local_ready=True,
    )
    refreshed = client.tunnels.refresh_lease(lease_id, requested_ttl_seconds=1200)
    released = client.tunnels.release_lease(lease_id)
    deleted = client.tunnels.delete_lease(lease_id)

    assert heartbeat["action"] in {"none", "restart_connector", "rotate_tunnel", "recreate_tunnel"}
    assert refreshed["lease_id"] == lease_id
    assert released["lease_id"] == lease_id
    assert deleted["lease_id"] == lease_id
