from __future__ import annotations

import pytest
from synth_ai import AsyncSynthClient, SynthClient
from synth_ai.sdk.tunnels import TunnelProvider, TunnelsClient


def test_tunnels_client_managed_and_synth_flows(
    backend_url: str,
    api_key: str,
) -> None:
    client = TunnelsClient(api_key=api_key, backend_base=backend_url)

    health = client.health()
    assert health["status"] == "ok"
    assert "ngrok" in health["providers"]

    created_tunnel = client.create(subdomain="sdk-integration", local_port=8080)
    assert created_tunnel["public_url"] == "https://sdk-integration.ngrok.app"

    listed_tunnels = client.list()
    assert [item["id"] for item in listed_tunnels] == [created_tunnel["id"]]

    rotated = client.rotate(local_port=8080, reason="integration-test")
    assert rotated["status"] == "rotated"

    managed_lease = client.create_lease(
        client_instance_id="client-managed-1",
        local_host="127.0.0.1",
        local_port=8080,
        app_name="gepa-runtime",
        provider_preference=TunnelProvider.NGROK,
        requested_ttl_seconds=900,
        reuse_connector=True,
        idempotency_key="idem-1",
    )
    assert managed_lease["provider_preference"] == "ngrok"
    assert managed_lease["app_name"] == "gepa-runtime"

    listed_leases = client.list_leases(client_instance_id="client-managed-1")
    assert [item["lease_id"] for item in listed_leases] == [managed_lease["lease_id"]]

    heartbeat = client.heartbeat(
        managed_lease["lease_id"],
        connected_to_edge=True,
        gateway_ready=True,
        local_ready=True,
    )
    assert heartbeat["status"] == "heartbeat_ok"

    refreshed = client.refresh_lease(managed_lease["lease_id"], requested_ttl_seconds=1200)
    assert refreshed["requested_ttl_seconds"] == 1200

    released = client.release_lease(managed_lease["lease_id"])
    assert released["status"] == "released"

    deleted_lease = client.delete_lease(managed_lease["lease_id"])
    assert deleted_lease["status"] == "deleted"

    synth_lease = client.create_synth_lease(
        client_instance_id="client-synth-1",
        local_host="127.0.0.1",
        local_port=9000,
        metadata={"use_case": "ngrok-and-synthtunnel-coverage"},
        capabilities={"max_inflight": 32},
    )
    assert synth_lease["lease_id"].startswith("synth-lease-")
    assert synth_lease["worker_token"].startswith("worker-")

    synth_status = client.get_synth_lease(synth_lease["lease_id"])
    assert synth_status["status"] == "PENDING"

    refreshed_worker = client.refresh_synth_worker_token(synth_lease["lease_id"])
    assert refreshed_worker["worker_token"].endswith("-refreshed")

    closed_synth = client.close_synth_lease(synth_lease["lease_id"])
    assert closed_synth["status"] == "EXPIRED"

    deleted_tunnel = client.delete(created_tunnel["id"])
    assert deleted_tunnel["status"] == "deleted"


@pytest.mark.asyncio
async def test_async_synth_client_tunnel_smoke(backend_url: str, api_key: str) -> None:
    client = AsyncSynthClient(api_key=api_key, base_url=backend_url)

    health = await client.tunnels.health()
    assert health["status"] == "ok"

    synth_lease = await client.tunnels.create_synth_lease(
        client_instance_id="async-client",
        local_host="127.0.0.1",
        local_port=9100,
    )
    assert synth_lease["lease_id"].startswith("synth-lease-")

    queue_status = await client.pools.get_queue_status()
    assert queue_status["running"] == 1

    composed = SynthClient(api_key=api_key, base_url=backend_url)
    assert composed.tunnels.health()["status"] == "ok"
