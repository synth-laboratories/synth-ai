from __future__ import annotations

import asyncio

import pytest

from synth_ai.sdk.optimization.policy import v1 as policy_v1


class _StubHttpClient:
    last_path: str | None = None

    def __init__(self, _base_url: str, _api_key: str, timeout: float = 30.0) -> None:
        self.timeout = timeout

    async def __aenter__(self) -> _StubHttpClient:
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        return None

    async def get(self, path: str, *, params=None):  # noqa: ANN001
        _StubHttpClient.last_path = path
        return {
            "object": "runtime_route_compatibility",
            "version": "v2",
            "routes": [
                {
                    "logical_path": "/api/v2/runtime/containers/:container_id/rollouts/:rollout_id/checkpoint/dump",
                    "methods": ["POST"],
                    "target_path": "/api/v2/containers/:container_id/rollouts/:rollout_id/checkpoint/dump",
                    "compatibility_type": "alias",
                    "status": "active",
                },
                {
                    "logical_path": "/api/v2/runtime/containers/:container_id/rollouts/:rollout_id/checkpoint/restore",
                    "methods": ["POST"],
                    "target_path": "/api/v2/containers/:container_id/rollouts/:rollout_id/checkpoint/restore",
                    "compatibility_type": "alias",
                    "status": "active",
                },
            ],
        }


def test_runtime_compatibility_path_uses_v2_runtime_endpoint(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(policy_v1, "RustCoreHttpClient", _StubHttpClient)
    payload = asyncio.run(
        policy_v1.PolicyOptimizationOnlineSession.runtime_compatibility_async(
            backend_url="http://localhost:8080",
            api_key="test-key",
            api_version="v2",
        )
    )
    assert payload.get("object") == "runtime_route_compatibility"
    assert _StubHttpClient.last_path == "/v2/runtime/compatibility"


def test_runtime_compatibility_rejects_v1_api_version() -> None:
    with pytest.raises(ValueError, match="api_version='v2'"):
        asyncio.run(
            policy_v1.PolicyOptimizationOnlineSession.runtime_compatibility_async(
                backend_url="http://localhost:8080",
                api_key="test-key",
                api_version="v1",
            )
        )


def test_runtime_compatibility_includes_runtime_container_checkpoint_aliases(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(policy_v1, "RustCoreHttpClient", _StubHttpClient)
    payload = asyncio.run(
        policy_v1.PolicyOptimizationOnlineSession.runtime_compatibility_async(
            backend_url="http://localhost:8080",
            api_key="test-key",
            api_version="v2",
        )
    )
    routes = payload.get("routes", [])
    assert any(
        route.get("logical_path")
        == "/api/v2/runtime/containers/:container_id/rollouts/:rollout_id/checkpoint/dump"
        and route.get("target_path")
        == "/api/v2/containers/:container_id/rollouts/:rollout_id/checkpoint/dump"
        for route in routes
    )
    assert any(
        route.get("logical_path")
        == "/api/v2/runtime/containers/:container_id/rollouts/:rollout_id/checkpoint/restore"
        and route.get("target_path")
        == "/api/v2/containers/:container_id/rollouts/:rollout_id/checkpoint/restore"
        for route in routes
    )
