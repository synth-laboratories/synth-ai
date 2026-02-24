from __future__ import annotations

from typing import Any

import pytest

import synth_ai
import synth_ai.client as canonical_client


def test_legacy_top_level_symbol_removed() -> None:
    with pytest.raises(AttributeError):
        _ = synth_ai.PolicyOptimizationJob


def test_algorithm_wrappers_only_available_in_recipes_namespace() -> None:
    with pytest.raises(AttributeError):
        _ = synth_ai.optimization.GepaOnlineSession
    with pytest.raises(AttributeError):
        _ = synth_ai.optimization.MiproOnlineSession

    assert hasattr(synth_ai.recipes, "GepaOnlineSession")
    assert hasattr(synth_ai.recipes, "MiproOnlineSession")


def test_frontdoor_sync_client_exposes_canonical_namespaces(monkeypatch: Any) -> None:
    class _StubInferenceClient:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass

        async def create_chat_completion(self, **kwargs: Any) -> dict[str, Any]:
            return {"ok": True, "payload": kwargs}

    class _StubInferenceJobsClient:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass

        async def create_job(self, **kwargs: Any) -> dict[str, Any]:
            return {"job_id": "ij_sync", "kwargs": kwargs}

        async def get_job(self, job_id: str) -> dict[str, Any]:
            return {"job_id": job_id, "state": "done"}

        async def create_job_from_request(self, request: Any) -> dict[str, Any]:
            return {"job_id": "ij_req", "request": request}

        async def create_job_from_path(self, **kwargs: Any) -> dict[str, Any]:
            return {"job_id": "ij_path", "kwargs": kwargs}

        async def list_artifacts(self, job_id: str, **kwargs: Any) -> dict[str, Any]:
            return {"job_id": job_id, "artifacts": [], "kwargs": kwargs}

        async def download_artifact(self, *args: Any, **kwargs: Any) -> bytes:
            return b"artifact"

    class _StubGraphsClient:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass

    class _StubVerifiersClient:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass

    class _StubPoolsClient:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass

    monkeypatch.setattr(canonical_client, "InferenceClient", _StubInferenceClient)
    monkeypatch.setattr(canonical_client, "InferenceJobsClient", _StubInferenceJobsClient)
    monkeypatch.setattr(canonical_client, "GraphsClient", _StubGraphsClient)
    monkeypatch.setattr(canonical_client, "VerifiersClient", _StubVerifiersClient)
    monkeypatch.setattr(canonical_client, "ContainerPoolsClient", _StubPoolsClient)

    client = canonical_client.SynthClient(
        api_key="sk_test_sync",
        base_url="http://example.test",
    )

    assert hasattr(client, "optimization")
    assert hasattr(client.optimization, "systems")
    assert hasattr(client.optimization, "offline")
    assert hasattr(client.optimization, "online")
    assert hasattr(client, "pools")
    assert hasattr(client.pools, "harbor")
    assert hasattr(client.pools, "openenv")
    assert hasattr(client.pools, "horizons")
    assert hasattr(client.pools, "arbitrary")
    assert not hasattr(client.pools, "sandbox")
    assert hasattr(client, "container")
    assert hasattr(client.container, "hosted")
    assert hasattr(client.container, "local")
    assert hasattr(client.container, "pools")
    assert hasattr(client.container, "tunnels")
    assert hasattr(client.container, "synth_tunnel")
    assert not hasattr(client, "containers")
    assert not hasattr(client, "container_pools")

    response = client.inference.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "hello"}],
    )
    assert response["ok"] is True
    assert response["payload"]["model"] == "gpt-4o-mini"


@pytest.mark.asyncio
async def test_frontdoor_async_client_exposes_canonical_namespaces(monkeypatch: Any) -> None:
    class _StubInferenceClient:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass

        async def create_chat_completion(self, **kwargs: Any) -> dict[str, Any]:
            return {"ok": True, "payload": kwargs}

    class _StubInferenceJobsClient:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass

        async def create_job(self, **kwargs: Any) -> dict[str, Any]:
            return {"job_id": "ij_async", "kwargs": kwargs}

        async def get_job(self, job_id: str) -> dict[str, Any]:
            return {"job_id": job_id, "state": "done"}

        async def create_job_from_request(self, request: Any) -> dict[str, Any]:
            return {"job_id": "ij_req", "request": request}

        async def create_job_from_path(self, **kwargs: Any) -> dict[str, Any]:
            return {"job_id": "ij_path", "kwargs": kwargs}

        async def list_artifacts(self, job_id: str, **kwargs: Any) -> dict[str, Any]:
            return {"job_id": job_id, "artifacts": [], "kwargs": kwargs}

        async def download_artifact(self, *args: Any, **kwargs: Any) -> bytes:
            return b"artifact"

    class _StubAsyncGraphsClient:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass

    class _StubAsyncVerifiersClient:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass

    class _StubPoolsClient:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass

    monkeypatch.setattr(canonical_client, "InferenceClient", _StubInferenceClient)
    monkeypatch.setattr(canonical_client, "InferenceJobsClient", _StubInferenceJobsClient)
    monkeypatch.setattr(canonical_client, "AsyncGraphsClient", _StubAsyncGraphsClient)
    monkeypatch.setattr(canonical_client, "AsyncVerifiersClient", _StubAsyncVerifiersClient)
    monkeypatch.setattr(canonical_client, "ContainerPoolsClient", _StubPoolsClient)

    client = canonical_client.AsyncSynthClient(
        api_key="sk_test_async",
        base_url="http://example.test",
    )

    assert hasattr(client, "optimization")
    assert hasattr(client.optimization, "systems")
    assert hasattr(client.optimization, "offline")
    assert hasattr(client.optimization, "online")
    assert hasattr(client, "pools")
    assert not hasattr(client.pools, "sandbox")
    assert hasattr(client, "container")
    assert hasattr(client.container, "hosted")
    assert hasattr(client.container, "local")
    assert hasattr(client.container, "pools")
    assert hasattr(client.container, "tunnels")
    assert hasattr(client.container, "synth_tunnel")
    assert not hasattr(client, "containers")
    assert not hasattr(client, "container_pools")

    response = await client.inference.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "hello"}],
    )
    assert response["ok"] is True
    assert response["payload"]["model"] == "gpt-4o-mini"


def test_pool_target_helpers_use_expected_runtime_and_template(monkeypatch: Any) -> None:
    class _StubPoolsClient:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            self.calls: list[dict[str, Any]] = []

        def create_assembly(self, **kwargs: Any) -> dict[str, Any]:
            self.calls.append({"method": "create_assembly", "kwargs": kwargs})
            suffix = kwargs.get("template_name") or "arbitrary"
            return {"assembly_id": f"asm_{suffix}"}

        def create_pool(self, request: dict[str, Any]) -> dict[str, Any]:
            self.calls.append({"method": "create_pool", "request": request})
            return {"pool_id": f"pool_{request['assembly_id']}"}

    monkeypatch.setattr(canonical_client, "ContainerPoolsClient", _StubPoolsClient)
    client = canonical_client.SynthClient(api_key="sk_test_sync", base_url="http://example.test")

    harbor = client.pools.harbor.create_from_data_source(data_source_id="ds_harbor")
    openenv = client.pools.openenv.create_from_data_source(data_source_id="ds_openenv")
    horizons = client.pools.horizons.create_from_data_source(data_source_id="ds_horizons")
    arbitrary = client.pools.arbitrary.create_from_data_source(data_source_id="ds_arbitrary")

    assert harbor["target"] == "harbor"
    assert openenv["target"] == "openenv"
    assert horizons["target"] == "horizons"
    assert arbitrary["target"] == "arbitrary"

    calls = client.pools.raw.calls
    create_assembly_calls = [c["kwargs"] for c in calls if c["method"] == "create_assembly"]
    assert create_assembly_calls[0]["runtime_type"] == "managed_template"
    assert create_assembly_calls[0]["template_name"] == "harbor"
    assert create_assembly_calls[1]["runtime_type"] == "managed_template"
    assert create_assembly_calls[1]["template_name"] == "openenv"
    assert create_assembly_calls[2]["runtime_type"] == "managed_template"
    assert create_assembly_calls[2]["template_name"] == "archipelago"
    assert create_assembly_calls[3]["runtime_type"] == "custom_container"
    assert create_assembly_calls[3]["template_name"] is None


@pytest.mark.asyncio
async def test_async_pools_target_helpers_support_nested_async_calls(monkeypatch: Any) -> None:
    class _StubPoolsClient:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            self.calls: list[dict[str, Any]] = []

        def create_assembly(self, **kwargs: Any) -> dict[str, Any]:
            self.calls.append({"method": "create_assembly", "kwargs": kwargs})
            return {"assembly_id": "asm_harbor"}

        def create_pool(self, request: dict[str, Any]) -> dict[str, Any]:
            self.calls.append({"method": "create_pool", "request": request})
            return {"pool_id": "pool_harbor"}

    monkeypatch.setattr(canonical_client, "ContainerPoolsClient", _StubPoolsClient)
    client = canonical_client.AsyncSynthClient(
        api_key="sk_test_async",
        base_url="http://example.test",
    )

    result = await client.pools.harbor.create_from_data_source(data_source_id="ds_harbor")
    assert result["target"] == "harbor"
    assert result["pool"]["pool_id"] == "pool_harbor"


def test_synth_tunnel_symbol_uses_synth_tunnel_backend(monkeypatch: Any) -> None:
    seen: dict[str, Any] = {}

    async def _fake_create(*args: Any, **kwargs: Any) -> dict[str, Any]:
        seen.update(kwargs)
        return {"url": "https://st.usesynth.ai/s/rt_test"}

    monkeypatch.setattr(canonical_client.TunneledContainer, "create", _fake_create)
    tunnel = canonical_client.SynthTunnel(
        api_key="sk_test_sync",
        base_url="http://example.test",
    ).open(local_port=8101)

    assert tunnel["url"].startswith("https://")
    assert seen["backend"] == canonical_client.TunnelBackend.SynthTunnel


def test_pools_agent_rollouts_namespace_uses_global_rollout_routes(monkeypatch: Any) -> None:
    class _StubPoolsClient:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            self._backend_base = "https://api.example.test"
            self._timeout = 30.0
            self._api_key = "sk_test_sync"
            self.calls: list[dict[str, Any]] = []

        def _request(
            self,
            method: str,
            path: str,
            *,
            json_body: dict[str, Any] | None = None,
            params: dict[str, Any] | None = None,
        ) -> dict[str, Any]:
            self.calls.append(
                {"method": method, "path": path, "json_body": json_body, "params": params}
            )
            if path == "/v1/rollouts" and method == "POST":
                return {"rollout_id": "rt_123"}
            if path == "/v1/rollouts/rt_123" and method == "GET":
                return {"rollout_id": "rt_123", "status": "completed"}
            if path == "/v1/rollouts/rt_123/artifacts" and method == "GET":
                return {"artifacts": [{"path": "generated_classifier.py"}]}
            return {}

    monkeypatch.setattr(canonical_client, "ContainerPoolsClient", _StubPoolsClient)
    client = canonical_client.SynthClient(api_key="sk_test_sync", base_url="http://example.test")

    created = client.pools.agent_rollouts.create(
        {
            "policy": {"instruction": "Classify banking intent"},
            "params": {"dataset": "x", "task_id": "y"},
        }
    )
    status = client.pools.agent_rollouts.get("rt_123")
    artifacts = client.pools.agent_rollouts.artifacts("rt_123")

    assert created["rollout_id"] == "rt_123"
    assert status["status"] == "completed"
    assert artifacts["artifacts"][0]["path"] == "generated_classifier.py"
    assert client.pools.raw.calls[0]["path"] == "/v1/rollouts"
    assert client.pools.raw.calls[1]["path"] == "/v1/rollouts/rt_123"
    assert client.pools.raw.calls[2]["path"] == "/v1/rollouts/rt_123/artifacts"


def test_pools_tasks_namespace_routes_to_pool_task_endpoints(monkeypatch: Any) -> None:
    class _StubPoolsClient:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            self._backend_base = "https://api.example.test"
            self._timeout = 30.0
            self._api_key = "sk_test_sync"
            self.calls: list[dict[str, Any]] = []

        def _request(
            self,
            method: str,
            path: str,
            *,
            json_body: dict[str, Any] | None = None,
            params: dict[str, Any] | None = None,
        ) -> dict[str, Any]:
            self.calls.append(
                {"method": method, "path": path, "json_body": json_body, "params": params}
            )
            if method == "GET" and path.endswith("/tasks"):
                return {"items": [{"task_id": "task_1"}]}
            return {"ok": True}

    monkeypatch.setattr(canonical_client, "ContainerPoolsClient", _StubPoolsClient)
    client = canonical_client.SynthClient(api_key="sk_test_sync", base_url="http://example.test")

    listed = client.pools.tasks.list("pool_1")
    updated = client.pools.tasks.update("pool_1", "task_1", {"backend": "harbor"})

    assert listed["items"][0]["task_id"] == "task_1"
    assert updated["ok"] is True
    assert client.pools.raw.calls[0]["path"] == "/v1/pools/pool_1/tasks"
    assert client.pools.raw.calls[1]["path"] == "/v1/pools/pool_1/tasks/task_1"


def test_pools_rollout_create_rejects_legacy_request_keys(monkeypatch: Any) -> None:
    class _StubPoolsClient:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass

        def create_rollout(self, pool_id: str, request: dict[str, Any]) -> dict[str, Any]:
            return {"pool_id": pool_id, "request": request}

        def _request(
            self,
            method: str,
            path: str,
            *,
            json_body: dict[str, Any] | None = None,
            params: dict[str, Any] | None = None,
        ) -> dict[str, Any]:
            return {"method": method, "path": path, "json_body": json_body, "params": params}

    monkeypatch.setattr(canonical_client, "ContainerPoolsClient", _StubPoolsClient)
    client = canonical_client.SynthClient(api_key="sk_test_sync", base_url="http://example.test")

    with pytest.raises(ValueError, match="unsupported keys"):
        client.pools.rollouts.create("pool_1", {"task_app_url": "https://legacy.example.test"})

    with pytest.raises(ValueError, match="unsupported keys"):
        client.pools.agent_rollouts.create({"task_ref": {"dataset": "x", "task_id": "y"}})
