from __future__ import annotations

from typing import Any

import pytest

import synth_ai
import synth_ai.client as canonical_client


def test_legacy_top_level_symbol_removed() -> None:
    with pytest.raises(AttributeError):
        _ = synth_ai.PolicyOptimizationJob


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

    response = await client.inference.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "hello"}],
    )
    assert response["ok"] is True
    assert response["payload"]["model"] == "gpt-4o-mini"
