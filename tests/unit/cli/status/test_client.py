from __future__ import annotations

import httpx
import pytest


def _make_response(method: str, path: str, status: int, payload) -> httpx.Response:
    request = httpx.Request(method, f"https://example.com{path}")
    return httpx.Response(status, json=payload, request=request)


def _stub_async_client(monkeypatch, client_mod, plan):
    """Patch httpx.AsyncClient with a stub that serves canned responses."""

    class StubAsyncClient:
        def __init__(self, *_, base_url=None, headers=None, timeout=None):
            self.base_url = base_url
            self.headers = headers
            self.timeout = timeout

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        async def aclose(self):
            return None

        async def get(self, path, params=None):
            assert plan, "Unexpected GET call"
            method, expected_path, matcher = plan.pop(0)
            assert method == "GET"
            assert expected_path == path
            if isinstance(matcher, dict):
                expected_params = matcher.get("params")
                if expected_params is not None:
                    assert expected_params == params
                response = matcher["response"]
            else:
                response = matcher
            return response

        async def post(self, path, json=None):
            assert plan, "Unexpected POST call"
            method, expected_path, matcher = plan.pop(0)
            assert method == "POST"
            assert expected_path == path
            return matcher if isinstance(matcher, httpx.Response) else matcher["response"]

    monkeypatch.setattr(client_mod.httpx, "AsyncClient", StubAsyncClient)


@pytest.mark.asyncio
async def test_list_jobs_success(monkeypatch, status_modules):
    config = status_modules["config"]
    client_mod = status_modules["client"]

    job_payload = {
        "jobs": [
            {"job_id": "job_123", "status": "running", "training_type": "rl_online"},
            {"job_id": "job_456", "status": "queued", "training_type": "sft_offline"},
        ]
    }

    plan = [
        (
            "GET",
            "/learning/jobs",
            {
                "params": {
                    "status": "running",
                    "type": "rl_online",
                    "created_after": "2025-01-01T00:00:00",
                    "limit": 5,
                },
                "response": _make_response("GET", "/learning/jobs", 200, job_payload),
            },
        )
    ]

    _stub_async_client(monkeypatch, client_mod, plan)

    cfg = config.BackendConfig(base_url="https://example.com/api", api_key="secret", timeout=12.0)

    async with client_mod.StatusAPIClient(cfg) as client:
        jobs = await client.list_jobs(
            status="running",
            job_type="rl_online",
            created_after="2025-01-01T00:00:00",
            limit=5,
        )

    assert jobs == job_payload["jobs"]
    assert not plan  # All planned calls consumed.


@pytest.mark.asyncio
async def test_get_job_raises_status_error(monkeypatch, status_modules):
    config = status_modules["config"]
    client_mod = status_modules["client"]
    errors = status_modules["errors"]

    plan = [
        (
            "GET",
            "/learning/jobs/job_789",
            _make_response("GET", "/learning/jobs/job_789", 404, {"detail": "Not Found"}),
        )
    ]

    _stub_async_client(monkeypatch, client_mod, plan)

    cfg = config.BackendConfig(base_url="https://example.com/api", api_key=None, timeout=10.0)

    async with client_mod.StatusAPIClient(cfg) as client:
        with pytest.raises(errors.StatusAPIError) as exc:
            await client.get_job("job_789")

    assert "Not Found" in str(exc.value)
    assert exc.value.status_code == 404
    assert not plan


@pytest.mark.asyncio
async def test_list_models_all(monkeypatch, status_modules):
    config = status_modules["config"]
    client_mod = status_modules["client"]

    plan = [
        (
            "GET",
            "/learning/models",
            {
                "params": {"limit": 10},
                "response": _make_response(
                    "GET",
                    "/learning/models",
                    200,
                    {"models": [{"id": "model-1", "base_model": "Qwen/Qwen3-4B"}]},
                ),
            },
        ),
    ]

    _stub_async_client(monkeypatch, client_mod, plan)

    cfg = config.BackendConfig(base_url="https://example.com/api", api_key="secret")

    async with client_mod.StatusAPIClient(cfg) as client:
        models = await client.list_models(limit=10)

    assert models == [{"id": "model-1", "base_model": "Qwen/Qwen3-4B"}]
    assert not plan


@pytest.mark.asyncio
async def test_list_models_rl(monkeypatch, status_modules):
    config = status_modules["config"]
    client_mod = status_modules["client"]

    plan = [
        (
            "GET",
            "/learning/models/rl",
            {
                "params": None,
                "response": _make_response(
                    "GET",
                    "/learning/models/rl",
                    200,
                    {"models": [{"id": "rl-model", "base_model": "Qwen/Qwen3-4B"}], "count": 1},
                ),
            },
        )
    ]

    _stub_async_client(monkeypatch, client_mod, plan)

    cfg = config.BackendConfig(base_url="https://example.com/api", api_key="secret")

    async with client_mod.StatusAPIClient(cfg) as client:
        models = await client.list_models(model_type="rl")

    assert models == [{"id": "rl-model", "base_model": "Qwen/Qwen3-4B"}]
    assert not plan


@pytest.mark.asyncio
async def test_list_job_runs_uses_jobs_facade(monkeypatch, status_modules):
    config = status_modules["config"]
    client_mod = status_modules["client"]

    plan = [
        (
            "GET",
            "/jobs/job_123/runs",
            _make_response(
                "GET",
                "/jobs/job_123/runs",
                200,
                {"runs": [{"id": "run-1", "status": "succeeded"}]},
            ),
        )
    ]

    _stub_async_client(monkeypatch, client_mod, plan)

    cfg = config.BackendConfig(base_url="https://example.com/api", api_key="secret")

    async with client_mod.StatusAPIClient(cfg) as client:
        runs = await client.list_job_runs("job_123")

    assert runs == [{"id": "run-1", "status": "succeeded"}]
    assert not plan


@pytest.mark.asyncio
async def test_cancel_job_posts(monkeypatch, status_modules):
    config = status_modules["config"]
    client_mod = status_modules["client"]

    plan = [
        (
            "POST",
            "/learning/jobs/job_123/cancel",
            _make_response("POST", "/learning/jobs/job_123/cancel", 200, {"message": "ok"}),
        )
    ]

    _stub_async_client(monkeypatch, client_mod, plan)

    cfg = config.BackendConfig(base_url="https://example.com/api", api_key="secret")

    async with client_mod.StatusAPIClient(cfg) as client:
        resp = await client.cancel_job("job_123")

    assert resp == {"message": "ok"}
    assert not plan
