from typing import Any

import pytest

pytestmark = pytest.mark.unit

from synth_ai.learning.client import LearningClient
from synth_ai.learning.rl import RlClient


@pytest.mark.asyncio
async def test_learning_client_create_job_rejects_unknown_model(monkeypatch):
    async def _factory(*args, **kwargs):  # pragma: no cover - should not run
        raise AssertionError("HTTP client should not be constructed")

    monkeypatch.setattr("synth_ai.learning.client.AsyncHttpClient", _factory)

    client = LearningClient(base_url="https://api.example.com", api_key="sk-test")
    with pytest.raises(ValueError):
        await client.create_job(
            training_type="sft_offline",
            model="Unknown/Model",
            training_file_id="file-1",
            hyperparameters={"n_epochs": 1},
        )


@pytest.mark.asyncio
async def test_rl_client_create_job_rejects_unknown_model(monkeypatch):
    async def _factory(*args, **kwargs):  # pragma: no cover - should not run
        raise AssertionError("HTTP client should not be constructed")

    monkeypatch.setattr("synth_ai.learning.rl.client.AsyncHttpClient", _factory)

    client = RlClient(base_url="https://api.example.com", api_key="sk-test")
    with pytest.raises(ValueError):
        await client.create_job(
            model="Unknown/Model",
            task_app_url="https://task",
            trainer={"batch_size": 1, "group_size": 2},
        )


@pytest.mark.asyncio
async def test_learning_client_create_job_valid(monkeypatch):
    class DummyHTTP:
        def __init__(self, *args, **kwargs) -> None:
            self.calls: list[tuple[str, dict[str, Any]]] = []

        async def __aenter__(self) -> "DummyHTTP":
            return self

        async def __aexit__(self, exc_type, exc, tb) -> None:
            return None

        async def post_json(self, url: str, json: dict[str, Any]) -> dict[str, Any]:
            self.calls.append((url, json))
            return {"id": "job-42"}

    dummy_client = DummyHTTP()
    monkeypatch.setattr(
        "synth_ai.learning.client.AsyncHttpClient",
        lambda *args, **kwargs: dummy_client,
    )

    client = LearningClient(base_url="https://api.example.com", api_key="sk-test")
    resp = await client.create_job(
        training_type="sft_offline",
        model="Qwen/Qwen3-0.6B",
        training_file_id="file-123",
        hyperparameters={"n_epochs": 2, "learning_rate": 5e-6},
        metadata={"tags": ["demo"]},
    )

    assert resp == {"id": "job-42"}
    assert dummy_client.calls, "Expected create_job to POST JSON payload"
    url, payload = dummy_client.calls[0]
    assert url == "/api/learning/jobs"
    assert payload["model"] == "Qwen/Qwen3-0.6B"
    assert payload["hyperparameters"]["n_epochs"] == 2
    assert payload["metadata"] == {"tags": ["demo"]}


@pytest.mark.asyncio
async def test_learning_client_requires_step_hyperparameter(monkeypatch):
    async def _factory(*args, **kwargs):  # pragma: no cover - should not run
        raise AssertionError("HTTP client should not be constructed")

    monkeypatch.setattr("synth_ai.learning.client.AsyncHttpClient", _factory)

    client = LearningClient(base_url="https://api.example.com", api_key="sk-test")
    with pytest.raises(ValueError):
        await client.create_job(
            training_type="sft_offline",
            model="Qwen/Qwen3-0.6B",
            training_file_id="file-1",
            hyperparameters={"learning_rate": 1e-5},
        )
