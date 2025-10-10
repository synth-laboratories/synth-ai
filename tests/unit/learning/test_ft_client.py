from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pytest

pytestmark = pytest.mark.unit

from synth_ai.learning.sft.client import FtClient


class DummyHTTPClient:
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.calls: List[Tuple[str, Dict[str, Any], Dict[str, Any]]] = []
        self.json_calls: List[Tuple[str, Dict[str, Any]]] = []

    async def __aenter__(self) -> "DummyHTTPClient":
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:  # pragma: no cover - no cleanup required
        return None

    async def post_multipart(self, url: str, data: Dict[str, Any], files: Dict[str, Any]) -> Dict[str, Any]:
        self.calls.append((url, data, files))
        return {"id": "file-123"}

    async def post_json(self, url: str, json: Dict[str, Any]) -> Dict[str, Any]:
        self.json_calls.append((url, json))
        return {"id": "job-123"}


@pytest.mark.asyncio
async def test_upload_training_file_validates_and_posts(monkeypatch, tmp_path: Path) -> None:
    dataset = {
        "messages": [
            {"role": "system", "content": "hi"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call-1",
                        "type": "function",
                        "function": {
                            "name": "interact",
                            "arguments": "{\"actions\": [\"noop\"]}",
                        },
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "call-1", "content": {"result": "ok"}},
        ]
    }
    path = tmp_path / "train.jsonl"
    path.write_text(json.dumps(dataset) + "\n", encoding="utf-8")

    dummy_clients: List[DummyHTTPClient] = []

    def _factory(*args: Any, **kwargs: Any) -> DummyHTTPClient:
        client = DummyHTTPClient(*args, **kwargs)
        dummy_clients.append(client)
        return client

    monkeypatch.setattr("synth_ai.learning.sft.client.AsyncHttpClient", _factory)

    client = FtClient(base_url="https://api.example.com", api_key="sk-test")
    file_id = await client.upload_training_file(path)

    assert file_id == "file-123"
    assert dummy_clients, "HTTP client was not instantiated"
    recorded = dummy_clients[0].calls
    assert recorded, "Multipart request not performed"
    url, data, files = recorded[0]
    assert url == "/api/learning/files"
    assert data["purpose"] == "fine-tune"
    assert "file" in files
    filename, content, content_type = files["file"]
    assert filename == path.name
    assert isinstance(content, (bytes, bytearray))
    assert content_type == "application/jsonl"


@pytest.mark.asyncio
async def test_upload_training_file_rejects_invalid_jsonl(monkeypatch, tmp_path: Path) -> None:
    invalid = {"messages": []}
    path = tmp_path / "bad.jsonl"
    path.write_text(json.dumps(invalid) + "\n", encoding="utf-8")

    def _factory(*args: Any, **kwargs: Any) -> DummyHTTPClient:  # pragma: no cover - should not be called
        raise AssertionError("HTTP client should not be constructed for invalid data")

    monkeypatch.setattr("synth_ai.learning.sft.client.AsyncHttpClient", _factory)

    client = FtClient(base_url="https://api.example.com", api_key="sk-test")
    with pytest.raises(ValueError) as excinfo:
        await client.upload_training_file(path)
    assert str(path) in str(excinfo.value)


@pytest.mark.asyncio
async def test_create_sft_job_rejects_unknown_model():
    client = FtClient(base_url="https://api.example.com", api_key="sk-test")
    with pytest.raises(ValueError):
        await client.create_sft_job(
            model="Unknown/Model",
            training_file_id="file-1",
            hyperparameters={},
        )


@pytest.mark.asyncio
async def test_create_sft_job_valid(monkeypatch):
    dummy_clients: List[DummyHTTPClient] = []

    def _factory(*args: Any, **kwargs: Any) -> DummyHTTPClient:
        client = DummyHTTPClient(*args, **kwargs)
        dummy_clients.append(client)
        return client

    monkeypatch.setattr("synth_ai.learning.sft.client.AsyncHttpClient", _factory)

    client = FtClient(base_url="https://api.example.com", api_key="sk-test")
    resp = await client.create_sft_job(
        model="Qwen/Qwen3-0.6B",
        training_file_id="file-1",
        hyperparameters={"n_epochs": 3, "learning_rate": 1e-5},
        metadata={"tags": ["demo"]},
    )

    assert resp == {"id": "job-123"}
    assert dummy_clients, "HTTP client was not instantiated"
    json_calls = dummy_clients[0].json_calls
    assert json_calls, "create_sft_job should post JSON payload"
    url, payload = json_calls[0]
    assert url == "/api/learning/jobs"
    assert payload["model"] == "Qwen/Qwen3-0.6B"
    assert payload["hyperparameters"]["n_epochs"] == 3
    assert payload["metadata"] == {"tags": ["demo"]}
