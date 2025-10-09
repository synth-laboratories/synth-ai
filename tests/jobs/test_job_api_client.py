import asyncio
import json
from typing import Any, Dict, Optional

import pytest

from synth_ai.api.models.supported import UnsupportedModelError
from synth_ai.jobs.client import JobsClient


class FakeHttp:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []
        self._response: Any = {"ok": True}

    def respond_with(self, value: Any) -> None:
        self._response = value

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):  # noqa: ANN001
        return None

    async def get(self, path: str, *, params: Optional[Dict[str, Any]] = None, headers: Optional[Dict[str, str]] = None) -> Any:
        self.calls.append({"method": "GET", "path": path, "params": params, "headers": headers})
        return self._response

    async def post_json(self, path: str, *, json: Dict[str, Any], headers: Optional[Dict[str, str]] = None) -> Any:  # type: ignore[override]
        self.calls.append({"method": "POST", "path": path, "json": json, "headers": headers})
        return self._response

    async def post_multipart(self, path: str, *, data: Dict[str, Any], files: Dict[str, tuple[str, bytes, Optional[str]]], headers: Optional[Dict[str, str]] = None) -> Any:  # type: ignore[override]
        # Store non-binary summary for assertion
        files_summary = {k: (v[0], len(v[1]), v[2]) for k, v in files.items()}
        self.calls.append({"method": "POST", "path": path, "data": data, "files": files_summary, "headers": headers})
        return self._response

    async def delete(self, path: str, *, headers: Optional[Dict[str, str]] = None) -> Any:
        self.calls.append({"method": "DELETE", "path": path, "headers": headers})
        return self._response


@pytest.mark.asyncio
async def test_files_upload_and_list():
    fake = FakeHttp()
    async with JobsClient(base_url="https://backend", api_key="k", http=fake) as client:
        # Upload
        resp = await client.files.upload(filename="data.jsonl", content=b"{}\n", purpose="sft_training", content_type="application/jsonl")
        assert resp == {"ok": True}
        assert fake.calls[-1]["path"] == "/api/files"
        assert fake.calls[-1]["method"] == "POST"
        assert fake.calls[-1]["data"]["purpose"] == "sft_training"
        assert fake.calls[-1]["files"]["file"][0] == "data.jsonl"
        assert fake.calls[-1]["files"]["file"][1] == len(b"{}\n")
        assert fake.calls[-1]["files"]["file"][2] == "application/jsonl"

        # List
        _ = await client.files.list(purpose="sft_training", after="file-1", limit=5)
        call = fake.calls[-1]
        assert call["method"] == "GET"
        assert call["path"] == "/api/files"
        assert call["params"] == {"purpose": "sft_training", "after": "file-1", "limit": 5}


@pytest.mark.asyncio
async def test_sft_jobs_endpoints():
    fake = FakeHttp()
    async with JobsClient(base_url="https://backend", api_key="k", http=fake) as client:
        # Create
        _ = await client.sft.create(
            training_file="file-abc",
            model="Qwen/Qwen3-0.6B",
            validation_file="file-val",
            hyperparameters={"n_epochs": 3},
            suffix="demo",
            integrations={"wandb": {"project": "p"}},
            metadata={"k": "v"},
            idempotency_key="idemp-sft",
        )
        call = fake.calls[-1]
        assert call["method"] == "POST"
        assert call["path"] == "/api/sft/jobs"
        assert call["headers"]["Idempotency-Key"] == "idemp-sft"
        assert call["json"]["training_file"] == "file-abc"
        assert call["json"]["model"] == "Qwen/Qwen3-0.6B"
        assert call["json"]["validation_file"] == "file-val"
        assert call["json"]["hyperparameters"] == {"n_epochs": 3}
        assert call["json"]["suffix"] == "demo"
        assert call["json"]["integrations"] == {"wandb": {"project": "p"}}
        assert call["json"]["metadata"] == {"k": "v"}

        # List with filters
        _ = await client.sft.list(status="queued", model="Qwen/Qwen3-0.6B", file_id="file-abc", created_after=10, created_before=20, after="sft-1", limit=7)
        call = fake.calls[-1]
        assert call["method"] == "GET"
        assert call["path"] == "/api/sft/jobs"
        assert call["params"] == {
            "limit": 7,
            "status": "queued",
            "model": "Qwen/Qwen3-0.6B",
            "file_id": "file-abc",
            "created_after": 10,
            "created_before": 20,
            "after": "sft-1",
        }

        # Retrieve
        _ = await client.sft.retrieve("sft-xyz")
        call = fake.calls[-1]
        assert call["method"] == "GET"
        assert call["path"] == "/api/sft/jobs/sft-xyz"

        # Cancel
        _ = await client.sft.cancel("sft-xyz")
        call = fake.calls[-1]
        assert call["method"] == "POST"
        assert call["path"] == "/api/sft/jobs/sft-xyz/cancel"
        assert call["json"] == {}

        # Events
        _ = await client.sft.list_events("sft-xyz", since_seq=5, limit=100)
        call = fake.calls[-1]
        assert call["method"] == "GET"
        assert call["path"] == "/api/sft/jobs/sft-xyz/events"
        assert call["params"] == {"since_seq": 5, "limit": 100}

        # Checkpoints
        _ = await client.sft.checkpoints("sft-xyz", after="10", limit=3)
        call = fake.calls[-1]
        assert call["method"] == "GET"
        assert call["path"] == "/api/sft/jobs/sft-xyz/checkpoints"
        assert call["params"] == {"after": "10", "limit": 3}


@pytest.mark.asyncio
async def test_rl_jobs_endpoints():
    fake = FakeHttp()
    async with JobsClient(base_url="https://backend", api_key="k", http=fake) as client:
        # Create
        _ = await client.rl.create(
            model="Qwen/Qwen3-0.6B",
            endpoint_base_url="https://taskapp",
            trainer_id="trainer-1",
            trainer={"batch_size": 32},
            job_config_id="cfg-1",
            config={"env": {"name": "CrafterClassic"}},
            metadata={"note": "demo"},
            idempotency_key="idemp-rl",
        )
        call = fake.calls[-1]
        assert call["method"] == "POST"
        assert call["path"] == "/api/rl/jobs"
        assert call["headers"]["Idempotency-Key"] == "idemp-rl"
        assert call["json"]["model"] == "Qwen/Qwen3-0.6B"
        assert call["json"]["endpoint_base_url"] == "https://taskapp"
        assert call["json"]["trainer_id"] == "trainer-1"
        assert call["json"]["trainer"] == {"batch_size": 32}
        assert call["json"]["job_config_id"] == "cfg-1"
        assert call["json"]["config"] == {"env": {"name": "CrafterClassic"}}
        assert call["json"]["metadata"] == {"note": "demo"}

        # List
        _ = await client.rl.list(status="running", model="Qwen/Qwen3-0.6B", created_after=1, created_before=2, after="rl-1", limit=9)
        call = fake.calls[-1]
        assert call["method"] == "GET"
        assert call["path"] == "/api/rl/jobs"
        assert call["params"] == {
            "limit": 9,
            "status": "running",
            "model": "Qwen/Qwen3-0.6B",
            "created_after": 1,
            "created_before": 2,
            "after": "rl-1",
        }

        # Retrieve
        _ = await client.rl.retrieve("rl-abc")
        call = fake.calls[-1]
        assert call["method"] == "GET"
        assert call["path"] == "/api/rl/jobs/rl-abc"

        # Cancel
        _ = await client.rl.cancel("rl-abc")
        call = fake.calls[-1]
        assert call["method"] == "POST"
        assert call["path"] == "/api/rl/jobs/rl-abc/cancel"
        assert call["json"] == {}

        # Events
        _ = await client.rl.list_events("rl-abc", since_seq=0, limit=200)
        call = fake.calls[-1]
        assert call["method"] == "GET"
        assert call["path"] == "/api/rl/jobs/rl-abc/events"
        assert call["params"] == {"since_seq": 0, "limit": 200}

        # Metrics
        _ = await client.rl.metrics("rl-abc", after_step=42, limit=50)
        call = fake.calls[-1]
        assert call["method"] == "GET"
        assert call["path"] == "/api/rl/jobs/rl-abc/metrics"
        assert call["params"] == {"after_step": 42, "limit": 50}


@pytest.mark.asyncio
async def test_models_endpoints():
    fake = FakeHttp()
    async with JobsClient(base_url="https://backend", api_key="k", http=fake) as client:
        # List
        _ = await client.models.list(source="sft", base_model="Qwen/Qwen3-0.6B", status="ready", after="ft:1", limit=11)
        call = fake.calls[-1]
        assert call["method"] == "GET"
        assert call["path"] == "/api/models"
        assert call["params"] == {
            "limit": 11,
            "source": "sft",
            "base_model": "Qwen/Qwen3-0.6B",
            "status": "ready",
            "after": "ft:1",
        }

        # Retrieve
        _ = await client.models.retrieve("ft:Qwen/Qwen3-0.6B:suffix")
        call = fake.calls[-1]
        assert call["method"] == "GET"
        assert call["path"] == "/api/models/ft:Qwen/Qwen3-0.6B:suffix"

        # Delete
        _ = await client.models.delete("ft:Qwen/Qwen3-0.6B:suffix")
        call = fake.calls[-1]
        assert call["method"] == "DELETE"
        assert call["path"] == "/api/models/ft:Qwen/Qwen3-0.6B:suffix"

        # List jobs
        _ = await client.models.list_jobs("ft:Qwen/Qwen3-0.6B:suffix", after="j1", limit=3)
        call = fake.calls[-1]
        assert call["method"] == "GET"
        assert call["path"] == "/api/models/ft:Qwen/Qwen3-0.6B:suffix/jobs"
        assert call["params"] == {"limit": 3, "after": "j1"}


@pytest.mark.asyncio
async def test_sft_create_rejects_unknown_model():
    fake = FakeHttp()
    async with JobsClient(base_url="https://backend", api_key="k", http=fake) as client:
        with pytest.raises(UnsupportedModelError):
            await client.sft.create(
                training_file="file-abc",
                model="Unknown/Model",
                hyperparameters={"n_epochs": 1},
            )
    assert not fake.calls, "HTTP request should not be sent for invalid model"


@pytest.mark.asyncio
async def test_rl_create_rejects_unknown_model():
    fake = FakeHttp()
    async with JobsClient(base_url="https://backend", api_key="k", http=fake) as client:
        with pytest.raises(UnsupportedModelError):
            await client.rl.create(
                model="Unknown/Model",
                endpoint_base_url="https://task",
                trainer_id="trainer-1",
            )
    assert not fake.calls, "HTTP request should not be sent for invalid model"
