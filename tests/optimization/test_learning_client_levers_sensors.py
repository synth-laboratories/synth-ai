from __future__ import annotations

import asyncio
import json
from typing import Any

from synth_ai.sdk.optimization.internal.learning import client as learning_client
from synth_ai.sdk.optimization.internal.learning.client import LearningClient


def test_create_or_update_lever_posts_payload(monkeypatch) -> None:
    captured: dict[str, Any] = {}

    class _FakeHttpClient:
        def __init__(self, *_args: Any, **_kwargs: Any) -> None:
            return

        async def __aenter__(self) -> "_FakeHttpClient":
            return self

        async def __aexit__(self, exc_type, exc, tb) -> None:
            _ = (exc_type, exc, tb)

        async def post_json(self, path: str, json: dict[str, Any]) -> Any:
            captured["path"] = path
            captured["json"] = json
            return {
                "lever_id": json.get("lever_id", "lever_1"),
                "kind": json.get("kind", "prompt"),
                "version": json.get("version", 1),
                "scope": json.get("scope", []),
            }

    monkeypatch.setattr(learning_client, "RustCoreHttpClient", _FakeHttpClient)
    client = LearningClient(base_url="https://example.com", api_key="key")
    payload = {
        "lever_id": "lever_123",
        "kind": "prompt",
        "version": 5,
        "scope": [{"kind": "org", "id": "org_1"}],
    }
    result = asyncio.run(client.create_or_update_lever("opt_1", payload))
    assert captured["path"] == "/api/v1/optimizers/opt_1/levers"
    assert captured["json"]["lever_id"] == "lever_123"
    assert result.lever_id == "lever_123"
    assert result.version == 5


def test_resolve_lever_builds_query(monkeypatch) -> None:
    captured: dict[str, Any] = {}

    class _FakeHttpClient:
        def __init__(self, *_args: Any, **_kwargs: Any) -> None:
            return

        async def __aenter__(self) -> "_FakeHttpClient":
            return self

        async def __aexit__(self, exc_type, exc, tb) -> None:
            _ = (exc_type, exc, tb)

        async def get(self, path: str, params: dict[str, Any] | None = None) -> Any:
            captured["path"] = path
            captured["params"] = params
            return {
                "lever_id": "lever_456",
                "kind": "constraint",
                "version": 2,
                "scope": json.loads(params["scope"]) if params and params.get("scope") else [],
            }

    monkeypatch.setattr(learning_client, "RustCoreHttpClient", _FakeHttpClient)
    client = LearningClient(base_url="https://example.com", api_key="key")
    scope_param = [{"kind": "job", "id": "job_1"}]
    result = asyncio.run(client.resolve_lever("opt_2", "lever_456", scope=scope_param))
    assert captured["path"] == "/api/v1/optimizers/opt_2/levers"
    assert captured["params"]["lever_id"] == "lever_456"
    assert json.loads(captured["params"]["scope"]) == scope_param
    assert result is not None
    assert result.kind.value == "constraint"


def test_emit_sensor_frame_returns_frame(monkeypatch) -> None:
    payload_frame = {
        "scope": [{"kind": "job", "id": "job_2"}],
        "sensors": [
            {
                "sensor_id": "reward.main",
                "kind": "reward",
                "scope": [{"kind": "job", "id": "job_2"}],
                "value": {"reward": 0.7},
            }
        ],
        "lever_versions": {"lever_abc": 3},
    }

    class _FakeHttpClient:
        def __init__(self, *_args: Any, **_kwargs: Any) -> None:
            return

        async def __aenter__(self) -> "_FakeHttpClient":
            return self

        async def __aexit__(self, exc_type, exc, tb) -> None:
            _ = (exc_type, exc, tb)

        async def post_json(self, path: str, json: dict[str, Any]) -> Any:
            assert path == "/api/v1/optimizers/opt_3/sensors"
            return {
                **json,
                "frame_id": "frame_42",
            }

    monkeypatch.setattr(learning_client, "RustCoreHttpClient", _FakeHttpClient)
    client = LearningClient(base_url="https://example.com", api_key="key")
    frame = asyncio.run(client.emit_sensor_frame("opt_3", payload_frame))
    assert frame.frame_id == "frame_42"
    assert frame.lever_versions["lever_abc"] == 3
    assert frame.sensors[0].sensor_id == "reward.main"


def test_list_sensor_frames_parses_list(monkeypatch) -> None:
    responses = [
        {
            "scope": [{"kind": "job", "id": "job_3"}],
            "sensors": [],
            "lever_versions": {},
            "frame_id": "frame_list",
        }
    ]

    class _FakeHttpClient:
        def __init__(self, *_args: Any, **_kwargs: Any) -> None:
            return

        async def __aenter__(self) -> "_FakeHttpClient":
            return self

        async def __aexit__(self, exc_type, exc, tb) -> None:
            _ = (exc_type, exc, tb)

        async def get(self, path: str, params: dict[str, Any] | None = None) -> Any:
            assert path == "/api/v1/optimizers/opt_4/sensors"
            assert params is not None and params.get("limit") == "10"
            return responses

    monkeypatch.setattr(learning_client, "RustCoreHttpClient", _FakeHttpClient)
    client = LearningClient(base_url="https://example.com", api_key="key")
    frames = asyncio.run(client.list_sensor_frames("opt_4", limit=10))
    assert len(frames) == 1
    assert frames[0].frame_id == "frame_list"
