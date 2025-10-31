import os
import asyncio
import pytest
from unittest.mock import AsyncMock, patch
from fastapi.testclient import TestClient

from examples.task_apps.crafter.task_app.synth_envs_hosted.hosted_app import create_app


@pytest.mark.asyncio
async def test_rollout_normalizes_malformed_inference_urls():
    os.environ["DEV_ENVIRONMENT_API_KEY"] = "sk_env_TEST"
    app = create_app()
    client = TestClient(app, headers={"X-API-Key": "sk_env_TEST"})

    malformed = [
        "https://host?cid=trace_run-r1/v1/chat/completions",
        "https://host:8000?cid=trace_run-r2/v1/chat/completions&foo=bar",
        "https://host?cid=trace_run-r3/v1/chat/completions?other=param",
    ]

    captured = []

    async def capture_post(*args, **kwargs):
        url = args[0] if args else kwargs.get("url")
        captured.append(url)
        import json as _json
        class _Resp:
            def __init__(self):
                self.status_code = 200
                self.headers = {"content-type": "application/json"}
                self._data = {"choices": [{"message": {"content": "ok"}}], "usage": {"total_tokens": 10}}
                self._body = _json.dumps(self._data).encode()
            def json(self):
                return self._data
            def raise_for_status(self):
                return None
            @property
            def content(self):
                return self._body
            @property
            def text(self):
                try:
                    return self._body.decode()
                except Exception:
                    return ""
        return _Resp()

    with patch(
        "examples.task_apps.crafter.task_app.synth_envs_hosted.inference.openai_client.httpx.AsyncClient"
    ) as mock_client_class, patch(
        "synth_ai.task.auth.allowed_environment_api_keys", return_value={"sk_env_TEST"}
    ), patch(
        "synth_ai.task.auth.is_api_key_header_authorized", return_value=True
    ):
        inst = AsyncMock()
        inst.post = capture_post
        mock_client_class.return_value.__aenter__.return_value = inst
        mock_client_class.return_value.__aexit__.return_value = None

        def one_rollout(u: str):
            payload = {
                "run_id": "run-proof",
                "env": {"env_name": "crafter", "config": {"env_params": {"max_steps_per_episode": 1}}},
                "policy": {
                    "policy_name": "crafter-react",
                    "config": {"inference_url": u, "model": "Qwen/Qwen3-4B"},
                },
                "ops": ["agent", "env"],
                "record": {"return_trace": True, "trace_format": "structured"},
                "on_done": "reset",
                "mode": "rl",
            }
            resp = client.post("/rollout", json=payload)
            assert resp.status_code == 200, resp.text

        # Run a few rollouts per malformed url
        for u in malformed:
            for _ in range(3):
                one_rollout(u)

    # Verify captured URLs are normalized
    assert captured, "No HTTP calls captured"
    for url in captured:
        base, *rest = url.split("?", 1)
        assert base.endswith("/v1/chat/completions"), f"Bad path: {url}"
        if rest:
            q = rest[0]
            assert "cid=" in q, f"cid missing: {url}"
            assert "/" not in q, f"Query contains path: {url}"


