from __future__ import annotations

import asyncio
import os
import tempfile
from typing import Any

import pytest
from fastapi.testclient import TestClient
from unittest.mock import AsyncMock, patch

from examples.task_apps.crafter.task_app.synth_envs_hosted.hosted_app import create_app


@pytest.mark.asyncio
@pytest.mark.xfail(reason="SQLite backend is expected to lock under parallel load", strict=False)
async def test_sqlite_tracing_locks_under_parallel_rollouts():
    pytest.xfail("SQLite backend intentionally limited; see parallel_sqlite.md for migration details.")
    # Force local sqlite tracing (non-libsql) in a temp location
    with tempfile.TemporaryDirectory() as d:
        os.environ["SYNTH_TRACING_DISABLED"] = "0"
        os.environ["SYNTH_TRACES_DIR"] = d
        os.environ["SYNTH_TRACES_DB"] = f"sqlite+aiosqlite:///{os.path.join(d, 'traces.db')}"
        os.environ["DEV_ENVIRONMENT_API_KEY"] = "sk_env_TEST"

        # Skip if Crafter env isn't installed in test runner
        try:
            import crafter  # type: ignore  # noqa: F401
        except Exception:
            pytest.skip("crafter package not installed; skipping sqlite lock reproduction test")

        app = create_app()
        client = TestClient(app, headers={"X-API-Key": "sk_env_TEST"})

        # Patch outbound HTTP to avoid real inference; return minimal OK body fast
        async def fake_post(*args: Any, **kwargs: Any):
            import json as _json

            class _Resp:
                def __init__(self):
                    self.status_code = 200
                    self.headers = {"content-type": "application/json"}
                    self._data = {
                        "choices": [
                            {
                                "message": {
                                    "role": "assistant",
                                    "content": "",
                                    "tool_calls": [
                                        {
                                            "id": "call_norm",
                                            "type": "function",
                                            "function": {"name": "interact_many", "arguments": "{}"},
                                        }
                                    ],
                                },
                                "finish_reason": "tool_calls",
                            }
                        ],
                        "usage": {"total_tokens": 10},
                    }
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
            inst.post = fake_post
            mock_client_class.return_value.__aenter__.return_value = inst
            mock_client_class.return_value.__aexit__.return_value = None

            def rollout_req() -> int:
                payload = {
                    "run_id": "run-lock-test",
                    "env": {"env_name": "crafter", "config": {"env_params": {"max_steps_per_episode": 2}}},
                    "policy": {
                        "policy_name": "crafter-react",
                        "config": {"inference_url": "http://localhost:8000/api/inference/v1/chat/completions", "model": "Qwen/Qwen3-4B"},
                    },
                    "ops": ["agent", "env"] * 2,
                    "record": {"return_trace": True, "trace_format": "structured"},
                    "on_done": "reset",
                    "mode": "rl",
                }
                resp = client.post("/rollout", json=payload)
                return resp.status_code

            # Hammer with concurrent rollouts to trigger sqlite writer contention
            async def one():
                return await asyncio.get_running_loop().run_in_executor(None, rollout_req)

            results = await asyncio.gather(*(one() for _ in range(16)))

            # Expect at least one failure (500) due to sqlite 'database is locked' under parallel load
            assert any(code >= 500 for code in results), f"Expected failures; got {results}"

