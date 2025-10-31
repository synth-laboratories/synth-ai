from __future__ import annotations

import json
import pytest


@pytest.mark.asyncio
async def test_mock_rl_trainer_emits_tool_calls_and_sets_cid() -> None:
    # Import the mock server by file path to avoid package side effects
    import importlib.util
    import sys as _sys
    _path = "/Users/joshpurtell/Documents/GitHub/synth-ai/synth_ai/cli/commands/smoke/core.py"
    spec = importlib.util.spec_from_file_location("smoke_core_isolated", _path)
    assert spec and spec.loader
    smoke_core = importlib.util.module_from_spec(spec)
    _sys.modules[spec.name] = smoke_core
    spec.loader.exec_module(smoke_core)  # type: ignore[arg-type]

    trainer = smoke_core.MockRLTrainer(port=0, backend="synthetic")
    app = trainer._build_app()  # FastAPI app

    # Build a minimal OpenAI chat request with a function tool to trigger tool_calls
    body = {
        "model": "gpt-5-nano",
        "messages": [
            {"role": "system", "content": "test system"},
            {"role": "user", "content": "test user"},
        ],
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "interact_many",
                    "description": "Execute a short sequence of actions",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "actions": {
                                "type": "array",
                                "items": {"type": "string"},
                            }
                        },
                        "required": ["actions"],
                    },
                },
            }
        ],
    }

    # Use ASGI transport to call the FastAPI app directly
    import httpx

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        cid = "test-correlation-123"
        resp = await client.post(f"/v1/chat/completions?cid={cid}", json=body)
        if resp.status_code != 200:
            try:
                print("MOCK_SERVER_ERROR:", resp.json())
            except Exception:
                print("MOCK_SERVER_ERROR_TEXT:", resp.text)
        assert resp.status_code == 200
        data = resp.json()

        # choices[0].message.tool_calls should be present and non-empty
        assert isinstance(data, dict)
        choices = data.get("choices")
        assert isinstance(choices, list) and len(choices) > 0
        message = choices[0].get("message", {})
        tool_calls = message.get("tool_calls")
        assert isinstance(tool_calls, list) and len(tool_calls) > 0
        # finish_reason should indicate tool_calls
        assert choices[0].get("finish_reason") in {"tool_calls", "stop"}
        # Ensure cid propagated in response meta
        synth_meta = data.get("synth")
        assert isinstance(synth_meta, dict)
        assert synth_meta.get("cid") == cid


