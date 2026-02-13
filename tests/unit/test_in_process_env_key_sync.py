import socket

import httpx
import pytest
import os

from synth_ai.sdk.localapi._impl.in_process import InProcessTaskApp
from synth_ai.sdk.localapi._impl.server import TaskAppConfig, create_task_app


def _free_port() -> int:
    s = socket.socket()
    try:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])
    finally:
        s.close()


@pytest.mark.asyncio
async def test_in_process_sets_environment_api_key_before_startup(monkeypatch: pytest.MonkeyPatch) -> None:
    """If caller provides api_key, ensure the server uses that exact key (no startup mint mismatch)."""

    monkeypatch.delenv("ENVIRONMENT_API_KEY", raising=False)
    monkeypatch.delenv("DEV_ENVIRONMENT_API_KEY", raising=False)
    monkeypatch.delenv("ENVIRONMENT_API_KEY_ALIASES", raising=False)

    expected_key = "env_test_unit_key"

    async def rollout(request, fastapi_request):  # pragma: no cover - not exercised in this test
        return {"trace_correlation_id": getattr(request, "trace_correlation_id", ""), "reward_info": {}}

    cfg = TaskAppConfig(
        app_id="unit-env-key-sync",
        name="Unit Env Key Sync",
        description="Unit test task app",
        provide_taskset_description=lambda: {"splits": ["train"], "sizes": {"train": 1}},
        provide_task_instances=lambda seeds: [],
        rollout=rollout,
    )
    app = create_task_app(cfg)

    port = _free_port()
    async with InProcessTaskApp(
        app=app,
        port=port,
        host="127.0.0.1",
        api_key=expected_key,
        tunnel_mode="local",
    ) as task_app:
        assert task_app.url

        # The in-process runner should sync the key into ENVIRONMENT_API_KEY before startup.
        assert (os.environ.get("ENVIRONMENT_API_KEY") or "").strip() == expected_key

        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.get(
                f"{task_app.url.rstrip('/')}/health",
                headers={"X-API-Key": expected_key, "Authorization": f"Bearer {expected_key}"},
            )
        assert resp.status_code == 200
