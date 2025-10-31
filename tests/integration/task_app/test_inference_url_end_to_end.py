import asyncio
import pytest
from fastapi.testclient import TestClient

from examples.task_apps.crafter.task_app.synth_envs_hosted.hosted_app import create_app


@pytest.mark.asyncio
async def test_policy_step_fixes_malformed_urls_under_load():
    app = create_app()
    client = TestClient(app)

    # Malformed URLs from logs
    bad_urls = [
        "https://host?cid=trace_run-1/v1/chat/completions",
        "https://host:8000?cid=trace_run-2/v1/chat/completions&foo=bar",
        "https://host?cid=trace_run-3/v1/chat/completions?other=param",
    ]

    def create_policy(url: str) -> str:
        payload = {
            "policy_name": "crafter-react",
            "config": {"inference_url": url, "model": "Qwen/Qwen3-4B"},
            "rl_run_id": "run-test",
            "mode": "rl",
        }
        resp = client.post("/policy/create", json=payload)
        assert resp.status_code == 200, resp.text
        return resp.json()["policy_id"]

    def step_policy(policy_id: str) -> dict:
        payload = {
            "policy_id": policy_id,
            "observation": {"env_name": "crafter", "step": 0},
            "dry_run": True,
        }
        resp = client.post("/policy/step", json=payload)
        assert resp.status_code == 200, resp.text
        return resp.json()

    # Run concurrently to stress the app
    async def one_round(u: str):
        pid = await asyncio.get_running_loop().run_in_executor(None, create_policy, u)
        data = await asyncio.get_running_loop().run_in_executor(None, step_policy, pid)
        meta = data.get("meta", {})
        url = str(meta.get("inference_url"))
        assert url, "meta.inference_url missing"
        # Path must be correct
        assert url.split("?")[0].endswith("/v1/chat/completions"), f"Bad path: {url}"
        # Query contains cid and no path segments
        if "?" in url:
            q = url.split("?", 1)[1]
            assert "cid=" in q, f"cid missing: {url}"
            assert "/" not in q, f"Query contains path: {url}"

    await asyncio.gather(*(one_round(u) for u in bad_urls for _ in range(8)))


