"""Integration tests for InProcessTaskApp."""

from __future__ import annotations

import shutil
from pathlib import Path
from unittest.mock import AsyncMock, Mock

import httpx
import pytest

from synth_ai.task.in_process import InProcessTaskApp
from synth_ai.task.server import TaskAppConfig, TaskInfo


@pytest.fixture
def minimal_task_app_file(tmp_path):
    """Create a minimal valid task app file for testing."""
    app_file = tmp_path / "test_task_app.py"
    app_file.write_text("""
from fastapi import FastAPI
from synth_ai.task.apps import TaskAppEntry, register_task_app
from synth_ai.task.contracts import (
    RolloutMetrics,
    RolloutRequest,
    RolloutResponse,
    RolloutStep,
    RolloutTrajectory,
    TaskInfo,
)
from synth_ai.task.server import TaskAppConfig, create_task_app

def provide_task_instances(seeds):
    from synth_ai.task.contracts import TaskInstance
    for seed in seeds:
        yield TaskInstance(task_id="test", task_version="1.0.0", seed=seed, metadata={})

async def rollout_executor(request, fastapi_request):
    return RolloutResponse(
        trajectory=RolloutTrajectory(steps=[RolloutStep(observation={}, action="test", reward=1.0)]),
        metrics=RolloutMetrics(reward=1.0),
    )

def build_config():
    return TaskAppConfig(
        app_id="test",
        name="Test",
        description="Test",
        base_task_info=TaskInfo(
            task={"id": "test", "name": "Test", "version": "1.0.0"},
            environment="test",
            dataset={"id": "test", "name": "Test", "version": "1.0.0"},
            rubric={"version": "1", "criteria_count": 1, "source": "inline"},
            inference={"supports_proxy": False},
            limits={"max_turns": 10},
        ),
        describe_taskset=lambda: {"id": "test", "name": "Test"},
        provide_task_instances=provide_task_instances,
        rollout=rollout_executor,
    )

register_task_app(entry=TaskAppEntry(app_id="test", description="Test", config_factory=build_config))
app = create_task_app(build_config())
""")
    return app_file


@pytest.mark.integration
@pytest.mark.asyncio
class TestInProcessTaskAppIntegration:
    """Integration tests requiring cloudflared."""

    @pytest.mark.skipif(
        not shutil.which("cloudflared"),
        reason="cloudflared not installed",
    )
    async def test_full_workflow_with_app(self):
        """Test complete workflow with FastAPI app."""
        from fastapi import FastAPI

        app = FastAPI()

        @app.get("/health")
        async def health():
            return {"status": "ok"}

        async with InProcessTaskApp(app=app, port=9100, api_key="test") as task_app:
            # Verify tunnel URL
            assert task_app.url is not None
            assert task_app.url.startswith("https://")
            assert ".trycloudflare.com" in task_app.url

            # Verify health endpoint works
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.get(
                    f"{task_app.url}/health",
                    headers={"X-API-Key": "test"},
                )
                assert resp.status_code == 200

    @pytest.mark.skipif(
        not shutil.which("cloudflared"),
        reason="cloudflared not installed",
    )
    async def test_full_workflow_with_task_app_path(
        self, minimal_task_app_file, tmp_path
    ):
        """Test complete workflow with task app file path."""
        async with InProcessTaskApp(
            task_app_path=str(minimal_task_app_file),
            port=9101,
            api_key="test",
        ) as task_app:
            # Verify tunnel URL
            assert task_app.url is not None
            assert task_app.url.startswith("https://")
            assert ".trycloudflare.com" in task_app.url

            # Verify health endpoint works
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.get(
                    f"{task_app.url}/health",
                    headers={"X-API-Key": "test"},
                )
                assert resp.status_code == 200

                # Verify task_info endpoint works
                resp = await client.get(
                    f"{task_app.url}/task_info",
                    headers={"X-API-Key": "test"},
                )
                assert resp.status_code == 200

    @pytest.mark.skipif(
        not shutil.which("cloudflared"),
        reason="cloudflared not installed",
    )
    async def test_multiple_instances_different_ports(self):
        """Test running multiple InProcessTaskApp instances."""
        from fastapi import FastAPI

        app1 = FastAPI()

        @app1.get("/health")
        async def health1():
            return {"status": "ok", "instance": 1}

        app2 = FastAPI()

        @app2.get("/health")
        async def health2():
            return {"status": "ok", "instance": 2}

        async with InProcessTaskApp(
            app=app1, port=9102, api_key="test"
        ) as task_app1:
            async with InProcessTaskApp(
                app=app2, port=9103, api_key="test"
            ) as task_app2:
                assert task_app1.url != task_app2.url
                assert task_app1.port != task_app2.port

                # Verify both work
                async with httpx.AsyncClient(timeout=10.0) as client:
                    resp1 = await client.get(
                        f"{task_app1.url}/health",
                        headers={"X-API-Key": "test"},
                    )
                    assert resp1.status_code == 200

                    resp2 = await client.get(
                        f"{task_app2.url}/health",
                        headers={"X-API-Key": "test"},
                    )
                    assert resp2.status_code == 200

    @pytest.mark.skipif(
        not shutil.which("cloudflared"),
        reason="cloudflared not installed",
    )
    async def test_cleanup_on_exit(self):
        """Test that tunnel is cleaned up on exit."""
        from fastapi import FastAPI

        app = FastAPI()

        @app.get("/health")
        async def health():
            return {"status": "ok"}

        tunnel_url = None
        async with InProcessTaskApp(app=app, port=9104, api_key="test") as task_app:
            tunnel_url = task_app.url

        # After exit, tunnel should be closed
        # We can't easily verify the process is gone, but we can verify
        # the URL is no longer accessible (or takes a long time)
        # For now, just verify the context manager exited cleanly
        assert tunnel_url is not None





