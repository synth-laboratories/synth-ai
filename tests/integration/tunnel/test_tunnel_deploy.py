"""Integration tests for Cloudflare Tunnel deployment."""
import asyncio
import os
from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch

import httpx
import pytest

from synth_ai.cfgs import CFDeployCfg
from synth_ai.cloudflare import _wait_for_health_check, deploy_app_tunnel


@pytest.fixture
def minimal_task_app(tmp_path):
    """Create a minimal valid task app for testing."""
    app_file = tmp_path / "test_task_app.py"
    app_file.write_text("""
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


@pytest.mark.asyncio
class TestQuickTunnelDeployment:
    """Integration tests for quick tunnel deployment."""
    
    @patch("synth_ai.cloudflare.start_uvicorn_background")
    @patch("synth_ai.cloudflare.open_quick_tunnel")
    @patch("synth_ai.cloudflare.wait_for_health_check")
    async def test_deploys_quick_tunnel_successfully(
        self,
        mock_health_check,
        mock_open_tunnel,
        mock_start_uvicorn,
        minimal_task_app,
        tmp_path,
    ):
        """Should successfully deploy quick tunnel and write URL to .env."""
        # Setup mocks
        mock_start_uvicorn.return_value = None
        mock_health_check.return_value = None
        
        mock_proc = Mock()
        mock_open_tunnel.return_value = ("https://test-abc123.trycloudflare.com", mock_proc)
        
        env_file = tmp_path / ".env"
        
        cfg = CFDeployCfg.create(
            task_app_path=minimal_task_app,
            env_api_key="test-key-123",
            mode="quick",
            port=8001,
        )
        
        url = await deploy_app_tunnel(cfg, env_file, keep_alive=False)
        
        assert url == "https://test-abc123.trycloudflare.com"
        mock_start_uvicorn.assert_called_once()
        mock_health_check.assert_called_once_with("127.0.0.1", 8001, "test-key-123")
        mock_open_tunnel.assert_called_once_with(8001)
        
        # Verify .env was written
        assert env_file.exists()
        content = env_file.read_text()
        assert "TASK_APP_URL=https://test-abc123.trycloudflare.com" in content
    
    @patch("synth_ai.cloudflare.start_uvicorn_background")
    @patch("synth_ai.cloudflare.open_quick_tunnel")
    @patch("synth_ai.cloudflare.wait_for_health_check")
    @patch("synth_ai.cloudflare.stop_tunnel")
    async def test_cleans_up_on_error(
        self,
        mock_stop_tunnel,
        mock_health_check,
        mock_open_tunnel,
        mock_start_uvicorn,
        minimal_task_app,
        tmp_path,
    ):
        """Should clean up tunnel process on error."""
        mock_start_uvicorn.return_value = None
        mock_health_check.return_value = None
        
        mock_open_tunnel.side_effect = RuntimeError("Tunnel failed")
        
        cfg = CFDeployCfg.create(
            task_app_path=minimal_task_app,
            env_api_key="test-key-123",
            mode="quick",
        )
        
        with pytest.raises(RuntimeError) as exc_info:
            await deploy_app_tunnel(cfg, tmp_path / ".env", keep_alive=False)
        
        assert "Failed to deploy tunnel" in str(exc_info.value)


@pytest.mark.asyncio
class TestHealthCheck:
    """Tests for health check functionality."""
    
    @patch("synth_ai.cloudflare.httpx.AsyncClient")
    async def test_waits_for_health_endpoint(self, mock_client_class):
        """Should wait for health endpoint to respond."""
        # Mock successful response after a few attempts
        mock_client = AsyncMock()
        mock_response = Mock()
        mock_response.status_code = 200
        
        call_count = 0
        async def mock_get(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise httpx.RequestError("Connection refused")
            return mock_response
        
        mock_client.__aenter__.return_value.get = mock_get
        mock_client_class.return_value = mock_client
        
        await wait_for_health_check("127.0.0.1", 8000, "test-key", timeout=5.0)
        
        assert call_count == 3
    
    @patch("synth_ai.cloudflare.httpx.AsyncClient")
    async def test_accepts_400_as_server_up(self, mock_client_class):
        """Should accept 400 status as server being up (auth error is ok)."""
        mock_client = AsyncMock()
        mock_response = Mock()
        mock_response.status_code = 400  # Auth error means server is up
        
        mock_client.__aenter__.return_value.get = AsyncMock(return_value=mock_response)
        mock_client_class.return_value = mock_client
        
        # Should not raise
        await wait_for_health_check("127.0.0.1", 8000, "test-key", timeout=5.0)
    
    @patch("synth_ai.cloudflare.httpx.AsyncClient")
    async def test_timeout_when_health_check_fails(self, mock_client_class):
        """Should raise RuntimeError if health check times out."""
        mock_client = AsyncMock()
        mock_client.__aenter__.return_value.get = AsyncMock(
            side_effect=httpx.RequestError("Connection refused")
        )
        mock_client_class.return_value = mock_client
        
        with pytest.raises(RuntimeError) as exc_info:
            await wait_for_health_check("127.0.0.1", 8000, "test-key", timeout=0.5)
        
        assert "Health check failed" in str(exc_info.value)


@pytest.mark.integration
@pytest.mark.asyncio
class TestEndToEndQuickTunnel:
    """End-to-end integration tests (require cloudflared installed)."""
    
    @pytest.mark.skipif(
        not os.path.exists("/opt/homebrew/bin/cloudflared") and not os.path.exists("/usr/local/bin/cloudflared"),
        reason="cloudflared not installed",
    )
    async def test_real_quick_tunnel_deployment(self, minimal_task_app, tmp_path):
        """Test actual quick tunnel deployment with real cloudflared."""
        # This test requires cloudflared to be installed
        # It will create a real tunnel, so we need to clean up
        
        env_file = tmp_path / ".env.test"
        
        cfg = CFDeployCfg.create(
            task_app_path=minimal_task_app,
            env_api_key="test-key-123",
            mode="quick",
            port=8002,  # Use different port to avoid conflicts
            trace=False,
        )
        
        try:
            url = await deploy_app_tunnel(cfg, env_file, keep_alive=False)
            
            # Verify URL format
            assert url.startswith("https://")
            assert url.endswith(".trycloudflare.com")
            
            # Verify .env was written
            assert env_file.exists()
            content = env_file.read_text()
            assert f"TASK_APP_URL={url}" in content
            
        finally:
            # Cleanup
            from synth_ai.cloudflare import _TUNNEL_PROCESSES, stop_tunnel

            if 8002 in _TUNNEL_PROCESSES:
                stop_tunnel(_TUNNEL_PROCESSES[8002])
                _TUNNEL_PROCESSES.pop(8002, None)
            
            # Kill any remaining processes
            import subprocess
            subprocess.run(["pkill", "-f", "cloudflared.*8002"], capture_output=True)
            subprocess.run(["pkill", "-f", "uvicorn.*8002"], capture_output=True)

