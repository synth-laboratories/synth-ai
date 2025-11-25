"""Unit tests for InProcessTaskApp."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch

import pytest
from fastapi import FastAPI

from synth_ai.sdk.task.in_process import InProcessTaskApp
from synth_ai.sdk.task.server import TaskAppConfig, TaskInfo


@pytest.fixture
def mock_fastapi_app():
    """Create a mock FastAPI app."""
    app = FastAPI()
    return app


@pytest.fixture
def mock_task_app_config():
    """Create a mock TaskAppConfig."""
    return TaskAppConfig(
        app_id="test",
        name="Test Task App",
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
        provide_task_instances=lambda seeds: [],
        rollout=AsyncMock(return_value=Mock()),
    )


class TestInProcessTaskAppInit:
    """Tests for InProcessTaskApp initialization."""

    def test_init_validates_exactly_one_input(self, mock_task_app_config):
        """Should raise ValueError if multiple or no inputs provided."""
        # No inputs
        with pytest.raises(ValueError, match="exactly one"):
            InProcessTaskApp()

        # Multiple inputs
        app = FastAPI()
        config = mock_task_app_config
        with pytest.raises(ValueError, match="exactly one"):
            InProcessTaskApp(app=app, config=config)

    def test_init_validates_port_range(self):
        """Should validate port is in valid range."""
        app = FastAPI()

        # Port too low
        with pytest.raises(ValueError, match="Port must be in range"):
            InProcessTaskApp(app=app, port=1023)

        # Port too high
        with pytest.raises(ValueError, match="Port must be in range"):
            InProcessTaskApp(app=app, port=65536)

        # Valid port
        task_app = InProcessTaskApp(app=app, port=8114)
        assert task_app.port == 8114

    def test_init_validates_host(self):
        """Should validate host is allowed."""
        app = FastAPI()

        # Invalid host
        with pytest.raises(ValueError, match="Host must be"):
            InProcessTaskApp(app=app, host="192.168.1.1")

        # Valid hosts
        for host in ("127.0.0.1", "localhost", "0.0.0.0"):
            task_app = InProcessTaskApp(app=app, host=host)
            assert task_app.host == host

    def test_init_validates_tunnel_mode(self):
        """Should validate tunnel_mode."""
        app = FastAPI()

        # Invalid mode
        with pytest.raises(ValueError, match="tunnel_mode must be"):
            InProcessTaskApp(app=app, tunnel_mode="invalid")

        # Valid mode
        task_app = InProcessTaskApp(app=app, tunnel_mode="quick")
        assert task_app.tunnel_mode == "quick"

    def test_init_validates_task_app_path_exists(self, tmp_path):
        """Should validate task_app_path exists."""
        # Non-existent file
        fake_path = tmp_path / "nonexistent.py"
        with pytest.raises(FileNotFoundError):
            InProcessTaskApp(task_app_path=str(fake_path))

        # Wrong extension
        wrong_ext = tmp_path / "test.txt"
        wrong_ext.write_text("test")
        with pytest.raises(ValueError, match="must be a .py file"):
            InProcessTaskApp(task_app_path=str(wrong_ext))

    def test_init_stores_parameters(self, mock_fastapi_app):
        """Should store initialization parameters."""
        task_app = InProcessTaskApp(
            app=mock_fastapi_app,
            port=9000,
            host="127.0.0.1",
            tunnel_mode="quick",
            api_key="test-key",
            health_check_timeout=60.0,
        )

        assert task_app.port == 9000
        assert task_app.host == "127.0.0.1"
        assert task_app.tunnel_mode == "quick"
        assert task_app.api_key == "test-key"
        assert task_app.health_check_timeout == 60.0
        assert task_app.url is None


@pytest.mark.asyncio
class TestInProcessTaskAppContextManager:
    """Tests for InProcessTaskApp context manager."""

    @patch("synth_ai.task.in_process.stop_tunnel")
    @patch("synth_ai.task.in_process.open_quick_tunnel")
    @patch("synth_ai.task.in_process.ensure_cloudflared_installed")
    @patch("synth_ai.task.in_process.wait_for_health_check")
    @patch("synth_ai.task.in_process.threading.Thread")
    async def test_init_with_app(
        self,
        mock_thread,
        mock_health_check,
        mock_ensure_cloudflared,
        mock_open_tunnel,
        mock_stop_tunnel,
        mock_fastapi_app,
    ):
        """Should accept FastAPI app directly."""
        mock_open_tunnel.return_value = ("https://test.trycloudflare.com", Mock())

        async with InProcessTaskApp(app=mock_fastapi_app, port=9001) as task_app:
            assert task_app.url == "https://test.trycloudflare.com"
            mock_thread.assert_called_once()
            mock_thread.return_value.start.assert_called_once()
            mock_health_check.assert_called_once()
            mock_ensure_cloudflared.assert_called_once()
            mock_open_tunnel.assert_called_once()

        # Verify cleanup
        mock_stop_tunnel.assert_called_once()

    @patch("synth_ai.task.in_process.stop_tunnel")
    @patch("synth_ai.task.in_process.open_quick_tunnel")
    @patch("synth_ai.task.in_process.ensure_cloudflared_installed")
    @patch("synth_ai.task.in_process.wait_for_health_check")
    @patch("synth_ai.task.in_process.threading.Thread")
    @patch("synth_ai.task.in_process.create_task_app")
    async def test_init_with_config(
        self,
        mock_create_app,
        mock_start_uvicorn,
        mock_health_check,
        mock_ensure_cloudflared,
        mock_open_tunnel,
        mock_stop_tunnel,
        mock_task_app_config,
        mock_fastapi_app,
    ):
        """Should accept TaskAppConfig."""
        mock_create_app.return_value = mock_fastapi_app
        mock_open_tunnel.return_value = ("https://test.trycloudflare.com", Mock())

        async with InProcessTaskApp(config=mock_task_app_config, port=9002) as task_app:
            assert task_app.url == "https://test.trycloudflare.com"
            mock_create_app.assert_called_once_with(mock_task_app_config)

    @patch("synth_ai.task.in_process.stop_tunnel")
    @patch("synth_ai.task.in_process.open_quick_tunnel")
    @patch("synth_ai.task.in_process.ensure_cloudflared_installed")
    @patch("synth_ai.task.in_process.wait_for_health_check")
    @patch("synth_ai.task.in_process.threading.Thread")
    @patch("synth_ai.task.in_process.create_task_app")
    async def test_init_with_config_factory(
        self,
        mock_create_app,
        mock_start_uvicorn,
        mock_health_check,
        mock_ensure_cloudflared,
        mock_open_tunnel,
        mock_stop_tunnel,
        mock_task_app_config,
        mock_fastapi_app,
    ):
        """Should accept config factory function."""
        mock_create_app.return_value = mock_fastapi_app
        mock_open_tunnel.return_value = ("https://test.trycloudflare.com", Mock())

        def build_config():
            return mock_task_app_config

        async with InProcessTaskApp(config_factory=build_config, port=9003) as task_app:
            assert task_app.url == "https://test.trycloudflare.com"
            mock_create_app.assert_called_once()

    @patch("synth_ai.task.in_process.stop_tunnel")
    @patch("synth_ai.task.in_process.open_quick_tunnel")
    @patch("synth_ai.task.in_process.ensure_cloudflared_installed")
    @patch("synth_ai.task.in_process.wait_for_health_check")
    @patch("synth_ai.task.in_process.threading.Thread")
    @patch("synth_ai.task.in_process.get_asgi_app")
    @patch("synth_ai.task.in_process.load_file_to_module")
    @patch("synth_ai.task.in_process.configure_import_paths")
    async def test_init_with_task_app_path(
        self,
        mock_configure_paths,
        mock_load_module,
        mock_get_app,
        mock_start_uvicorn,
        mock_health_check,
        mock_ensure_cloudflared,
        mock_open_tunnel,
        mock_stop_tunnel,
        mock_fastapi_app,
        tmp_path,
    ):
        """Should load task app from file path."""
        task_app_file = tmp_path / "test_task_app.py"
        task_app_file.write_text("app = FastAPI()")

        mock_module = Mock()
        mock_load_module.return_value = mock_module
        mock_get_app.return_value = mock_fastapi_app
        mock_open_tunnel.return_value = ("https://test.trycloudflare.com", Mock())

        async with InProcessTaskApp(
            task_app_path=str(task_app_file), port=9004
        ) as task_app:
            assert task_app.url == "https://test.trycloudflare.com"
            mock_load_module.assert_called_once()
            mock_get_app.assert_called_once_with(mock_module)

    @patch("synth_ai.task.in_process.stop_tunnel")
    @patch("synth_ai.task.in_process.open_quick_tunnel")
    @patch("synth_ai.task.in_process.ensure_cloudflared_installed")
    @patch("synth_ai.task.in_process.wait_for_health_check")
    @patch("synth_ai.task.in_process.threading.Thread")
    async def test_cleanup_on_exception(
        self,
        mock_start_uvicorn,
        mock_health_check,
        mock_ensure_cloudflared,
        mock_open_tunnel,
        mock_stop_tunnel,
        mock_fastapi_app,
    ):
        """Should clean up tunnel even if exception occurs."""
        mock_tunnel_proc = Mock()
        mock_open_tunnel.return_value = ("https://test.trycloudflare.com", mock_tunnel_proc)

        try:
            async with InProcessTaskApp(app=mock_fastapi_app, port=9005):
                raise ValueError("Test exception")
        except ValueError:
            pass

        # Verify tunnel was stopped
        mock_stop_tunnel.assert_called_once_with(mock_tunnel_proc)

    @patch("synth_ai.task.in_process.open_quick_tunnel")
    @patch("synth_ai.task.in_process.ensure_cloudflared_installed")
    @patch("synth_ai.task.in_process.wait_for_health_check")
    @patch("synth_ai.task.in_process.threading.Thread")
    async def test_health_check_timeout(
        self,
        mock_start_uvicorn,
        mock_health_check,
        mock_ensure_cloudflared,
        mock_open_tunnel,
        mock_fastapi_app,
    ):
        """Should raise RuntimeError if health check times out."""
        import asyncio

        async def slow_health_check(*args, **kwargs):
            await asyncio.sleep(2.0)  # Longer than timeout
            raise RuntimeError("Health check failed")

        mock_health_check.side_effect = slow_health_check

        with pytest.raises(RuntimeError, match="health check"):
            async with InProcessTaskApp(
                app=mock_fastapi_app, port=9006, health_check_timeout=0.5
            ):
                pass

    @patch("synth_ai.task.in_process.stop_tunnel")
    @patch("synth_ai.task.in_process.open_quick_tunnel")
    @patch("synth_ai.task.in_process.ensure_cloudflared_installed")
    @patch("synth_ai.task.in_process.wait_for_health_check")
    @patch("synth_ai.task.in_process.threading.Thread")
    @patch("synth_ai.task.in_process._is_port_available")
    @patch("synth_ai.task.in_process._find_available_port")
    async def test_port_conflict_handling(
        self,
        mock_find_port,
        mock_is_available,
        mock_start_uvicorn,
        mock_health_check,
        mock_ensure_cloudflared,
        mock_open_tunnel,
        mock_stop_tunnel,
        mock_fastapi_app,
    ):
        """Should handle port conflicts gracefully."""
        # Simulate port conflict
        mock_is_available.return_value = False
        mock_find_port.return_value = 9008
        mock_open_tunnel.return_value = ("https://test.trycloudflare.com", Mock())

        async with InProcessTaskApp(app=mock_fastapi_app, port=9007, auto_find_port=True) as task_app:
            # Should have found different port
            assert task_app.port == 9008
            mock_find_port.assert_called_once()

    @patch("synth_ai.task.in_process.stop_tunnel")
    @patch("synth_ai.task.in_process.open_quick_tunnel")
    @patch("synth_ai.task.in_process.ensure_cloudflared_installed")
    @patch("synth_ai.task.in_process.wait_for_health_check")
    @patch("synth_ai.task.in_process.threading.Thread")
    async def test_uses_custom_api_key(
        self,
        mock_start_uvicorn,
        mock_health_check,
        mock_ensure_cloudflared,
        mock_open_tunnel,
        mock_stop_tunnel,
        mock_fastapi_app,
    ):
        """Should use custom API key for health checks."""
        mock_open_tunnel.return_value = ("https://test.trycloudflare.com", Mock())

        async with InProcessTaskApp(
            app=mock_fastapi_app, port=9008, api_key="custom-key"
        ) as task_app:
            # Verify health check called with custom key
            mock_health_check.assert_called_once()
            call_args = mock_health_check.call_args
            assert call_args[0][2] == "custom-key"  # api_key is 3rd positional arg


@pytest.mark.asyncio
class TestInProcessTaskAppHelpers:
    """Tests for InProcessTaskApp helper methods."""

    def test_get_api_key_from_env(self, mock_fastapi_app):
        """Should get API key from environment."""
        import os

        os.environ["ENVIRONMENT_API_KEY"] = "env-key"
        task_app = InProcessTaskApp(app=mock_fastapi_app)
        assert task_app._get_api_key() == "env-key"
        del os.environ["ENVIRONMENT_API_KEY"]

    def test_get_api_key_default(self, mock_fastapi_app):
        """Should use default API key if not in environment."""
        import os

        if "ENVIRONMENT_API_KEY" in os.environ:
            del os.environ["ENVIRONMENT_API_KEY"]
        task_app = InProcessTaskApp(app=mock_fastapi_app)
        assert task_app._get_api_key() == "test"

