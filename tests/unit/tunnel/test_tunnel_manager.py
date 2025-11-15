"""Unit tests for TunnelManager."""

import asyncio
import os
import subprocess
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import httpx
import pytest

from synth_ai.tunnel.manager import TunnelManager
from synth_ai.tunnel.models import Tunnel


@pytest.fixture
def mock_api_key():
    """Mock API key for tests."""
    return "sk_test_1234567890"


@pytest.fixture
def mock_backend_url():
    """Mock backend URL for tests."""
    return "http://localhost:8000"


@pytest.fixture
def tunnel_manager(mock_api_key, mock_backend_url):
    """Create TunnelManager instance for testing."""
    with patch.dict(os.environ, {"SYNTH_API_KEY": mock_api_key}):
        return TunnelManager(api_key=mock_api_key, backend_url=mock_backend_url)


@pytest.fixture
def mock_tunnel_response():
    """Mock tunnel API response."""
    return {
        "id": "tunnel-uuid-123",
        "hostname": "test-abc123.trycloudflare.com",
        "quick_tunnel_url": "https://test-abc123.trycloudflare.com",
        "tunnel_type": "quick",
        "local_port": 8114,
        "local_host": "127.0.0.1",
        "status": "active",
        "created_at": datetime.now().isoformat(),
        "last_used_at": datetime.now().isoformat(),
    }


@pytest.fixture
def mock_managed_tunnel_response():
    """Mock managed tunnel API response."""
    return {
        "id": "managed-uuid-456",
        "hostname": "my-company.usesynth.ai",
        "tunnel_type": "managed",
        "local_port": 8114,
        "local_host": "127.0.0.1",
        "status": "active",
        "created_at": datetime.now().isoformat(),
    }


@pytest.fixture
def mock_process():
    """Mock subprocess.Popen."""
    proc = Mock(spec=subprocess.Popen)
    proc.poll.return_value = None  # Process is running
    proc.pid = 12345
    proc.terminate = Mock()
    proc.wait = Mock(return_value=0)
    proc.kill = Mock()
    return proc


class TestTunnelManagerGetOrCreate:
    """Test TunnelManager.get_or_create_tunnel()."""

    @pytest.mark.asyncio
    async def test_reuses_existing_healthy_tunnel(
        self,
        tunnel_manager,
        mock_tunnel_response,
        mock_process,
    ):
        """Should reuse existing healthy tunnel from DB."""
        # Mock registry to return existing tunnel
        tunnel_manager.registry.find_active_tunnel = AsyncMock(
            return_value=Tunnel.from_dict(mock_tunnel_response)
        )
        tunnel_manager.registry.update_last_used = AsyncMock()
        
        # Mock health checker to return healthy
        tunnel_manager.health_checker.check_tunnel_health = AsyncMock(return_value=True)
        
        tunnel = await tunnel_manager.get_or_create_tunnel(
            port=8114,
            reuse_existing=True,
        )
        
        assert tunnel.url == "https://test-abc123.trycloudflare.com"
        assert tunnel.tunnel_type == "quick"
        tunnel_manager.registry.find_active_tunnel.assert_called_once_with(
            port=8114,
            local_host="127.0.0.1",
        )
        tunnel_manager.registry.update_last_used.assert_called_once_with("tunnel-uuid-123")

    @pytest.mark.asyncio
    async def test_restarts_unhealthy_quick_tunnel(
        self,
        tunnel_manager,
        mock_tunnel_response,
        mock_process,
    ):
        """Should restart unhealthy quick tunnel."""
        existing_tunnel = Tunnel.from_dict(mock_tunnel_response)
        existing_tunnel.process = mock_process
        
        tunnel_manager.registry.find_active_tunnel = AsyncMock(return_value=existing_tunnel)
        tunnel_manager.health_checker.check_tunnel_health = AsyncMock(return_value=False)
        
        # Mock tunnel creation
        new_url = "https://new-tunnel-xyz.trycloudflare.com"
        new_proc = Mock(spec=subprocess.Popen)
        new_proc.pid = 67890
        
        tunnel_manager._create_quick_tunnel = AsyncMock(
            return_value=Tunnel(
                id="new-tunnel-id",
                url=new_url,
                tunnel_type="quick",
                port=8114,
                local_host="127.0.0.1",
                process=new_proc,
            )
        )
        tunnel_manager.registry.update_tunnel_url = AsyncMock()
        
        tunnel = await tunnel_manager.get_or_create_tunnel(
            port=8114,
            reuse_existing=True,
        )
        
        assert tunnel.url == new_url
        # Should have killed old process
        mock_process.terminate.assert_called_once()
        tunnel_manager.registry.update_tunnel_url.assert_called_once()

    @pytest.mark.asyncio
    async def test_errors_on_unhealthy_managed_tunnel(
        self,
        tunnel_manager,
        mock_managed_tunnel_response,
    ):
        """Should raise error for unhealthy managed tunnel."""
        existing_tunnel = Tunnel.from_dict(mock_managed_tunnel_response)
        
        tunnel_manager.registry.find_active_tunnel = AsyncMock(return_value=existing_tunnel)
        tunnel_manager.health_checker.check_tunnel_health = AsyncMock(return_value=False)
        
        with pytest.raises(RuntimeError, match="Managed tunnel.*is unhealthy"):
            await tunnel_manager.get_or_create_tunnel(
                port=8114,
                reuse_existing=True,
            )

    @pytest.mark.asyncio
    async def test_prefers_managed_tunnel_when_available(
        self,
        tunnel_manager,
        mock_managed_tunnel_response,
    ):
        """Should prefer managed tunnel over quick tunnel."""
        tunnel_manager.registry.find_active_tunnel = AsyncMock(return_value=None)
        tunnel_manager._get_or_create_managed_tunnel = AsyncMock(
            return_value=Tunnel.from_dict(mock_managed_tunnel_response)
        )
        
        tunnel = await tunnel_manager.get_or_create_tunnel(
            port=8114,
            prefer_managed=True,
        )
        
        assert tunnel.tunnel_type == "managed"
        assert tunnel.hostname == "my-company.usesynth.ai"
        tunnel_manager._get_or_create_managed_tunnel.assert_called_once()

    @pytest.mark.asyncio
    async def test_falls_back_to_quick_when_managed_fails(
        self,
        tunnel_manager,
        mock_process,
    ):
        """Should fallback to quick tunnel when managed tunnel creation fails."""
        tunnel_manager.registry.find_active_tunnel = AsyncMock(return_value=None)
        tunnel_manager._get_or_create_managed_tunnel = AsyncMock(
            side_effect=Exception("Managed tunnel creation failed")
        )
        
        quick_url = "https://quick-tunnel.trycloudflare.com"
        quick_proc = mock_process
        
        tunnel_manager._create_quick_tunnel = AsyncMock(
            return_value=Tunnel(
                id="quick-id",
                url=quick_url,
                tunnel_type="quick",
                port=8114,
                local_host="127.0.0.1",
                process=quick_proc,
            )
        )
        
        tunnel = await tunnel_manager.get_or_create_tunnel(
            port=8114,
            prefer_managed=True,
        )
        
        assert tunnel.tunnel_type == "quick"
        assert tunnel.url == quick_url

    @pytest.mark.asyncio
    async def test_creates_new_tunnel_when_none_exist(
        self,
        tunnel_manager,
        mock_process,
    ):
        """Should create new tunnel when none exist."""
        tunnel_manager.registry.find_active_tunnel = AsyncMock(return_value=None)
        tunnel_manager._get_or_create_managed_tunnel = AsyncMock(return_value=None)
        
        quick_url = "https://new-tunnel.trycloudflare.com"
        quick_proc = mock_process
        
        tunnel_manager._create_quick_tunnel = AsyncMock(
            return_value=Tunnel(
                id="new-id",
                url=quick_url,
                tunnel_type="quick",
                port=8114,
                local_host="127.0.0.1",
                process=quick_proc,
            )
        )
        
        tunnel = await tunnel_manager.get_or_create_tunnel(port=8114)
        
        assert tunnel.url == quick_url
        tunnel_manager._create_quick_tunnel.assert_called_once()

    @pytest.mark.asyncio
    async def test_skips_reuse_when_reuse_existing_false(
        self,
        tunnel_manager,
        mock_process,
    ):
        """Should skip reuse check when reuse_existing=False."""
        tunnel_manager._get_or_create_managed_tunnel = AsyncMock(return_value=None)
        
        quick_url = "https://new-tunnel.trycloudflare.com"
        quick_proc = mock_process
        
        tunnel_manager._create_quick_tunnel = AsyncMock(
            return_value=Tunnel(
                id="new-id",
                url=quick_url,
                tunnel_type="quick",
                port=8114,
                local_host="127.0.0.1",
                process=quick_proc,
            )
        )
        
        tunnel = await tunnel_manager.get_or_create_tunnel(
            port=8114,
            reuse_existing=False,
        )
        
        tunnel_manager.registry.find_active_tunnel.assert_not_called()
        tunnel_manager._create_quick_tunnel.assert_called_once()


class TestTunnelManagerManagedTunnel:
    """Test managed tunnel creation logic."""

    @pytest.mark.asyncio
    async def test_gets_existing_managed_tunnel(
        self,
        tunnel_manager,
        mock_managed_tunnel_response,
    ):
        """Should return existing managed tunnel if found."""
        tunnel_manager.registry.find_managed_tunnel = AsyncMock(
            return_value=Tunnel.from_dict(mock_managed_tunnel_response)
        )
        
        tunnel = await tunnel_manager._get_or_create_managed_tunnel(
            port=8114,
            local_host="127.0.0.1",
        )
        
        assert tunnel.tunnel_type == "managed"
        tunnel_manager.registry.find_managed_tunnel.assert_called_once_with(port=8114)

    @pytest.mark.asyncio
    async def test_creates_new_managed_tunnel(
        self,
        tunnel_manager,
        mock_managed_tunnel_response,
    ):
        """Should create new managed tunnel if none exists."""
        tunnel_manager.registry.find_managed_tunnel = AsyncMock(return_value=None)
        tunnel_manager.registry.create_managed_tunnel = AsyncMock(
            return_value=Tunnel.from_dict(mock_managed_tunnel_response)
        )
        
        tunnel = await tunnel_manager._get_or_create_managed_tunnel(
            port=8114,
            local_host="127.0.0.1",
        )
        
        assert tunnel.tunnel_type == "managed"
        tunnel_manager.registry.create_managed_tunnel.assert_called_once_with(
            subdomain="tunnel-8114",
            local_port=8114,
            local_host="127.0.0.1",
        )


class TestTunnelManagerQuickTunnel:
    """Test quick tunnel creation logic."""

    @pytest.mark.asyncio
    @patch("synth_ai.tunnel.manager.open_quick_tunnel_with_dns_verification")
    async def test_creates_quick_tunnel_with_db_storage(
        self,
        mock_open_tunnel,
        tunnel_manager,
        mock_process,
    ):
        """Should create quick tunnel and store in DB."""
        url = "https://quick-tunnel.trycloudflare.com"
        mock_open_tunnel.return_value = (url, mock_process)
        
        tunnel_manager.registry.create_quick_tunnel = AsyncMock(
            return_value=Tunnel(
                id="db-tunnel-id",
                url=url,
                tunnel_type="quick",
                port=8114,
                local_host="127.0.0.1",
            )
        )
        
        tunnel = await tunnel_manager._create_quick_tunnel(
            port=8114,
            local_host="127.0.0.1",
        )
        
        assert tunnel.url == url
        assert tunnel.process == mock_process
        tunnel_manager.registry.create_quick_tunnel.assert_called_once_with(
            url=url,
            port=8114,
            local_host="127.0.0.1",
            process_pid=mock_process.pid,
        )

    @pytest.mark.asyncio
    @patch("synth_ai.tunnel.manager.open_quick_tunnel_with_dns_verification")
    async def test_creates_quick_tunnel_without_db_when_no_api_key(
        self,
        mock_open_tunnel,
        mock_process,
    ):
        """Should create quick tunnel without DB when no API key."""
        manager = TunnelManager(api_key=None)
        url = "https://quick-tunnel.trycloudflare.com"
        mock_open_tunnel.return_value = (url, mock_process)
        
        tunnel = await manager._create_quick_tunnel(
            port=8114,
            local_host="127.0.0.1",
        )
        
        assert tunnel.url == url
        assert tunnel.process == mock_process
        # Should not try to store in DB
        assert manager.registry.api_key is None


class TestTunnelManagerRestart:
    """Test tunnel restart logic."""

    @pytest.mark.asyncio
    @patch("synth_ai.tunnel.manager.open_quick_tunnel_with_dns_verification")
    async def test_restarts_dead_quick_tunnel(
        self,
        mock_open_tunnel,
        tunnel_manager,
        mock_process,
    ):
        """Should restart dead quick tunnel."""
        old_tunnel = Tunnel(
            id="old-id",
            url="https://old-tunnel.trycloudflare.com",
            tunnel_type="quick",
            port=8114,
            local_host="127.0.0.1",
            process=mock_process,
            tunnel_id="db-old-id",
        )
        mock_process.poll.return_value = 1  # Process is dead
        
        new_url = "https://new-tunnel.trycloudflare.com"
        new_proc = Mock(spec=subprocess.Popen)
        new_proc.pid = 99999
        mock_open_tunnel.return_value = (new_url, new_proc)
        
        tunnel_manager.registry.update_tunnel_url = AsyncMock()
        
        tunnel = await tunnel_manager._restart_quick_tunnel(old_tunnel, port=8114)
        
        assert tunnel.url == new_url
        mock_process.terminate.assert_called_once()
        tunnel_manager.registry.update_tunnel_url.assert_called_once_with(
            tunnel_id="db-old-id",
            new_url=new_url,
            new_pid=new_proc.pid,
        )

    @pytest.mark.asyncio
    async def test_restart_kills_running_process(
        self,
        tunnel_manager,
        mock_process,
    ):
        """Should kill running process before restarting."""
        old_tunnel = Tunnel(
            id="old-id",
            url="https://old-tunnel.trycloudflare.com",
            tunnel_type="quick",
            port=8114,
            local_host="127.0.0.1",
            process=mock_process,
            tunnel_id="db-old-id",
        )
        mock_process.poll.return_value = None  # Process still running
        
        new_url = "https://new-tunnel.trycloudflare.com"
        new_proc = Mock(spec=subprocess.Popen)
        new_proc.pid = 88888
        
        tunnel_manager._create_quick_tunnel = AsyncMock(
            return_value=Tunnel(
                id="new-id",
                url=new_url,
                tunnel_type="quick",
                port=8114,
                local_host="127.0.0.1",
                process=new_proc,
            )
        )
        
        await tunnel_manager._restart_quick_tunnel(old_tunnel, port=8114)
        
        mock_process.terminate.assert_called_once()
        # Should wait for process to terminate
        mock_process.wait.assert_called_once()


class TestTunnelManagerCleanup:
    """Test tunnel cleanup logic."""

    @pytest.mark.asyncio
    async def test_cleanup_expired_tunnels(
        self,
        tunnel_manager,
        mock_process,
    ):
        """Should clean up expired tunnels."""
        expired_tunnel = Tunnel(
            id="expired-id",
            url="https://expired.trycloudflare.com",
            tunnel_type="quick",
            port=8114,
            local_host="127.0.0.1",
            process=mock_process,
            tunnel_id="db-expired-id",
        )
        
        tunnel_manager.registry.find_expired_tunnels = AsyncMock(
            return_value=[expired_tunnel]
        )
        tunnel_manager.registry.mark_expired = AsyncMock()
        
        await tunnel_manager.cleanup_expired_tunnels()
        
        mock_process.terminate.assert_called_once()
        tunnel_manager.registry.mark_expired.assert_called_once_with("expired-id")


