"""Unit tests for TunnelRegistry."""

import asyncio
from datetime import datetime
from unittest.mock import AsyncMock, Mock, patch

import httpx
import pytest

try:
from synth_ai.tunnel.models import Tunnel
from synth_ai.tunnel.registry import TunnelRegistry
except ImportError:
    pytest.skip("synth_ai.tunnel module not implemented yet", allow_module_level=True)


@pytest.fixture
def mock_api_key():
    """Mock API key."""
    return "sk_test_1234567890"


@pytest.fixture
def mock_backend_url():
    """Mock backend URL."""
    return "http://localhost:8000"


@pytest.fixture
def registry(mock_api_key, mock_backend_url):
    """Create TunnelRegistry instance."""
    return TunnelRegistry(api_key=mock_api_key, backend_url=mock_backend_url)


class TestTunnelRegistryFindActive:
    """Test finding active tunnels."""

    @pytest.mark.asyncio
    async def test_finds_managed_tunnel_first(
        self,
        registry,
    ):
        """Should prefer managed tunnels over quick tunnels."""
        managed_response = {
            "id": "managed-id",
            "hostname": "managed.usesynth.ai",
            "tunnel_type": "managed",
            "local_port": 8114,
            "local_host": "127.0.0.1",
            "status": "active",
            "created_at": datetime.now().isoformat(),
        }
        quick_response = {
            "id": "quick-id",
            "hostname": "quick.trycloudflare.com",
            "quick_tunnel_url": "https://quick.trycloudflare.com",
            "tunnel_type": "quick",
            "local_port": 8114,
            "local_host": "127.0.0.1",
            "status": "active",
            "created_at": datetime.now().isoformat(),
        }
        
        with patch("httpx.AsyncClient") as mock_client:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = [managed_response, quick_response]
            
            mock_client.return_value.__aenter__.return_value.get.return_value = mock_response
            
            tunnel = await registry.find_active_tunnel(port=8114)
            
            assert tunnel is not None
            assert tunnel.tunnel_type == "managed"
            assert tunnel.hostname == "managed.usesynth.ai"

    @pytest.mark.asyncio
    async def test_finds_quick_tunnel_when_no_managed(
        self,
        registry,
    ):
        """Should return quick tunnel when no managed tunnel exists."""
        quick_response = {
            "id": "quick-id",
            "hostname": "quick.trycloudflare.com",
            "quick_tunnel_url": "https://quick.trycloudflare.com",
            "tunnel_type": "quick",
            "local_port": 8114,
            "local_host": "127.0.0.1",
            "status": "active",
            "created_at": datetime.now().isoformat(),
        }
        
        with patch("httpx.AsyncClient") as mock_client:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = [quick_response]
            
            mock_client.return_value.__aenter__.return_value.get.return_value = mock_response
            
            tunnel = await registry.find_active_tunnel(port=8114)
            
            assert tunnel is not None
            assert tunnel.tunnel_type == "quick"

    @pytest.mark.asyncio
    async def test_returns_none_when_no_tunnels(
        self,
        registry,
    ):
        """Should return None when no tunnels found."""
        with patch("httpx.AsyncClient") as mock_client:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = []
            
            mock_client.return_value.__aenter__.return_value.get.return_value = mock_response
            
            tunnel = await registry.find_active_tunnel(port=8114)
            
            assert tunnel is None

    @pytest.mark.asyncio
    async def test_returns_none_when_no_api_key(
        self,
    ):
        """Should return None when no API key."""
        registry = TunnelRegistry(api_key=None)
        tunnel = await registry.find_active_tunnel(port=8114)
        assert tunnel is None

    @pytest.mark.asyncio
    async def test_filters_by_port_and_host(
        self,
        registry,
    ):
        """Should filter tunnels by port and local_host."""
        with patch("httpx.AsyncClient") as mock_client:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = []
            
            mock_get = mock_client.return_value.__aenter__.return_value.get
            mock_get.return_value = mock_response
            
            await registry.find_active_tunnel(port=8114, local_host="127.0.0.1")
            
            # Verify query params
            call_args = mock_get.call_args
            assert call_args[1]["params"]["local_port"] == 8114
            assert call_args[1]["params"]["local_host"] == "127.0.0.1"
            assert call_args[1]["params"]["status"] == "active"


class TestTunnelRegistryCreateQuick:
    """Test quick tunnel creation."""

    @pytest.mark.asyncio
    async def test_creates_quick_tunnel(
        self,
        registry,
    ):
        """Should create quick tunnel via backend API."""
        response_data = {
            "id": "new-quick-id",
            "hostname": "new-quick.trycloudflare.com",
            "quick_tunnel_url": "https://new-quick.trycloudflare.com",
            "tunnel_type": "quick",
            "local_port": 8114,
            "local_host": "127.0.0.1",
            "status": "active",
            "created_at": datetime.now().isoformat(),
        }
        
        with patch("httpx.AsyncClient") as mock_client:
            mock_response = Mock()
            mock_response.status_code = 201
            mock_response.json.return_value = response_data
            
            mock_post = mock_client.return_value.__aenter__.return_value.post
            mock_post.return_value = mock_response
            
            tunnel = await registry.create_quick_tunnel(
                url="https://new-quick.trycloudflare.com",
                port=8114,
                local_host="127.0.0.1",
                process_pid=12345,
            )
            
            assert tunnel.id == "new-quick-id"
            assert tunnel.url == "https://new-quick.trycloudflare.com"
            
            # Verify request
            call_args = mock_post.call_args
            assert call_args[1]["json"]["local_port"] == 8114
            assert call_args[1]["json"]["quick_tunnel_url"] == "https://new-quick.trycloudflare.com"
            assert call_args[1]["json"]["quick_tunnel_process_pid"] == 12345

    @pytest.mark.asyncio
    async def test_raises_error_when_no_api_key(
        self,
    ):
        """Should raise error when no API key."""
        registry = TunnelRegistry(api_key=None)
        
        with pytest.raises(ValueError, match="API key required"):
            await registry.create_quick_tunnel(
                url="https://test.trycloudflare.com",
                port=8114,
                local_host="127.0.0.1",
                process_pid=12345,
            )


class TestTunnelRegistryCreateManaged:
    """Test managed tunnel creation."""

    @pytest.mark.asyncio
    async def test_creates_managed_tunnel(
        self,
        registry,
    ):
        """Should create managed tunnel via backend API."""
        response_data = {
            "id": "managed-id",
            "hostname": "my-company.usesynth.ai",
            "tunnel_type": "managed",
            "local_port": 8114,
            "local_host": "127.0.0.1",
            "status": "active",
            "created_at": datetime.now().isoformat(),
        }
        
        with patch("httpx.AsyncClient") as mock_client:
            mock_response = Mock()
            mock_response.status_code = 201
            mock_response.json.return_value = response_data
            
            mock_post = mock_client.return_value.__aenter__.return_value.post
            mock_post.return_value = mock_response
            
            tunnel = await registry.create_managed_tunnel(
                subdomain="my-company",
                local_port=8114,
                local_host="127.0.0.1",
            )
            
            assert tunnel.tunnel_type == "managed"
            assert tunnel.hostname == "my-company.usesynth.ai"
            
            # Verify request
            call_args = mock_post.call_args
            assert call_args[1]["json"]["subdomain"] == "my-company"
            assert call_args[1]["json"]["local_port"] == 8114

    @pytest.mark.asyncio
    async def test_raises_error_when_no_api_key(
        self,
    ):
        """Should raise error when no API key."""
        registry = TunnelRegistry(api_key=None)
        
        with pytest.raises(ValueError, match="API key required"):
            await registry.create_managed_tunnel(
                subdomain="test",
                local_port=8114,
                local_host="127.0.0.1",
            )


class TestTunnelRegistryUpdate:
    """Test tunnel update operations."""

    @pytest.mark.asyncio
    async def test_updates_last_used(
        self,
        registry,
    ):
        """Should update last_used_at timestamp."""
        with patch("httpx.AsyncClient") as mock_client:
            mock_response = Mock()
            mock_response.status_code = 200
            
            mock_post = mock_client.return_value.__aenter__.return_value.post
            mock_post.return_value = mock_response
            
            await registry.update_last_used("tunnel-id")
            
            mock_post.assert_called_once()
            call_args = mock_post.call_args
            assert "/tunnels/tunnel-id/touch" in call_args[0][0]

    @pytest.mark.asyncio
    async def test_updates_tunnel_url(
        self,
        registry,
    ):
        """Should update tunnel URL and PID."""
        with patch("httpx.AsyncClient") as mock_client:
            mock_response = Mock()
            mock_response.status_code = 200
            
            mock_patch = mock_client.return_value.__aenter__.return_value.patch
            mock_patch.return_value = mock_response
            
            await registry.update_tunnel_url(
                tunnel_id="tunnel-id",
                new_url="https://new-url.trycloudflare.com",
                new_pid=99999,
            )
            
            mock_patch.assert_called_once()
            call_args = mock_patch.call_args
            assert call_args[1]["json"]["quick_tunnel_url"] == "https://new-url.trycloudflare.com"
            assert call_args[1]["json"]["quick_tunnel_process_pid"] == 99999

    @pytest.mark.asyncio
    async def test_update_skips_when_no_api_key(
        self,
    ):
        """Should skip update when no API key."""
        registry = TunnelRegistry(api_key=None)
        await registry.update_last_used("tunnel-id")  # Should not raise


class TestTunnelRegistryExpired:
    """Test expired tunnel operations."""

    @pytest.mark.asyncio
    async def test_finds_expired_tunnels(
        self,
        registry,
    ):
        """Should find expired tunnels."""
        expired_data = [
            {
                "id": "expired-1",
                "hostname": "expired1.trycloudflare.com",
                "tunnel_type": "quick",
                "local_port": 8114,
                "local_host": "127.0.0.1",
                "status": "expired",
                "created_at": datetime.now().isoformat(),
            },
        ]
        
        with patch("httpx.AsyncClient") as mock_client:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = expired_data
            
            mock_get = mock_client.return_value.__aenter__.return_value.get
            mock_get.return_value = mock_response
            
            tunnels = await registry.find_expired_tunnels()
            
            assert len(tunnels) == 1
            assert tunnels[0].id == "expired-1"

    @pytest.mark.asyncio
    async def test_marks_tunnel_as_expired(
        self,
        registry,
    ):
        """Should mark tunnel as expired."""
        with patch("httpx.AsyncClient") as mock_client:
            mock_response = Mock()
            mock_response.status_code = 200
            
            mock_post = mock_client.return_value.__aenter__.return_value.post
            mock_post.return_value = mock_response
            
            await registry.mark_expired("tunnel-id")
            
            mock_post.assert_called_once()
            call_args = mock_post.call_args
            assert "/tunnels/tunnel-id/expire" in call_args[0][0]


