"""Unit tests for TunnelHealthChecker."""

import asyncio
from unittest.mock import AsyncMock, Mock, patch

import httpx
import pytest

from synth_ai.tunnel.health import TunnelHealthChecker
from synth_ai.tunnel.models import Tunnel


@pytest.fixture
def health_checker():
    """Create TunnelHealthChecker instance."""
    return TunnelHealthChecker()


@pytest.fixture
def quick_tunnel():
    """Create quick tunnel for testing."""
    proc = Mock()
    proc.poll.return_value = None  # Process running
    
    return Tunnel(
        id="quick-id",
        url="https://quick.trycloudflare.com",
        tunnel_type="quick",
        port=8114,
        local_host="127.0.0.1",
        process=proc,
    )


@pytest.fixture
def managed_tunnel():
    """Create managed tunnel for testing."""
    return Tunnel(
        id="managed-id",
        url="https://managed.usesynth.ai",
        tunnel_type="managed",
        port=8114,
        local_host="127.0.0.1",
        process=None,  # Managed tunnels don't have process
    )


class TestTunnelHealthChecker:
    """Test TunnelHealthChecker.check_tunnel_health()."""

    @pytest.mark.asyncio
    async def test_checks_process_for_quick_tunnel(
        self,
        health_checker,
        quick_tunnel,
    ):
        """Should check process status for quick tunnels."""
        quick_tunnel.process.poll.return_value = None  # Process running
        
        with patch("httpx.AsyncClient") as mock_client:
            mock_response = Mock()
            mock_response.status_code = 200
            
            mock_get = mock_client.return_value.__aenter__.return_value.get
            mock_get.return_value = mock_response
            
            is_healthy = await health_checker.check_tunnel_health(quick_tunnel)
            
            assert is_healthy is True
            quick_tunnel.process.poll.assert_called_once()

    @pytest.mark.asyncio
    async def test_detects_dead_process(
        self,
        health_checker,
        quick_tunnel,
    ):
        """Should detect dead process."""
        quick_tunnel.process.poll.return_value = 1  # Process dead
        
        is_healthy = await health_checker.check_tunnel_health(quick_tunnel)
        
        assert is_healthy is False

    @pytest.mark.asyncio
    async def test_checks_http_health_for_quick_tunnel(
        self,
        health_checker,
        quick_tunnel,
    ):
        """Should check HTTP health endpoint for quick tunnels."""
        quick_tunnel.process.poll.return_value = None
        
        with patch("httpx.AsyncClient") as mock_client:
            mock_response = Mock()
            mock_response.status_code = 200
            
            mock_get = mock_client.return_value.__aenter__.return_value.get
            mock_get.return_value = mock_response
            
            is_healthy = await health_checker.check_tunnel_health(quick_tunnel)
            
            assert is_healthy is True
            mock_get.assert_called_once_with(
                "https://quick.trycloudflare.com/health",
                timeout=30.0,
            )

    @pytest.mark.asyncio
    async def test_checks_http_health_for_managed_tunnel(
        self,
        health_checker,
        managed_tunnel,
    ):
        """Should check HTTP health for managed tunnels."""
        with patch("httpx.AsyncClient") as mock_client:
            mock_response = Mock()
            mock_response.status_code = 200
            
            mock_get = mock_client.return_value.__aenter__.return_value.get
            mock_get.return_value = mock_response
            
            is_healthy = await health_checker.check_tunnel_health(managed_tunnel)
            
            assert is_healthy is True
            mock_get.assert_called_once_with(
                "https://managed.usesynth.ai/health",
                timeout=30.0,
            )

    @pytest.mark.asyncio
    async def test_detects_http_failure(
        self,
        health_checker,
        quick_tunnel,
    ):
        """Should detect HTTP health check failure."""
        quick_tunnel.process.poll.return_value = None
        
        with patch("httpx.AsyncClient") as mock_client:
            mock_get = mock_client.return_value.__aenter__.return_value.get
            mock_get.side_effect = httpx.RequestError("Connection failed")
            
            is_healthy = await health_checker.check_tunnel_health(quick_tunnel)
            
            assert is_healthy is False

    @pytest.mark.asyncio
    async def test_detects_non_200_status(
        self,
        health_checker,
        quick_tunnel,
    ):
        """Should detect non-200 HTTP status."""
        quick_tunnel.process.poll.return_value = None
        
        with patch("httpx.AsyncClient") as mock_client:
            mock_response = Mock()
            mock_response.status_code = 500
            
            mock_get = mock_client.return_value.__aenter__.return_value.get
            mock_get.return_value = mock_response
            
            is_healthy = await health_checker.check_tunnel_health(quick_tunnel)
            
            assert is_healthy is False

    @pytest.mark.asyncio
    async def test_handles_timeout(
        self,
        health_checker,
        quick_tunnel,
    ):
        """Should handle timeout gracefully."""
        quick_tunnel.process.poll.return_value = None
        
        with patch("httpx.AsyncClient") as mock_client:
            mock_get = mock_client.return_value.__aenter__.return_value.get
            mock_get.side_effect = httpx.TimeoutException("Timeout")
            
            is_healthy = await health_checker.check_tunnel_health(quick_tunnel, timeout=5.0)
            
            assert is_healthy is False


class TestTunnelHealthCheckerRetry:
    """Test TunnelHealthChecker.check_tunnel_with_retry()."""

    @pytest.mark.asyncio
    async def test_retries_on_failure(
        self,
        health_checker,
        quick_tunnel,
    ):
        """Should retry health check on failure."""
        quick_tunnel.process.poll.return_value = None
        
        call_count = 0
        
        async def mock_check(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                return False
            return True
        
        health_checker.check_tunnel_health = mock_check
        
        is_healthy = await health_checker.check_tunnel_with_retry(
            quick_tunnel,
            max_retries=3,
            retry_delay=0.1,
        )
        
        assert is_healthy is True
        assert call_count == 3

    @pytest.mark.asyncio
    async def test_fails_after_max_retries(
        self,
        health_checker,
        quick_tunnel,
    ):
        """Should fail after max retries."""
        quick_tunnel.process.poll.return_value = None
        
        health_checker.check_tunnel_health = AsyncMock(return_value=False)
        
        is_healthy = await health_checker.check_tunnel_with_retry(
            quick_tunnel,
            max_retries=3,
            retry_delay=0.1,
        )
        
        assert is_healthy is False
        assert health_checker.check_tunnel_health.call_count == 3

    @pytest.mark.asyncio
    async def test_succeeds_on_first_try(
        self,
        health_checker,
        quick_tunnel,
    ):
        """Should succeed on first try without retries."""
        quick_tunnel.process.poll.return_value = None
        
        health_checker.check_tunnel_health = AsyncMock(return_value=True)
        
        is_healthy = await health_checker.check_tunnel_with_retry(
            quick_tunnel,
            max_retries=3,
            retry_delay=0.1,
        )
        
        assert is_healthy is True
        assert health_checker.check_tunnel_health.call_count == 1


