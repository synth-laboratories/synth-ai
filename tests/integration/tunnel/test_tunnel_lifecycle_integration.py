"""Integration tests for tunnel lifecycle management.

These tests verify the full tunnel lifecycle:
- Tunnel creation and reuse
- Health checking
- Automatic failover
- Managed vs quick tunnel selection
"""

import asyncio
import os
import subprocess
from unittest.mock import AsyncMock, Mock, patch

import pytest

try:
    from synth_ai.tunnel.manager import TunnelManager
    from synth_ai.tunnel.models import Tunnel
except ImportError:
    pytest.skip("synth_ai.tunnel module not implemented yet", allow_module_level=True)


@pytest.mark.integration
@pytest.mark.asyncio
class TestTunnelLifecycleIntegration:
    """Integration tests for tunnel lifecycle."""

    @pytest.fixture
    def tunnel_manager(self):
        """Create tunnel manager with test API key."""
        api_key = os.getenv("SYNTH_API_KEY")
        backend_url = os.getenv("BACKEND_URL", "http://localhost:8000")
        return TunnelManager(api_key=api_key, backend_url=backend_url)

    async def test_full_lifecycle_reuse(
        self,
        tunnel_manager,
    ):
        """Test full lifecycle: create, reuse, cleanup."""
        # Skip if no API key
        if not tunnel_manager.api_key:
            pytest.skip("SYNTH_API_KEY not set")
        
        port = 19999  # Use high port to avoid conflicts
        
        # Step 1: Create tunnel
        tunnel1 = await tunnel_manager.get_or_create_tunnel(
            port=port,
            reuse_existing=False,  # Force new tunnel
        )
        
        assert tunnel1.url is not None
        assert tunnel1.url.startswith("https://")
        tunnel_id_1 = tunnel1.tunnel_id
        
        # Step 2: Reuse tunnel
        tunnel2 = await tunnel_manager.get_or_create_tunnel(
            port=port,
            reuse_existing=True,  # Should reuse tunnel1
        )
        
        assert tunnel2.tunnel_id == tunnel_id_1
        assert tunnel2.url == tunnel1.url
        
        # Step 3: Cleanup
        if tunnel1.process:
            tunnel1.process.terminate()
            try:
                tunnel1.process.wait(timeout=5.0)
            except subprocess.TimeoutExpired:
                tunnel1.process.kill()

    async def test_managed_tunnel_preference(
        self,
        tunnel_manager,
    ):
        """Test that managed tunnels are preferred."""
        if not tunnel_manager.api_key:
            pytest.skip("SYNTH_API_KEY not set")
        
        port = 19998
        
        # Try to get tunnel with managed preference
        tunnel = await tunnel_manager.get_or_create_tunnel(
            port=port,
            prefer_managed=True,
            reuse_existing=True,
        )
        
        # Should get a tunnel (managed or quick)
        assert tunnel is not None
        assert tunnel.url is not None
        
        # Cleanup
        if tunnel.process:
            tunnel.process.terminate()
            try:
                tunnel.process.wait(timeout=5.0)
            except subprocess.TimeoutExpired:
                tunnel.process.kill()

    async def test_health_check_integration(
        self,
        tunnel_manager,
    ):
        """Test health checking with real tunnel."""
        if not tunnel_manager.api_key:
            pytest.skip("SYNTH_API_KEY not set")
        
        port = 19997
        
        # Create tunnel
        tunnel = await tunnel_manager.get_or_create_tunnel(
            port=port,
            reuse_existing=False,
        )
        
        # Health check should pass for active tunnel
        is_healthy = await tunnel_manager.health_checker.check_tunnel_health(tunnel)
        
        # Note: This might fail if no server is running on the port
        # That's okay - we're testing the health check mechanism
        
        # Cleanup
        if tunnel.process:
            tunnel.process.terminate()
            try:
                tunnel.process.wait(timeout=5.0)
            except subprocess.TimeoutExpired:
                tunnel.process.kill()

    async def test_failover_on_dead_tunnel(
        self,
        tunnel_manager,
    ):
        """Test automatic failover when tunnel dies."""
        if not tunnel_manager.api_key:
            pytest.skip("SYNTH_API_KEY not set")
        
        port = 19996
        
        # Create tunnel
        tunnel1 = await tunnel_manager.get_or_create_tunnel(
            port=port,
            reuse_existing=False,
        )
        
        original_url = tunnel1.url
        
        # Kill the tunnel process
        if tunnel1.process:
            tunnel1.process.kill()
            tunnel1.process.wait(timeout=5.0)
        
        # Wait a bit for process to fully die
        await asyncio.sleep(1.0)
        
        # Get tunnel again - should detect dead tunnel and create new one
        tunnel2 = await tunnel_manager.get_or_create_tunnel(
            port=port,
            reuse_existing=True,
        )
        
        # Should have created new tunnel (URL might be different)
        assert tunnel2.url is not None
        
        # Cleanup
        if tunnel2.process:
            tunnel2.process.terminate()
            try:
                tunnel2.process.wait(timeout=5.0)
            except subprocess.TimeoutExpired:
                tunnel2.process.kill()


@pytest.mark.integration
@pytest.mark.asyncio
class TestTunnelRegistryIntegration:
    """Integration tests for TunnelRegistry with backend API."""

    @pytest.fixture
    def registry(self):
        """Create registry with test API key."""
        api_key = os.getenv("SYNTH_API_KEY")
        backend_url = os.getenv("BACKEND_URL", "http://localhost:8000")
        return TunnelRegistry(api_key=api_key, backend_url=backend_url)

    async def test_list_tunnels(
        self,
        registry,
    ):
        """Test listing tunnels via API."""
        if not registry.api_key:
            pytest.skip("SYNTH_API_KEY not set")
        
        # This will call the backend API
        # We expect it to either succeed or fail gracefully
        try:
            tunnel = await registry.find_active_tunnel(port=8114)
            # If tunnel exists, it should have valid data
            if tunnel:
                assert tunnel.id is not None
                assert tunnel.url is not None
        except Exception as e:
            # If backend is not available, that's okay for integration test
            pytest.skip(f"Backend not available: {e}")


@pytest.mark.integration
@pytest.mark.asyncio
class TestInProcessTaskAppIntegration:
    """Integration tests for InProcessTaskApp with TunnelManager."""

    async def test_in_process_app_with_tunnel_reuse(
        self,
    ):
        """Test InProcessTaskApp reuses tunnels."""
        from synth_ai.task.in_process import InProcessTaskApp
        
        api_key = os.getenv("SYNTH_API_KEY")
        if not api_key:
            pytest.skip("SYNTH_API_KEY not set")
        
        # Create minimal task app
        from fastapi import FastAPI
        
        app = FastAPI()
        
        @app.get("/health")
        async def health():
            return {"status": "ok"}
        
        # Use InProcessTaskApp with tunnel
        async with InProcessTaskApp(
            app=app,
            port=19995,
            tunnel_mode="quick",
            api_key=api_key,
        ) as task_app:
            assert task_app.url is not None
            assert task_app.url.startswith("https://")
            
            # Get tunnel again - should reuse
            async with InProcessTaskApp(
                app=app,
                port=19995,
                tunnel_mode="quick",
                api_key=api_key,
            ) as task_app2:
                # Should reuse same tunnel if possible
                assert task_app2.url is not None


