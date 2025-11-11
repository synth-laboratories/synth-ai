"""
Backend API client for Cloudflare Tunnel provisioning.
"""
import os
from typing import Any, Dict, Optional

import httpx

# Default backend URL - can be overridden via SYNTH_BACKEND_BASE_URL
BACKEND_BASE_URL = os.getenv("SYNTH_BACKEND_BASE_URL", "https://api.usesynth.ai")


async def create_tunnel(
    synth_api_key: str,
    port: int,
    subdomain: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Create a managed Cloudflare tunnel via Synth backend API.
    
    Args:
        synth_api_key: Synth API key for authentication
        port: Local port the tunnel will forward to
        subdomain: Optional custom subdomain (e.g., "my-company")
    
    Returns:
        Dict containing:
        - tunnel_token: Token for cloudflared
        - hostname: Public hostname (e.g., "cust-abc123.usesynth.ai")
        - access_client_id: Cloudflare Access client ID (if Access enabled)
        - access_client_secret: Cloudflare Access client secret (if Access enabled)
    
    Raises:
        httpx.HTTPStatusError: If API request fails
    """
    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.post(
            f"{BACKEND_BASE_URL}/api/v1/tunnels",
            headers={"Authorization": f"Bearer {synth_api_key}"},
            json={
                "port": port,
                "subdomain": subdomain,
                "mode": "managed",
                "enable_access": True,
            },
        )
        response.raise_for_status()
        return response.json()

