"""
Cloudflare Tunnel client module.

Provides functions to spawn and manage cloudflared processes for quick and managed tunnels.
"""
import os
import re
import shutil
import subprocess
import sys
import time
from typing import Optional, Tuple


_URL_RE = re.compile(r"https://[a-z0-9-]+\.trycloudflare\.com", re.I)


def _which_cloudflared() -> str:
    """Find cloudflared binary in PATH or common install locations."""
    path = shutil.which("cloudflared")
    if path:
        return path
    
    # Check common install locations
    common_paths = [
        "/usr/local/bin/cloudflared",
        "/opt/homebrew/bin/cloudflared",
        os.path.expanduser("~/bin/cloudflared"),
    ]
    for common_path in common_paths:
        if os.path.exists(common_path) and os.access(common_path, os.X_OK):
            return common_path
    
    raise FileNotFoundError(
        "cloudflared not found. Install it:\n"
        "  macOS: brew install cloudflare/cloudflare/cloudflared\n"
        "  Linux/Windows: https://developers.cloudflare.com/cloudflare-one/networks/connectors/cloudflare-tunnel/do-more-with-tunnels/local-management/create-local-tunnel/"
    )


def open_quick_tunnel(port: int, wait_s: float = 10.0) -> Tuple[str, subprocess.Popen]:
    """
    Open a quick (ephemeral) Cloudflare tunnel.
    
    Args:
        port: Local port to tunnel to
        wait_s: Maximum time to wait for URL in seconds
    
    Returns:
        Tuple of (public_url, process_handle)
    
    Raises:
        FileNotFoundError: If cloudflared is not installed
        RuntimeError: If tunnel fails to start or URL cannot be parsed
    """
    bin_path = _which_cloudflared()
    proc = subprocess.Popen(
        [bin_path, "tunnel", "--url", f"http://127.0.0.1:{port}"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    
    start = time.time()
    url: Optional[str] = None
    
    # Stream stdout to detect the trycloudflare URL
    while time.time() - start < wait_s:
        if proc.poll() is not None:
            # Process exited early
            stdout, _ = proc.communicate()
            raise RuntimeError(
                f"cloudflared exited early with code {proc.returncode}. "
                f"Output: {stdout[:500] if stdout else 'no output'}"
            )
        
        line = proc.stdout.readline()
        if not line:
            time.sleep(0.05)
            continue
        
        match = _URL_RE.search(line)
        if match:
            url = match.group(0)
            break
    
    if not url:
        proc.terminate()
        stdout, _ = proc.communicate(timeout=2.0)
        raise RuntimeError(
            f"Failed to parse trycloudflare URL from cloudflared output after {wait_s}s. "
            f"Output: {stdout[:500] if stdout else 'no output'}"
        )
    
    return url, proc


def open_managed_tunnel(tunnel_token: str) -> subprocess.Popen:
    """
    Open a managed (named) Cloudflare tunnel using a token.
    
    Args:
        tunnel_token: Cloudflare tunnel token from backend API
    
    Returns:
        Process handle for the tunnel
    
    Raises:
        FileNotFoundError: If cloudflared is not installed
    """
    bin_path = _which_cloudflared()
    # cloudflared v2023.4+ accepts --token for named tunnels
    return subprocess.Popen(
        [bin_path, "tunnel", "run", "--token", tunnel_token],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )


def stop_tunnel(proc: Optional[subprocess.Popen]) -> None:
    """
    Gracefully stop a tunnel process.
    
    Args:
        proc: Process handle to terminate, or None
    """
    if proc is None:
        return
    
    if proc.poll() is None:
        # Process is still running
        proc.terminate()
        try:
            proc.wait(timeout=5.0)
        except subprocess.TimeoutExpired:
            # Force kill if graceful termination fails
            proc.kill()
            proc.wait()

