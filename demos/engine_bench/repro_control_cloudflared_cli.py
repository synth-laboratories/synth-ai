#!/usr/bin/env python3
"""
CONTROL TEST: Verify cloudflared CLI works directly.

This script spawns cloudflared via subprocess (NOT the Rust wrapper)
to confirm that the CLI itself is functional. This is the known-working case.

Expected output:
    ✅ cloudflared CLI creates a tunnel successfully
    
If this fails, the issue is with cloudflared installation, not the Rust wrapper.

Usage:
    cd /Users/joshpurtell/Documents/GitHub/synth-ai
    .venv/bin/python demos/engine_bench/repro_control_cloudflared_cli.py
"""

import asyncio
import re
import subprocess
import sys
import time


def find_cloudflared() -> str | None:
    """Find cloudflared binary path."""
    import shutil
    
    # Try common locations
    paths = [
        "/opt/homebrew/bin/cloudflared",
        "/usr/local/bin/cloudflared",
        shutil.which("cloudflared"),
    ]
    
    for path in paths:
        if path:
            try:
                result = subprocess.run([path, "--version"], capture_output=True, timeout=5)
                if result.returncode == 0:
                    return path
            except Exception:
                pass
    
    return None


async def test_quick_tunnel_via_cli():
    """Create a quick tunnel using cloudflared CLI directly."""
    
    cf_path = find_cloudflared()
    if not cf_path:
        print("ERROR: cloudflared not found. Install with: brew install cloudflared")
        return False
    
    print(f"cloudflared path: {cf_path}")
    
    # Get version
    version = subprocess.run([cf_path, "--version"], capture_output=True, text=True)
    print(f"cloudflared version: {version.stdout.strip()}")
    
    print("\n" + "-" * 50)
    print("Starting cloudflared tunnel via CLI...")
    print("Looking for trycloudflare.com URL in output...")
    print("-" * 50 + "\n")
    
    # Start cloudflared with a dummy URL (we don't need a real server)
    proc = subprocess.Popen(
        [cf_path, "tunnel", "--url", "http://localhost:19999"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    
    tunnel_url = None
    start_time = time.time()
    timeout = 20.0
    
    try:
        while time.time() - start_time < timeout:
            line = proc.stdout.readline()
            if not line:
                await asyncio.sleep(0.1)
                continue
            
            line = line.strip()
            print(f"  [cloudflared] {line}")
            
            # Look for the tunnel URL pattern
            # Example: https://xxx-yyy-zzz.trycloudflare.com
            match = re.search(r'(https://[a-z0-9-]+\.trycloudflare\.com)', line)
            if match:
                tunnel_url = match.group(1)
                print(f"\n✅ FOUND TUNNEL URL: {tunnel_url}")
                break
            
            # Also check for error messages
            if "error" in line.lower() or "failed" in line.lower():
                print(f"\n⚠️  Error in cloudflared output")
        
        if not tunnel_url:
            print(f"\n❌ No tunnel URL found after {timeout}s")
            return False
        
        return True
        
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()


async def main():
    print("=" * 60)
    print("CONTROL TEST: cloudflared CLI (bypassing Rust wrapper)")
    print("Expected: This should SUCCEED (cloudflared works directly)")
    print("=" * 60)
    
    success = await test_quick_tunnel_via_cli()
    
    print("\n" + "=" * 60)
    print("RESULT")
    print("=" * 60)
    
    if success:
        print("\n✅ CONTROL PASSED: cloudflared CLI works correctly")
        print("\nThis confirms the issue is in the Rust synth_ai_py wrapper,")
        print("NOT in cloudflared itself.")
        return 0
    else:
        print("\n❌ CONTROL FAILED: cloudflared CLI itself has issues")
        print("\nThis is unexpected - cloudflared should work directly.")
        print("Check: brew install cloudflared, or network connectivity.")
        return 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
