#!/usr/bin/env python3
"""
Test if SYNTH_API_KEY has access to managed tunnels.
"""

import asyncio
import os
import sys
from pathlib import Path

# Try to load .env file
env_file = Path(__file__).parent / ".env"
if env_file.exists():
    print(f"Found .env file: {env_file}")
    from dotenv import load_dotenv
    load_dotenv(env_file)
    print("Loaded .env file")
else:
    print("No .env file found in synth-ai directory")
    print("Checking environment variables...")

synth_api_key = os.environ.get("SYNTH_API_KEY")
if not synth_api_key:
    print("\n❌ SYNTH_API_KEY not found in environment")
    print("   Set it in .env file or export SYNTH_API_KEY=...")
    sys.exit(1)

print(f"\n✅ Found SYNTH_API_KEY: {synth_api_key[:20]}...{synth_api_key[-4:] if len(synth_api_key) > 24 else ''}")

# Test managed tunnel access
async def test_managed_tunnel_access():
    """Test if we can access managed tunnels."""
    print("\n" + "=" * 80)
    print("  TESTING MANAGED TUNNEL ACCESS")
    print("=" * 80)
    
    try:
        from synth_ai.cloudflare import create_tunnel, BACKEND_URL_BASE
        import httpx
        
        print(f"\n1. Backend URL: {BACKEND_URL_BASE}")
        print(f"2. Testing tunnel creation endpoint...")
        
        # First, try to list existing tunnels
        list_url = f"{BACKEND_URL_BASE}/api/v1/tunnels/"
        print(f"\n   Testing list tunnels endpoint: {list_url}")
        
        async with httpx.AsyncClient(timeout=10.0) as client:
            try:
                response = await client.get(
                    list_url,
                    headers={"Authorization": f"Bearer {synth_api_key}"},
                )
                if response.status_code == 200:
                    tunnels = response.json()
                    print(f"   ✅ Successfully listed tunnels")
                    print(f"   Found {len(tunnels)} existing tunnel(s)")
                    for tunnel in tunnels:
                        print(f"     - {tunnel.get('hostname', 'unknown')} (port {tunnel.get('local_port', '?')})")
                    return True
                else:
                    print(f"   ❌ Failed with status {response.status_code}")
                    print(f"   Response: {response.text[:500]}")
                    return False
            except Exception as e:
                print(f"   ❌ Error: {e}")
                import traceback
                traceback.print_exc()
                return False
        
    except ImportError as e:
        print(f"\n❌ Failed to import synth_ai.cloudflare: {e}")
        return False
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    try:
        result = asyncio.run(test_managed_tunnel_access())
        sys.exit(0 if result else 1)
    except KeyboardInterrupt:
        print("\n\nInterrupted")
        sys.exit(130)
    except Exception as e:
        print(f"\n\nFATAL: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

