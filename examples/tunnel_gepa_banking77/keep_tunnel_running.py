#!/usr/bin/env python3
"""Helper script to deploy tunnel and keep it running while training runs."""
import asyncio
import os
import signal
import sys
import time
from pathlib import Path

# Add repo to path
repo_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(repo_root))

from synth_ai.cfgs import CloudflareTunnelDeployCfg
from synth_ai.cloudflare import deploy_app_tunnel, _TUNNEL_PROCESSES, stop_tunnel


async def main():
    """Deploy tunnel and keep it running."""
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8102
    env_file = Path(sys.argv[2]) if len(sys.argv) > 2 else Path(".env.tunnel")
    task_app_path = Path(sys.argv[3]) if len(sys.argv) > 3 else Path("examples/task_apps/banking77/banking77_task_app.py")
    env_api_key = os.environ.get("ENVIRONMENT_API_KEY", "test-key")
    
    cfg = CloudflareTunnelDeployCfg.create(
        task_app_path=task_app_path,
        env_api_key=env_api_key,
        mode="quick",
        port=port,
        trace=False
    )
    
    try:
        print(f"🚀 Deploying tunnel on port {port}...")
        url = await deploy_app_tunnel(cfg, env_file)
        print(f"✅ Tunnel ready: {url}")
        print(f"📝 URL written to: {env_file}")
        print(f"⏳ Keeping tunnel running... (Press Ctrl+C to stop)")
        
        # Keep process alive
        while True:
            # Check if processes are still running
            if port in _TUNNEL_PROCESSES:
                proc = _TUNNEL_PROCESSES[port]
                if proc.poll() is not None:
                    print(f"❌ Tunnel process exited with code {proc.returncode}")
                    break
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n🛑 Stopping tunnel...")
    finally:
        if port in _TUNNEL_PROCESSES:
            stop_tunnel(_TUNNEL_PROCESSES[port])
            _TUNNEL_PROCESSES.pop(port, None)


if __name__ == "__main__":
    asyncio.run(main())



