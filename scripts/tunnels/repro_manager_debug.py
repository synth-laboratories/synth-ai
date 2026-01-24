"""Debug repro: TunnelManager.open with explicit cloudflared process tracking.

Run with verbose logging:
  TUNNEL_DEBUG=1 uv run python scripts/tunnels/repro_manager_debug.py
"""
from __future__ import annotations

import asyncio
import logging
import os
import subprocess
import sys
import threading
import time
from http.server import HTTPServer, SimpleHTTPRequestHandler

LOCAL_PORT = 8001

# Configure logging based on TUNNEL_DEBUG env var
if os.environ.get("TUNNEL_DEBUG"):
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s.%(msecs)03d [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stderr,
    )
    # Enable debug logging for synth_ai tunnel modules
    for name in ["synth_ai.core.tunnels", "httpx", "httpcore"]:
        logging.getLogger(name).setLevel(logging.DEBUG)

class H(SimpleHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.end_headers()
        self.wfile.write(b'{"ok":1}')
    def log_message(self, *_): pass

def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

def check_cloudflared() -> str:
    r = subprocess.run(["pgrep", "-f", "cloudflared"], capture_output=True, text=True)
    pids = r.stdout.strip()
    return pids if pids else "none"

async def main() -> None:
    log(f"cloudflared_before_cleanup={check_cloudflared()}")
    
    # Cleanup
    subprocess.run(["pkill", "-9", "-f", "cloudflared"], capture_output=True)
    subprocess.run(["sh", "-c", "lsof -ti:8016 | xargs kill -9 2>/dev/null"], capture_output=True)
    subprocess.run(["sh", "-c", f"lsof -ti:{LOCAL_PORT} | xargs kill -9 2>/dev/null"], capture_output=True)
    await asyncio.sleep(0.5)
    
    log(f"cloudflared_after_cleanup={check_cloudflared()}")
    
    # Start local server
    threading.Thread(target=lambda: HTTPServer(("127.0.0.1", LOCAL_PORT), H).serve_forever(), daemon=True).start()
    log("local_server_started")
    
    log(f"cloudflared_before_import={check_cloudflared()}")
    
    from synth_ai.core.tunnels.manager import TunnelManager
    
    log(f"cloudflared_after_import={check_cloudflared()}")
    
    manager = TunnelManager()
    
    log(f"cloudflared_after_manager_init={check_cloudflared()}")
    
    log("calling_open")
    try:
        # Use context manager for proper cleanup
        async with manager:
            log("await_start")
            handle = await asyncio.wait_for(
                manager.open(
                    local_port=LOCAL_PORT,
                    verify_local=True,
                    verify_public=False,
                    progress=True,
                    local_timeout=10.0,
                ),
                timeout=30.0,
            )
            log("await_done")
            log(f"handle_url={handle.url}")
            log(f"cloudflared_running={check_cloudflared()}")
            
            # Simulate some work
            await asyncio.sleep(1)
            log("work_done")
        
        # Context manager calls shutdown() which stops connector + gateway
        log(f"cloudflared_after_shutdown={check_cloudflared()}")
        log("done")
    except asyncio.TimeoutError:
        log("TIMEOUT")
        log(f"cloudflared_at_timeout={check_cloudflared()}")

if __name__ == "__main__":
    os.environ.setdefault("NO_PROXY", "localhost,127.0.0.1")
    asyncio.run(main())
