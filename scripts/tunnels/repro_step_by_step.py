"""Step-by-step isolation of TunnelManager flow.

Each step has its own timeout. We find exactly where it hangs.
"""
from __future__ import annotations

import asyncio
import os
import subprocess
import sys
import threading
import time
from http.server import HTTPServer, SimpleHTTPRequestHandler

LOCAL_PORT = 8001

class H(SimpleHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.end_headers()
        self.wfile.write(b'{"ok":1}')
    def log_message(self, *_): pass

def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

async def main() -> None:
    # Cleanup
    subprocess.run(["pkill", "-9", "-f", "cloudflared"], capture_output=True)
    subprocess.run(["sh", "-c", "lsof -ti:8016 | xargs kill -9 2>/dev/null"], capture_output=True)
    subprocess.run(["sh", "-c", f"lsof -ti:{LOCAL_PORT} | xargs kill -9 2>/dev/null"], capture_output=True)
    await asyncio.sleep(0.5)

    # Start local server
    threading.Thread(target=lambda: HTTPServer(("127.0.0.1", LOCAL_PORT), H).serve_forever(), daemon=True).start()
    log(f"step=local_server port={LOCAL_PORT}")

    from synth_ai.core.tunnels.gateway import TunnelGateway, ensure_gateway_running
    from synth_ai.core.tunnels.connector import TunnelConnector, ensure_connector_running
    from synth_ai.core.tunnels.backend_client import get_lease_client
    from synth_ai.core.tunnels.manager import _get_client_instance_id
    import httpx

    # Step 1: Create lease
    log("step=create_lease start")
    client = get_lease_client()
    try:
        lease = await asyncio.wait_for(
            client.create_lease(
                client_instance_id=_get_client_instance_id(),
                local_host="127.0.0.1",
                local_port=LOCAL_PORT,
            ),
            timeout=15.0,
        )
        log(f"step=create_lease done route={lease.route_prefix}")
    except asyncio.TimeoutError:
        log("step=create_lease TIMEOUT")
        return

    # Step 2: Start gateway
    log("step=gateway start")
    try:
        gw = await asyncio.wait_for(ensure_gateway_running(8016, force=True), timeout=10.0)
        gw.add_route(lease.route_prefix, "127.0.0.1", LOCAL_PORT)
        log(f"step=gateway done running={gw.is_running}")
    except asyncio.TimeoutError:
        log("step=gateway TIMEOUT")
        return

    # Step 3: Verify local
    log("step=verify_local start")
    try:
        async with httpx.AsyncClient(timeout=5.0, trust_env=False) as hx:
            resp = await asyncio.wait_for(
                hx.get(f"http://127.0.0.1:{LOCAL_PORT}/health"),
                timeout=10.0,
            )
            log(f"step=verify_local done status={resp.status_code}")
    except asyncio.TimeoutError:
        log("step=verify_local TIMEOUT")
        return

    # Step 4: Start connector
    log("step=connector start")
    try:
        await asyncio.wait_for(
            ensure_connector_running(lease.tunnel_token),
            timeout=30.0,
        )
        from synth_ai.core.tunnels.connector import get_connector
        conn = get_connector()
        log(f"step=connector done connected={conn.is_connected}")
    except asyncio.TimeoutError:
        log("step=connector TIMEOUT")
        return

    # Step 5: Test gateway with subprocess (was deadlock)
    log("step=gateway_curl start")
    url = f"http://127.0.0.1:8016{lease.route_prefix}/health"
    r = subprocess.run(["curl", "-s", "-m", "5", url], capture_output=True, text=True)
    log(f"step=gateway_curl done exit={r.returncode} body={r.stdout.strip()}")

    # Step 6: Test public URL
    log("step=public_url start")
    from synth_ai.core.tunnels.cloudflare import resolve_hostname_with_explicit_resolvers
    ip = await resolve_hostname_with_explicit_resolvers(lease.hostname)
    public_url = f"https://{lease.hostname}{lease.route_prefix}/health"
    r = subprocess.run([
        "curl", "-s", "-m", "15",
        "--resolve", f"{lease.hostname}:443:{ip}",
        public_url,
    ], capture_output=True, text=True)
    log(f"step=public_url done exit={r.returncode} body={r.stdout.strip()}")

    # Cleanup
    await gw.stop()
    await conn.stop()
    log("step=cleanup done")

if __name__ == "__main__":
    os.environ.setdefault("NO_PROXY", "localhost,127.0.0.1")
    asyncio.run(main())
