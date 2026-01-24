"""Repro: Public URL through Cloudflare edge using --resolve.

Run with:
  SYNTH_API_KEY=... SYNTH_BACKEND_URL=... uv run python scripts/tunnels/repro_public_url.py
"""
from __future__ import annotations

import asyncio
import subprocess
import threading
from http.server import HTTPServer, SimpleHTTPRequestHandler

LOCAL_PORT = 8001


class HealthHandler(SimpleHTTPRequestHandler):
    def do_GET(self) -> None:  # noqa: N802 (SimpleHTTPRequestHandler signature)
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(b'{"status":"ok"}')

    def log_message(self, *_: object) -> None:
        return


def start_local_server() -> None:
    server = HTTPServer(("127.0.0.1", LOCAL_PORT), HealthHandler)
    server.serve_forever()


async def main() -> None:
    from synth_ai.core.tunnels.backend_client import get_lease_client
    from synth_ai.core.tunnels.connector import TunnelConnector
    from synth_ai.core.tunnels.gateway import TunnelGateway
    from synth_ai.core.tunnels.manager import _get_client_instance_id
    from synth_ai.core.tunnels.cloudflare import resolve_hostname_with_explicit_resolvers

    threading.Thread(target=start_local_server, daemon=True).start()
    print(f"local_server=127.0.0.1:{LOCAL_PORT}")

    client = get_lease_client()
    lease = await client.create_lease(
        client_instance_id=_get_client_instance_id(),
        local_host="127.0.0.1",
        local_port=LOCAL_PORT,
    )
    print(f"lease_route={lease.route_prefix}")

    gateway = TunnelGateway(port=8016)
    await gateway.start(force=True)
    gateway.add_route(lease.route_prefix, "127.0.0.1", LOCAL_PORT)

    connector = TunnelConnector()
    await connector.start(lease.tunnel_token, timeout=45)

    ip = await resolve_hostname_with_explicit_resolvers(lease.hostname)
    url = f"https://{lease.hostname}{lease.route_prefix}/health"
    result = subprocess.run(
        ["curl", "-s", "-m", "15", "--resolve", f"{lease.hostname}:443:{ip}", url],
        capture_output=True,
        text=True,
        check=False,
    )
    print(f"exit={result.returncode} body={result.stdout.strip()}")

    await gateway.stop()
    await connector.stop()


if __name__ == "__main__":
    asyncio.run(main())
