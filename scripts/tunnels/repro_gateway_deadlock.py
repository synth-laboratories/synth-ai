"""Repro: gateway deadlock when caller blocks the main thread.

Expected: curl returns a JSON body quickly.
If it hangs, the gateway event loop is blocked.
"""
from __future__ import annotations

import asyncio
import subprocess
import threading
from http.server import HTTPServer, SimpleHTTPRequestHandler

LOCAL_PORT = 8001
GATEWAY_PORT = 8016


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
    from synth_ai.core.tunnels.gateway import TunnelGateway

    threading.Thread(target=start_local_server, daemon=True).start()

    gateway = TunnelGateway(port=GATEWAY_PORT)
    await gateway.start(force=True)
    gateway.add_route("/s/test", "127.0.0.1", LOCAL_PORT)

    url = f"http://127.0.0.1:{GATEWAY_PORT}/s/test/health"
    result = subprocess.run(
        ["curl", "-s", "-m", "5", url],
        capture_output=True,
        text=True,
        check=False,
    )
    print(f"exit={result.returncode} body={result.stdout.strip()}")

    await gateway.stop()


if __name__ == "__main__":
    asyncio.run(main())
