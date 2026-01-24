"""Repro: TunnelManager.open end-to-end with timeouts.

Run with:
  SYNTH_API_KEY=... SYNTH_BACKEND_URL=... uv run python scripts/tunnels/repro_manager_open.py

For verbose logging:
  SYNTH_API_KEY=... SYNTH_BACKEND_URL=... TUNNEL_DEBUG=1 uv run python scripts/tunnels/repro_manager_open.py
"""
from __future__ import annotations

import asyncio
import logging
import os
import sys
import threading
import time as time_module
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
    # Also enable httpx debug logging
    logging.getLogger("httpx").setLevel(logging.DEBUG)
    logging.getLogger("httpcore").setLevel(logging.DEBUG)
else:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s.%(msecs)03d [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stderr,
    )

logger = logging.getLogger("repro")


def log(msg: str) -> None:
    """Log with timestamp and flush."""
    print(f"[{time_module.strftime('%H:%M:%S')}] {msg}", flush=True)
    sys.stderr.flush()


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


def list_threads() -> None:
    """Log all active threads."""
    threads = threading.enumerate()
    log(f"Active threads ({len(threads)}):")
    for t in threads:
        log(f"  - {t.name} (daemon={t.daemon}, alive={t.is_alive()})")


def list_tasks() -> None:
    """Log all asyncio tasks."""
    try:
        tasks = asyncio.all_tasks()
        log(f"Active asyncio tasks ({len(tasks)}):")
        for t in tasks:
            log(f"  - {t.get_name()} (done={t.done()}, cancelled={t.cancelled()})")
    except RuntimeError:
        log("No running event loop to list tasks")


async def main() -> None:
    import subprocess

    log("=" * 60)
    log("REPRO: TunnelManager.open() hang investigation")
    log("=" * 60)

    log("cleanup_start")
    # Cleanup stale processes
    subprocess.run(["pkill", "-9", "-f", "cloudflared"], capture_output=True)
    subprocess.run(
        ["sh", "-c", "lsof -ti:8016 | xargs kill -9 2>/dev/null"], capture_output=True
    )
    subprocess.run(
        ["sh", "-c", "lsof -ti:8001 | xargs kill -9 2>/dev/null"], capture_output=True
    )
    await asyncio.sleep(0.3)
    log("cleanup_done")

    log("import_start")
    from synth_ai.core.tunnels.manager import TunnelManager

    log("import_done")

    threading.Thread(target=start_local_server, daemon=True, name="local-http-server").start()
    log(f"local_server=127.0.0.1:{LOCAL_PORT}")

    list_threads()

    log("manager_create")
    manager = TunnelManager()
    log("manager_created")

    timeout_seconds = 45.0

    try:
        log("open_start - calling manager.open()")
        log(f"  local_port={LOCAL_PORT}")
        log(f"  verify_local=True")
        log(f"  verify_public=False")
        log(f"  timeout={timeout_seconds}s")

        handle = await asyncio.wait_for(
            manager.open(
                local_port=LOCAL_PORT,
                verify_local=True,
                verify_public=False,
                progress=True,
                local_timeout=10.0,
            ),
            timeout=timeout_seconds,
        )

        log("=" * 60)
        log("SUCCESS: manager.open() returned!")
        log("=" * 60)
        log(f"handle.url = {handle.url}")
        log(f"handle.hostname = {handle.hostname}")
        log(f"handle.lease.lease_id = {handle.lease.lease_id}")

        list_threads()
        list_tasks()

        log("Shutting down (closes tunnel + stops connector/gateway)...")
        await manager.shutdown()
        log("shutdown=ok")

    except asyncio.TimeoutError:
        log("=" * 60)
        log(f"TIMEOUT: manager.open() did not return within {timeout_seconds}s")
        log("=" * 60)

        list_threads()
        list_tasks()

    except Exception as e:
        log("=" * 60)
        log(f"ERROR: {type(e).__name__}: {e}")
        log("=" * 60)
        import traceback

        traceback.print_exc()

        list_threads()
        list_tasks()

    finally:
        log("Repro script exiting")
        sys.stdout.flush()
        sys.stderr.flush()


if __name__ == "__main__":
    os.environ.setdefault("NO_PROXY", "localhost,127.0.0.1")

    log("Starting asyncio.run(main())")
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        log("KeyboardInterrupt received")
    finally:
        log("asyncio.run() completed")
        list_threads()
