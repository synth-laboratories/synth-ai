from __future__ import annotations

import socket
import threading
import time
from collections.abc import Iterator

import httpx
import pytest
import uvicorn

from testing.fake_backend import TEST_API_KEY, build_fake_backend


@pytest.fixture(scope="session")
def api_key() -> str:
    return TEST_API_KEY


@pytest.fixture(scope="session")
def backend_url() -> Iterator[str]:
    host = "127.0.0.1"
    with socket.socket() as sock:
        sock.bind((host, 0))
        port = sock.getsockname()[1]

    app = build_fake_backend()
    config = uvicorn.Config(app, host=host, port=port, log_level="error")
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()

    base_url = f"http://{host}:{port}"
    deadline = time.time() + 10
    while time.time() < deadline:
        try:
            response = httpx.get(f"{base_url}/__test__/health", timeout=1.0)
            if response.status_code == 200:
                break
        except httpx.HTTPError:
            time.sleep(0.1)
    else:
        server.should_exit = True
        thread.join(timeout=5)
        raise RuntimeError("fake backend did not start")

    yield base_url

    server.should_exit = True
    thread.join(timeout=5)


@pytest.fixture(autouse=True)
def reset_fake_backend(backend_url: str) -> Iterator[None]:
    response = httpx.post(f"{backend_url}/__test__/reset", timeout=5.0)
    response.raise_for_status()
    yield
