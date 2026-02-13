"""Unit tests for InProcessTaskApp env override semantics."""

import asyncio
import socket

import pytest
from synth_ai.sdk.localapi._impl.in_process import InProcessTaskApp


def _pick_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


class _FakeServer:
    def __init__(self, config):  # type: ignore[no-untyped-def]
        self.should_exit = False
        self.force_exit = False

    async def serve(self) -> None:
        return


async def _fake_health(*args, **kwargs) -> None:  # type: ignore[no-untyped-def]
    return


@pytest.mark.unit
def test_env_tunnel_mode_does_not_override_explicit_local_mode(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Explicit tunnel_mode should win over SYNTH_TUNNEL_MODE."""
    app_path = tmp_path / "task_app.py"
    app_path.write_text("from fastapi import FastAPI\n\napp = FastAPI()\n")

    monkeypatch.setenv("SYNTH_TUNNEL_MODE", "synthtunnel")
    monkeypatch.setenv("SYNTH_BACKEND_URL", "http://localhost:8000")

    async def _unexpected_create(*args, **kwargs):  # type: ignore[no-untyped-def]
        raise AssertionError("TunneledLocalAPI.create should not be called in local mode")

    monkeypatch.setattr(
        "synth_ai.sdk.localapi._impl.in_process.TunneledLocalAPI.create",
        _unexpected_create,
    )
    monkeypatch.setattr(
        "synth_ai.sdk.localapi._impl.in_process.uvicorn.Server",
        _FakeServer,
    )
    monkeypatch.setattr(
        "synth_ai.sdk.localapi._impl.in_process._wait_for_local_health_check",
        _fake_health,
    )

    app = InProcessTaskApp(
        task_app_path=app_path,
        port=_pick_free_port(),
        host="127.0.0.1",
        tunnel_mode="local",
        api_key="env_test_key",
    )

    async def _run() -> str:
        async with app:
            assert app.url is not None
            return app.url

    url = asyncio.run(_run())
    assert url.startswith("http://127.0.0.1:")


@pytest.mark.unit
def test_env_tunnel_mode_still_overrides_default(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """SYNTH_TUNNEL_MODE should still override when using default settings."""
    app_path = tmp_path / "task_app.py"
    app_path.write_text("from fastapi import FastAPI\n\napp = FastAPI()\n")

    monkeypatch.setenv("SYNTH_TUNNEL_MODE", "local")
    monkeypatch.setenv("SYNTH_BACKEND_URL", "http://localhost:8000")

    async def _unexpected_create(*args, **kwargs):  # type: ignore[no-untyped-def]
        raise AssertionError(
            "TunneledLocalAPI.create should not be called when env forces local mode"
        )

    monkeypatch.setattr(
        "synth_ai.sdk.localapi._impl.in_process.TunneledLocalAPI.create",
        _unexpected_create,
    )
    monkeypatch.setattr(
        "synth_ai.sdk.localapi._impl.in_process.uvicorn.Server",
        _FakeServer,
    )
    monkeypatch.setattr(
        "synth_ai.sdk.localapi._impl.in_process._wait_for_local_health_check",
        _fake_health,
    )

    app = InProcessTaskApp(
        task_app_path=app_path,
        port=_pick_free_port(),
        host="127.0.0.1",
        # default tunnel_mode="synthtunnel"
        api_key="env_test_key",
    )

    async def _run() -> str:
        async with app:
            assert app.url is not None
            return app.url

    url = asyncio.run(_run())
    assert url.startswith("http://127.0.0.1:")
