from __future__ import annotations
from pathlib import Path

import pytest

from synth_ai.tracing_v3.config import (
    resolve_trace_db_auth_token,
    resolve_trace_db_settings,
    resolve_trace_db_url,
)
from synth_ai.tracing_v3.storage.config import StorageBackend, StorageConfig


def _clear_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Remove tracing-related environment variables for test isolation."""
    for key in [
        "SYNTH_TRACES_DB",
        "SYNTH_TRACES_DIR",
        "LIBSQL_URL",
        "LIBSQL_AUTH_TOKEN",
        "TURSO_DATABASE_URL",
        "TURSO_AUTH_TOKEN",
        "TRACING_DB_URL",
        "TRACING_DB_AUTH_TOKEN",
        "TURSO_LOCAL_DB_URL",
        "SQLD_DB_PATH",
        "TURSO_NATIVE",
    ]:
        monkeypatch.delenv(key, raising=False)


def test_explicit_sqlite_url_passthrough(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    _clear_env(monkeypatch)
    sqlite_url = f"sqlite+aiosqlite:///{tmp_path/'manual.db'}?mode=ro"
    monkeypatch.setenv("SYNTH_TRACES_DB", sqlite_url)

    resolved_url, token = resolve_trace_db_settings(ensure_dir=False)
    assert resolved_url == sqlite_url
    assert token is None
    assert resolve_trace_db_url(ensure_dir=False) == sqlite_url
    assert resolve_trace_db_auth_token() is None


def test_libsql_env_injects_auth_token(monkeypatch: pytest.MonkeyPatch):
    _clear_env(monkeypatch)
    monkeypatch.setenv("LIBSQL_URL", "libsql://tracer.example.turso.io")
    monkeypatch.setenv("LIBSQL_AUTH_TOKEN", "tkn_test")

    resolved_url, token = resolve_trace_db_settings(ensure_dir=False)
    assert resolved_url == "libsql://tracer.example.turso.io"
    assert token == "tkn_test"

    config = StorageConfig(connection_string=None, backend=None, turso_auth_token=None)
    assert config.backend == StorageBackend.TURSO_NATIVE
    assert config.get_connection_string() == "libsql://tracer.example.turso.io"
    assert config.get_backend_config()["auth_token"] == "tkn_test"


def test_embedded_auth_token_is_stripped(monkeypatch: pytest.MonkeyPatch):
    _clear_env(monkeypatch)
    monkeypatch.setenv(
        "LIBSQL_URL",
        "libsql://example-db.turso.io?auth_token=embedded&resource=traces",
    )

    resolved_url, token = resolve_trace_db_settings(ensure_dir=False)
    assert token == "embedded"
    # Remaining query params should be preserved (ordering may differ)
    assert resolved_url.startswith("libsql://example-db.turso.io")
    assert "auth_token" not in resolved_url
    assert "resource=traces" in resolved_url

    config = StorageConfig(connection_string=resolved_url, backend=None, turso_auth_token=None)
    assert config.get_backend_config()["auth_token"] == "embedded"


def test_sqlite_fallback_creates_directory(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    _clear_env(monkeypatch)
    traces_dir = tmp_path / "traces_dir"
    monkeypatch.setenv("SYNTH_TRACES_DIR", str(traces_dir))

    resolved_url, token = resolve_trace_db_settings()
    assert resolved_url.startswith("sqlite+aiosqlite:///")
    assert token is None
    assert traces_dir.exists()

    config = StorageConfig(connection_string=None, backend=None, turso_auth_token=None)
    assert config.backend == StorageBackend.SQLITE
    assert config.get_backend_config() == {}
