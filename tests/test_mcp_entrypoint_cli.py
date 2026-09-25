"""MCP console entry points: --help/--version exit 0; bad config is one typed line."""

from __future__ import annotations

import subprocess
import sys

import pytest
from synth_ai import __version__
from synth_ai.mcp.research.server import (
    McpConfigurationError,
    _stdio_server,
    main,
    main_index,
)

_ENTRYPOINTS = [
    pytest.param(main_index, "synth-ai-index-mcp", id="index"),
    pytest.param(main, "synth-ai-research-mcp", id="research"),
]


@pytest.fixture(autouse=True)
def _clean_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in (
        "SYNTH_BACKEND_URL",
        "SYNTH_API_KEY",
        "SYNTH_INDEX_MCP_ENABLED",
        "SYNTH_INDEX_MCP_WRITE_ENABLED",
    ):
        monkeypatch.delenv(name, raising=False)


@pytest.mark.parametrize(("entrypoint", "prog"), _ENTRYPOINTS)
def test_help_prints_usage_and_exits_zero_without_configuration(
    entrypoint, prog: str, capsys: pytest.CaptureFixture[str]
) -> None:
    with pytest.raises(SystemExit) as exited:
        entrypoint(["--help"])
    assert exited.value.code == 0
    out = capsys.readouterr().out
    assert out.startswith(f"usage: {prog}")
    assert "SYNTH_BACKEND_URL" in out


@pytest.mark.parametrize(("entrypoint", "prog"), _ENTRYPOINTS)
def test_version_exits_zero(entrypoint, prog: str, capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exited:
        entrypoint(["--version"])
    assert exited.value.code == 0
    assert capsys.readouterr().out.strip() == f"{prog} {__version__}"


@pytest.mark.parametrize(("entrypoint", "prog"), _ENTRYPOINTS)
def test_unknown_argument_is_a_usage_error(entrypoint, prog: str) -> None:
    with pytest.raises(SystemExit) as exited:
        entrypoint(["--bogus"])
    assert exited.value.code == 2


def test_missing_backend_url_is_typed_error() -> None:
    with pytest.raises(McpConfigurationError, match="explicit SYNTH_BACKEND_URL"):
        _stdio_server(index_only=True)
    assert issubclass(McpConfigurationError, ValueError)


@pytest.mark.parametrize(
    ("entrypoint", "prog", "env"),
    [
        (main_index, "synth-ai-index-mcp", {}),
        (main, "synth-ai-research-mcp", {"SYNTH_INDEX_MCP_ENABLED": "true"}),
    ],
)
def test_missing_backend_url_exits_with_one_line(
    entrypoint,
    prog: str,
    env: dict[str, str],
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    for name, value in env.items():
        monkeypatch.setenv(name, value)
    with pytest.raises(SystemExit) as exited:
        entrypoint([])
    assert exited.value.code == 2
    err = capsys.readouterr().err
    assert err == f"{prog}: error: Index MCP requires explicit SYNTH_BACKEND_URL\n"


def test_installed_style_process_has_no_traceback() -> None:
    code = "from synth_ai.mcp.research.server import main_index; main_index()"
    completed = subprocess.run(
        [sys.executable, "-c", code],
        env={"PATH": "/usr/bin:/bin"},
        input="",
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert completed.returncode == 2
    assert "Traceback" not in completed.stderr
    assert completed.stderr.strip().splitlines() == [
        "synth-ai-index-mcp: error: Index MCP requires explicit SYNTH_BACKEND_URL"
    ]
