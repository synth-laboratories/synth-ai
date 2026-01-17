import subprocess

import synth_ai.cli.turso as turso_module
import synth_ai.core.tracing_v3.sqld as sqld_module
from click.testing import CliRunner
from synth_ai.cli import cli


def test_turso_reports_existing_binary(monkeypatch):
    runner = CliRunner()

    monkeypatch.setattr(turso_module, "find_sqld_binary", lambda: "/opt/sqld")
    monkeypatch.setattr(turso_module, "_get_sqld_version", lambda _: "sqld version v0.26.2")

    result = runner.invoke(cli, ["turso"])

    assert result.exit_code == 0
    assert "Turso sqld detected" in result.output
    assert "No action taken" in result.output


def test_turso_installs_when_missing(monkeypatch):
    runner = CliRunner()
    state = {"sqld": False, "cli": False}

    def fake_find():
        return "/home/test/.local/bin/sqld" if state["sqld"] else None

    def fake_which(cmd: str):
        if cmd == "brew":
            return "/opt/homebrew/bin/brew"
        if cmd == "turso":
            return "/opt/homebrew/bin/turso" if state["cli"] else None
        return None

    def fake_run(cmd, check=True):
        if cmd[:3] == ["/opt/homebrew/bin/brew", "install", "tursodatabase/tap/turso"]:
            state["cli"] = True
            return subprocess.CompletedProcess(cmd, 0)
        raise AssertionError(f"Unexpected subprocess.run command {cmd}")

    class PopenStub:
        def __init__(self, *args, **kwargs):
            self.returncode = 0

        def communicate(self, timeout=None):
            state["sqld"] = True
            return ("downloaded", "")

        def terminate(self):
            self.returncode = -15

        def kill(self):
            self.returncode = -9

    monkeypatch.setattr(turso_module, "find_sqld_binary", fake_find)
    monkeypatch.setattr(sqld_module, "find_sqld_binary", fake_find)
    monkeypatch.setattr(sqld_module.shutil, "which", fake_which)
    monkeypatch.setattr(sqld_module.subprocess, "run", fake_run)
    monkeypatch.setattr(sqld_module.subprocess, "Popen", lambda *a, **k: PopenStub())

    # Avoid creating real temp files; provide NamedTemporaryFile-like API
    class Tmp:
        name = "/tmp/synth_sqld_test.db"

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def close(self):
            return None

    class SocketStub:
        def __init__(self, *args, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_val, exc_tb):
            return False

        def bind(self, address):
            return None

        def getsockname(self):
            return ("127.0.0.1", 19099)

    monkeypatch.setattr(
        sqld_module.tempfile,
        "NamedTemporaryFile",
        lambda prefix, suffix, delete: Tmp(),
    )
    monkeypatch.setattr(sqld_module.socket, "socket", lambda *a, **k: SocketStub())

    result = runner.invoke(cli, ["turso"])

    assert result.exit_code == 0
    assert "Installing Turso CLI" in result.output
    assert "sqld available" in result.output
