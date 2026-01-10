from pathlib import Path

import pytest
import synth_ai.cli.task_app as task_app
from click import ClickException


def _write_executable(path: Path, content: str) -> None:
    path.write_text(content)
    path.chmod(path.stat().st_mode | 0o111)


def _make_real(dir_path: Path) -> Path:
    dir_path.mkdir(parents=True, exist_ok=True)
    real_path = dir_path / "modal"
    _write_executable(real_path, "#!/bin/sh\necho real\n")
    return real_path


def test_modal_command_prefix_uses_real_modal_when_available(tmp_path, monkeypatch, capsys):
    real_path = _make_real(tmp_path / "real")
    monkeypatch.setenv("PATH", str(real_path.parent))

    prefix = task_app._modal_command_prefix("modal")
    assert prefix == [str(real_path.resolve())]

    output = capsys.readouterr().out
    assert f"selected={real_path.resolve()}" in output


def test_modal_command_prefix_trusts_explicit_path(tmp_path):
    real_path = _make_real(tmp_path / "real")
    prefix = task_app._modal_command_prefix(str(real_path))
    assert prefix == [str(real_path.resolve())]


def test_modal_command_prefix_requires_modal_when_not_found(monkeypatch):
    monkeypatch.setenv("PATH", "")

    with pytest.raises(ClickException) as exc:
        task_app._modal_command_prefix("modal")

    assert "Modal CLI not found" in str(exc.value)


def test_modal_command_prefix_custom_cli_name(tmp_path, monkeypatch):
    custom_dir = tmp_path / "custom"
    custom_dir.mkdir()
    custom_cli = custom_dir / "custom-modal"
    _write_executable(custom_cli, "#!/bin/sh\necho custom\n")

    monkeypatch.setenv("PATH", str(custom_dir))

    prefix = task_app._modal_command_prefix("custom-modal")
    assert prefix == [str(custom_cli)]
