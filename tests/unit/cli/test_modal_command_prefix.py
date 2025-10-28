

import os
import sys
from pathlib import Path

import pytest
from click import ClickException

from synth_ai.cli import task_apps


@pytest.fixture(autouse=True)
def _reset_modal_env(monkeypatch):
    task_apps._is_modal_shim.cache_clear()
    monkeypatch.delenv("SYNTH_FORCE_MODAL_WRAPPER", raising=False)


def _write_executable(path: Path, content: str) -> None:
    path.write_text(content)
    path.chmod(path.stat().st_mode | 0o111)


def _make_wrapper_shim(dir_path: Path) -> Path:
    dir_path.mkdir(parents=True, exist_ok=True)
    shim_path = dir_path / "modal"
    _write_executable(shim_path, "#!/bin/sh\nsynth_ai.cli._modal_wrapper\n")
    return shim_path


def _make_modal_main_shim(dir_path: Path) -> Path:
    dir_path.mkdir(parents=True, exist_ok=True)
    shim_path = dir_path / "modal"
    _write_executable(
        shim_path,
        "#!/usr/bin/env python3\nfrom modal.__main__ import main\nif __name__ == '__main__':\n    main()\n",
    )
    return shim_path


def _make_real(dir_path: Path) -> Path:
    dir_path.mkdir(parents=True, exist_ok=True)
    real_path = dir_path / "modal"
    _write_executable(real_path, "#!/bin/sh\necho real\n")
    return real_path


def test_find_modal_executable_prefers_non_shim(tmp_path, monkeypatch):
    shim_path = _make_wrapper_shim(tmp_path / "shim")
    real_path = _make_real(tmp_path / "real")

    monkeypatch.setenv("PATH", os.pathsep.join([str(shim_path.parent), str(real_path.parent)]))

    preferred, shim_candidate = task_apps._find_modal_executable("modal")
    assert preferred is not None
    assert Path(preferred) == real_path
    assert shim_candidate is not None
    assert Path(shim_candidate) == shim_path


def test_find_modal_executable_trusts_explicit_path(tmp_path):
    real_path = _make_real(tmp_path / "real")
    preferred, shim_candidate = task_apps._find_modal_executable(str(real_path))
    assert preferred == str(real_path.resolve())
    assert shim_candidate is None


def test_is_modal_shim_detects_modal_main_stub(tmp_path, monkeypatch):
    shim_path = _make_modal_main_shim(tmp_path / ".venv" / "bin")
    monkeypatch.setenv("VIRTUAL_ENV", str(tmp_path / ".venv"))

    assert task_apps._is_modal_shim(str(shim_path)) is True


def test_modal_command_prefix_falls_back_to_wrapper_with_shim(tmp_path, monkeypatch, capsys):
    shim_path = _make_wrapper_shim(tmp_path / "shim")
    monkeypatch.setenv("PATH", str(shim_path.parent))
    monkeypatch.setattr(task_apps.importlib.util, "find_spec", lambda name: object() if name == "modal" else None)

    prefix = task_apps._modal_command_prefix("modal")
    assert prefix == [sys.executable, "-m", "synth_ai.cli._modal_wrapper"]

    output = capsys.readouterr().out
    assert "Using synth-ai modal shim" in output
    assert "selected=module-wrapper" in output


def test_modal_command_prefix_errors_when_only_shim_and_no_spec(tmp_path, monkeypatch):
    shim_path = _make_wrapper_shim(tmp_path / "shim")
    monkeypatch.setenv("PATH", str(shim_path.parent))
    monkeypatch.setattr(task_apps.importlib.util, "find_spec", lambda name: None)

    with pytest.raises(ClickException) as exc:
        task_apps._modal_command_prefix("modal")

    assert "synth-ai shim" in str(exc.value)


def test_modal_command_prefix_uses_real_modal_when_available(tmp_path, monkeypatch, capsys):
    real_path = _make_real(tmp_path / "real")
    monkeypatch.setenv("PATH", str(real_path.parent))
    monkeypatch.setattr(task_apps.importlib.util, "find_spec", lambda name: object() if name == "modal" else None)

    prefix = task_apps._modal_command_prefix("modal")
    assert prefix == [str(real_path.resolve())]

    output = capsys.readouterr().out
    assert f"selected={real_path.resolve()}" in output


def test_modal_command_prefix_trusts_explicit_path(tmp_path):
    real_path = _make_real(tmp_path / "real")
    prefix = task_apps._modal_command_prefix(str(real_path))
    assert prefix == [str(real_path.resolve())]


def test_modal_command_prefix_force_wrapper(monkeypatch):
    monkeypatch.setenv("SYNTH_FORCE_MODAL_WRAPPER", "1")
    prefix = task_apps._modal_command_prefix("modal")
    assert prefix == [sys.executable, "-m", "synth_ai.cli._modal_wrapper"]


def test_modal_command_prefix_requires_modal_when_not_found(monkeypatch):
    monkeypatch.setenv("PATH", "")
    monkeypatch.setattr(task_apps.importlib.util, "find_spec", lambda name: None)

    with pytest.raises(ClickException) as exc:
        task_apps._modal_command_prefix("modal")

    assert "Modal CLI not found" in str(exc.value)


def test_modal_command_prefix_custom_cli_name(tmp_path, monkeypatch):
    custom_dir = tmp_path / "custom"
    custom_dir.mkdir()
    custom_cli = custom_dir / "custom-modal"
    _write_executable(custom_cli, "#!/bin/sh\necho custom\n")

    monkeypatch.setenv("PATH", str(custom_dir))

    prefix = task_apps._modal_command_prefix("custom-modal")
    assert prefix == [str(custom_cli)]
