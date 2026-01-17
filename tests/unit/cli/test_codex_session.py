"""Tests for codex command session management and TOML config."""

import tempfile
from pathlib import Path

import pytest
from click.testing import CliRunner
from synth_ai.core.agents.codex import _load_session_config


@pytest.fixture()
def runner() -> CliRunner:
    return CliRunner()


@pytest.fixture()
def mock_env():
    """Mock environment variables."""
    return {
        "PATH": "/usr/bin:/bin",
        "HOME": "/home/test",
    }


@pytest.fixture()
def temp_toml_file():
    """Create a temporary TOML file."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".toml", delete=False) as f:
        yield Path(f.name)
    Path(f.name).unlink(missing_ok=True)


def test_load_session_config_default():
    """Test that default $20 cost limit is returned when no config exists."""
    config = _load_session_config(None)
    assert config == {"limit_cost_usd": 20.0}


def test_load_session_config_from_file(temp_toml_file):
    """Test loading session config from TOML file."""
    temp_toml_file.write_text("""
[session]
limit_cost_usd = 50.0
limit_tokens = 100000
limit_gpu_hours = 10.0
""")

    config = _load_session_config(temp_toml_file)
    assert config["limit_cost_usd"] == 50.0
    assert config["limit_tokens"] == 100000
    assert config["limit_gpu_hours"] == 10.0


def test_load_session_config_defaults_to_20(temp_toml_file):
    """Test that cost limit defaults to $20 if not specified."""
    temp_toml_file.write_text("""
[session]
limit_tokens = 50000
""")

    config = _load_session_config(temp_toml_file)
    assert config["limit_cost_usd"] == 20.0  # Default
    assert config["limit_tokens"] == 50000


def test_load_session_config_finds_codex_toml(tmp_path, monkeypatch):
    """Test that codex.toml is found in current directory."""
    codex_toml = tmp_path / "codex.toml"
    codex_toml.write_text("""
[session]
limit_cost_usd = 30.0
""")

    monkeypatch.chdir(tmp_path)
    config = _load_session_config(None)
    assert config["limit_cost_usd"] == 30.0


def test_load_session_config_finds_synth_toml(tmp_path, monkeypatch):
    """Test that synth.toml is found if codex.toml doesn't exist."""
    synth_toml = tmp_path / "synth.toml"
    synth_toml.write_text("""
[session]
limit_cost_usd = 40.0
""")

    monkeypatch.chdir(tmp_path)
    config = _load_session_config(None)
    assert config["limit_cost_usd"] == 40.0


# Note: Session creation/ending is tested in integration tests
# These unit tests focus on config loading which is the critical logic
