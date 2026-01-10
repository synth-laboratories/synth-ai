"""Tests for opencode command session management and TOML config."""

import pytest
from click.testing import CliRunner
from synth_ai.core.agents.opencode import _load_session_config


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


def test_opencode_load_session_config_default():
    """Test that default $20 cost limit is returned when no config exists."""
    config = _load_session_config(None)
    assert config == {"limit_cost_usd": 20.0}


def test_opencode_load_session_config_from_file(tmp_path):
    """Test loading session config from TOML file."""
    temp_toml = tmp_path / "opencode.toml"
    temp_toml.write_text("""
[session]
limit_cost_usd = 75.0
limit_tokens = 150000
limit_gpu_hours = 8.0
""")

    config = _load_session_config(temp_toml)
    assert config["limit_cost_usd"] == 75.0
    assert config["limit_tokens"] == 150000
    assert config["limit_gpu_hours"] == 8.0


# Note: Session creation/ending is tested in integration tests
# These unit tests focus on config loading which is the critical logic
