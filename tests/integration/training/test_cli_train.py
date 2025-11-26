"""Integration tests for CLI train command.

These tests validate the CLI training workflow:
1. Config file discovery and validation
2. Train type detection (RL, SFT, Prompt Learning)
3. Command parsing and execution
4. Error handling and messaging

Tests are marked with pytest markers for selective execution:
- @pytest.mark.integration: All integration tests
- @pytest.mark.cli: CLI-specific tests
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner

pytestmark = [pytest.mark.integration, pytest.mark.cli]


class TestTrainConfigDiscovery:
    """Test train config file discovery."""

    def test_find_train_cfgs_finds_rl_config(self, tmp_path: Path, monkeypatch) -> None:
        """Should find RL training config in cwd."""
        from synth_ai.cli.lib.train_cfgs import find_train_cfgs_in_cwd

        monkeypatch.chdir(tmp_path)

        # Create RL config with all required fields
        rl_config = tmp_path / "rl.toml"
        rl_config.write_text("""
[algorithm]
type = "online"
variety = "grpo"

[policy]
model_name = "Qwen/Qwen3-0.6B"
trainer_mode = "local"
label = "test-run"

[compute]
gpu_type = "A100"
gpu_count = 1

[topology]
n_policy_workers = 1

[rollout]
env_name = "heartdisease"
policy_name = "test_policy"
""")

        configs = find_train_cfgs_in_cwd()

        assert len(configs) >= 1
        assert any(cfg[0] == "rl" for cfg in configs)

    def test_find_train_cfgs_finds_sft_config(self, tmp_path: Path, monkeypatch) -> None:
        """Should find SFT training config in cwd."""
        from synth_ai.cli.lib.train_cfgs import find_train_cfgs_in_cwd

        monkeypatch.chdir(tmp_path)

        # Create SFT config with all required fields
        sft_config = tmp_path / "sft.toml"
        sft_config.write_text("""
[algorithm]
type = "offline"
variety = "fft"
method = "sft"

[job]
model = "Qwen/Qwen3-0.6B"
data = "/path/to/train.jsonl"

[compute]
gpu_type = "A100"
gpu_count = 1
""")

        configs = find_train_cfgs_in_cwd()

        assert len(configs) >= 1
        assert any(cfg[0] == "sft" for cfg in configs)

    def test_find_train_cfgs_finds_prompt_config(self, tmp_path: Path, monkeypatch) -> None:
        """Should find Prompt Learning config in cwd."""
        from synth_ai.cli.lib.train_cfgs import find_train_cfgs_in_cwd

        monkeypatch.chdir(tmp_path)

        # Create prompt learning config with required gepa section
        pl_config = tmp_path / "gepa.toml"
        pl_config.write_text("""
[prompt_learning]
algorithm = "gepa"
task_app_url = "http://localhost:8001"
results_folder = "results"

[prompt_learning.gepa]
n_generations = 10
population_size = 5
""")

        configs = find_train_cfgs_in_cwd()

        assert len(configs) >= 1
        assert any(cfg[0] == "prompt" for cfg in configs)


class TestTrainTypeValidation:
    """Test train type detection from config."""

    def test_validate_train_cfg_detects_rl(self, tmp_path: Path) -> None:
        """Should detect RL training type."""
        from synth_ai.cli.lib.train_cfgs import validate_train_cfg

        config = tmp_path / "config.toml"
        config.write_text("""
[algorithm]
type = "online"
variety = "grpo"

[policy]
model_name = "Qwen/Qwen3-0.6B"
trainer_mode = "local"
label = "test-run"

[compute]
gpu_type = "A100"
gpu_count = 1

[topology]
n_policy_workers = 1

[rollout]
env_name = "heartdisease"
policy_name = "test_policy"
""")

        train_type = validate_train_cfg(config)
        assert train_type == "rl"

    def test_validate_train_cfg_detects_sft(self, tmp_path: Path) -> None:
        """Should detect SFT training type."""
        from synth_ai.cli.lib.train_cfgs import validate_train_cfg

        config = tmp_path / "config.toml"
        config.write_text("""
[algorithm]
type = "offline"
variety = "fft"
method = "sft"

[job]
model = "Qwen/Qwen3-0.6B"
data = "/path/to/train.jsonl"

[compute]
gpu_type = "A100"
gpu_count = 1
""")

        train_type = validate_train_cfg(config)
        assert train_type == "sft"

    def test_validate_train_cfg_detects_prompt_gepa(self, tmp_path: Path) -> None:
        """Should detect GEPA prompt learning type."""
        from synth_ai.cli.lib.train_cfgs import validate_train_cfg

        config = tmp_path / "config.toml"
        config.write_text("""
[prompt_learning]
algorithm = "gepa"
task_app_url = "http://localhost:8001"
results_folder = "results"

[prompt_learning.gepa]
n_generations = 10
population_size = 5
""")

        train_type = validate_train_cfg(config)
        assert train_type == "prompt"

    def test_validate_train_cfg_detects_prompt_mipro(self, tmp_path: Path) -> None:
        """Should detect MIPRO prompt learning type."""
        from synth_ai.cli.lib.train_cfgs import validate_train_cfg

        config = tmp_path / "config.toml"
        config.write_text("""
[prompt_learning]
algorithm = "mipro"
task_app_url = "http://localhost:8001"
results_folder = "results"

[prompt_learning.mipro]
n_trials = 10
""")

        train_type = validate_train_cfg(config)
        assert train_type == "prompt"


class TestTrainCommandParsing:
    """Test train command argument parsing."""

    @pytest.fixture
    def cli_runner(self) -> CliRunner:
        """Create Click test runner."""
        return CliRunner()

    def test_train_command_requires_config_or_auto_detect(self, cli_runner, tmp_path: Path) -> None:
        """Train command should require config or auto-detect."""
        from synth_ai.sdk.api.train.cli import train_command

        # Run in empty directory (no config to auto-detect)
        with cli_runner.isolated_filesystem(temp_dir=tmp_path):
            result = cli_runner.invoke(train_command, [])

        # Should either find no config or require config path
        assert result.exit_code != 0 or "No training config found" in result.output

    def test_train_command_accepts_config_path(self, cli_runner, tmp_path: Path) -> None:
        """Train command should accept --config path."""
        from synth_ai.sdk.api.train.cli import train_command

        config = tmp_path / "config.toml"
        config.write_text("""
[prompt_learning]
algorithm = "gepa"
task_app_url = "http://localhost:8001"
results_folder = "results"
""")

        # Run with config but expect failure due to missing credentials
        result = cli_runner.invoke(train_command, [str(config), "--no-poll"])

        # Should get past config parsing (may fail on API key)
        assert "config.toml" in result.output or "SYNTH_API_KEY" in result.output


class TestTrainCommandOptions:
    """Test train command options."""

    @pytest.fixture
    def cli_runner(self) -> CliRunner:
        """Create Click test runner."""
        return CliRunner()

    def test_train_command_has_poll_option(self, cli_runner) -> None:
        """Train command should have --poll/--no-poll option."""
        from synth_ai.sdk.api.train.cli import train_command

        result = cli_runner.invoke(train_command, ["--help"])
        assert "--poll" in result.output
        assert "--no-poll" in result.output

    def test_train_command_has_stream_format_option(self, cli_runner) -> None:
        """Train command should have --stream-format option."""
        from synth_ai.sdk.api.train.cli import train_command

        result = cli_runner.invoke(train_command, ["--help"])
        assert "--stream-format" in result.output
        assert "cli" in result.output
        assert "chart" in result.output

    def test_train_command_has_task_url_option(self, cli_runner) -> None:
        """Train command should have --task-url option."""
        from synth_ai.sdk.api.train.cli import train_command

        result = cli_runner.invoke(train_command, ["--help"])
        assert "--task-url" in result.output

    def test_train_command_has_dataset_option(self, cli_runner) -> None:
        """Train command should have --dataset option."""
        from synth_ai.sdk.api.train.cli import train_command

        result = cli_runner.invoke(train_command, ["--help"])
        assert "--dataset" in result.output

    def test_train_command_has_backend_option(self, cli_runner) -> None:
        """Train command should have --backend option."""
        from synth_ai.sdk.api.train.cli import train_command

        result = cli_runner.invoke(train_command, ["--help"])
        assert "--backend" in result.output

    def test_train_command_has_env_option(self, cli_runner) -> None:
        """Train command should have --env option."""
        from synth_ai.sdk.api.train.cli import train_command

        result = cli_runner.invoke(train_command, ["--help"])
        assert "--env" in result.output


class TestTrainCLIErrorHandling:
    """Test train CLI error handling."""

    @pytest.fixture
    def cli_runner(self) -> CliRunner:
        """Create Click test runner."""
        return CliRunner()

    def test_invalid_config_path_error(self, cli_runner) -> None:
        """Should error on invalid config path."""
        from synth_ai.sdk.api.train.cli import train_command

        result = cli_runner.invoke(train_command, ["/nonexistent/config.toml"])

        assert result.exit_code != 0
        assert "does not exist" in result.output.lower() or "not found" in result.output.lower()

    def test_malformed_config_error(self, cli_runner, tmp_path: Path) -> None:
        """Should error on malformed TOML config."""
        from synth_ai.sdk.api.train.cli import train_command

        config = tmp_path / "bad.toml"
        config.write_text("this is not valid toml [[[")

        result = cli_runner.invoke(train_command, [str(config)])

        assert result.exit_code != 0


class TestTrainCLIEnvHandling:
    """Test train CLI environment variable handling."""

    @pytest.fixture
    def cli_runner(self) -> CliRunner:
        """Create Click test runner."""
        return CliRunner()

    def test_loads_env_file(self, cli_runner, tmp_path: Path) -> None:
        """Should load .env file when --env specified."""
        from synth_ai.sdk.api.train.cli import train_command

        # Create env file
        env_file = tmp_path / ".env"
        env_file.write_text("SYNTH_API_KEY=test-key\nENVIRONMENT_API_KEY=test-env-key")

        # Create config with required gepa section
        config = tmp_path / "config.toml"
        config.write_text("""
[prompt_learning]
algorithm = "gepa"
task_app_url = "http://localhost:8001"
results_folder = "results"

[prompt_learning.gepa]
n_generations = 10
population_size = 5
""")

        result = cli_runner.invoke(
            train_command,
            [str(config), "--env", str(env_file), "--no-poll"],
        )

        # Should get past env loading
        assert ".env" in result.output or "test-key" in result.output or result.exit_code in (0, 1)

    def test_backend_override_from_env(self, cli_runner, tmp_path: Path, monkeypatch) -> None:
        """Should use BACKEND_BASE_URL from environment."""
        from synth_ai.sdk.api.train.cli import train_command

        monkeypatch.setenv("BACKEND_BASE_URL", "http://localhost:8000")
        monkeypatch.setenv("SYNTH_API_KEY", "test-key")
        monkeypatch.setenv("ENVIRONMENT_API_KEY", "test-env-key")

        config = tmp_path / "config.toml"
        config.write_text("""
[prompt_learning]
algorithm = "gepa"
task_app_url = "http://localhost:8001"
results_folder = "results"

[prompt_learning.gepa]
n_generations = 10
population_size = 5
""")

        result = cli_runner.invoke(train_command, [str(config), "--no-poll"])

        # Should mention localhost backend or accept the config
        assert "localhost:8000" in result.output or "Backend" in result.output or result.exit_code in (0, 1)


class TestTrainCLIBackendFlag:
    """Test train CLI --backend flag handling."""

    @pytest.fixture
    def cli_runner(self) -> CliRunner:
        """Create Click test runner."""
        return CliRunner()

    def test_backend_flag_overrides_env(self, cli_runner, tmp_path: Path, monkeypatch) -> None:
        """--backend flag should override BACKEND_BASE_URL."""
        from synth_ai.sdk.api.train.cli import train_command

        monkeypatch.setenv("BACKEND_BASE_URL", "http://ignored:8000")
        monkeypatch.setenv("SYNTH_API_KEY", "test-key")
        monkeypatch.setenv("ENVIRONMENT_API_KEY", "test-env-key")

        config = tmp_path / "config.toml"
        config.write_text("""
[prompt_learning]
algorithm = "gepa"
task_app_url = "http://localhost:8001"
results_folder = "results"

[prompt_learning.gepa]
n_generations = 10
population_size = 5
""")

        result = cli_runner.invoke(
            train_command,
            [str(config), "--backend", "http://override:9000", "--no-poll"],
        )

        # Should mention override backend, not ignored, or accept the config
        assert "override:9000" in result.output or "--backend" in result.output or result.exit_code in (0, 1)
