"""Unit tests for smoke configuration validation."""

from __future__ import annotations

import pytest

from synth_ai.api.train.configs.rl import SmokeConfig


class TestSmokeConfig:
    """Test smoke configuration model."""

    def test_empty_smoke_config(self) -> None:
        """Test that empty smoke config is valid."""
        config = SmokeConfig()
        assert config is not None
        # All fields should be None by default
        assert config.task_url is None
        assert config.env_name is None
        assert config.max_steps is None

    def test_basic_smoke_config(self) -> None:
        """Test basic smoke configuration."""
        config = SmokeConfig(
            task_url="http://localhost:8001",
            env_name="crafter",
            policy_name="crafter-react",
            max_steps=10,
            use_mock=True,
        )
        assert config.task_url == "http://localhost:8001"
        assert config.env_name == "crafter"
        assert config.policy_name == "crafter-react"
        assert config.max_steps == 10
        assert config.use_mock is True

    def test_smoke_config_with_task_app_autostart(self) -> None:
        """Test smoke config with task app auto-start settings."""
        config = SmokeConfig(
            task_app_name="grpo-crafter",
            task_app_port=8765,
            task_app_env_file=".env",
            task_app_force=True,
            task_url="http://localhost:8765",
        )
        assert config.task_app_name == "grpo-crafter"
        assert config.task_app_port == 8765
        assert config.task_app_env_file == ".env"
        assert config.task_app_force is True
        assert config.task_url == "http://localhost:8765"

    def test_smoke_config_with_sqld_autostart(self) -> None:
        """Test smoke config with sqld auto-start settings."""
        config = SmokeConfig(
            sqld_auto_start=True,
            sqld_db_path="./traces/local.db",
            sqld_hrana_port=8080,
            sqld_http_port=8081,
        )
        assert config.sqld_auto_start is True
        assert config.sqld_db_path == "./traces/local.db"
        assert config.sqld_hrana_port == 8080
        assert config.sqld_http_port == 8081

    def test_smoke_config_full_setup(self) -> None:
        """Test smoke config with all auto-start features."""
        config = SmokeConfig(
            # Test parameters
            task_url="http://localhost:8765",
            env_name="crafter",
            policy_name="crafter-react",
            max_steps=10,
            policy="mock",
            model="gpt-5-nano",
            mock_backend="openai",
            mock_port=0,
            return_trace=True,
            use_mock=True,
            # Task app auto-start
            task_app_name="grpo-crafter",
            task_app_port=8765,
            task_app_env_file=".env",
            task_app_force=True,
            # sqld auto-start
            sqld_auto_start=True,
            sqld_db_path="./traces/local.db",
            sqld_hrana_port=8080,
            sqld_http_port=8081,
        )
        
        # Verify test parameters
        assert config.task_url == "http://localhost:8765"
        assert config.env_name == "crafter"
        assert config.policy_name == "crafter-react"
        assert config.max_steps == 10
        assert config.policy == "mock"
        assert config.model == "gpt-5-nano"
        assert config.mock_backend == "openai"
        assert config.mock_port == 0
        assert config.return_trace is True
        assert config.use_mock is True
        
        # Verify task app settings
        assert config.task_app_name == "grpo-crafter"
        assert config.task_app_port == 8765
        assert config.task_app_env_file == ".env"
        assert config.task_app_force is True
        
        # Verify sqld settings
        assert config.sqld_auto_start is True
        assert config.sqld_db_path == "./traces/local.db"
        assert config.sqld_hrana_port == 8080
        assert config.sqld_http_port == 8081

    def test_smoke_config_from_dict(self) -> None:
        """Test creating smoke config from dictionary."""
        data = {
            "task_url": "http://localhost:8001",
            "env_name": "math",
            "max_steps": 5,
            "task_app_name": "math-task-app",
            "task_app_port": 9000,
            "sqld_auto_start": True,
        }
        config = SmokeConfig(**data)
        assert config.task_url == "http://localhost:8001"
        assert config.env_name == "math"
        assert config.max_steps == 5
        assert config.task_app_name == "math-task-app"
        assert config.task_app_port == 9000
        assert config.sqld_auto_start is True

    def test_smoke_config_partial_task_app(self) -> None:
        """Test smoke config with only some task app fields."""
        config = SmokeConfig(
            task_app_name="grpo-crafter",
            # Other fields left as None
        )
        assert config.task_app_name == "grpo-crafter"
        assert config.task_app_port is None  # Should use default in smoke command
        assert config.task_app_env_file is None
        assert config.task_app_force is None

    def test_smoke_config_serialization(self) -> None:
        """Test that smoke config can be serialized."""
        config = SmokeConfig(
            task_url="http://localhost:8765",
            task_app_name="grpo-crafter",
            task_app_port=8765,
            sqld_auto_start=True,
            sqld_hrana_port=8080,
        )
        data = config.model_dump(exclude_none=True)
        assert "task_url" in data
        assert "task_app_name" in data
        assert "task_app_port" in data
        assert "sqld_auto_start" in data
        assert "sqld_hrana_port" in data
        # Fields that are None should be excluded
        assert "env_name" not in data
        assert "sqld_db_path" not in data

    def test_smoke_config_type_validation(self) -> None:
        """Test that smoke config validates types."""
        # Test integer fields
        config = SmokeConfig(max_steps=10, task_app_port=8765)
        assert isinstance(config.max_steps, int)
        assert isinstance(config.task_app_port, int)
        
        # Test boolean fields
        config = SmokeConfig(use_mock=True, sqld_auto_start=False, task_app_force=True)
        assert isinstance(config.use_mock, bool)
        assert isinstance(config.sqld_auto_start, bool)
        assert isinstance(config.task_app_force, bool)
        
        # Test string fields
        config = SmokeConfig(
            task_url="http://localhost:8765",
            env_name="crafter",
            task_app_name="grpo-crafter",
            sqld_db_path="./traces/local.db",
        )
        assert isinstance(config.task_url, str)
        assert isinstance(config.env_name, str)
        assert isinstance(config.task_app_name, str)
        assert isinstance(config.sqld_db_path, str)


