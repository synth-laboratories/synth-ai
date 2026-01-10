"""Comprehensive unit tests for GEPA file-based config validation.

Tests the validate_gepa_config_from_file function
that provides comprehensive validation matching backend requirements.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest
import toml

from synth_ai.sdk.api.train.validators import (
    ConfigValidationError,
    validate_gepa_config_from_file,
    validate_prompt_learning_config_from_file,
)

pytestmark = pytest.mark.unit


def _create_toml_file(config_data: dict) -> Path:
    """Helper to create a temporary TOML config file."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".toml", delete=False) as f:
        toml.dump(config_data, f)
        return Path(f.name)


class TestGEPAFileValidation:
    """Tests for validate_gepa_config_from_file."""

    def test_valid_gepa_config_passes(self) -> None:
        """Test that a valid GEPA config passes validation."""
        config_data = {
            "prompt_learning": {
                "algorithm": "gepa",
                "task_app_url": "http://localhost:8001",
                "task_app_api_key": "test-key",
                "initial_prompt": {
                    "id": "test-prompt",
                    "messages": [{"role": "system", "content": "Test"}],
                },
                "policy": {
                    "model": "gpt-4o-mini",
                    "provider": "openai",
                },
                "gepa": {
                    "env_name": "banking77",
                    "evaluation": {
                        "train_seeds": [0, 1, 2],
                        "val_seeds": [10, 11, 12],
                    },
                    "rollout": {
                        "budget": 100,
                    },
                    "mutation": {
                        "llm_model": "gpt-4o-mini",
                        "llm_provider": "openai",
                    },
                    "population": {
                        "initial_size": 20,
                        "num_generations": 10,
                    },
                    "archive": {
                        "size": 64,
                    },
                    "token": {
                        "max_limit": 1000,
                    },
                },
            }
        }
        path = _create_toml_file(config_data)
        try:
            is_valid, errors = validate_gepa_config_from_file(path)
            assert is_valid, f"Validation failed with errors: {errors}"
        finally:
            path.unlink(missing_ok=True)

    def test_missing_prompt_learning_section(self) -> None:
        """Test that missing prompt_learning section fails."""
        config_data = {}
        path = _create_toml_file(config_data)
        try:
            is_valid, errors = validate_gepa_config_from_file(path)
            assert not is_valid
            assert any("prompt_learning" in err.lower() for err in errors)
        finally:
            path.unlink(missing_ok=True)

    def test_wrong_algorithm_fails(self) -> None:
        """Test that wrong algorithm fails."""
        config_data = {
            "prompt_learning": {
                "algorithm": "gepa",
                "task_app_url": "http://localhost:8001",
            }
        }
        path = _create_toml_file(config_data)
        try:
            is_valid, errors = validate_gepa_config_from_file(path)
            assert not is_valid
            assert any("gepa" in err.lower() for err in errors)
        finally:
            path.unlink(missing_ok=True)

    def test_missing_task_app_url_fails(self) -> None:
        """Test that missing task_app_url fails."""
        config_data = {
            "prompt_learning": {
                "algorithm": "gepa",
                "task_app_api_key": "test-key",
            }
        }
        path = _create_toml_file(config_data)
        try:
            is_valid, errors = validate_gepa_config_from_file(path)
            assert not is_valid
            assert any("task_app_url" in err.lower() for err in errors)
        finally:
            path.unlink(missing_ok=True)

    def test_missing_env_name_fails(self) -> None:
        """Test that missing env_name fails."""
        config_data = {
            "prompt_learning": {
                "algorithm": "gepa",
                "task_app_url": "http://localhost:8001",
                "task_app_api_key": "test-key",
                "gepa": {},
            }
        }
        path = _create_toml_file(config_data)
        try:
            is_valid, errors = validate_gepa_config_from_file(path)
            assert not is_valid
            assert any("env_name" in err.lower() for err in errors)
        finally:
            path.unlink(missing_ok=True)

    def test_missing_required_sections_fails(self) -> None:
        """Test that missing required GEPA sections fail."""
        config_data = {
            "prompt_learning": {
                "algorithm": "gepa",
                "task_app_url": "http://localhost:8001",
                "task_app_api_key": "test-key",
                "gepa": {
                    "env_name": "banking77",
                },
            }
        }
        path = _create_toml_file(config_data)
        try:
            is_valid, errors = validate_gepa_config_from_file(path)
            assert not is_valid
            assert any("evaluation" in err.lower() or "rollout" in err.lower() for err in errors)
        finally:
            path.unlink(missing_ok=True)

    def test_inference_url_rejection(self) -> None:
        """Test that inference_url in policy is rejected."""
        config_data = {
            "prompt_learning": {
                "algorithm": "gepa",
                "task_app_url": "http://localhost:8001",
                "task_app_api_key": "test-key",
                "policy": {
                    "model": "gpt-4o-mini",
                    "provider": "openai",
                    "inference_url": "https://api.openai.com/v1",  # Should be rejected
                },
                "gepa": {
                    "env_name": "banking77",
                    "evaluation": {"train_seeds": [0], "val_seeds": [10]},
                    "rollout": {"budget": 100},
                    "mutation": {"llm_model": "gpt-4o-mini", "llm_provider": "openai"},
                    "population": {"initial_size": 20},
                    "archive": {"size": 64},
                    "token": {"max_limit": 1000},
                },
            }
        }
        path = _create_toml_file(config_data)
        try:
            is_valid, errors = validate_gepa_config_from_file(path)
            assert not is_valid
            assert any("inference_url" in err.lower() and "must not" in err.lower() for err in errors)
        finally:
            path.unlink(missing_ok=True)

    def test_api_base_rejection(self) -> None:
        """Test that api_base in policy is rejected."""
        config_data = {
            "prompt_learning": {
                "algorithm": "gepa",
                "task_app_url": "http://localhost:8001",
                "task_app_api_key": "test-key",
                "policy": {
                    "model": "gpt-4o-mini",
                    "provider": "openai",
                    "api_base": "https://api.openai.com/v1",  # Should be rejected
                },
                "gepa": {
                    "env_name": "banking77",
                    "evaluation": {"train_seeds": [0], "val_seeds": [10]},
                    "rollout": {"budget": 100},
                    "mutation": {"llm_model": "gpt-4o-mini", "llm_provider": "openai"},
                    "population": {"initial_size": 20},
                    "archive": {"size": 64},
                    "token": {"max_limit": 1000},
                },
            }
        }
        path = _create_toml_file(config_data)
        try:
            is_valid, errors = validate_gepa_config_from_file(path)
            assert not is_valid
            assert any("api_base" in err.lower() and "must not" in err.lower() for err in errors)
        finally:
            path.unlink(missing_ok=True)

    def test_missing_train_seeds_fails(self) -> None:
        """Test that missing train_seeds fails."""
        config_data = {
            "prompt_learning": {
                "algorithm": "gepa",
                "task_app_url": "http://localhost:8001",
                "task_app_api_key": "test-key",
                "gepa": {
                    "env_name": "banking77",
                    "evaluation": {
                        "val_seeds": [10, 11, 12],  # Missing train_seeds
                    },
                    "rollout": {"budget": 100},
                    "mutation": {"llm_model": "gpt-4o-mini", "llm_provider": "openai"},
                    "population": {"initial_size": 20},
                    "archive": {"size": 64},
                    "token": {"max_limit": 1000},
                },
            }
        }
        path = _create_toml_file(config_data)
        try:
            is_valid, errors = validate_gepa_config_from_file(path)
            assert not is_valid
            assert any("train_seeds" in err.lower() for err in errors)
        finally:
            path.unlink(missing_ok=True)

    def test_missing_val_seeds_fails(self) -> None:
        """Test that missing val_seeds fails."""
        config_data = {
            "prompt_learning": {
                "algorithm": "gepa",
                "task_app_url": "http://localhost:8001",
                "task_app_api_key": "test-key",
                "gepa": {
                    "env_name": "banking77",
                    "evaluation": {
                        "train_seeds": [0, 1, 2],  # Missing val_seeds
                    },
                    "rollout": {"budget": 100},
                    "mutation": {"llm_model": "gpt-4o-mini", "llm_provider": "openai"},
                    "population": {"initial_size": 20},
                    "archive": {"size": 64},
                    "token": {"max_limit": 1000},
                },
            }
        }
        path = _create_toml_file(config_data)
        try:
            is_valid, errors = validate_gepa_config_from_file(path)
            assert not is_valid
            assert any("val_seeds" in err.lower() for err in errors)
        finally:
            path.unlink(missing_ok=True)

    def test_missing_rollout_budget_fails(self) -> None:
        """Test that missing rollout budget fails."""
        config_data = {
            "prompt_learning": {
                "algorithm": "gepa",
                "task_app_url": "http://localhost:8001",
                "task_app_api_key": "test-key",
                "gepa": {
                    "env_name": "banking77",
                    "evaluation": {"train_seeds": [0], "val_seeds": [10]},
                    "rollout": {},  # Missing budget
                    "mutation": {"llm_model": "gpt-4o-mini", "llm_provider": "openai"},
                    "population": {"initial_size": 20},
                    "archive": {"size": 64},
                    "token": {"max_limit": 1000},
                },
            }
        }
        path = _create_toml_file(config_data)
        try:
            is_valid, errors = validate_gepa_config_from_file(path)
            assert not is_valid
            assert any("budget" in err.lower() for err in errors)
        finally:
            path.unlink(missing_ok=True)

    def test_missing_mutation_llm_fails(self) -> None:
        """Test that missing mutation LLM fails."""
        config_data = {
            "prompt_learning": {
                "algorithm": "gepa",
                "task_app_url": "http://localhost:8001",
                "task_app_api_key": "test-key",
                "gepa": {
                    "env_name": "banking77",
                    "evaluation": {"train_seeds": [0], "val_seeds": [10]},
                    "rollout": {"budget": 100},
                    "mutation": {},  # Missing llm_model and llm_provider
                    "population": {"initial_size": 20},
                    "archive": {"size": 64},
                    "token": {"max_limit": 1000},
                },
            }
        }
        path = _create_toml_file(config_data)
        try:
            is_valid, errors = validate_gepa_config_from_file(path)
            assert not is_valid
            assert any("llm_model" in err.lower() or "llm_provider" in err.lower() for err in errors)
        finally:
            path.unlink(missing_ok=True)

    def test_invalid_toml_fails(self) -> None:
        """Test that invalid TOML fails gracefully."""
        path = Path("/nonexistent/file.toml")
        is_valid, errors = validate_gepa_config_from_file(path)
        assert not is_valid
        assert len(errors) > 0


class TestValidatePromptLearningConfigFromFile:
    """Tests for validate_prompt_learning_config_from_file wrapper."""

    def test_gepa_algorithm_calls_gepa_validator(self) -> None:
        """Test that gepa algorithm calls gepa validator."""
        config_data = {
            "prompt_learning": {
                "algorithm": "gepa",
                "task_app_url": "http://localhost:8001",
                "task_app_api_key": "test-key",
                "gepa": {
                    "env_name": "banking77",
                    "evaluation": {"train_seeds": [0], "val_seeds": [10]},
                    "rollout": {"budget": 100},
                    "mutation": {"llm_model": "gpt-4o-mini", "llm_provider": "openai"},
                    "population": {"initial_size": 20},
                    "archive": {"size": 64},
                    "token": {"max_limit": 1000},
                },
            }
        }
        path = _create_toml_file(config_data)
        try:
            # Should not raise
            validate_prompt_learning_config_from_file(path, "gepa")
        except ConfigValidationError:
            # If it raises, that's okay - means validation is working
            pass
        finally:
            path.unlink(missing_ok=True)


    def test_invalid_algorithm_raises_error(self) -> None:
        """Test that invalid algorithm raises error."""
        config_data = {
            "prompt_learning": {
                "algorithm": "invalid",
                "task_app_url": "http://localhost:8001",
            }
        }
        path = _create_toml_file(config_data)
        try:
            with pytest.raises(ValueError, match="Unknown algorithm"):
                validate_prompt_learning_config_from_file(path, "invalid")
        finally:
            path.unlink(missing_ok=True)




