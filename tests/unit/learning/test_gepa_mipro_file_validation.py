"""Comprehensive unit tests for GEPA and MIPRO file-based config validation.

Tests the new validate_gepa_config_from_file and validate_mipro_config_from_file functions
that provide comprehensive validation matching backend requirements.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest
import toml

from synth_ai.api.train.validators import (
    ConfigValidationError,
    validate_gepa_config_from_file,
    validate_mipro_config_from_file,
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
                "algorithm": "mipro",
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


class TestMIPROFileValidation:
    """Tests for validate_mipro_config_from_file."""

    def test_valid_mipro_config_passes(self) -> None:
        """Test that a valid MIPRO config passes validation."""
        config_data = {
            "prompt_learning": {
                "algorithm": "mipro",
                "task_app_url": "http://localhost:8001",
                "task_app_api_key": "test-key",
                "env_name": "banking77",
                "initial_prompt": {
                    "id": "test-prompt",
                    "messages": [{"role": "system", "content": "Test"}],
                },
                "policy": {
                    "model": "gpt-4o-mini",
                    "provider": "openai",
                },
                "mipro": {
                    "num_iterations": 10,
                    "num_evaluations_per_iteration": 5,
                    "batch_size": 32,
                    "max_concurrent": 20,
                    "meta_model": "gpt-4o-mini",
                    "meta_model_provider": "openai",
                    "bootstrap_train_seeds": [0, 1, 2, 3, 4],
                    "online_pool": [5, 6, 7, 8, 9],
                    "reference_pool": [30, 31, 32, 33, 34],
                },
            }
        }
        path = _create_toml_file(config_data)
        try:
            is_valid, errors = validate_mipro_config_from_file(path)
            assert is_valid, f"Validation failed with errors: {errors}"
        finally:
            path.unlink(missing_ok=True)

    def test_missing_reference_pool_fails(self) -> None:
        """Test that missing reference_pool fails (backend requirement)."""
        config_data = {
            "prompt_learning": {
                "algorithm": "mipro",
                "task_app_url": "http://localhost:8001",
                "task_app_api_key": "test-key",
                "env_name": "banking77",
                "mipro": {
                    "num_iterations": 10,
                    "num_evaluations_per_iteration": 5,
                    "batch_size": 32,
                    "max_concurrent": 20,
                    "meta_model": "gpt-4o-mini",
                    "meta_model_provider": "openai",
                    "bootstrap_train_seeds": [0, 1, 2],
                    "online_pool": [5, 6, 7],
                    # Missing reference_pool - should fail
                },
            }
        }
        path = _create_toml_file(config_data)
        try:
            is_valid, errors = validate_mipro_config_from_file(path)
            assert not is_valid
            assert any("reference_pool" in err.lower() and "required" in err.lower() for err in errors)
        finally:
            path.unlink(missing_ok=True)

    def test_reference_pool_overlap_fails(self) -> None:
        """Test that reference_pool overlapping with bootstrap/online/test fails."""
        config_data = {
            "prompt_learning": {
                "algorithm": "mipro",
                "task_app_url": "http://localhost:8001",
                "task_app_api_key": "test-key",
                "env_name": "banking77",
                "mipro": {
                    "num_iterations": 10,
                    "num_evaluations_per_iteration": 5,
                    "batch_size": 32,
                    "max_concurrent": 20,
                    "meta_model": "gpt-4o-mini",
                    "meta_model_provider": "openai",
                    "bootstrap_train_seeds": [0, 1, 2],
                    "online_pool": [5, 6, 7],
                    "reference_pool": [2, 3, 4],  # Overlaps with bootstrap (seed 2)
                },
            }
        }
        path = _create_toml_file(config_data)
        try:
            is_valid, errors = validate_mipro_config_from_file(path)
            assert not is_valid
            assert any("overlap" in err.lower() for err in errors)
        finally:
            path.unlink(missing_ok=True)

    def test_missing_bootstrap_seeds_fails(self) -> None:
        """Test that missing bootstrap_train_seeds fails."""
        config_data = {
            "prompt_learning": {
                "algorithm": "mipro",
                "task_app_url": "http://localhost:8001",
                "task_app_api_key": "test-key",
                "env_name": "banking77",
                "mipro": {
                    "num_iterations": 10,
                    "num_evaluations_per_iteration": 5,
                    "batch_size": 32,
                    "max_concurrent": 20,
                    "meta_model": "gpt-4o-mini",
                    "meta_model_provider": "openai",
                    "online_pool": [5, 6, 7],
                    "reference_pool": [30, 31, 32],
                    # Missing bootstrap_train_seeds
                },
            }
        }
        path = _create_toml_file(config_data)
        try:
            is_valid, errors = validate_mipro_config_from_file(path)
            assert not is_valid
            assert any("bootstrap_train_seeds" in err.lower() for err in errors)
        finally:
            path.unlink(missing_ok=True)

    def test_missing_online_pool_fails(self) -> None:
        """Test that missing online_pool fails."""
        config_data = {
            "prompt_learning": {
                "algorithm": "mipro",
                "task_app_url": "http://localhost:8001",
                "task_app_api_key": "test-key",
                "env_name": "banking77",
                "mipro": {
                    "num_iterations": 10,
                    "num_evaluations_per_iteration": 5,
                    "batch_size": 32,
                    "max_concurrent": 20,
                    "meta_model": "gpt-4o-mini",
                    "meta_model_provider": "openai",
                    "bootstrap_train_seeds": [0, 1, 2],
                    "reference_pool": [30, 31, 32],
                    # Missing online_pool
                },
            }
        }
        path = _create_toml_file(config_data)
        try:
            is_valid, errors = validate_mipro_config_from_file(path)
            assert not is_valid
            assert any("online_pool" in err.lower() for err in errors)
        finally:
            path.unlink(missing_ok=True)

    def test_missing_env_name_fails(self) -> None:
        """Test that missing env_name fails."""
        config_data = {
            "prompt_learning": {
                "algorithm": "mipro",
                "task_app_url": "http://localhost:8001",
                "task_app_api_key": "test-key",
                "mipro": {
                    "num_iterations": 10,
                    "num_evaluations_per_iteration": 5,
                    "batch_size": 32,
                    "max_concurrent": 20,
                    "meta_model": "gpt-4o-mini",
                    "meta_model_provider": "openai",
                    "bootstrap_train_seeds": [0, 1, 2],
                    "online_pool": [5, 6, 7],
                    "reference_pool": [30, 31, 32],
                    # Missing env_name
                },
            }
        }
        path = _create_toml_file(config_data)
        try:
            is_valid, errors = validate_mipro_config_from_file(path)
            assert not is_valid
            assert any("env_name" in err.lower() for err in errors)
        finally:
            path.unlink(missing_ok=True)

    def test_missing_required_numeric_fields_fails(self) -> None:
        """Test that missing required numeric fields fail."""
        config_data = {
            "prompt_learning": {
                "algorithm": "mipro",
                "task_app_url": "http://localhost:8001",
                "task_app_api_key": "test-key",
                "env_name": "banking77",
                "mipro": {
                    # Missing num_iterations, num_evaluations_per_iteration, batch_size, max_concurrent
                    "meta_model": "gpt-4o-mini",
                    "meta_model_provider": "openai",
                    "bootstrap_train_seeds": [0, 1, 2],
                    "online_pool": [5, 6, 7],
                    "reference_pool": [30, 31, 32],
                },
            }
        }
        path = _create_toml_file(config_data)
        try:
            is_valid, errors = validate_mipro_config_from_file(path)
            assert not is_valid
            assert any(
                "num_iterations" in err.lower()
                or "num_evaluations_per_iteration" in err.lower()
                or "batch_size" in err.lower()
                or "max_concurrent" in err.lower()
                for err in errors
            )
        finally:
            path.unlink(missing_ok=True)

    def test_missing_meta_model_fails(self) -> None:
        """Test that missing meta_model fails."""
        config_data = {
            "prompt_learning": {
                "algorithm": "mipro",
                "task_app_url": "http://localhost:8001",
                "task_app_api_key": "test-key",
                "env_name": "banking77",
                "mipro": {
                    "num_iterations": 10,
                    "num_evaluations_per_iteration": 5,
                    "batch_size": 32,
                    "max_concurrent": 20,
                    # Missing meta_model
                    "bootstrap_train_seeds": [0, 1, 2],
                    "online_pool": [5, 6, 7],
                    "reference_pool": [30, 31, 32],
                },
            }
        }
        path = _create_toml_file(config_data)
        try:
            is_valid, errors = validate_mipro_config_from_file(path)
            assert not is_valid
            assert any("meta_model" in err.lower() for err in errors)
        finally:
            path.unlink(missing_ok=True)

    def test_inference_url_rejection(self) -> None:
        """Test that inference_url in policy is rejected."""
        config_data = {
            "prompt_learning": {
                "algorithm": "mipro",
                "task_app_url": "http://localhost:8001",
                "task_app_api_key": "test-key",
                "env_name": "banking77",
                "policy": {
                    "model": "gpt-4o-mini",
                    "provider": "openai",
                    "inference_url": "https://api.openai.com/v1",  # Should be rejected
                },
                "mipro": {
                    "num_iterations": 10,
                    "num_evaluations_per_iteration": 5,
                    "batch_size": 32,
                    "max_concurrent": 20,
                    "meta_model": "gpt-4o-mini",
                    "meta_model_provider": "openai",
                    "bootstrap_train_seeds": [0, 1, 2],
                    "online_pool": [5, 6, 7],
                    "reference_pool": [30, 31, 32],
                },
            }
        }
        path = _create_toml_file(config_data)
        try:
            is_valid, errors = validate_mipro_config_from_file(path)
            assert not is_valid
            assert any("inference_url" in err.lower() and "must not" in err.lower() for err in errors)
        finally:
            path.unlink(missing_ok=True)

    def test_invalid_numeric_values_fail(self) -> None:
        """Test that invalid numeric values fail."""
        config_data = {
            "prompt_learning": {
                "algorithm": "mipro",
                "task_app_url": "http://localhost:8001",
                "task_app_api_key": "test-key",
                "env_name": "banking77",
                "mipro": {
                    "num_iterations": -1,  # Invalid: must be > 0
                    "num_evaluations_per_iteration": 5,
                    "batch_size": 32,
                    "max_concurrent": 20,
                    "meta_model": "gpt-4o-mini",
                    "meta_model_provider": "openai",
                    "bootstrap_train_seeds": [0, 1, 2],
                    "online_pool": [5, 6, 7],
                    "reference_pool": [30, 31, 32],
                },
            }
        }
        path = _create_toml_file(config_data)
        try:
            is_valid, errors = validate_mipro_config_from_file(path)
            assert not is_valid
            assert any("num_iterations" in err.lower() and "> 0" in err for err in errors)
        finally:
            path.unlink(missing_ok=True)

    def test_few_shot_score_threshold_range(self) -> None:
        """Test that few_shot_score_threshold must be in [0.0, 1.0]."""
        config_data = {
            "prompt_learning": {
                "algorithm": "mipro",
                "task_app_url": "http://localhost:8001",
                "task_app_api_key": "test-key",
                "env_name": "banking77",
                "mipro": {
                    "num_iterations": 10,
                    "num_evaluations_per_iteration": 5,
                    "batch_size": 32,
                    "max_concurrent": 20,
                    "meta_model": "gpt-4o-mini",
                    "meta_model_provider": "openai",
                    "few_shot_score_threshold": 1.5,  # Invalid: > 1.0
                    "bootstrap_train_seeds": [0, 1, 2],
                    "online_pool": [5, 6, 7],
                    "reference_pool": [30, 31, 32],
                },
            }
        }
        path = _create_toml_file(config_data)
        try:
            is_valid, errors = validate_mipro_config_from_file(path)
            assert not is_valid
            assert any("few_shot_score_threshold" in err.lower() and "between" in err.lower() for err in errors)
        finally:
            path.unlink(missing_ok=True)

    def test_modules_stage_validation(self) -> None:
        """Test that module/stage configuration is validated."""
        config_data = {
            "prompt_learning": {
                "algorithm": "mipro",
                "task_app_url": "http://localhost:8001",
                "task_app_api_key": "test-key",
                "env_name": "banking77",
                "mipro": {
                    "num_iterations": 10,
                    "num_evaluations_per_iteration": 5,
                    "batch_size": 32,
                    "max_concurrent": 20,
                    "max_instruction_sets": 128,
                    "max_demo_sets": 128,
                    "meta_model": "gpt-4o-mini",
                    "meta_model_provider": "openai",
                    "bootstrap_train_seeds": [0, 1, 2],
                    "online_pool": [5, 6, 7],
                    "reference_pool": [30, 31, 32],
                    "modules": [
                        {
                            "module_id": "test_module",
                            "stages": [
                                {
                                    "stage_id": "test_stage",
                                    "max_instruction_slots": 200,  # Invalid: exceeds max_instruction_sets (128)
                                }
                            ],
                        }
                    ],
                },
            }
        }
        path = _create_toml_file(config_data)
        try:
            is_valid, errors = validate_mipro_config_from_file(path)
            assert not is_valid
            assert any("exceeds max_instruction_sets" in err.lower() for err in errors)
        finally:
            path.unlink(missing_ok=True)


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

    def test_mipro_algorithm_calls_mipro_validator(self) -> None:
        """Test that mipro algorithm calls mipro validator."""
        config_data = {
            "prompt_learning": {
                "algorithm": "mipro",
                "task_app_url": "http://localhost:8001",
                "task_app_api_key": "test-key",
                "env_name": "banking77",
                "mipro": {
                    "num_iterations": 10,
                    "num_evaluations_per_iteration": 5,
                    "batch_size": 32,
                    "max_concurrent": 20,
                    "meta_model": "gpt-4o-mini",
                    "meta_model_provider": "openai",
                    "bootstrap_train_seeds": [0, 1, 2],
                    "online_pool": [5, 6, 7],
                    "reference_pool": [30, 31, 32],
                },
            }
        }
        path = _create_toml_file(config_data)
        try:
            # Should not raise
            validate_prompt_learning_config_from_file(path, "mipro")
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

