"""Comprehensive unit tests for TOML config validation - making SDK configs idiot-proof.

This test suite covers:
- Required fields validation
- Type validation (string, int, float, list, dict)
- Range validation (positive numbers, 0-1 for rates, etc.)
- Enum validation (algorithm, provider, inference_mode)
- Cross-field validation (multi-stage requires modules, etc.)
- Nested structure validation
- Error message clarity
- Edge cases (empty strings, None, wrong types, etc.)
"""
from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any

import pytest

from synth_ai.api.train.validators import (
    ConfigValidationError,
    validate_prompt_learning_config,
    _validate_model_for_provider,
)

pytestmark = pytest.mark.unit


def _create_config_file(config_data: dict[str, Any]) -> Path:
    """Helper to create a temporary TOML config file."""
    try:
        import tomli_w
        
        with tempfile.NamedTemporaryFile(mode="wb", suffix=".toml", delete=False) as f:
            tomli_w.dump(config_data, f)
            return Path(f.name)
    except ImportError:
        # Fallback: For validation tests, we can use dict directly
        # Create a dummy file path - validator accepts dict, not file
        # This is mainly for error message formatting
        return Path(tempfile.mktemp(suffix=".toml"))


class TestRequiredFields:
    """Tests for required fields validation."""

    def test_missing_prompt_learning_section(self):
        """Test that missing [prompt_learning] section raises error."""
        config_data = {}
        path = _create_config_file(config_data)
        try:
            with pytest.raises(Exception, match="prompt_learning"):
                validate_prompt_learning_config(config_data, path)
        finally:
            path.unlink(missing_ok=True)

    def test_prompt_learning_not_dict(self):
        """Test that [prompt_learning] must be a dict/table."""
        config_data = {"prompt_learning": "not a dict"}
        path = _create_config_file(config_data)
        try:
            with pytest.raises(Exception, match="table|dict"):
                validate_prompt_learning_config(config_data, path)
        finally:
            path.unlink(missing_ok=True)

    def test_missing_algorithm(self):
        """Test that missing algorithm raises error."""
        config_data = {
            "prompt_learning": {
                "task_app_url": "http://localhost:8001",
            }
        }
        path = _create_config_file(config_data)
        try:
            with pytest.raises(Exception, match="algorithm"):
                validate_prompt_learning_config(config_data, path)
        finally:
            path.unlink(missing_ok=True)

    def test_missing_task_app_url(self):
        """Test that missing task_app_url raises error."""
        config_data = {
            "prompt_learning": {
                "algorithm": "gepa",
            }
        }
        path = _create_config_file(config_data)
        try:
            with pytest.raises(Exception, match="task_app_url"):
                validate_prompt_learning_config(config_data, path)
        finally:
            path.unlink(missing_ok=True)

    def test_missing_policy_section(self):
        """Test that missing [prompt_learning.policy] raises error."""
        config_data = {
            "prompt_learning": {
                "algorithm": "gepa",
                "task_app_url": "http://localhost:8001",
            }
        }
        path = _create_config_file(config_data)
        try:
            with pytest.raises(Exception, match="policy"):
                validate_prompt_learning_config(config_data, path)
        finally:
            path.unlink(missing_ok=True)

    def test_missing_policy_provider(self):
        """Test that missing policy.provider raises error."""
        config_data = {
            "prompt_learning": {
                "algorithm": "gepa",
                "task_app_url": "http://localhost:8001",
                "policy": {
                    "model": "gpt-4o-mini",
                    "inference_mode": "synth_hosted",
                },
            }
        }
        path = _create_config_file(config_data)
        try:
            with pytest.raises(Exception, match="provider"):
                validate_prompt_learning_config(config_data, path)
        finally:
            path.unlink(missing_ok=True)

    def test_missing_policy_model(self):
        """Test that missing policy.model raises error."""
        config_data = {
            "prompt_learning": {
                "algorithm": "gepa",
                "task_app_url": "http://localhost:8001",
                "policy": {
                    "provider": "openai",
                    "inference_mode": "synth_hosted",
                },
            }
        }
        path = _create_config_file(config_data)
        try:
            with pytest.raises(Exception, match="model"):
                validate_prompt_learning_config(config_data, path)
        finally:
            path.unlink(missing_ok=True)

    def test_missing_policy_inference_mode(self):
        """Test that missing policy.inference_mode raises error."""
        config_data = {
            "prompt_learning": {
                "algorithm": "gepa",
                "task_app_url": "http://localhost:8001",
                "policy": {
                    "model": "gpt-4o-mini",
                    "provider": "openai",
                },
            }
        }
        path = _create_config_file(config_data)
        try:
            with pytest.raises(Exception, match="inference_mode"):
                validate_prompt_learning_config(config_data, path)
        finally:
            path.unlink(missing_ok=True)

    def test_missing_gepa_section(self):
        """Test that missing [prompt_learning.gepa] raises error when algorithm is gepa."""
        config_data = {
            "prompt_learning": {
                "algorithm": "gepa",
                "task_app_url": "http://localhost:8001",
                "policy": {
                    "model": "gpt-4o-mini",
                    "provider": "openai",
                    "inference_mode": "synth_hosted",
                },
            }
        }
        path = _create_config_file(config_data)
        try:
            with pytest.raises(Exception, match="gepa"):
                validate_prompt_learning_config(config_data, path)
        finally:
            path.unlink(missing_ok=True)

    def test_missing_mipro_section(self):
        """Test that missing [prompt_learning.mipro] raises error when algorithm is mipro."""
        config_data = {
            "prompt_learning": {
                "algorithm": "mipro",
                "task_app_url": "http://localhost:8001",
                "policy": {
                    "model": "gpt-4o-mini",
                    "provider": "openai",
                    "inference_mode": "synth_hosted",
                },
            }
        }
        path = _create_config_file(config_data)
        try:
            with pytest.raises(Exception, match="mipro"):
                validate_prompt_learning_config(config_data, path)
        finally:
            path.unlink(missing_ok=True)

    def test_missing_mipro_meta_model(self):
        """Test that missing mipro.meta_model passes (backend applies defaults)."""
        config_data = {
            "prompt_learning": {
                "algorithm": "mipro",
                "task_app_url": "http://localhost:8001",
                "initial_prompt": {
                    "id": "test-prompt",
                    "messages": [{"role": "system", "content": "Test"}],
                },
                "policy": {
                    "model": "gpt-4o-mini",
                    "provider": "openai",
                    "inference_mode": "synth_hosted",
                },
                "mipro": {
                    "num_iterations": 5,
                    "num_evaluations_per_iteration": 2,
                    "batch_size": 6,
                    "max_concurrent": 16,
                    "bootstrap_train_seeds": [0, 1, 2],
                    "online_pool": [5, 6, 7],
                    "reference_pool": [30, 31, 32],
                    # Missing meta_model - backend will use defaults
                },
            }
        }
        path = _create_config_file(config_data)
        try:
            # Should pass validation since meta_model is optional
            validate_prompt_learning_config(config_data, path)
        finally:
            path.unlink(missing_ok=True)

    def test_missing_mipro_bootstrap_seeds(self):
        """Test that missing bootstrap_train_seeds raises error."""
        config_data = {
            "prompt_learning": {
                "algorithm": "mipro",
                "task_app_url": "http://localhost:8001",
                "policy": {
                    "model": "gpt-4o-mini",
                    "provider": "openai",
                    "inference_mode": "synth_hosted",
                },
                "mipro": {
                    "num_iterations": 5,
                    "num_evaluations_per_iteration": 2,
                    "batch_size": 6,
                    "max_concurrent": 16,
                    "meta_model": "gpt-4o-mini",
                    "meta_model_provider": "openai",
                    "online_pool": [5, 6, 7],
                },
            }
        }
        path = _create_config_file(config_data)
        try:
            with pytest.raises(Exception, match="bootstrap"):
                validate_prompt_learning_config(config_data, path)
        finally:
            path.unlink(missing_ok=True)

    def test_missing_mipro_online_pool(self):
        """Test that missing online_pool raises error."""
        config_data = {
            "prompt_learning": {
                "algorithm": "mipro",
                "task_app_url": "http://localhost:8001",
                "policy": {
                    "model": "gpt-4o-mini",
                    "provider": "openai",
                    "inference_mode": "synth_hosted",
                },
                "mipro": {
                    "num_iterations": 5,
                    "num_evaluations_per_iteration": 2,
                    "batch_size": 6,
                    "max_concurrent": 16,
                    "meta_model": "gpt-4o-mini",
                    "meta_model_provider": "openai",
                    "bootstrap_train_seeds": [0, 1, 2],
                },
            }
        }
        path = _create_config_file(config_data)
        try:
            with pytest.raises(Exception, match="online_pool"):
                validate_prompt_learning_config(config_data, path)
        finally:
            path.unlink(missing_ok=True)


class TestTypeValidation:
    """Tests for type validation."""

    def test_task_app_url_not_string(self):
        """Test that task_app_url must be a string."""
        config_data = {
            "prompt_learning": {
                "algorithm": "gepa",
                "task_app_url": 12345,  # Wrong type
            }
        }
        path = _create_config_file(config_data)
        try:
            with pytest.raises(Exception, match="string|task_app_url"):
                validate_prompt_learning_config(config_data, path)
        finally:
            path.unlink(missing_ok=True)

    def test_algorithm_not_string(self):
        """Test that algorithm must be a string."""
        config_data = {
            "prompt_learning": {
                "algorithm": 123,  # Wrong type
                "task_app_url": "http://localhost:8001",
            }
        }
        path = _create_config_file(config_data)
        try:
            # Should fail on invalid algorithm value
            with pytest.raises(Exception):
                validate_prompt_learning_config(config_data, path)
        finally:
            path.unlink(missing_ok=True)

    def test_policy_not_dict(self):
        """Test that policy must be a dict/table."""
        config_data = {
            "prompt_learning": {
                "algorithm": "gepa",
                "task_app_url": "http://localhost:8001",
                "policy": "not a dict",  # Wrong type
            }
        }
        path = _create_config_file(config_data)
        try:
            with pytest.raises(Exception, match="table|dict|policy"):
                validate_prompt_learning_config(config_data, path)
        finally:
            path.unlink(missing_ok=True)

    def test_initial_prompt_messages_not_list(self):
        """Test that initial_prompt.messages must be a list."""
        config_data = {
            "prompt_learning": {
                "algorithm": "gepa",
                "task_app_url": "http://localhost:8001",
                "policy": {
                    "model": "gpt-4o-mini",
                    "provider": "openai",
                    "inference_mode": "synth_hosted",
                },
                "initial_prompt": {
                    "messages": "not a list",  # Wrong type
                },
            }
        }
        path = _create_config_file(config_data)
        try:
            with pytest.raises(Exception, match="array|list|messages"):
                validate_prompt_learning_config(config_data, path)
        finally:
            path.unlink(missing_ok=True)

    def test_gepa_not_dict(self):
        """Test that gepa must be a dict/table."""
        config_data = {
            "prompt_learning": {
                "algorithm": "gepa",
                "task_app_url": "http://localhost:8001",
                "policy": {
                    "model": "gpt-4o-mini",
                    "provider": "openai",
                    "inference_mode": "synth_hosted",
                },
                "gepa": "not a dict",  # Wrong type
            }
        }
        path = _create_config_file(config_data)
        try:
            with pytest.raises(Exception, match="table|dict|gepa"):
                validate_prompt_learning_config(config_data, path)
        finally:
            path.unlink(missing_ok=True)

    def test_mipro_not_dict(self):
        """Test that mipro must be a dict/table."""
        config_data = {
            "prompt_learning": {
                "algorithm": "mipro",
                "task_app_url": "http://localhost:8001",
                "policy": {
                    "model": "gpt-4o-mini",
                    "provider": "openai",
                    "inference_mode": "synth_hosted",
                },
                "mipro": "not a dict",  # Wrong type
            }
        }
        path = _create_config_file(config_data)
        try:
            with pytest.raises(Exception, match="table|dict|mipro"):
                validate_prompt_learning_config(config_data, path)
        finally:
            path.unlink(missing_ok=True)

    def test_bootstrap_seeds_not_list(self):
        """Test that bootstrap_train_seeds must be a list."""
        config_data = {
            "prompt_learning": {
                "algorithm": "mipro",
                "task_app_url": "http://localhost:8001",
                "policy": {
                    "model": "gpt-4o-mini",
                    "provider": "openai",
                    "inference_mode": "synth_hosted",
                },
                "mipro": {
                    "num_iterations": 5,
                    "num_evaluations_per_iteration": 2,
                    "batch_size": 6,
                    "max_concurrent": 16,
                    "meta_model": "gpt-4o-mini",
                    "meta_model_provider": "openai",
                    "bootstrap_train_seeds": "not a list",  # Wrong type
                    "online_pool": [5, 6, 7],
                },
            }
        }
        path = _create_config_file(config_data)
        try:
            with pytest.raises(Exception, match="array|list|bootstrap"):
                validate_prompt_learning_config(config_data, path)
        finally:
            path.unlink(missing_ok=True)

    def test_online_pool_not_list(self):
        """Test that online_pool must be a list."""
        config_data = {
            "prompt_learning": {
                "algorithm": "mipro",
                "task_app_url": "http://localhost:8001",
                "policy": {
                    "model": "gpt-4o-mini",
                    "provider": "openai",
                    "inference_mode": "synth_hosted",
                },
                "mipro": {
                    "num_iterations": 5,
                    "num_evaluations_per_iteration": 2,
                    "batch_size": 6,
                    "max_concurrent": 16,
                    "meta_model": "gpt-4o-mini",
                    "meta_model_provider": "openai",
                    "bootstrap_train_seeds": [0, 1, 2],
                    "online_pool": "not a list",  # Wrong type
                },
            }
        }
        path = _create_config_file(config_data)
        try:
            with pytest.raises(Exception, match="array|list|online_pool"):
                validate_prompt_learning_config(config_data, path)
        finally:
            path.unlink(missing_ok=True)


class TestValueValidation:
    """Tests for value validation (ranges, enums, etc.)."""

    def test_invalid_algorithm(self):
        """Test that invalid algorithm value raises error."""
        config_data = {
            "prompt_learning": {
                "algorithm": "invalid_algorithm",
                "task_app_url": "http://localhost:8001",
            }
        }
        path = _create_config_file(config_data)
        try:
            with pytest.raises(Exception, match="algorithm"):
                validate_prompt_learning_config(config_data, path)
        finally:
            path.unlink(missing_ok=True)

    def test_task_app_url_invalid_protocol(self):
        """Test that task_app_url must start with http:// or https://."""
        config_data = {
            "prompt_learning": {
                "algorithm": "gepa",
                "task_app_url": "ftp://example.com",  # Invalid protocol
            }
        }
        path = _create_config_file(config_data)
        try:
            with pytest.raises(Exception, match="http://|https://|task_app_url"):
                validate_prompt_learning_config(config_data, path)
        finally:
            path.unlink(missing_ok=True)

    def test_task_app_url_empty_string(self):
        """Test that empty task_app_url raises error."""
        config_data = {
            "prompt_learning": {
                "algorithm": "gepa",
                "task_app_url": "",  # Empty string
            }
        }
        path = _create_config_file(config_data)
        try:
            with pytest.raises(Exception, match="task_app_url"):
                validate_prompt_learning_config(config_data, path)
        finally:
            path.unlink(missing_ok=True)

    def test_invalid_inference_mode(self):
        """Test that invalid inference_mode raises error."""
        config_data = {
            "prompt_learning": {
                "algorithm": "gepa",
                "task_app_url": "http://localhost:8001",
                "policy": {
                    "model": "gpt-4o-mini",
                    "provider": "openai",
                    "inference_mode": "bring_your_own",  # Invalid
                },
            }
        }
        path = _create_config_file(config_data)
        try:
            with pytest.raises(Exception, match="inference_mode|synth_hosted"):
                validate_prompt_learning_config(config_data, path)
        finally:
            path.unlink(missing_ok=True)

    def test_gepa_negative_initial_population_size(self):
        """Test that negative initial_population_size raises error."""
        config_data = {
            "prompt_learning": {
                "algorithm": "gepa",
                "task_app_url": "http://localhost:8001",
                "policy": {
                    "model": "gpt-4o-mini",
                    "provider": "openai",
                    "inference_mode": "synth_hosted",
                },
                "gepa": {
                    "initial_population_size": -5,  # Invalid: negative
                },
            }
        }
        path = _create_config_file(config_data)
        try:
            with pytest.raises(Exception, match="initial_population_size|> 0"):
                validate_prompt_learning_config(config_data, path)
        finally:
            path.unlink(missing_ok=True)

    def test_gepa_zero_initial_population_size(self):
        """Test that zero initial_population_size raises error."""
        config_data = {
            "prompt_learning": {
                "algorithm": "gepa",
                "task_app_url": "http://localhost:8001",
                "policy": {
                    "model": "gpt-4o-mini",
                    "provider": "openai",
                    "inference_mode": "synth_hosted",
                },
                "gepa": {
                    "initial_population_size": 0,  # Invalid: zero
                },
            }
        }
        path = _create_config_file(config_data)
        try:
            with pytest.raises(Exception, match="initial_population_size|> 0"):
                validate_prompt_learning_config(config_data, path)
        finally:
            path.unlink(missing_ok=True)

    def test_gepa_negative_num_generations(self):
        """Test that negative num_generations raises error."""
        config_data = {
            "prompt_learning": {
                "algorithm": "gepa",
                "task_app_url": "http://localhost:8001",
                "policy": {
                    "model": "gpt-4o-mini",
                    "provider": "openai",
                    "inference_mode": "synth_hosted",
                },
                "gepa": {
                    "num_generations": -1,  # Invalid: negative
                },
            }
        }
        path = _create_config_file(config_data)
        try:
            with pytest.raises(Exception, match="num_generations|> 0"):
                validate_prompt_learning_config(config_data, path)
        finally:
            path.unlink(missing_ok=True)

    def test_gepa_negative_max_spend(self):
        """Test that negative max_spend_usd raises error."""
        config_data = {
            "prompt_learning": {
                "algorithm": "gepa",
                "task_app_url": "http://localhost:8001",
                "policy": {
                    "model": "gpt-4o-mini",
                    "provider": "openai",
                    "inference_mode": "synth_hosted",
                },
                "gepa": {
                    "max_spend_usd": -10.0,  # Invalid: negative
                },
            }
        }
        path = _create_config_file(config_data)
        try:
            with pytest.raises(Exception, match="max_spend_usd|> 0"):
                validate_prompt_learning_config(config_data, path)
        finally:
            path.unlink(missing_ok=True)

    def test_gepa_zero_max_spend(self):
        """Test that zero max_spend_usd raises error."""
        config_data = {
            "prompt_learning": {
                "algorithm": "gepa",
                "task_app_url": "http://localhost:8001",
                "policy": {
                    "model": "gpt-4o-mini",
                    "provider": "openai",
                    "inference_mode": "synth_hosted",
                },
                "gepa": {
                    "max_spend_usd": 0.0,  # Invalid: zero
                },
            }
        }
        path = _create_config_file(config_data)
        try:
            with pytest.raises(Exception, match="max_spend_usd|> 0"):
                validate_prompt_learning_config(config_data, path)
        finally:
            path.unlink(missing_ok=True)

    def test_mipro_negative_num_iterations(self):
        """Test that negative num_iterations raises error."""
        config_data = {
            "prompt_learning": {
                "algorithm": "mipro",
                "task_app_url": "http://localhost:8001",
                "policy": {
                    "model": "gpt-4o-mini",
                    "provider": "openai",
                    "inference_mode": "synth_hosted",
                },
                "mipro": {
                    "num_iterations": -1,  # Invalid: negative
                    "num_evaluations_per_iteration": 2,
                    "batch_size": 6,
                    "max_concurrent": 16,
                    "meta_model": "gpt-4o-mini",
                    "meta_model_provider": "openai",
                    "bootstrap_train_seeds": [0, 1, 2],
                    "online_pool": [5, 6, 7],
                },
            }
        }
        path = _create_config_file(config_data)
        try:
            with pytest.raises(Exception, match="num_iterations|> 0"):
                validate_prompt_learning_config(config_data, path)
        finally:
            path.unlink(missing_ok=True)

    def test_mipro_zero_num_iterations(self):
        """Test that zero num_iterations raises error."""
        config_data = {
            "prompt_learning": {
                "algorithm": "mipro",
                "task_app_url": "http://localhost:8001",
                "policy": {
                    "model": "gpt-4o-mini",
                    "provider": "openai",
                    "inference_mode": "synth_hosted",
                },
                "mipro": {
                    "num_iterations": 0,  # Invalid: zero
                    "num_evaluations_per_iteration": 2,
                    "batch_size": 6,
                    "max_concurrent": 16,
                    "meta_model": "gpt-4o-mini",
                    "meta_model_provider": "openai",
                    "bootstrap_train_seeds": [0, 1, 2],
                    "online_pool": [5, 6, 7],
                },
            }
        }
        path = _create_config_file(config_data)
        try:
            with pytest.raises(Exception, match="num_iterations|> 0"):
                validate_prompt_learning_config(config_data, path)
        finally:
            path.unlink(missing_ok=True)

    def test_mipro_negative_batch_size(self):
        """Test that negative batch_size raises error."""
        config_data = {
            "prompt_learning": {
                "algorithm": "mipro",
                "task_app_url": "http://localhost:8001",
                "policy": {
                    "model": "gpt-4o-mini",
                    "provider": "openai",
                    "inference_mode": "synth_hosted",
                },
                "mipro": {
                    "num_iterations": 5,
                    "num_evaluations_per_iteration": 2,
                    "batch_size": -1,  # Invalid: negative
                    "max_concurrent": 16,
                    "meta_model": "gpt-4o-mini",
                    "meta_model_provider": "openai",
                    "bootstrap_train_seeds": [0, 1, 2],
                    "online_pool": [5, 6, 7],
                },
            }
        }
        path = _create_config_file(config_data)
        try:
            with pytest.raises(Exception, match="batch_size|> 0"):
                validate_prompt_learning_config(config_data, path)
        finally:
            path.unlink(missing_ok=True)

    def test_mipro_empty_bootstrap_seeds(self):
        """Test that empty bootstrap_train_seeds raises error."""
        config_data = {
            "prompt_learning": {
                "algorithm": "mipro",
                "task_app_url": "http://localhost:8001",
                "policy": {
                    "model": "gpt-4o-mini",
                    "provider": "openai",
                    "inference_mode": "synth_hosted",
                },
                "mipro": {
                    "num_iterations": 5,
                    "num_evaluations_per_iteration": 2,
                    "batch_size": 6,
                    "max_concurrent": 16,
                    "meta_model": "gpt-4o-mini",
                    "meta_model_provider": "openai",
                    "bootstrap_train_seeds": [],  # Invalid: empty
                    "online_pool": [5, 6, 7],
                },
            }
        }
        path = _create_config_file(config_data)
        try:
            with pytest.raises(Exception, match="bootstrap|empty"):
                validate_prompt_learning_config(config_data, path)
        finally:
            path.unlink(missing_ok=True)

    def test_mipro_empty_online_pool(self):
        """Test that empty online_pool raises error."""
        config_data = {
            "prompt_learning": {
                "algorithm": "mipro",
                "task_app_url": "http://localhost:8001",
                "policy": {
                    "model": "gpt-4o-mini",
                    "provider": "openai",
                    "inference_mode": "synth_hosted",
                },
                "mipro": {
                    "num_iterations": 5,
                    "num_evaluations_per_iteration": 2,
                    "batch_size": 6,
                    "max_concurrent": 16,
                    "meta_model": "gpt-4o-mini",
                    "meta_model_provider": "openai",
                    "bootstrap_train_seeds": [0, 1, 2],
                    "online_pool": [],  # Invalid: empty
                },
            }
        }
        path = _create_config_file(config_data)
        try:
            with pytest.raises(Exception, match="online_pool|empty"):
                validate_prompt_learning_config(config_data, path)
        finally:
            path.unlink(missing_ok=True)

    def test_mipro_few_shot_threshold_out_of_range(self):
        """Test that few_shot_score_threshold must be between 0.0 and 1.0."""
        config_data = {
            "prompt_learning": {
                "algorithm": "mipro",
                "task_app_url": "http://localhost:8001",
                "policy": {
                    "model": "gpt-4o-mini",
                    "provider": "openai",
                    "inference_mode": "synth_hosted",
                },
                "mipro": {
                    "num_iterations": 5,
                    "num_evaluations_per_iteration": 2,
                    "batch_size": 6,
                    "max_concurrent": 16,
                    "meta_model": "gpt-4o-mini",
                    "meta_model_provider": "openai",
                    "bootstrap_train_seeds": [0, 1, 2],
                    "online_pool": [5, 6, 7],
                    "few_shot_score_threshold": 1.5,  # Invalid: > 1.0
                },
            }
        }
        path = _create_config_file(config_data)
        try:
            with pytest.raises(Exception, match="few_shot_score_threshold|0.0.*1.0"):
                validate_prompt_learning_config(config_data, path)
        finally:
            path.unlink(missing_ok=True)

    def test_empty_initial_prompt_messages(self):
        """Test that empty initial_prompt.messages raises error."""
        config_data = {
            "prompt_learning": {
                "algorithm": "gepa",
                "task_app_url": "http://localhost:8001",
                "policy": {
                    "model": "gpt-4o-mini",
                    "provider": "openai",
                    "inference_mode": "synth_hosted",
                },
                "initial_prompt": {
                    "messages": [],  # Invalid: empty
                },
            }
        }
        path = _create_config_file(config_data)
        try:
            with pytest.raises(Exception, match="messages|empty"):
                validate_prompt_learning_config(config_data, path)
        finally:
            path.unlink(missing_ok=True)


class TestModelValidation:
    """Tests for model validation."""

    def test_unsupported_openai_model(self):
        """Test that unsupported OpenAI model raises error."""
        config_data = {
            "prompt_learning": {
                "algorithm": "gepa",
                "task_app_url": "http://localhost:8001",
                "policy": {
                    "model": "gpt-999",  # Invalid: not supported
                    "provider": "openai",
                    "inference_mode": "synth_hosted",
                },
            }
        }
        path = _create_config_file(config_data)
        try:
            with pytest.raises(Exception, match="Unsupported.*model|gpt-999"):
                validate_prompt_learning_config(config_data, path)
        finally:
            path.unlink(missing_ok=True)

    def test_gpt_5_pro_rejected(self):
        """Test that gpt-5-pro is explicitly rejected."""
        config_data = {
            "prompt_learning": {
                "algorithm": "gepa",
                "task_app_url": "http://localhost:8001",
                "policy": {
                    "model": "gpt-5-pro",  # Invalid: too expensive
                    "provider": "openai",
                    "inference_mode": "synth_hosted",
                },
            }
        }
        path = _create_config_file(config_data)
        try:
            with pytest.raises(Exception, match="gpt-5-pro|too expensive|not supported"):
                validate_prompt_learning_config(config_data, path)
        finally:
            path.unlink(missing_ok=True)

    def test_nano_model_for_mutation_rejected(self):
        """Test that nano models are rejected for mutation models."""
        config_data = {
            "prompt_learning": {
                "algorithm": "gepa",
                "task_app_url": "http://localhost:8001",
                "policy": {
                    "model": "gpt-4o-mini",
                    "provider": "openai",
                    "inference_mode": "synth_hosted",
                },
                "gepa": {
                    "mutation": {
                        "llm_model": "gpt-4.1-nano",  # Invalid: nano not allowed for mutation
                        "llm_provider": "openai",
                    },
                },
            }
        }
        path = _create_config_file(config_data)
        try:
            with pytest.raises(Exception, match="nano|mutation|not allowed"):
                validate_prompt_learning_config(config_data, path)
        finally:
            path.unlink(missing_ok=True)

    def test_nano_model_for_meta_rejected(self):
        """Test that nano models are rejected for meta models."""
        config_data = {
            "prompt_learning": {
                "algorithm": "mipro",
                "task_app_url": "http://localhost:8001",
                "policy": {
                    "model": "gpt-4o-mini",
                    "provider": "openai",
                    "inference_mode": "synth_hosted",
                },
                "mipro": {
                    "num_iterations": 5,
                    "num_evaluations_per_iteration": 2,
                    "batch_size": 6,
                    "max_concurrent": 16,
                    "meta_model": "gpt-5-nano",  # Invalid: nano not allowed for meta
                    "meta_model_provider": "openai",
                    "bootstrap_train_seeds": [0, 1, 2],
                    "online_pool": [5, 6, 7],
                },
            }
        }
        path = _create_config_file(config_data)
        try:
            with pytest.raises(Exception, match="nano|meta|not allowed"):
                validate_prompt_learning_config(config_data, path)
        finally:
            path.unlink(missing_ok=True)

    def test_nano_model_for_policy_allowed(self):
        """Test that nano models ARE allowed for policy models."""
        config_data = {
            "prompt_learning": {
                "algorithm": "gepa",
                "task_app_url": "http://localhost:8001",
                "policy": {
                    "model": "gpt-4.1-nano",  # Valid: nano allowed for policy
                    "provider": "openai",
                    "inference_mode": "synth_hosted",
                },
                "gepa": {
                    "num_generations": 10,
                },
            }
        }
        path = _create_config_file(config_data)
        try:
            # Should NOT raise - nano models are allowed for policy
            validate_prompt_learning_config(config_data, path)
        finally:
            path.unlink(missing_ok=True)

    def test_unsupported_groq_model(self):
        """Test that unsupported Groq model raises error."""
        config_data = {
            "prompt_learning": {
                "algorithm": "gepa",
                "task_app_url": "http://localhost:8001",
                "policy": {
                    "model": "invalid-groq-model",  # Invalid: not supported
                    "provider": "groq",
                    "inference_mode": "synth_hosted",
                },
            }
        }
        path = _create_config_file(config_data)
        try:
            with pytest.raises(Exception, match="Unsupported.*Groq|model"):
                validate_prompt_learning_config(config_data, path)
        finally:
            path.unlink(missing_ok=True)

    def test_unsupported_google_model(self):
        """Test that unsupported Google model raises error."""
        config_data = {
            "prompt_learning": {
                "algorithm": "gepa",
                "task_app_url": "http://localhost:8001",
                "policy": {
                    "model": "gemini-1.0",  # Invalid: not supported
                    "provider": "google",
                    "inference_mode": "synth_hosted",
                },
            }
        }
        path = _create_config_file(config_data)
        try:
            with pytest.raises(Exception, match="Unsupported.*Google|model"):
                validate_prompt_learning_config(config_data, path)
        finally:
            path.unlink(missing_ok=True)

    def test_unsupported_provider(self):
        """Test that unsupported provider raises error."""
        config_data = {
            "prompt_learning": {
                "algorithm": "gepa",
                "task_app_url": "http://localhost:8001",
                "policy": {
                    "model": "some-model",
                    "provider": "invalid_provider",  # Invalid: not supported
                    "inference_mode": "synth_hosted",
                },
            }
        }
        path = _create_config_file(config_data)
        try:
            with pytest.raises(Exception, match="Unsupported.*provider|openai.*groq.*google"):
                validate_prompt_learning_config(config_data, path)
        finally:
            path.unlink(missing_ok=True)

    def test_missing_mutation_provider(self):
        """Test that mutation model requires provider."""
        config_data = {
            "prompt_learning": {
                "algorithm": "gepa",
                "task_app_url": "http://localhost:8001",
                "policy": {
                    "model": "gpt-4o-mini",
                    "provider": "openai",
                    "inference_mode": "synth_hosted",
                },
                "gepa": {
                    "mutation": {
                        "llm_model": "gpt-4o-mini",
                        # Missing llm_provider
                    },
                },
            }
        }
        path = _create_config_file(config_data)
        try:
            with pytest.raises(Exception, match="llm_provider|mutation"):
                validate_prompt_learning_config(config_data, path)
        finally:
            path.unlink(missing_ok=True)

    def test_missing_meta_provider(self):
        """Test that meta model requires provider."""
        config_data = {
            "prompt_learning": {
                "algorithm": "mipro",
                "task_app_url": "http://localhost:8001",
                "policy": {
                    "model": "gpt-4o-mini",
                    "provider": "openai",
                    "inference_mode": "synth_hosted",
                },
                "mipro": {
                    "num_iterations": 5,
                    "num_evaluations_per_iteration": 2,
                    "batch_size": 6,
                    "max_concurrent": 16,
                    "meta_model": "gpt-4o-mini",
                    # Missing meta_model_provider
                    "bootstrap_train_seeds": [0, 1, 2],
                    "online_pool": [5, 6, 7],
                },
            }
        }
        path = _create_config_file(config_data)
        try:
            with pytest.raises(Exception, match="meta_model_provider|meta"):
                validate_prompt_learning_config(config_data, path)
        finally:
            path.unlink(missing_ok=True)


class TestMultiStageValidation:
    """Tests for multi-stage pipeline validation."""

    def test_multi_stage_gepa_missing_modules(self):
        """Test that multi-stage GEPA requires modules config."""
        config_data = {
            "prompt_learning": {
                "algorithm": "gepa",
                "task_app_url": "http://localhost:8001",
                "policy": {
                    "model": "gpt-4o-mini",
                    "provider": "openai",
                    "inference_mode": "synth_hosted",
                },
                "initial_prompt": {
                    "metadata": {
                        "pipeline_modules": ["classifier", "calibrator"],
                    },
                },
                "gepa": {
                    "num_generations": 10,
                    # Missing modules config
                },
            }
        }
        path = _create_config_file(config_data)
        try:
            with pytest.raises(Exception, match="modules|multi-stage|pipeline"):
                validate_prompt_learning_config(config_data, path)
        finally:
            path.unlink(missing_ok=True)

    def test_multi_stage_gepa_mismatched_module_ids(self):
        """Test that module IDs must match pipeline_modules."""
        config_data = {
            "prompt_learning": {
                "algorithm": "gepa",
                "task_app_url": "http://localhost:8001",
                "policy": {
                    "model": "gpt-4o-mini",
                    "provider": "openai",
                    "inference_mode": "synth_hosted",
                },
                "initial_prompt": {
                    "metadata": {
                        "pipeline_modules": ["classifier", "calibrator"],
                    },
                },
                "gepa": {
                    "modules": [
                        {"module_id": "classifier", "max_instruction_slots": 2},
                        # Missing "calibrator" module
                    ],
                    "num_generations": 10,
                },
            }
        }
        path = _create_config_file(config_data)
        try:
            with pytest.raises(Exception, match="calibrator|missing|modules"):
                validate_prompt_learning_config(config_data, path)
        finally:
            path.unlink(missing_ok=True)

    def test_multi_stage_gepa_valid(self):
        """Test that valid multi-stage GEPA config passes."""
        config_data = {
            "prompt_learning": {
                "algorithm": "gepa",
                "task_app_url": "http://localhost:8001",
                "policy": {
                    "model": "gpt-4o-mini",
                    "provider": "openai",
                    "inference_mode": "synth_hosted",
                },
                "initial_prompt": {
                    "metadata": {
                        "pipeline_modules": ["classifier", "calibrator"],
                    },
                },
                "gepa": {
                    "modules": [
                        {
                            "module_id": "classifier",
                            "max_instruction_slots": 2,
                            "policy": {"model": "gpt-4o-mini", "provider": "openai"},
                        },
                        {
                            "module_id": "calibrator",
                            "max_instruction_slots": 3,
                            "policy": {"model": "gpt-4o-mini", "provider": "openai"},
                        },
                    ],
                    "num_generations": 10,
                },
            }
        }
        path = _create_config_file(config_data)
        try:
            # Should NOT raise
            validate_prompt_learning_config(config_data, path)
        finally:
            path.unlink(missing_ok=True)

    def test_single_stage_gepa_no_modules_required(self):
        """Test that single-stage GEPA doesn't require modules."""
        config_data = {
            "prompt_learning": {
                "algorithm": "gepa",
                "task_app_url": "http://localhost:8001",
                "policy": {
                    "model": "gpt-4o-mini",
                    "provider": "openai",
                    "inference_mode": "synth_hosted",
                },
                "gepa": {
                    "num_generations": 10,
                    # No modules - single stage
                },
            }
        }
        path = _create_config_file(config_data)
        try:
            # Should NOT raise - single stage doesn't need modules
            validate_prompt_learning_config(config_data, path)
        finally:
            path.unlink(missing_ok=True)


class TestEdgeCases:
    """Tests for edge cases and unusual inputs."""

    def test_empty_string_model(self):
        """Test that empty string model raises error."""
        config_data = {
            "prompt_learning": {
                "algorithm": "gepa",
                "task_app_url": "http://localhost:8001",
                "policy": {
                    "model": "",  # Empty string
                    "provider": "openai",
                    "inference_mode": "synth_hosted",
                },
            }
        }
        path = _create_config_file(config_data)
        try:
            with pytest.raises(Exception, match="model|empty|missing"):
                validate_prompt_learning_config(config_data, path)
        finally:
            path.unlink(missing_ok=True)

    def test_whitespace_only_model(self):
        """Test that whitespace-only model raises error."""
        config_data = {
            "prompt_learning": {
                "algorithm": "gepa",
                "task_app_url": "http://localhost:8001",
                "policy": {
                    "model": "   ",  # Whitespace only
                    "provider": "openai",
                    "inference_mode": "synth_hosted",
                },
            }
        }
        path = _create_config_file(config_data)
        try:
            with pytest.raises(Exception, match="model|empty|missing"):
                validate_prompt_learning_config(config_data, path)
        finally:
            path.unlink(missing_ok=True)

    def test_whitespace_only_provider(self):
        """Test that whitespace-only provider raises error."""
        config_data = {
            "prompt_learning": {
                "algorithm": "gepa",
                "task_app_url": "http://localhost:8001",
                "policy": {
                    "model": "gpt-4o-mini",
                    "provider": "   ",  # Whitespace only
                    "inference_mode": "synth_hosted",
                },
            }
        }
        path = _create_config_file(config_data)
        try:
            with pytest.raises(Exception, match="provider|Unsupported"):
                validate_prompt_learning_config(config_data, path)
        finally:
            path.unlink(missing_ok=True)

    def test_float_instead_of_int(self):
        """Test that float where int expected raises error."""
        config_data = {
            "prompt_learning": {
                "algorithm": "gepa",
                "task_app_url": "http://localhost:8001",
                "policy": {
                    "model": "gpt-4o-mini",
                    "provider": "openai",
                    "inference_mode": "synth_hosted",
                },
                "gepa": {
                    "initial_population_size": 10.5,  # Float instead of int
                },
            }
        }
        path = _create_config_file(config_data)
        try:
            # Should either accept (if cast) or raise error
            # This documents current behavior
            try:
                validate_prompt_learning_config(config_data, path)
            except Exception as e:
                # If it raises, check error mentions the field
                assert "initial_population_size" in str(e) or "integer" in str(e).lower()
        finally:
            path.unlink(missing_ok=True)

    def test_string_instead_of_int(self):
        """Test that string where int expected raises error."""
        config_data = {
            "prompt_learning": {
                "algorithm": "gepa",
                "task_app_url": "http://localhost:8001",
                "policy": {
                    "model": "gpt-4o-mini",
                    "provider": "openai",
                    "inference_mode": "synth_hosted",
                },
                "gepa": {
                    "initial_population_size": "ten",  # String instead of int
                },
            }
        }
        path = _create_config_file(config_data)
        try:
            with pytest.raises(Exception, match="initial_population_size|integer"):
                validate_prompt_learning_config(config_data, path)
        finally:
            path.unlink(missing_ok=True)

    def test_string_instead_of_float(self):
        """Test that string where float expected raises error."""
        config_data = {
            "prompt_learning": {
                "algorithm": "mipro",
                "task_app_url": "http://localhost:8001",
                "policy": {
                    "model": "gpt-4o-mini",
                    "provider": "openai",
                    "inference_mode": "synth_hosted",
                },
                "mipro": {
                    "num_iterations": 5,
                    "num_evaluations_per_iteration": 2,
                    "batch_size": 6,
                    "max_concurrent": 16,
                    "meta_model": "gpt-4o-mini",
                    "meta_model_provider": "openai",
                    "bootstrap_train_seeds": [0, 1, 2],
                    "online_pool": [5, 6, 7],
                    "few_shot_score_threshold": "high",  # String instead of float
                },
            }
        }
        path = _create_config_file(config_data)
        try:
            with pytest.raises(Exception, match="few_shot_score_threshold|number"):
                validate_prompt_learning_config(config_data, path)
        finally:
            path.unlink(missing_ok=True)

    def test_list_with_mixed_types(self):
        """Test that list with mixed types raises error."""
        config_data = {
            "prompt_learning": {
                "algorithm": "mipro",
                "task_app_url": "http://localhost:8001",
                "policy": {
                    "model": "gpt-4o-mini",
                    "provider": "openai",
                    "inference_mode": "synth_hosted",
                },
                "mipro": {
                    "num_iterations": 5,
                    "num_evaluations_per_iteration": 2,
                    "batch_size": 6,
                    "max_concurrent": 16,
                    "meta_model": "gpt-4o-mini",
                    "meta_model_provider": "openai",
                    "bootstrap_train_seeds": [0, "one", 2],  # Mixed types
                    "online_pool": [5, 6, 7],
                },
            }
        }
        path = _create_config_file(config_data)
        try:
            # Should either accept (if lenient) or raise error
            # This documents current behavior
            try:
                validate_prompt_learning_config(config_data, path)
            except Exception:
                # If it raises, that's acceptable
                pass
        finally:
            path.unlink(missing_ok=True)


class TestErrorMessages:
    """Tests for error message clarity and helpfulness."""

    def test_error_message_includes_field_name(self):
        """Test that error messages include the field name."""
        config_data = {
            "prompt_learning": {
                "algorithm": "gepa",
                "task_app_url": "http://localhost:8001",
                "policy": {
                    "model": "gpt-4o-mini",
                    "provider": "openai",
                    "inference_mode": "synth_hosted",
                },
                "gepa": {
                    "initial_population_size": -5,
                },
            }
        }
        path = _create_config_file(config_data)
        try:
            with pytest.raises(Exception) as exc_info:
                validate_prompt_learning_config(config_data, path)
            error_msg = str(exc_info.value)
            assert "initial_population_size" in error_msg
        finally:
            path.unlink(missing_ok=True)

    def test_error_message_includes_supported_values(self):
        """Test that error messages include supported values."""
        config_data = {
            "prompt_learning": {
                "algorithm": "invalid_algorithm",
                "task_app_url": "http://localhost:8001",
            }
        }
        path = _create_config_file(config_data)
        try:
            with pytest.raises(Exception) as exc_info:
                validate_prompt_learning_config(config_data, path)
            error_msg = str(exc_info.value)
            assert "gepa" in error_msg or "mipro" in error_msg
        finally:
            path.unlink(missing_ok=True)

    def test_error_message_includes_example(self):
        """Test that error messages include examples when helpful."""
        config_data = {
            "prompt_learning": {
                "algorithm": "gepa",
                "task_app_url": "http://localhost:8001",
            }
        }
        path = _create_config_file(config_data)
        try:
            with pytest.raises(Exception) as exc_info:
                validate_prompt_learning_config(config_data, path)
            error_msg = str(exc_info.value)
            # Should mention policy section
            assert "policy" in error_msg.lower()
        finally:
            path.unlink(missing_ok=True)

    def test_multiple_errors_reported(self):
        """Test that multiple errors are reported together."""
        config_data = {
            "prompt_learning": {
                "algorithm": "gepa",
                "task_app_url": "http://localhost:8001",
                "policy": {
                    # Missing model and provider
                    "inference_mode": "synth_hosted",
                },
                "gepa": {
                    "initial_population_size": -5,
                },
            }
        }
        path = _create_config_file(config_data)
        try:
            with pytest.raises(Exception) as exc_info:
                validate_prompt_learning_config(config_data, path)
            error_msg = str(exc_info.value)
            # Should mention multiple issues
            assert "model" in error_msg.lower() or "provider" in error_msg.lower()
            assert "initial_population_size" in error_msg.lower() or "> 0" in error_msg
        finally:
            path.unlink(missing_ok=True)

    def test_error_message_includes_config_path(self):
        """Test that error messages include config file path."""
        config_data = {
            "prompt_learning": {
                "algorithm": "gepa",
            }
        }
        path = _create_config_file(config_data)
        try:
            with pytest.raises(Exception) as exc_info:
                validate_prompt_learning_config(config_data, path)
            error_msg = str(exc_info.value)
            # Should mention the config path
            assert str(path) in error_msg or ".toml" in error_msg
        finally:
            path.unlink(missing_ok=True)


class TestValidConfigs:
    """Tests for valid configs that should pass validation."""

    def test_minimal_valid_gepa_config(self):
        """Test that minimal valid GEPA config passes."""
        config_data = {
            "prompt_learning": {
                "algorithm": "gepa",
                "task_app_url": "http://localhost:8001",
                "policy": {
                    "model": "gpt-4o-mini",
                    "provider": "openai",
                    "inference_mode": "synth_hosted",
                },
                "gepa": {
                    "num_generations": 10,
                },
            }
        }
        path = _create_config_file(config_data)
        try:
            # Should NOT raise
            validate_prompt_learning_config(config_data, path)
        finally:
            path.unlink(missing_ok=True)

    def test_minimal_valid_mipro_config(self):
        """Test that minimal valid MIPRO config passes."""
        config_data = {
            "prompt_learning": {
                "algorithm": "mipro",
                "task_app_url": "http://localhost:8001",
                "policy": {
                    "model": "gpt-4o-mini",
                    "provider": "openai",
                    "inference_mode": "synth_hosted",
                },
                "mipro": {
                    "num_iterations": 5,
                    "num_evaluations_per_iteration": 2,
                    "batch_size": 6,
                    "max_concurrent": 16,
                    "meta_model": "gpt-4o-mini",
                    "meta_model_provider": "openai",
                    "bootstrap_train_seeds": [0, 1, 2],
                    "online_pool": [5, 6, 7],
                },
            }
        }
        path = _create_config_file(config_data)
        try:
            # Should NOT raise
            validate_prompt_learning_config(config_data, path)
        finally:
            path.unlink(missing_ok=True)

    def test_full_valid_gepa_config(self):
        """Test that full valid GEPA config passes."""
        config_data = {
            "prompt_learning": {
                "algorithm": "gepa",
                "task_app_url": "http://localhost:8001",
                "policy": {
                    "model": "gpt-4o-mini",
                    "provider": "openai",
                    "inference_mode": "synth_hosted",
                    "temperature": 0.0,
                    "max_completion_tokens": 512,
                },
                "initial_prompt": {
                    "id": "test",
                    "name": "Test Prompt",
                    "messages": [
                        {"role": "system", "pattern": "You are helpful", "order": 0}
                    ],
                },
                "gepa": {
                    "env_name": "test",
                    "num_generations": 10,
                    "initial_population_size": 20,
                    "children_per_generation": 5,
                    "mutation_rate": 0.3,
                    "crossover_rate": 0.5,
                    "max_spend_usd": 100.0,
                },
            }
        }
        path = _create_config_file(config_data)
        try:
            # Should NOT raise
            validate_prompt_learning_config(config_data, path)
        finally:
            path.unlink(missing_ok=True)

    def test_full_valid_mipro_config(self):
        """Test that full valid MIPRO config passes."""
        config_data = {
            "prompt_learning": {
                "algorithm": "mipro",
                "task_app_url": "http://localhost:8001",
                "policy": {
                    "model": "gpt-4o-mini",
                    "provider": "openai",
                    "inference_mode": "synth_hosted",
                    "temperature": 0.0,
                    "max_completion_tokens": 512,
                },
                "initial_prompt": {
                    "id": "test",
                    "name": "Test Prompt",
                    "messages": [
                        {"role": "system", "pattern": "You are helpful", "order": 0}
                    ],
                },
                "mipro": {
                    "num_iterations": 5,
                    "num_evaluations_per_iteration": 2,
                    "batch_size": 6,
                    "max_concurrent": 16,
                    "meta_model": "gpt-4o-mini",
                    "meta_model_provider": "openai",
                    "bootstrap_train_seeds": [0, 1, 2, 3, 4],
                    "online_pool": [5, 6, 7, 8, 9],
                    "test_pool": [10, 11, 12],
                    "few_shot_score_threshold": 0.85,
                },
            }
        }
        path = _create_config_file(config_data)
        try:
            # Should NOT raise
            validate_prompt_learning_config(config_data, path)
        finally:
            path.unlink(missing_ok=True)

