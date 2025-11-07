"""Unit tests for MIPRO spec mode config validation in SDK."""
import tempfile
from pathlib import Path

import pytest

from synth_ai.api.train.validators import validate_prompt_learning_config


class TestMIPROSpecConfigValidation:
    """Test MIPRO spec mode config validation in SDK."""

    def test_accept_spec_path_when_provided(self) -> None:
        """Test that spec_path is accepted when provided."""
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
                    "batch_size": 5,
                    "max_concurrent": 10,
                    "meta_model": "gpt-4o-mini",
                    "meta_model_provider": "openai",
                    "spec_path": "examples/task_apps/banking77_pipeline/banking77_pipeline_spec.json",
                    "spec_max_tokens": 5000,
                    "spec_include_examples": True,
                    "spec_priority_threshold": 8,
                    "bootstrap_train_seeds": [0, 1, 2],
                    "online_pool": [3, 4, 5],
                    "test_pool": [6, 7, 8],
                },
            }
        }
        with tempfile.NamedTemporaryFile(mode="w", suffix=".toml", delete=False) as f:
            path = Path(f.name)
        try:
            # Should not raise
            validate_prompt_learning_config(config_data, path)
        finally:
            path.unlink(missing_ok=True)
    
    def test_accept_config_without_spec_path(self) -> None:
        """Test that config without spec_path is accepted (spec is optional)."""
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
                    "batch_size": 5,
                    "max_concurrent": 10,
                    "meta_model": "gpt-4o-mini",
                    "meta_model_provider": "openai",
                    # spec_path not provided - spec is optional
                    "bootstrap_train_seeds": [0, 1, 2],
                    "online_pool": [3, 4, 5],
                    "test_pool": [6, 7, 8],
                },
            }
        }
        with tempfile.NamedTemporaryFile(mode="w", suffix=".toml", delete=False) as f:
            path = Path(f.name)
        try:
            # Should not raise
            validate_prompt_learning_config(config_data, path)
        finally:
            path.unlink(missing_ok=True)
    
    def test_spec_fields_optional(self) -> None:
        """Test that spec fields are optional."""
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
                    "batch_size": 5,
                    "max_concurrent": 10,
                    "meta_model": "gpt-4o-mini",
                    "meta_model_provider": "openai",
                    "spec_path": "path/to/spec.json",
                    # spec_max_tokens, spec_include_examples, spec_priority_threshold are optional
                    "bootstrap_train_seeds": [0, 1, 2],
                    "online_pool": [3, 4, 5],
                    "test_pool": [6, 7, 8],
                },
            }
        }
        with tempfile.NamedTemporaryFile(mode="w", suffix=".toml", delete=False) as f:
            path = Path(f.name)
        try:
            # Should not raise
            validate_prompt_learning_config(config_data, path)
        finally:
            path.unlink(missing_ok=True)


